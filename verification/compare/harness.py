#!/usr/bin/env python3
"""
Verification harness: run Fortran + WebGPU on case variants, compare, report.

Usage:
  python3 verification/compare/harness.py <suite.json> [--only NAME] [--skip-fortran]

suite.json:
{
  "cases": [
    {"name": "stat1d_baldock", "base": "stat1d_slope",
     "overrides": {"break": "baldock"},
     "vars": ["H","D","Qb"], "tol": {"H": 1.0, "default": 2.0},
     "records": "last"}
  ]
}

Metrics per variable (comparison region excludes the last two s-rows and, for
2D, one lateral row each side): rmse, nrmse (% of reference range), bias,
maxabs, r (Pearson). Pass iff nrmse <= tol[var] (or tol.default).
Artifacts: verification/results/<name>/{webgpu outputs, metrics.json, plots}.

Optional per-case keys:
  "ref": compare against a different committed reference dir (default: the
    case name). Fortran is never (re)run for a row whose ref != name.
  "copy_bcf": false skips copying the reference *.bcf into the case dir
    (default true), so run.ts exercises the TS wavemaker generation branch
    instead of reuse. Derived-name rows only (name != base) — the derived
    work dir is scrubbed of stale *.bcf; a name==base row would risk
    deleting committed fixture files, so the scrub is skipped there.
The surfbeat1d_jons_r0_gen row in suites/all.json is the generation-path
parity gate (random=0, TS tables vs the committed r0 Fortran reference
fields). Like every harness row it needs a WebGPU-capable deno — local
only, not CI; the CPU table-parity gate that DOES run in CI is
test/wavemaker_parity.test.ts. Run it with:
  python3 verification/compare/harness.py verification/suites/all.json \
      --only surfbeat1d_jons_r0_gen --skip-fortran
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "verification" / "cases"
REF = ROOT / "verification" / "reference"
RESULTS = ROOT / "verification" / "results"


def read_params(path):
    d = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith(("%", "#", "!")) or "=" not in s:
            continue
        k, v = s.split("=", 1)
        d[k.strip()] = v.strip()
    return d


def write_case(base_dir, out_dir, overrides, extra_vars):
    out_dir.mkdir(parents=True, exist_ok=True)
    # keep the wave-bc file the base params reference (e.g. jonswap.txt)
    keep_txt = {"params.txt"}
    bcfile = read_params(base_dir / "params.txt").get("bcfile")
    if bcfile:
        keep_txt.add(bcfile)
    for f in base_dir.iterdir():
        if f.suffix in (".dat", ".txt") and f.name not in keep_txt:
            continue
        if f.name == "params.txt":
            continue
        shutil.copy(f, out_dir / f.name)
    lines = base_dir.joinpath("params.txt").read_text().splitlines()
    # split off output var block (keep anything after it, e.g. nmeanvar block)
    try:
        nv_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("nglobalvar"))
        nvars = int(lines[nv_idx].split("=")[1])
        var_block = [l.strip() for l in lines[nv_idx + 1 : nv_idx + 1 + nvars]]
        head = lines[:nv_idx]
        tail = lines[nv_idx + 1 + nvars :]
    except StopIteration:
        head, var_block, tail = lines, [], []
    # apply overrides
    keys_done = set()
    out_lines = []
    for l in head:
        s = l.strip()
        replaced = False
        for k, v in overrides.items():
            if s.lower().startswith(k.lower() + " ") or s.lower().startswith(k.lower() + "="):
                if v is None:
                    replaced = True  # drop line
                else:
                    out_lines.append(f"{k:<10s} = {v}")
                    replaced = True
                keys_done.add(k)
                break
        if not replaced:
            out_lines.append(l)
    for k, v in overrides.items():
        if k not in keys_done and v is not None:
            out_lines.append(f"{k:<10s} = {v}")
    for v in extra_vars:
        if v not in var_block:
            var_block.append(v)
    out_lines.append(f"nglobalvar = {len(var_block)}")
    out_lines.extend(var_block)
    out_lines.extend(tail)
    (out_dir / "params.txt").write_text("\n".join(out_lines) + "\n")
    return var_block


def deno_bin():
    """deno on PATH, falling back to the historical ~/.deno/bin/deno."""
    return shutil.which("deno") or str(Path.home() / ".deno/bin/deno")


def interior_mask(nx, ny):
    """Flat comparison mask: drop the last two s-rows and, in 2D, one lateral
    row each side. Shared with verification/compare/mission.py so both
    harnesses score on exactly the same region."""
    mask2d = np.ones((ny + 1, nx + 1), bool)
    mask2d[:, -2:] = False
    if ny > 0:
        mask2d[0, :] = False
        mask2d[-1, :] = False
    return mask2d.ravel()


def load_var(d, v, n):
    return np.fromfile(d / f"{v}.dat").reshape(-1, n)


def metrics(f, w):
    d = w - f
    rng = float(f.max() - f.min())
    rmse = float(np.sqrt((d**2).mean()))
    # degenerate reference (constant/zero field): fall back to absolute scale
    denom = rng if rng > 1e-6 else 1.0
    out = {
        "rmse": rmse,
        "nrmse_pct": 100 * rmse / denom,
        "bias": float(d.mean()),
        "maxabs": float(np.abs(d).max()),
        "degenerate_range": bool(rng <= 1e-6),
    }
    if f.std() > 1e-12 and w.std() > 1e-12:
        out["r"] = float(np.corrcoef(f.ravel(), w.ravel())[0, 1])
    return out


def run_case(case, skip_fortran=False):
    name = case["name"]
    ref_name = case.get("ref", name)
    base = CASES / case["base"]
    work_case = CASES / name
    extra_vars = case.get("vars", [])
    if work_case != base:
        var_block = write_case(base, work_case, case.get("overrides", {}), extra_vars)
    else:
        var_block = [
            v for v in read_params(base / "params.txt")
        ]  # unused; vars read from output files
    p = read_params(work_case / "params.txt")
    nx, ny = int(p["nx"]), int(p["ny"])
    n = (nx + 1) * (ny + 1)

    if ref_name == name and (not skip_fortran or not (REF / name).exists()):
        subprocess.run(
            [str(ROOT / "fortran" / "run_case.sh"), name],
            check=True,
            capture_output=True,
        )
    # spectral wave BCs: the WebGPU model must read the exact E_series/q_series
    # realization the Fortran run just generated (random seed differs per run).
    # copy_bcf=false instead exercises run.ts's TS-generation branch (only
    # valid with random=0, where TS and Fortran produce the same realization).
    if case.get("copy_bcf", True):
        for bcf in (REF / ref_name).glob("*.bcf"):
            shutil.copy(bcf, work_case / bcf.name)
    elif work_case != base:
        for bcf in work_case.glob("*.bcf"):
            bcf.unlink()
    out_dir = RESULTS / name
    r = subprocess.run(
        [
            deno_bin(), "run", "--allow-all",
            str(ROOT / "src/cli/run.ts"), str(work_case), "--out", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return {"name": name, "error": (r.stderr or r.stdout)[-2000:]}

    tol = case.get("tol", {})
    default_tol = tol.get("default", 2.0)
    variables = case.get("compare", None) or [
        f.stem for f in out_dir.glob("*.dat")
    ]
    res = {"name": name, "vars": {}, "pass": True}
    # comparison mask: drop last two s-rows; drop one lateral row each side (2D)
    mask = interior_mask(nx, ny)
    for v in variables:
        fr = REF / ref_name / f"{v}.dat"
        if not fr.exists():
            # OCE-247: a listed variable with no reference file is a FAIL,
            # not a skip — the roller-energy mean went uncompared for a year
            # because this silently passed over it
            res["vars"][v] = {"error": f"no reference file {fr.name}", "pass": False}
            res["pass"] = False
            continue
        f = load_var(REF / ref_name, v, n)[-1][mask]
        w = load_var(out_dir, v, n)[-1][mask]
        m = metrics(f, w)
        vtol = tol.get(v, default_tol)
        m["tol_pct"] = vtol
        m["pass"] = bool(m["nrmse_pct"] <= vtol)
        if not m["pass"]:
            res["pass"] = False
        res["vars"][v] = m
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    suite_path = sys.argv[1]
    only = None
    skip_fortran = "--skip-fortran" in sys.argv
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    suite = json.loads(Path(suite_path).read_text())
    rows = []
    for case in suite["cases"]:
        if only and case["name"] != only:
            continue
        print(f"=== {case['name']}")
        res = run_case(case, skip_fortran)
        rows.append(res)
        if "error" in res:
            print("  ERROR:", res["error"][:500])
            continue
        for v, m in res["vars"].items():
            flag = "PASS" if m["pass"] else "FAIL"
            print(
                f"  {v:10s} nrmse={m['nrmse_pct']:8.4f}% (tol {m['tol_pct']}%) "
                f"maxabs={m['maxabs']:.2e} {flag}"
            )
        print(f"  => {'PASS' if res['pass'] else 'FAIL'}")
    # summary table
    summary = RESULTS / "summary.md"
    with open(summary, "a") as fh:
        for r in rows:
            if "error" in r:
                fh.write(f"| {r['name']} | ERROR | | |\n")
                continue
            if not r["vars"]:
                fh.write(f"| {r['name']} | NO-DATA | | |\n")
                continue
            worst = max(r["vars"].items(), key=lambda kv: kv[1]["nrmse_pct"] / kv[1]["tol_pct"])
            fh.write(
                f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | "
                f"worst {worst[0]} {worst[1]['nrmse_pct']:.3f}% | tol {worst[1]['tol_pct']}% |\n"
            )
    nfail = sum(1 for r in rows if not r.get("pass", False))
    print(f"\n{len(rows) - nfail}/{len(rows)} cases pass")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
