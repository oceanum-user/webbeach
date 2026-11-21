#!/usr/bin/env python3
"""Assemble the verification report from per-case metrics.json + plots.
Usage: report.py <suite.json> — writes verification/results/REPORT.md and
plots for each case's worst + key variables."""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "verification" / "results"

CASE_DESCRIPTIONS = {
    "stat1d_slope": "1D stationary waves, planar beach, Baldock breaker + roller",
    "stat1d_janssen": "as stat1d_slope with Janssen-Battjes breaker",
    "stat1d_norollerfric": "as stat1d_slope, roller off, wave friction fw=0.05",
    "stat1d_upwind1": "as stat1d_slope with first-order upwind scheme",
    "stat2d_refr": "2D stationary oblique waves (dir0=300, m=16), refraction on planar beach",
    "smoke1d": "1D stationary waves + flow: setup/setdown balance",
    "flow2d_longshore": "2D oblique stationary waves + flow: longshore current",
    "surfbeat1d_bichrom": "1D surfbeat, bichromatic groups + bound IG wave",
    "surfbeat1d_jons": "1D surfbeat, JONSWAP spectrum (identical E/q boundary files)",
    "sb1d_roelvink1": "surfbeat formulation: break=roelvink1",
    "sb1d_roelvink2": "surfbeat formulation: break=roelvink2",
    "sb1d_daly": "surfbeat formulation: break=roelvink_daly (advected Qb)",
    "sb1d_janssen": "surfbeat formulation: break=janssen",
    "sb1d_norol": "surfbeat formulation: roller=0",
    "sb1d_abs2d": "surfbeat 1D with front=abs_2d (reduces to abs_1d as in Fortran)",
    "rip2d": "2D surfbeat rip current: barred beach with channel, abs_2d front",
    "rip2d_abs1d": "rip case variant with abs_1d front",
}

PLOT_VARS = {
    "stat1d_slope": ["H", "D", "Fx"],
    "stat2d_refr": ["H", "thetamean"],
    "smoke1d": ["zs", "u"],
    "flow2d_longshore": ["ve", "H"],
    "surfbeat1d_bichrom": ["zs_var", "H_mean", "ue_mean"],
    "surfbeat1d_jons": ["zs_var", "H_mean"],
    "rip2d": ["ue_mean", "zs_mean", "H_mean"],
}


def main():
    suite = json.loads(Path(sys.argv[1]).read_text())
    rows = []
    for case in suite["cases"]:
        mj = RESULTS / case["name"] / "metrics.json"
        if not mj.exists():
            rows.append((case["name"], None))
            continue
        rows.append((case["name"], json.loads(mj.read_text())))

    lines = [
        "# Verification report: WebGPU XBeach vs Fortran reference",
        "",
        f"Date: {date.today().isoformat()}. Metrics computed on the final output",
        "record of each case over the interior comparison region (last two",
        "s-rows and one lateral row per side excluded). nRMSE = RMSE normalised",
        "by the reference field range; pass criterion nRMSE <= tolerance",
        "(per-variable, see suite definition). Reference: XBeach trunk r-latest",
        "built from this checkout (f64); WebGPU model computes in f32.",
        "",
        "| case | description | result | worst var (nRMSE / tol) |",
        "|---|---|---|---|",
    ]
    npass = 0
    for name, res in rows:
        desc = CASE_DESCRIPTIONS.get(name, "")
        if res is None:
            lines.append(f"| {name} | {desc} | MISSING | |")
            continue
        ok = res["pass"]
        npass += ok
        worst = max(
            res["vars"].items(), key=lambda kv: kv[1]["nrmse_pct"] / kv[1]["tol_pct"]
        )
        lines.append(
            f"| {name} | {desc} | {'PASS' if ok else 'FAIL'} | "
            f"{worst[0]} ({worst[1]['nrmse_pct']:.3f}% / {worst[1]['tol_pct']}%) |"
        )
    lines.append("")
    lines.append(f"**{npass}/{len(rows)} cases pass.**")
    lines.append("")
    lines.append("## Per-case metrics")
    for name, res in rows:
        if res is None:
            continue
        lines.append(f"\n### {name}\n")
        lines.append("| var | nRMSE % | bias | max abs diff | r | pass |")
        lines.append("|---|---|---|---|---|---|")
        for v, m in sorted(res["vars"].items()):
            r = m.get("r")
            lines.append(
                f"| {v} | {m['nrmse_pct']:.4f} | {m['bias']:+.2e} | "
                f"{m['maxabs']:.2e} | {'' if r is None else f'{r:.5f}'} | "
                f"{'y' if m['pass'] else 'N'} |"
            )
        plots = PLOT_VARS.get(name)
        if plots:
            subprocess.run(
                [sys.executable, str(ROOT / "verification/compare/plots.py"), name, *plots],
                check=False,
                capture_output=True,
            )
            for v in plots:
                p = RESULTS / name / "plots" / f"{v}.png"
                if p.exists():
                    lines.append(f"\n![{name} {v}](./{name}/plots/{v}.png)")
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS / 'REPORT.md'} — {npass}/{len(rows)} pass")


if __name__ == "__main__":
    main()
