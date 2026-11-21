#!/usr/bin/env python3
"""Generate comparison plots (Fortran vs WebGPU) for verification cases.
Usage: plots.py <case> <var> [<var>...] [--ny NY] — writes PNGs into
verification/results/<case>/plots/."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

def read_params(path):
    d = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith(("%", "#")) and "=" in s:
            k, v = s.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def main():
    case = sys.argv[1]
    vars_ = [a for a in sys.argv[2:] if not a.startswith("--")]
    p = read_params(ROOT / "verification/cases" / case / "params.txt")
    nx, ny = int(p["nx"]), int(p["ny"])
    n = (nx + 1) * (ny + 1)
    dx = float(p.get("dx", 1))
    ref = ROOT / "verification/reference" / case
    web = ROOT / "verification/results" / case
    out = web / "plots"
    out.mkdir(parents=True, exist_ok=True)
    for v in vars_:
        try:
            f = np.fromfile(ref / f"{v}.dat").reshape(-1, ny + 1, nx + 1)[-1]
            w = np.fromfile(web / f"{v}.dat").reshape(-1, ny + 1, nx + 1)[-1]
        except FileNotFoundError:
            print(f"skip {v} (missing)")
            continue
        if ny == 0:
            x = np.arange(nx + 1) * dx
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                           height_ratios=[3, 1])
            ax1.plot(x, f[0], "k-", lw=1.8, label="XBeach (Fortran)")
            ax1.plot(x, w[0], "r--", lw=1.2, label="WebGPU")
            ax1.set_ylabel(v); ax1.legend(); ax1.grid(alpha=.3)
            ax1.set_title(f"{case}: {v}")
            ax2.plot(x, w[0] - f[0], "b-", lw=0.9)
            ax2.set_ylabel("diff"); ax2.set_xlabel("x [m]"); ax2.grid(alpha=.3)
        else:
            fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
            vmin, vmax = np.percentile(f, [1, 99])
            for ax, data, title in zip(axs[:2], [f, w], ["Fortran", "WebGPU"]):
                im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                               extent=[0, (nx+1)*dx, 0, (ny+1)*float(p.get("dy",1))])
                ax.set_title(f"{v} — {title}")
                fig.colorbar(im, ax=ax, shrink=0.85)
            d = w - f
            lim = np.abs(d).max() or 1
            im = axs[2].imshow(d, origin="lower", aspect="auto", cmap="RdBu_r",
                               vmin=-lim, vmax=lim,
                               extent=[0, (nx+1)*dx, 0, (ny+1)*float(p.get("dy",1))])
            axs[2].set_title("difference")
            fig.colorbar(im, ax=axs[2], shrink=0.85)
        fig.savefig(out / f"{v}.png", dpi=110)
        plt.close(fig)
        print(f"wrote {out / f'{v}.png'}")

if __name__ == "__main__":
    main()
