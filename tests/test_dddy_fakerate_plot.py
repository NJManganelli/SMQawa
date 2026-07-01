#!/usr/bin/env python
"""Plot test: legacy 2018 tau fake rate vs the Run2 average + per-year stat variations.

Renders the LNT->VT tau transfer factor (``LNTTau_to_VTTau_TransferFactor`` -- the tau
fake rate) as a function of ``tau_pt`` for each jet multiplicity, comparing:

* the legacy single-era **2018** correction
  (``src/qawa/data/dd/2018/WZ_inclusive_data_driven_2018.json``), and
* the new multi-era luminosity-weighted **Run2 average**
  (``src/qawa/data/dd/Run2/WZ_inclusive_data_driven_Run2.json``),

with a shaded band per era for its decorrelated statistical nuisance (``stat_{era}``).
Uses ``mplhep`` CMS styling.

Run directly to (re)generate the figure::

    python tests/test_dddy_fakerate_plot.py            # -> SMQawa/plots/dddy_fakerate_run2_vs_2018.{pdf,png}

The Run2 file can be regenerated with::

    python src/qawa/datadriven.py --derive \
        -i ../DCTools/config/inc-WZ/input_UL_2016APV-WZ_inclusive.yaml \
           ../DCTools/config/inc-WZ/input_UL_2016-WZ_inclusive.yaml \
           ../DCTools/config/inc-WZ/input_UL_2017-WZ_inclusive.yaml \
           ../DCTools/config/inc-WZ/input_UL_2018-WZ_inclusive.yaml \
        -o src/qawa/data/dd/Run2/WZ_inclusive_data_driven_Run2.json
"""
import os
from pathlib import Path

import numpy as np
import correctionlib
import matplotlib
matplotlib.use("Agg")  # headless / batch safe
import matplotlib.pyplot as plt
import mplhep as hep

from qawa.datadriven import discover_systematics

TF_NAME = "LNTTau_to_VTTau_TransferFactor"
_REPO = Path(__file__).resolve().parents[1]
_DD = _REPO / "src" / "qawa" / "data" / "dd"
OLD_2018_PATH = _DD / "2018" / "WZ_inclusive_data_driven_2018.json"
RUN2_PATH = _DD / "Run2" / "WZ_inclusive_data_driven_Run2.json"
OUT_PATH = _REPO / "plots" / "dddy_fakerate_run2_vs_2018.pdf"


def _eval_curve(corr, jet, tau_pts, systematic):
    """Evaluate a correction over a tau_pt grid at fixed jet multiplicity.

    Builds the positional args from the correction's *declared* input order so it works
    regardless of whether the file stores inputs as (tau_pt, jet, ...) or (jet, tau_pt, ...).
    """
    order = [i.name for i in corr.inputs]
    out = np.empty(len(tau_pts), dtype=float)
    for i, pt in enumerate(tau_pts):
        args = {"jet_multiplicity": float(jet), "tau_pt": float(pt), "systematic": systematic}
        out[i] = corr.evaluate(*[args[n] for n in order])
    return out


def make_plot(old_path=OLD_2018_PATH, run2_path=RUN2_PATH, out_path=OUT_PATH,
              jets=(0, 1), tau_max=150.0, n_points=300):
    old_path, run2_path, out_path = Path(old_path), Path(run2_path), Path(out_path)
    if not old_path.exists():
        raise FileNotFoundError(f"legacy 2018 correction not found: {old_path}")
    if not run2_path.exists():
        raise FileNotFoundError(f"Run2 averaged correction not found: {run2_path} (see module docstring to regenerate)")

    old_tf = correctionlib.CorrectionSet.from_file(str(old_path))[TF_NAME]
    run2_tf = correctionlib.CorrectionSet.from_file(str(run2_path))[TF_NAME]
    stat_systs, _ = discover_systematics(run2_path)

    tau = np.linspace(1.0, tau_max, n_points)
    band_colors = plt.cm.viridis(np.linspace(0.0, 0.85, max(len(stat_systs), 1)))

    hep.style.use("CMS")
    fig, axes = plt.subplots(1, len(jets), figsize=(7.0 * len(jets), 6.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, jet in zip(axes, jets):
        run2_nom = _eval_curve(run2_tf, jet, tau, "nominal")
        old_nom = _eval_curve(old_tf, jet, tau, "nominal")
        assert np.all(np.isfinite(run2_nom)) and np.all(np.isfinite(old_nom)), "non-finite fake-rate values"

        # per-era statistical bands around the Run2 nominal
        for color, syst in zip(band_colors, stat_systs):
            up = _eval_curve(run2_tf, jet, tau, f"{syst}Up")
            dn = _eval_curve(run2_tf, jet, tau, f"{syst}Down")
            ax.fill_between(tau, dn, up, alpha=0.30, color=color, linewidth=0,
                            label=syst.replace("stat_", "stat ") + " (Run2)")

        ax.plot(tau, run2_nom, color="black", lw=2.2, label="Run2 average (nominal)")
        ax.plot(tau, old_nom, color="tab:red", lw=2.0, ls="--", label="2018 legacy (nominal)")

        ax.set_xlabel(r"$p_{T}^{\tau}$ [GeV]")
        ax.text(0.95, 0.95, rf"$N_\mathrm{{jet}} = {jet}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=15)
        ax.set_xlim(0, tau_max)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize="x-small", ncol=1, loc="upper left")

    axes[0].set_ylabel(r"LNT$\to$VT $\tau$ fake rate")
    hep.cms.label(ax=axes[0], text="Preliminary", data=True, year="Run 2", com=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path))
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150)
    plt.close(fig)
    return out_path


def test_dddy_fakerate_plot():
    """Smoke test: the comparison figure renders and is written to disk."""
    out = make_plot()
    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".png").exists()


if __name__ == "__main__":
    out = make_plot()
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
