"""Trigger scale factor and b-tag efficiency measurements: correctionlib payload writers."""
import numpy as np
from coffea.util import coffea_console
from correctionlib import schemav2 as cs

TRIGGER_SF_FLAVOURS = ("mm", "me", "em", "ee")
TRIGGER_SF_SYSTEMATICS = ("nominal", "up", "down", "alpha_up", "alpha_down")
BTAG_EFF_SYSTEMATICS = ("nominal", "up", "down")
BTAG_EFF_FLAVOURS = (0, 4, 5)


def multibinning(inputs, edges, content):
    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=list(inputs),
        edges=[[float(e) for e in axis] for axis in edges],
        content=[float(v) for v in np.asarray(content, dtype=float).ravel(order="C")],
        flow="clamp",
    )


def trigger_sf_payload(grids, edges, description):
    """grids[flavour][systematic] is an (n_lead, n_trail) array over edges = (lead_edges, trail_edges)."""
    corrections = []
    for flavour in TRIGGER_SF_FLAVOURS:
        corrections.append(cs.Correction(
            name=f"SF_{flavour}",
            description=f"Di-lepton trigger scale factor (leading={flavour[0]}, trailing={flavour[1]})",
            version=1,
            inputs=[
                cs.Variable(name="lead_pt", type="real", description="Leading lepton pT [GeV]"),
                cs.Variable(name="trail_pt", type="real", description="Trailing lepton pT [GeV]"),
                cs.Variable(name="systematic", type="string", description=" / ".join(TRIGGER_SF_SYSTEMATICS)),
            ],
            output=cs.Variable(name="weight", type="real", description="Scale factor weight"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[cs.CategoryItem(key=systematic,
                                         value=multibinning(("lead_pt", "trail_pt"), edges, grids[flavour][systematic]))
                         for systematic in TRIGGER_SF_SYSTEMATICS],
            ),
        ))
    return cs.CorrectionSet(schema_version=2, description=description, corrections=corrections)


def btag_eff_payload(maps, edges, wps):
    """maps[process][systematic][wp][flavour] is an (n_pt, n_abseta) array over edges = (pt_edges, abseta_edges)."""
    def flavour_node(by_flavour):
        return cs.Category(
            nodetype="category",
            input="hadronFlavour",
            content=[cs.CategoryItem(key=flavour, value=multibinning(("pt", "abseta"), edges, by_flavour[flavour]))
                     for flavour in BTAG_EFF_FLAVOURS],
        )

    def category(name, keys, value):
        return cs.Category(nodetype="category", input=name,
                           content=[cs.CategoryItem(key=key, value=value(key)) for key in keys])

    data = category("process", sorted(maps), lambda process: category(
        "systematic", BTAG_EFF_SYSTEMATICS, lambda systematic: category(
            "working_point", wps, lambda wp: flavour_node(maps[process][systematic][wp]))))
    correction = cs.Correction(
        name="btag_eff",
        description="b-tagging efficiency per process, working point and hadron flavour",
        version=1,
        inputs=[
            cs.Variable(name="process", type="string", description="dataset name, or inclusive"),
            cs.Variable(name="systematic", type="string", description=" / ".join(BTAG_EFF_SYSTEMATICS)),
            cs.Variable(name="working_point", type="string", description="fixed working point"),
            cs.Variable(name="hadronFlavour", type="int", description="0 (light), 4 (c), 5 (b)"),
            cs.Variable(name="pt", type="real", description="jet pT [GeV]"),
            cs.Variable(name="abseta", type="real", description="jet |eta|"),
        ],
        output=cs.Variable(name="efficiency", type="real", description="tagging efficiency"),
        data=data,
    )
    return cs.CorrectionSet(schema_version=2, description="b-tagging efficiencies", corrections=[correction])


def write_payload(cset, path):
    with open(path, "w") as fout:
        fout.write(cset.model_dump_json(exclude_unset=True))
    coffea_console.print(f"Wrote {path}. A payload at the jsoncorrections location under src/qawa/data "
                         "is picked up by every later run of that era.", markup=False)
