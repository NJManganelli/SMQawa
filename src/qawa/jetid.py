import operator

import awkward as ak
import numpy as np

LT, GT, GE = operator.lt, operator.gt, operator.ge
INF = float("inf")

# AK4 PUPPI jet ID: a region spans |eta| above the previous abseta_max; Tight needs its tight cuts, TightLepVeto also lep_veto.
# UL*: TWiki JetID13TeVUL NanoAODv15 recipe, |eta| <= abseta_max. Summer24: the 2024 jetid.json, |eta| < abseta_max.
JET_ID_RECIPES = {
    "UL16PUPPI": dict(upper_inclusive=True, regions=[
        dict(abseta_max=2.4,
             tight=[("neHEF", LT, 0.9), ("neEmEF", LT, 0.9), ("multiplicity", GT, 1), ("chHEF", GT, 0.0), ("chMultiplicity", GT, 0)],
             lep_veto=[("muEF", LT, 0.8), ("chEmEF", LT, 0.8)]),
        dict(abseta_max=2.7, tight=[("neHEF", LT, 0.98), ("neEmEF", LT, 0.99)], lep_veto=[]),
        dict(abseta_max=3.0, tight=[("neMultiplicity", GE, 1)], lep_veto=[]),
        dict(abseta_max=INF, tight=[("neEmEF", LT, 0.9), ("neMultiplicity", GT, 2)], lep_veto=[]),
    ]),
    "UL1718PUPPI": dict(upper_inclusive=True, regions=[
        dict(abseta_max=2.6,
             tight=[("neHEF", LT, 0.9), ("neEmEF", LT, 0.9), ("multiplicity", GT, 1), ("chHEF", GT, 0.0), ("chMultiplicity", GT, 0)],
             lep_veto=[("muEF", LT, 0.8), ("chEmEF", LT, 0.8)]),
        dict(abseta_max=2.7, tight=[("neHEF", LT, 0.9), ("neEmEF", LT, 0.99)], lep_veto=[("muEF", LT, 0.8), ("chEmEF", LT, 0.8)]),
        dict(abseta_max=3.0, tight=[("neHEF", LT, 0.9999)], lep_veto=[]),
        dict(abseta_max=INF, tight=[("neEmEF", LT, 0.9), ("neMultiplicity", GT, 2)], lep_veto=[]),
    ]),
    "Summer24PUPPI": dict(upper_inclusive=False, regions=[
        dict(abseta_max=2.6,
             tight=[("chHEF", GE, 0.01), ("neHEF", LT, 0.99), ("neEmEF", LT, 0.9), ("chMultiplicity", GE, 1), ("multiplicity", GE, 2)],
             lep_veto=[("muEF", LT, 0.8), ("chEmEF", LT, 0.8)]),
        dict(abseta_max=2.7, tight=[("neHEF", LT, 0.9), ("neEmEF", LT, 0.99)], lep_veto=[("muEF", LT, 0.8), ("chEmEF", LT, 0.8)]),
        dict(abseta_max=3.0, tight=[("neHEF", LT, 0.99)], lep_veto=[]),
        dict(abseta_max=INF, tight=[("neEmEF", LT, 0.4), ("neMultiplicity", GE, 2)], lep_veto=[]),
    ]),
}

JET_FIELDS = ("eta", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity")


def jet_id_recipe(era, nanoAODversion):
    if era == "2016":
        return "UL16PUPPI"
    if era in ("2017", "2018"):
        return "UL1718PUPPI"
    if era == "2024":
        return "Summer24PUPPI"
    raise NotImplementedError(f"no AK4 PUPPI jet ID recipe for era {era}, NanoAOD {nanoAODversion}")


def recipe_jet_id(jets, recipe):
    """NanoAOD jetId encoding (2 = Tight, 4 = TightLepVeto) for the named recipe in JET_ID_RECIPES."""
    config = JET_ID_RECIPES[recipe]
    abseta = np.abs(jets.eta)
    multiplicity = jets.chMultiplicity + jets.neMultiplicity

    def passes(mask, cuts):
        for name, op, cut in cuts:
            mask = mask & op(multiplicity if name == "multiplicity" else jets[name], cut)
        return mask

    tight = ak.zeros_like(abseta, dtype=bool)
    lep_veto = ak.zeros_like(abseta, dtype=bool)
    below_previous = ak.zeros_like(abseta, dtype=bool)
    for region in config["regions"]:
        below = abseta <= region["abseta_max"] if config["upper_inclusive"] else abseta < region["abseta_max"]
        region_tight = passes(below & ~below_previous, region["tight"])
        tight = tight | region_tight
        lep_veto = lep_veto | passes(region_tight, region["lep_veto"])
        below_previous = below
    return 2 * ak.values_astype(tight, np.int32) + 4 * ak.values_astype(lep_veto, np.int32)


class JetID:
    """AK4 jet ID in the NanoAOD jetId encoding, from the stored branch, the jetid.json, or JET_ID_RECIPES."""

    def __init__(self, era, nanoAODversion, clibhandler=None):
        self.stored = nanoAODversion in [f"v{V}" for V in range(12)]
        self.cset = None
        self.recipe = None
        if self.stored:
            return
        if clibhandler is not None:
            try:
                self.cset = clibhandler.getCorrectionSet("jetid")
            except KeyError:
                self.cset = None
        if self.cset is None:
            self.recipe = jet_id_recipe(era, nanoAODversion)

    def __call__(self, jets):
        if self.stored:
            return jets.jetId
        if self.cset is None:
            return recipe_jet_id(jets, self.recipe)
        values = {name: jets[name] for name in JET_FIELDS}
        values["multiplicity"] = jets.chMultiplicity + jets.neMultiplicity
        tight = self.cset["AK4PUPPI_Tight"]
        lep_veto = self.cset["AK4PUPPI_TightLeptonVeto"]
        return ak.values_astype(
            2 * tight.evaluate(*[values[i.name] for i in tight.inputs])
            + 4 * lep_veto.evaluate(*[values[i.name] for i in lep_veto.inputs]),
            np.int32,
        )
