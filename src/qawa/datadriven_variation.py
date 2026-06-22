import correctionlib
import os
import awkward as ak
from pathlib import Path


class DataDrivenEventReweight:
    def __init__ (
            self,
            era: str = "2018",
            clibhandler = None,
    ):
        if clibhandler is not None:
            self.dd_estimator = clibhandler.getCorrectionSet("dddy").compound["LNTTau_TTau_DD_Estimate"]
        else:
            _data_path = Path(os.path.dirname(__file__)) / f"data/dd/{era}/WZ_inclusive_data_driven_{era}.json"
            assert _data_path.exists(), f"DataDrivenEventReweight could not find the expected json file: {str(_data_path)}"
            self.dd_estimator = correctionlib.CorrectionSet.from_file(str(_data_path)).compound["LNTTau_TTau_DD_Estimate"]
        
    def estimate_dd_DY(self, jet_multiplicity, tau_pt, systematic: str = None):
        if systematic is None:
            return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0.0), ak.fill_none(tau_pt, 0.0), "nominal")
        else:
            return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0.0), ak.fill_none(tau_pt, 0.0), systematic)
        # return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0), ak.fill_none(tau_pt, 0.0), systematic)

if __name__ == "__main__":
    from correctionlib import schemav2 as cs

    new_cset_corrections = [
        # --- First Correction ---
        cs.Correction(
            name="LNTTau_to_TTau_TransferFactor",
            version=1,
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
                cs.Variable(name="systematic", type="string"),
            ],
            output=cs.Variable(name="weight", type="real", description="Multiplicative weight for transfer factor estimating fake taus from Tight Tau to Loose Not Tight Tau rate"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[0.0, 0.0240346, 0.0297225, 0.0299458, 0.0317921, 0.0305019, 0.0320465, 0.0300869, 0.0429356,
                                     0.0, 0.0253023, 0.0244285, 0.0293068, 0.0287562, 0.0270025, 0.0218764, 0.0242165, 0.0357419,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYUp",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.0248818, 0.0308564, 0.0313777, 0.0336745, 0.0320314, 0.0354358, 0.0360421, 0.0518765,
                                0.0, 0.0267548, 0.0262391, 0.0318123, 0.0318713, 0.0293651, 0.0255855, 0.0307169, 0.0448783,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYDown",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.0231874, 0.0285886, 0.0285139, 0.0299097, 0.0289724, 0.0286572, 0.0241317, 0.0339947,
                                0.0, 0.0238498, 0.0226180, 0.0268013, 0.0256411, 0.0246399, 0.0181673, 0.0177161, 0.0266055,
                            ],
                            flow="clamp",
                        ),
                    },
                ],
            ),
        ),

        # --- Second Correction ---
        cs.Correction(
            name="LNTTau_HighMET_DY_to_Data_estimate",
            version=1,
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real"),
                cs.Variable(name="tau_pt", type="real"),
                cs.Variable(name="systematic", type="string"),
            ],
            output=cs.Variable(name="weight", type="real"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.992965, 0.992484, 0.991686, 0.990726, 0.98751, 0.984467, 0.980861, 0.980778,
                                0.0, 0.965156, 0.961381, 0.958212, 0.958427, 0.947141, 0.942788, 0.930479, 0.912021,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYUp",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.993336, 0.992953, 0.992303, 0.991555, 0.988307, 0.986319, 0.984485, 0.985298,
                                0.0, 0.966403, 0.96303, 0.960362, 0.961064, 0.949454, 0.946937, 0.937681, 0.920581,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYDown",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.992593, 0.992016, 0.991069, 0.989897, 0.986716, 0.982615, 0.977236, 0.976259,
                                0.0, 0.963909, 0.959732, 0.956062, 0.95579, 0.944828, 0.938639, 0.923277, 0.903461,
                            ],
                            flow="clamp",
                        ),
                    },
                ],
            ),
        ),
    ]

    new_cset_compound_corrections = [
        cs.CompoundCorrection(
            name="LNTTau_TTau_DD_Estimate",
            description="For generating data driven estimate of DrellYan background in a tight tau, high-MET region, weighting data from a loost-not-tight tau, high-MET region",
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
                cs.Variable(name="systematic", type="string", description="Systematic variation for the datadriven DY background estimation"),
            ],
            output=cs.Variable(name="weight", type="real", description="Estimated number of events coming from tau fakes of a DrellYan background"),
            inputs_update=[],
            input_op="*",
            output_op="*",
            stack=["LNTTau_to_TTau_TransferFactor", "LNTTau_HighMET_DY_to_Data_estimate"],
        )
    ]
    new_cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        corrections=new_cset_corrections,
        compound_corrections=new_cset_compound_corrections,
    )
    with open(f"WZ_inclusive_data_driven_2018_latest.json", "w") as fout:
        fout.write(new_cset.model_dump_json(exclude_unset=True))

    test =  DataDrivenEventReweight()
    import numpy as np
    test_njets = np.array([0, 1, 0, 1, 0, 1])
    test_tau_pt = np.array([ 25, 25, 80, 80, 110, 110])
    for syst in ["nominal", "DDDYUp", "DDDYDown", None] :
        try:
            test_weights = test.estimate_dd_DY(test_njets, test_tau_pt, syst)
            print("test_njets:", test_njets)
            print("test_tau_pt:", test_tau_pt)
            print("weights:", test_weights)
        except Exception as e:
            print(syst, e)

    # # Wrap them into a CorrectionSet
    # cset = cs.CorrectionSet(schema_version=2, corrections=new_cset_corrections)

    # # # Example evaluations
    # print(new_cset["LNTTau_TTau_DD_Estimate"].evaluate(1, 35.0, "nominal"))
    # # print(cset["LNTTau_to_TTau_TransferFactor"].evaluate(1, 35.0, "up"))
    # # print(cset["LNTTau_to_TTau_TransferFactor"].evaluate(1, 35.0, "down"))

    # 
