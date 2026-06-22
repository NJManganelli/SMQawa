# Ref: https://gitlab.cern.ch/cms-analysis/general/HiggsDNA/-/blob/86b60a287f2b579fdf8287a5e16a7d353ff23dba/higgs_dna/systematics/EGM_SS_systematics.py
import numpy as np
import awkward as ak
import correctionlib
import os
import sys

# sink is needed for expected "pt" argument in the add_systematic callable function, so that all others may be set; it can be set to anything since it's unused
# def EGM_Scale_Trad(sink, events, year=None, is_correction=True, restriction=None, is_electron=False, clibhandler=None):
#     if year in ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
#         if is_electron:
#             cset_name = f"electronSS_EtDependent"
#             egm_object = events.Electron
#         else:
#             cset_name = f"photonSS_EtDependent"
#             egm_object = events.Photon

#         if hasattr(egm_object, "eCorr"):
#             raise ValueError("The EGM_Scale_Trad correction is not compatible with NanoAODv9 Run2 UL samples.")
#     else:
#         raise NotImplementedError(f"Got unexpected era {year}")

#     # for later unflattening:
#     counts = ak.num(egm_object.pt)

#     run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
#     gain = ak.flatten(egm_object.seedGain)
#     SCeta = ak.flatten(egm_object.superclusterEta) if "superclusterEta" in egm_object.fields else ak.flatten(egm_object.eta + egm_object.deltaEtaSC)
#     r9 = ak.flatten(egm_object.r9)
#     pt_raw = ak.flatten(egm_object.pt_raw) if "pt_raw" in egm_object.fields else ak.flatten(egm_object.pt)

#     try:
#         cset = clibhandler.getCorrectionSet(cset_name)
#         scale_evaluator = cset.compound["Scale"]
#         smear_and_syst_evaluator = cset["SmearAndSyst"]
#     except KeyError as ke:
#         raise KeyError(f"Unable to locatione {cset_name} in clibhandler with available keys: {clibhandler.keys()}")
#     # REFERENCE:
#     # DATA
#     # scale = scale_evaluator.evaluate("scale", data_run, data_electrons.ScEta, data_electrons.r9, data_electrons.pt, data_electrons.seedGain,)
#     # print(f"Multiplicative scale (for data): {scale}")
#     # # -- Apply the scale to the electron pt
#     # data_pt_corrected = scale * data_electrons.pt
#     # # -- Smearing also affects the energy uncertainty in Data.
#     # smear = smear_and_syst_evaluator.evaluate("smear", data_electrons.pt * scale, data_electrons.r9, data_electrons.ScEta)
#     # data_energyErr_corrected = np.sqrt((data_electrons.energyErr)**2 + (data_electrons.energy * smear)**2) * scale

#     # MC
#     # smear = smear_and_syst_evaluator.evaluate("smear", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # # -- Calculate the nominal smearing factor.
#     # # Since the smearing is stochastic, a random number is needed for each event.
#     # random_numbers = rng.normal(loc=0.0, scale=1.0, size=len(mc_electrons.pt))
#     # smearing = 1 + smear * random_numbers
#     # mc_pt_corrected_nominal = mc_electrons.pt * smearing
#     # # -- Smearing also affects the energy uncertainty in MC.
#     # mc_energyErr_corrected = np.sqrt((mc_electrons.energyErr)**2 + (mc_electrons.energy * smear)**2) * smearing

#     # unc_smear = smear_and_syst_evaluator.evaluate("esmear", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # smear_up = smear_and_syst_evaluator.evaluate("smear_up", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # smear_down = smear_and_syst_evaluator.evaluate("smear_down", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # # Note: The relations are simple: smear_up = smear + unc_smear and smear_down = max(0, smear - unc_smear).
#     # # Note 2: In 2022, the "smear_down" variation can lead to negative smearing width in some cases, which is unphysical.
#     # # Therefore, we use max(smear - unc_smear, 0) to ensure the smearing width is non-negative.

#     # smearing_up   = 1 + smear_up * random_numbers  # we use the same random numbers as for the nominal smearing
#     # smearing_down = 1 + smear_down * random_numbers

#     # mc_pt_corrected_smearing_up   = mc_electrons.pt * smearing_up
#     # mc_pt_corrected_smearing_down = mc_electrons.pt * smearing_down

#     # # -- Smearing uncertainties also affects the energy uncertainty in MC
#     # mc_energyErr_corrected_smearing_up = np.sqrt((mc_electrons.energyErr)**2 + (mc_electrons.energy * smear_up)**2) * smearing_up
#     # mc_energyErr_corrected_smearing_down = np.sqrt((mc_electrons.energyErr)**2 + (mc_electrons.energy * smear_down)**2) * smearing_down

#     # # We now turn to the scale uncertainty, which is also evaluated on MC original variables (pt, r9, and ScEta) BUT applied on the smeared pt.
#     # unc_scale = smear_and_syst_evaluator.evaluate("escale", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # scale_up = smear_and_syst_evaluator.evaluate("scale_up", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # scale_down = smear_and_syst_evaluator.evaluate("scale_down", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
#     # # Note: The relations are simple: scale_up = 1 + unc_scale and scale_down = 1 - unc_scale.
#     # mc_pt_corrected_scale_up   = scale_up * mc_pt_corrected_nominal
#     # mc_pt_corrected_scale_down = scale_down * mc_pt_corrected_nominal
#     # # -- Scale uncertainties also affects the energy uncertainty in MC
#     # mc_energyErr_corrected_scale_up = mc_energyErr_corrected * scale_up
#     # mc_energyErr_corrected_scale_down = mc_energyErr_corrected * scale_down

#     if is_correction:
#         # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
#         if hasattr(events, "GenPart"):
#             raise ValueError("Scale corrections should only be applied to data!")

#         correction = scale_evaluator.evaluate("scale", run, SCeta, r9, pt_raw, gain)
#         pt_corr = pt_raw * correction

#         corrected_egm_object = egm_object
#         pt_corr = ak.unflatten(pt_corr, counts)
#         corrected_egm_object["eCorr"] = correction
#         corrected_egm_object["pt"] = pt_corr

#         if is_electron:
#             events["Electron"] = corrected_egm_object
#         else:
#             events["Photon"] = corrected_egm_object

#         return events

#     else:
#         if not hasattr(events, "GenPart"):
#             raise ValueError("Scale uncertainties should only be applied to MC!")

#         if is_electron:
#             uncertainty_up = smear_and_syst_evaluator.evaluate("scale_up", pt_raw, r9, SCeta)
#             uncertainty_down = smear_and_syst_evaluator.evaluate("scale_down", pt_raw, r9, SCeta)
#         else:
#             # Conservative scale uncertainties without Zmmg corrections
#             if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
#                 uncertainty_up = 1.005 * np.ones_like(ak.to_numpy(pt_raw))
#                 uncertainty_down = 0.995 * np.ones_like(ak.to_numpy(pt_raw))
#                 # logger.warning("Using conservative scale uncertainties of 0.5% to cover electron/photon energy scale discrepancies for Run2 samples \n")
#             else:
#                 uncertainty_up = 1.01 * np.ones_like(ak.to_numpy(pt_raw))
#                 uncertainty_down = 0.99 * np.ones_like(ak.to_numpy(pt_raw))
#                 # logger.warning("Using conservative scale uncertainties of 1% to cover electron/photon energy scale discrepancies for Run3 samples \n")

#         # Apply restriction if needed
#         if restriction is not None:
#             if restriction == "EB":
#                 uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEB))

#             elif restriction == "EE":
#                 uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEE))

#             uncertainty_up = np.where(uncMask, uncertainty_up, np.zeros_like(uncertainty_up))
#             uncertainty_down = np.where(uncMask, uncertainty_down, np.zeros_like(uncertainty_down))

#         corr_up_variation = uncertainty_up
#         corr_down_variation = uncertainty_down

#         # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
#         return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * pt_raw[:, None]


# def EGM_Smearing_Trad(sink, events, year="2022postEE", is_correction=True, is_electron=False, clibhandler=None):
#     if year in ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
#         if is_electron:
#             cset_name = f"electronSS_EtDependent"
#             egm_object = events.Electron
#         else:
#             cset_name = f"photonSS_EtDependent"
#             egm_object = events.Photon

#         if hasattr(egm_object, "eCorr"):
#             raise ValueError("The EGM_Scale_Trad correction is not compatible with NanoAODv9 Run2 UL samples.")
#     else:
#         raise NotImplementedError(f"Got unexpected era {year}")

#     # for later unflattening:
#     counts = ak.num(egm_object.pt)

#     SCeta = ak.flatten(egm_object.superclusterEta) if "superclusterEta" in egm_object.fields else ak.flatten(egm_object.eta + egm_object.deltaEtaSC)
#     # SCeta = ak.flatten(egm_object.ScEta) if "ScEta" in egm_object.fields else ak.flatten(egm_object.superclusterEta)
#     r9 = ak.flatten(egm_object.r9)
#     pt_raw = ak.flatten(egm_object.pt_raw) if "pt_raw" in egm_object.fields else ak.flatten(egm_object.pt)

#     try:
#         cset = clibhandler.getCorrectionSet(cset_name)
#         smear_and_syst_evaluator = cset["SmearAndSyst"]
#     except KeyError as ke:
#         raise KeyError(f"Unable to locatione {cset_name} in clibhandler with available keys: {clibhandler.keys()}")

#     # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
#     if len(SCeta) > 0:
#         seed = abs(np.float32(SCeta[0]).view("int32"))
#     else:
#         seed = 42
#     rng = np.random.default_rng(seed=seed)

#     smearing = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, SCeta)

#     if is_correction:
#         smearing_factor = rng.normal(loc=1., scale=smearing)
#         pt_corr = pt_raw * smearing_factor
#         corrected_egm_object = egm_object
#         pt_corr = ak.unflatten(pt_corr, counts)
#         rho_corr = ak.unflatten(smearing, counts)

#         # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
#         if hasattr(events, "GenPart"):  # this operation is here because if there is no "events.GenPart" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
#             corrected_egm_object["pt"] = pt_corr

#         corrected_egm_object["rho_smear"] = rho_corr

#         if is_electron:
#             events["Electron"] = corrected_egm_object
#         else:
#             events["Photon"] = corrected_egm_object
#         return events

#     else:
#         smear_up = smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, SCeta)
#         smear_down = smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, SCeta)

#         corr_up_variation = rng.normal(loc=1., scale=smear_up)
#         corr_down_variation = rng.normal(loc=1., scale=smear_down)

#         # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
#         return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * pt_raw[:, None]

def EGM_scale_and_smear_v9(sink, events, unc_type=None, is_correction=True, restriction=None, is_electron=False):
    if is_correction or restriction or (not is_electron) or (not unc_type):
        raise ValueError("v9 EGM_scale_and_smear narrowly works on Electron scale and smear systematics (MC bug included on dEscale(Up|Down)), without restriction, and cannot embed the correction")
    egm_object = events.Electron        
    if not hasattr(egm_object, "eCorr"):
        raise ValueError("The EGM_scale_and_smear is only compatible with NanoAODv9 Run2 UL samples.")

    # Buggy, 0 value... lets stick with energyErr as placeholder even if wrong until we switch to newer NanoAOD
    # egm_object["pt_scale_up"] = egm_object.pt + egm_object.dEscaleUp
    # egm_object["pt_scale_down"] = egm_object.pt + egm_object.dEscaleDown
    egm_object["pt_scale_up"] = egm_object.pt + egm_object.energyErr/np.cosh(egm_object.eta)
    egm_object["pt_scale_down"] = egm_object.pt - egm_object.energyErr/np.cosh(egm_object.eta)
    egm_object["pt_smear_up"] = egm_object.pt + egm_object.dEsigmaUp
    egm_object["pt_smear_down"] = egm_object.pt + egm_object.dEsigmaDown
    if is_electron:
        if unc_type:
            egm_object
            if unc_type == "Scale":
                if not (hasattr(egm_object, "pt_scale_up") and hasattr(egm_object, "pt_scale_down")):
                    raise ValueError(f"EGM Object collection missing pt_smear_down or pt_smear_up fields")
                return ak.zip({"pt": np.concatenate((ak.flatten(egm_object.pt_scale_up)[:, None],
                                                     ak.flatten(egm_object.pt_scale_down)[:, None],
                                                     ),axis=1),
                               "energyErr": np.concatenate((ak.flatten(egm_object.energyErr)[:, None],
                                                            ak.flatten(egm_object.energyErr)[:, None],
                                                            ),
                                                           axis=1),
                               }, depth_limit=1)
            if unc_type == "Smear":
                if not (hasattr(egm_object, "pt_smear_up") and hasattr(egm_object, "pt_smear_down")):
                    raise ValueError(f"EGM Object collection missing pt_smear_down or pt_smear_up fields")
                return ak.zip({"pt": np.concatenate((ak.flatten(egm_object.pt_smear_up)[:, None],
                                                     ak.flatten(egm_object.pt_smear_down)[:, None],
                                                     ),axis=1),
                               "energyErr": np.concatenate((ak.flatten(egm_object.energyErr)[:, None],
                                                            ak.flatten(egm_object.energyErr)[:, None],
                                                            ),
                                                           axis=1),
                               }, depth_limit=1)
            else:
                raise ValueError(f"unc_type must be one of Scale or Resolution, got {unc_type}")
    
# sink is needed for expected "pt" argument in the add_systematic callable function, so that all others may be set; it can be set to anything since it's unused
def EGM_scale_and_smear(sink, events, unc_type=None, is_correction=True, restriction=None, is_electron=False, clibhandler=None):
    # if year in ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]:
    if is_electron:
        cset_name = f"electronSS_EtDependent"
        egm_object = events.Electron
    else:
        cset_name = f"photonSS_EtDependent"
        egm_object = events.Photon
        
    if hasattr(egm_object, "eCorr"):
        raise ValueError("The EGM_Scale_Trad correction is not compatible with NanoAODv9 Run2 UL samples.")
    # else:
    #     raise NotImplementedError(f"Got unexpected era {year}")

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
    gain = ak.flatten(egm_object.seedGain)
    SCeta = ak.flatten(egm_object.superclusterEta) if "superclusterEta" in egm_object.fields else ak.flatten(egm_object.eta + egm_object.deltaEtaSC)
    # SCeta = ak.flatten(egm_object.ScEta) if "ScEta" in egm_object.fields else ak.flatten(egm_object.superclusterEta)
    r9 = ak.flatten(egm_object.r9)
    pt_raw = ak.flatten(egm_object.pt_raw) if "pt_raw" in egm_object.fields else ak.flatten(egm_object.pt)
    energy = ak.flatten(egm_object.energy)
    energyErr = ak.flatten(egm_object.energyErr)

    try:
        cset = clibhandler.getCorrectionSet(cset_name)
        scale_evaluator = cset.compound["Scale"]
        smear_and_syst_evaluator = cset["SmearAndSyst"]
    except KeyError as ke:
        raise KeyError(f"Unable to locatione {cset_name} in clibhandler with available keys: {clibhandler.keys()}")

    # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
    if len(SCeta) > 0:
        seed = abs(np.float32(SCeta[0]).view("int32"))
    else:
        seed = 42
    rng = np.random.default_rng(seed=seed)
    random_throw = rng.normal(loc=0., scale=1.)
    smear_correction = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, SCeta)


    if is_correction:
        corrected_egm_object = egm_object
        if hasattr(events, "GenPart"):
            ### MC PART ###
            # MC
            # smear = smear_and_syst_evaluator.evaluate("smear", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # # -- Calculate the nominal smearing factor.
            # # Since the smearing is stochastic, a random number is needed for each event.
            # random_numbers = rng.normal(loc=0.0, scale=1.0, size=len(mc_electrons.pt))
            # smearing = 1 + smear * random_numbers
            # mc_pt_corrected_nominal = mc_electrons.pt * smearing
            # # -- Smearing also affects the energy uncertainty in MC.
            # mc_energyErr_corrected = np.sqrt((mc_electrons.energyErr)**2 + (mc_electrons.energy * smear)**2) * smearing
        
            # unc_smear = smear_and_syst_evaluator.evaluate("esmear", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # smear_up = smear_and_syst_evaluator.evaluate("smear_up", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # smear_down = smear_and_syst_evaluator.evaluate("smear_down", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # # Note: The relations are simple: smear_up = smear + unc_smear and smear_down = max(0, smear - unc_smear).
            # # Note 2: In 2022, the "smear_down" variation can lead to negative smearing width in some cases, which is unphysical.
            # # Therefore, we use max(smear - unc_smear, 0) to ensure the smearing width is non-negative.
        
            # smearing_up   = 1 + smear_up * random_numbers  # we use the same random numbers as for the nominal smearing
            # smearing_down = 1 + smear_down * random_numbers
        
            # mc_pt_corrected_smearing_up   = mc_electrons.pt * smearing_up
            # mc_pt_corrected_smearing_down = mc_electrons.pt * smearing_down
        
            # # -- Smearing uncertainties also affects the energy uncertainty in MC
            # mc_energyErr_corrected_smearing_up = np.sqrt((mc_electrons.energyErr)**2 + (mc_electrons.energy * smear_up)**2) * smearing_up
            # mc_energyErr_corrected_smearing_down = np.sqrt((mc_electrons.energyErr)**2 + (mc_electrons.energy * smear_down)**2) * smearing_down
        
            # # We now turn to the scale uncertainty, which is also evaluated on MC original variables (pt, r9, and ScEta) BUT applied on the smeared pt.
            # unc_scale = smear_and_syst_evaluator.evaluate("escale", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # scale_up = smear_and_syst_evaluator.evaluate("scale_up", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # scale_down = smear_and_syst_evaluator.evaluate("scale_down", mc_electrons.pt, mc_electrons.r9, mc_electrons.ScEta)
            # # Note: The relations are simple: scale_up = 1 + unc_scale and scale_down = 1 - unc_scale.
            # mc_pt_corrected_scale_up   = scale_up * mc_pt_corrected_nominal
            # mc_pt_corrected_scale_down = scale_down * mc_pt_corrected_nominal
            # # -- Scale uncertainties also affects the energy uncertainty in MC
            # mc_energyErr_corrected_scale_up = mc_energyErr_corrected * scale_up
            # mc_energyErr_corrected_scale_down = mc_energyErr_corrected * scale_down
            smearing_factor = 1 + smear_correction * random_throw
            pt_corr = pt_raw * smearing_factor
            energyErr_corr = np.sqrt( energyErr**2 + (energy * smear_correction)**2 ) * smearing_factor

            # embed systematics to match NanoAODv9 format
            if is_electron:
                scale_up = smear_and_syst_evaluator.evaluate("scale_up", pt_raw, r9, SCeta)
                scale_down = smear_and_syst_evaluator.evaluate("scale_down", pt_raw, r9, SCeta)
                smear_up = smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, SCeta) * random_throw
                smear_down = smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, SCeta) * random_throw
            else:
                pass
                # Conservative scale uncertainties without Zmmg corrections
                # DISABLED: codoe for photons that depends on year
                # if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
                #     scale_up = 1.005 * np.ones_like(ak.to_numpy(pt_raw))
                #     scale_down = 0.995 * np.ones_like(ak.to_numpy(pt_raw))
                #     smear_up = np.ones_like(ak.to_numpy(pt_raw))
                #     smear_down = np.ones_like(ak.to_numpy(pt_raw))
                #     # logger.warning("Using conservative scale uncertainties of 0.5% to cover electron/photon energy scale discrepancies for Run2 samples \n")
                # else:
                #     scale_up = 1.01 * np.ones_like(ak.to_numpy(pt_raw))
                #     scale_down = 0.99 * np.ones_like(ak.to_numpy(pt_raw))
                #     smear_up = np.ones_like(ak.to_numpy(pt_raw))
                #     smear_down = np.ones_like(ak.to_numpy(pt_raw))
                #     # logger.warning("Using conservative scale uncertainties of 1% to cover electron/photon energy scale discrepancies for Run3 samples \n")
    
            # Apply restriction if needed
            if restriction is not None:
                if restriction == "EB":
                    uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEB))
    
                elif restriction == "EE":
                    uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEE))
    
                scale_up = np.where(uncMask, scale_up, np.zeros_like(scale_up))
                scale_down = np.where(uncMask, scale_down, np.zeros_like(scale_down))
    
                smear_up = np.where(uncMask, smear_up, np.zeros_like(smear_up))
                smear_down = np.where(uncMask, smear_down, np.zeros_like(smear_down))
            
            pt_corr_scale_up = pt_corr * scale_up
            pt_corr_scale_down = pt_corr * scale_down
            energyErr_corr_scale_up = energyErr_corr * scale_up
            energyErr_corr_scale_down = energyErr_corr * scale_down

            smearing_factor_up = 1 + smear_up * random_throw
            smearing_factor_down = 1 + smear_down * random_throw
            pt_corr_smear_up = pt_corr * smearing_factor_up
            pt_corr_smear_down = pt_corr * smearing_factor_down
            energyErr_corr_smear_up = np.sqrt( energyErr**2 + (energy * smear_up)**2 ) * smearing_factor_up
            energyErr_corr_smear_down = np.sqrt( energyErr**2 + (energy * smear_down)**2 ) * smearing_factor_down

            corrected_egm_object = egm_object
            corrected_egm_object["energyErr_orig"] = corrected_egm_object.energyErr
            corrected_egm_object["pt_orig"] = corrected_egm_object.pt
            # corrected_egm_object["eSmear"] = ak.unflatten(pt_corr / pt_raw, counts)
            # rho_corr = ak.unflatten(smearing, counts)
            corrected_egm_object["pt"] = ak.unflatten(pt_corr, counts)
            corrected_egm_object["energyErr"] = ak.unflatten(energyErr_corr, counts)
            corrected_egm_object["pt_smear_up"] = ak.unflatten(pt_corr_smear_up, counts)
            corrected_egm_object["energyErr_smear_up"] = ak.unflatten(energyErr_corr_smear_up, counts)
            corrected_egm_object["pt_smear_down"] = ak.unflatten(pt_corr_smear_down, counts)
            corrected_egm_object["energyErr_smear_down"] = ak.unflatten(energyErr_corr_smear_down, counts)
            corrected_egm_object["pt_scale_up"] = ak.unflatten(pt_corr_scale_up, counts)
            corrected_egm_object["energyErr_scale_up"] = ak.unflatten(energyErr_corr_scale_up, counts)
            corrected_egm_object["pt_scale_down"] = ak.unflatten(pt_corr_scale_down, counts)
            corrected_egm_object["energyErr_scale_down"] = ak.unflatten(energyErr_corr_scale_down, counts)
            # corrected_egm_object["dEscaleUp"] = ak.unflatten( #FIXME: need the dEscaleUp/Down dEsigmaUp/Down computed from energy? 
        else:
            # DATA
            # scale = scale_evaluator.evaluate("scale", data_run, data_electrons.ScEta, data_electrons.r9, data_electrons.pt, data_electrons.seedGain,)
            # data_pt_corrected = scale * data_electrons.pt
            # # -- Smearing also affects the energy uncertainty in Data.
            # smear = smear_and_syst_evaluator.evaluate("smear", data_electrons.pt * scale, data_electrons.r9, data_electrons.ScEta)
            # data_energyErr_corrected = np.sqrt((data_electrons.energyErr)**2 + (data_electrons.energy * smear)**2) * scale
            scale_correction = scale_evaluator.evaluate("scale", run, SCeta, r9, pt_raw, gain)
            pt_corr = pt_raw * scale_correction
            smear_correction = smear_and_syst_evaluator.evaluate("smear", pt_corr, r9, SCeta)
            energyErr_corr = np.sqrt( (energyErr**2 + (energy * smear_correction)**2 ) * scale_correction )

            corrected_egm_object["energyErr_orig"] = corrected_egm_object.energyErr
            corrected_egm_object["energyErr"] = ak.unflatten(energyErr_corr, counts)
            # corrected_egm_object["eCorr"] = correction
            corrected_egm_object["pt_orig"] = corrected_egm_object.pt
            corrected_egm_object["pt"] = ak.unflatten(pt_corr, counts)
            # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event! - from HiggsDNA implementation
            # rho_corr = ak.unflatten(smearing, counts)            
            # corrected_egm_object["rho_smear"] = ak.unflatten(rho_corr, counts)

        if is_electron:
            events["Electron"] = corrected_egm_object
        else:
            events["Photon"] = corrected_egm_object
        return events

    else:
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale and Smear uncertainties should only be applied to MC!")
        if unc_type:
            if unc_type == "Scale":
                if not (hasattr(egm_object, "pt_scale_up") and hasattr(egm_object, "pt_scale_down")):
                    raise ValueError(f"EGM Object collection missing pt_smear_down or pt_smear_up fields")
                return ak.zip({"pt": np.concatenate((ak.flatten(egm_object.pt_scale_up)[:, None],
                                                     ak.flatten(egm_object.pt_scale_down)[:, None],
                                                     ),axis=1),
                               "energyErr": np.concatenate((ak.flatten(egm_object.energyErr_scale_up)[:, None],
                                                            ak.flatten(egm_object.energyErr_scale_down)[:, None],
                                                            ),
                                                           axis=1),
                               }, depth_limit=1)
            if unc_type == "Smear":
                if not (hasattr(egm_object, "pt_smear_up") and hasattr(egm_object, "pt_smear_down")):
                    raise ValueError(f"EGM Object collection missing pt_smear_down or pt_smear_up fields")
                return ak.zip({"pt": np.concatenate((ak.flatten(egm_object.pt_smear_up)[:, None],
                                                     ak.flatten(egm_object.pt_smear_down)[:, None],
                                                     ),axis=1),
                               "energyErr": np.concatenate((ak.flatten(egm_object.energyErr_smear_up)[:, None],
                                                            ak.flatten(egm_object.energyErr_smear_down)[:, None],
                                                            ),
                                                           axis=1),
                               }, depth_limit=1)
            else:
                raise ValueError(f"unc_type must be one of Scale or Resolution, got {unc_type}")
