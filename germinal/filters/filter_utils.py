"""Filter utilities for structure prediction and interface quality assessment.

This module contains functions for running filters and computing metrics for a single design trajectory.
"""

import os
from tempfile import gettempdir
from typing import Any, Dict, Optional, Tuple, Sequence, Set, Union, List
import numpy as np
import torch
import torch.nn.functional as F
import ablang2
from ablang2.models.ablang2.vocab import ablang_vocab
from iglm import IgLM
from colabdesign.ablang.model import CustomAbLang
from germinal.utils import utils
from germinal.filters import af3, chai, protenix, pDockQ, pyrosetta_utils
from germinal.utils.io import IO, Trajectory
import sys
from pathlib import Path
from germinal.utils.overlay_path import resolve_mmdesign_overlay
_OVERLAY = resolve_mmdesign_overlay()
sys.path.insert(0, str(_OVERLAY))
from residue_map import map_hotspots_to_output, pdb_residue_table


def run_filters(
    trajectory: Trajectory,
    run_settings: dict,
    target_settings: dict,
    filter_set: dict,
    io: IO,
    trajectory_sequence: str,
    trajectory_pdb_af: str,
    target_len: int,
    multi_relax: bool = False,
    select_mode: str = "best",
    af3_seed_size: int = 5,
) -> Tuple[dict, dict, bool, str]:
    """Run filters and compute metrics for a single design trajectory.

    The pipeline:
    1) Predict complex structure using AF3 or Chai on the final sequence
    2) Relax the predicted structure
    3) Compute clashes, secondary structure content, and interface metrics
    4) Compute additional confidence metrics (pDockQ, pDockQ2, LIS/LIA)
    5) Compute hydrophobic patch and hotspot proximity metrics
    6) Aggregate all metrics and evaluate against the provided filter set

    Args:
        trajectory: Trajectory metadata for the current design.
        run_settings: Configuration for the run (model choice, CDRs, etc.).
        target_settings: Target/binder chains, hotspots, target length.
        filter_set: Mapping of metric name to threshold spec with an operator.
        io: IO/layout helper providing directory structure.
        trajectory_sequence: Final amino acid sequence of the binder.
        trajectory_pdb_af: Path to the multimer PDB file used for alignment.

    Returns:
        Tuple containing:
            - filter_metrics: Dict of aggregated metrics for the trajectory
            - filter_results: Dict mapping '<metric>_filter' to pass/fail booleans
            - accepted: True if all filters in the set passed, else False
            - external_relaxed_pdb: Path to the relaxed complex PDB file
    """
    # ========================== Run Chai-1 or AF3 with final sequence ==========================
    structures_directory = io.layout.trajectories / "structures"
    target_chain = target_settings["target_chain"]
    binder_chain = target_settings["binder_chain"]
    target_sequence = []
    sequences_from_pdb = utils.get_sequence_from_pdb(run_settings["starting_pdb_complex"])

    for ch in target_chain.split(","):    
        target_sequence.append(sequences_from_pdb[
            ch
        ])
    # H-CDR3 positions (1-indexed). For VL-first scFv the third CDR slot is L3;
    # H3 sits in the second set of CDRs. For nb and VH-first scFv the flat
    # `cdr_lengths[:-1]` suffix lands on H3.
    if run_settings["type"].lower() == "nb":
        h3_positions = run_settings["cdr_positions"][sum(run_settings["cdr_lengths"][:-1]) :]
    elif run_settings["type"].lower() == "scfv":
        if run_settings.get("vh_first", True):
            h3_positions = run_settings["cdr_positions"][
                sum(run_settings["cdr_lengths"][:2]) : sum(run_settings["cdr_lengths"][:3])
            ]
        else:
            h3_positions = run_settings["cdr_positions"][sum(run_settings["cdr_lengths"][:-1]) :]
    else:
        raise ValueError(
            f"Type {run_settings['type']} not supported, select either nb or scfv"
        )
    cdr3 = np.array(h3_positions) + 1

    external_pdb, external_metrics, ipsae = run_structure_prediction(
        trajectory_sequence=trajectory_sequence,
        target_sequence=target_sequence,
        target_chain=target_chain,
        binder_chain=binder_chain,
        structures_directory=structures_directory,
        design_name=trajectory.design_name,
        run_settings=run_settings,
        hotspot_residue = target_settings.get("hotspot_residue", None),
        target_len=target_len,
        select_mode=select_mode,
        af3_seed_size=af3_seed_size,
        h3_positions=h3_positions,
    )

    # Validate exact target identity and convert raw input numbering before contact filters.
    hotspot_map = map_hotspots_to_output(run_settings["starting_pdb_complex"], external_pdb,
                                        target_chain, target_settings.get("target_hotspots", ""))
    external_residues = pdb_residue_table(external_pdb)
    if "".join(r["letter"] for r in external_residues[binder_chain]) != trajectory_sequence:
        raise ValueError("external predictor binder sequence/chain does not match this design")
    contact_settings = dict(target_settings)
    contact_settings["hotspot_output_positions"] = hotspot_map["positions"]

    # ========================== FastRelax ==========================
    if multi_relax:
        relaxed_paths = pyrosetta_utils.pr_relax_parallel(
            external_pdb,
            str(structures_directory),
            trajectory.design_name,
            run_settings["dalphaball_path"],
            n_relax=run_settings.get("n_relax", 5),
        )
        (best_interface_scores, best_interface_AA, best_interface_residues,
         best_relaxed_pdb) = pyrosetta_utils.score_interface_ensemble(
            relaxed_paths, binder_chain, target_chain,
            score_mode=run_settings.get("relax_score_mode", "average"),
        )
        external_relaxed_pdb = best_relaxed_pdb

        clash_threshold = run_settings["clash_threshold"]
        num_clashes_trajectory = utils.calculate_clash_score(
            external_pdb, threshold=clash_threshold, only_ca=True
        )
        num_clashes_relaxed = utils.calculate_clash_score(
            external_relaxed_pdb, threshold=clash_threshold, only_ca=True
        )

        interface_metrics = {
            "interface_scores": best_interface_scores,
            "interface_AA": best_interface_AA,
            "interface_residues": best_interface_residues,
        }
    else:
        external_relaxed_pdb = os.path.join(
            structures_directory, trajectory.design_name + "_relaxed.pdb"
        )
        pyrosetta_utils.pr_relax(external_pdb, external_relaxed_pdb)

        # ========================== Calculate Clashes ==========================
        clash_threshold = run_settings["clash_threshold"]
        num_clashes_trajectory = utils.calculate_clash_score(
            external_pdb, threshold=clash_threshold, only_ca=True
        )
        num_clashes_relaxed = utils.calculate_clash_score(
            external_relaxed_pdb, threshold=clash_threshold, only_ca=True
        )

        # ========================== Calculate Interface Metrics ==========================
        interface_metric_names = ["interface_scores", "interface_AA", "interface_residues"]
        interface_metrics = {
            k: v
            for k, v in zip(
                interface_metric_names,
                pyrosetta_utils.score_interface(
                    external_relaxed_pdb, binder_chain, target_chain=target_chain
                ),
            )
        }

    # ========================== Secondary structure content ==========================
    ss_content = utils.calc_ss_percentage(
        external_pdb, run_settings, binder_chain, return_dict=True, target_chain=target_chain
    )

    # ========================== Calculate number of framework mutations ==========================
    n_framework_mutations, framework_mutations = get_framework_mutations(
        trajectory_sequence,
        run_settings["starting_binder_seq"],
        run_settings["cdr_positions"],
    )
    print("Framework mutations:", framework_mutations)

    # ========================== Calculate Binding Interface (CDR 3) near hotspot filter ==========================
    one_indexed_cdr_positions = np.array(run_settings["cdr_positions"]) + 1

    binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = (
        compute_hotspot_proximity(
            external_relaxed_pdb=external_relaxed_pdb,
            target_settings=contact_settings,
            binder_chain=binder_chain,
            target_chain=target_chain,
            one_indexed_cdr_positions=one_indexed_cdr_positions,
            cdr3=cdr3,
            distance_threshold=run_settings["hotspot_distance_threshold"],
            contact_distance=run_settings["residue_contact_distance"],
            min_hotspot_contacts=run_settings["min_cdr_hotspot_contacts"],
        )
    )

    # ========================== Calculate Interface CDR % ==========================
    percent_interface_is_cdr = utils.interface_cdrs(
        interface_metrics["interface_residues"],
        run_settings["cdr_positions"],
        h3_positions,
        binder_chain=binder_chain,
    )

    # ========================== Calculate pDockQ, pDockQ2, LIS/LIA ==========================
    pae_matrix = external_metrics.get("pae_matrix", np.array([[0.0]]))
    if not isinstance(pae_matrix, np.ndarray):
        pae_matrix = np.array(pae_matrix)
    has_valid_pae = pae_matrix.size > 1

    if has_valid_pae:
        pdockq_metrics, lis_metrics, pDockQ2_out = compute_pdockq_and_lis(
            external_pdb=external_pdb,
            external_metrics=external_metrics,
            binder_chain=binder_chain,
            ipsae=ipsae,
        )
        i_pae = pDockQ2_out["ifpae_norm"].mean()
        i_plddt = pDockQ2_out["ifplddt"].mean() / 100
    else:
        # PAE matrix not available (e.g. Protenix without full_data output).
        # Use ipsae values if available, otherwise None (fail-closed via
        # the new evaluate_filters None handling).
        pdockq_metrics = {
            "pDockQ2": ipsae["pdockq2"] if ipsae is not None else None,
        }
        lis_metrics = {
            "lis": ipsae.get("LIS") if ipsae is not None else None,
            "lia": None,
        }
        pDockQ2_out = None
        i_pae = None
        i_plddt = None
        print("Warning: PAE matrix not available, using ipsae metrics for pDockQ2/LIS")

    # ========================== Aggregate Confidence Metrics ==========================
    # SCIENCE_FIX_CHAI_NONE_METRICS: None/missing → leave None (fail-closed downstream); never [-1] on None
    def _m0(key, idx=0):
        v = external_metrics.get(key)
        if v is None:
            return None
        try:
            if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
                return v.item()
            return np.asarray(v).reshape(-1)[idx]
        except Exception:
            try:
                return float(v)
            except Exception:
                return None
    confidence_metrics = {
        "plddt": _m0("plddt"),
        "plddt_binder": _m0("plddt_binder"),
        "ptm": _m0("ptm", 0),
        "i_ptm": _m0("iptm", 0),
        "chain_ptm": _m0("chain_ptm", -1),
        "pae": _m0("pae"),
        "aggregate_score": _m0("aggregate_score", 0) if external_metrics.get("aggregate_score") is not None else _m0("agg_score"),
        "i_pae": i_pae,
        "i_plddt": i_plddt,
        "binder_pae": (
            external_metrics["binder_pae"].item()
            if external_metrics.get("binder_pae") is not None
            else None
        ),
        "ipsae": ipsae,
    }

    # ========================== Calculate Hydrophobic Patch Filter ==========================
    sap_score, cdr_sap, _, hydrophobic_patches_binder = pyrosetta_utils.get_sap_score(
        external_relaxed_pdb,
        binder_chain=binder_chain,
        only_binder=True,
        limit_sasa=run_settings["sap_limit_sasa"],
        patch_radius=run_settings["sap_patch_radius"],
        avg_sasa_patch_thr=run_settings["sap_avg_sasa_patch_thr"],
        cdrs=run_settings["cdr_positions"],
    )

    hydrophobic_patches_struct = []

    # ========================== Calculate RMSD of Binder between Multimer and External Predictor ==========================
    try:
        pyrosetta_utils.align_pdbs(
            external_pdb, trajectory_pdb_af, target_chain, target_chain
        )
        binder_rmsd = pyrosetta_utils.unaligned_rmsd(
            external_pdb, trajectory_pdb_af, binder_chain, binder_chain
        )

    except Exception:
        binder_rmsd = 100

    # ========================== Get Log-likelihood from AbLM ==========================
    # Default to "iglm" if config omits the key (e.g. older configs like
    # vhh_il3.yaml). Without this, run_settings["ablm_model"] raises KeyError
    # and crashes the whole trajectory at the LM-likelihood step.
    ablm_model_name = run_settings.get("ablm_model", "iglm")
    if ablm_model_name == "iglm":
        lm_ll = get_iglm_ll(
            sequence=trajectory_sequence,
            species_token=run_settings["iglm_species"],
            vh_first=run_settings["vh_first"],
            vh_len=run_settings["vh_len"],
            vl_len=run_settings["vl_len"],
        )
    elif ablm_model_name == "ablang":
        lm_ll = get_ablang_ll(
            sequence=trajectory_sequence,
            vh_first=run_settings["vh_first"],
            vh_len=run_settings["vh_len"],
            vl_len=run_settings["vl_len"],
        )
    else:
        raise ValueError(f"unrecognized ablm_model={ablm_model_name!r}; no substitute score")

    # ========================== Aggregate Filter Metrics ==========================
    filter_metrics = build_filter_metrics(
        confidence_metrics,
        interface_metrics,
        hydrophobic_patches_binder,
        hydrophobic_patches_struct,
        sap_score,
        cdr_sap,
        cdr3_hotspot_contacts,
        cdr_hotspot_contacts,
        pdockq_metrics,
        lis_metrics,
        percent_interface_is_cdr,
        ss_content,
        binder_rmsd,
        n_framework_mutations,
        framework_mutations,
        num_clashes_trajectory,
        num_clashes_relaxed,
        binder_near_hotspot,
        lm_ll,
    )

    # ========================== Evaluate Filter Set ==========================
    import hashlib
    filter_metrics["hotspot_mapping_version"] = hotspot_map["version"]
    filter_metrics["hotspot_mapping"] = hotspot_map
    filter_metrics["naturalness_method"] = "germinal.ablang.compute_pll" if ablm_model_name == "ablang" else "germinal.iglm.log_likelihood"
    filter_metrics["naturalness_sequence_sha256"] = hashlib.sha256(trajectory_sequence.encode()).hexdigest()
    filter_metrics["ipsae_parser_version"] = ipsae.get("parser_version") if ipsae else None
    filter_metrics["ipsae_evidence_path"] = ipsae.get("evidence_path") if ipsae else None
    filter_metrics["interface_definition"] = f"{binder_chain}_{target_chain.replace(',', '')}"
    accepted, filter_results = evaluate_filters(filter_set, filter_metrics)

    return filter_metrics, filter_results, accepted, external_relaxed_pdb, external_pdb


def build_filter_metrics(
    confidence_metrics: dict,
    interface_metrics: dict,
    hydrophobic_patches_binder,
    hydrophobic_patches_struct,
    sap_score,
    cdr_sap,
    cdr3_hotspot_contacts,
    cdr_hotspot_contacts,
    pdockq_metrics: dict,
    lis_metrics: dict,
    percent_interface_is_cdr,
    ss_content: dict,
    binder_rmsd,
    n_framework_mutations,
    framework_mutations,
    num_clashes_trajectory,
    num_clashes_relaxed,
    binder_near_hotspot,
    lm_ll,
) -> Dict[str, Any]:
    """
    Aggregate all metrics into comprehensive evaluation dict (floats rounded to 4 decimals).

    Returns:
        Dict[str, Any]: Confidence, interface, structural, biological, and sequence metrics
    """
    ipsae = confidence_metrics["ipsae"]
    metrics = {
        # confidence
        "external_plddt": confidence_metrics["plddt"],
        "external_ptm": confidence_metrics["ptm"],
        "external_iptm": confidence_metrics["i_ptm"],
        "external_pae": confidence_metrics["pae"],
        "external_aggregate_score": confidence_metrics["aggregate_score"],
        "external_i_pae": confidence_metrics["i_pae"],
        "external_i_plddt": confidence_metrics["i_plddt"],
        "external_plddt_binder": confidence_metrics["plddt_binder"],
        "external_chain_ptm": confidence_metrics["chain_ptm"],
        "external_binder_pae": confidence_metrics["binder_pae"],
        "ipsae": None if ipsae is None else ipsae["ipsae"],
        # structure + interface
        "binder_near_hotspot": binder_near_hotspot,
        "clashes_unrelaxed": num_clashes_trajectory,
        "clashes": num_clashes_relaxed,  # relaxed clashes
        "binder_score": interface_metrics["interface_scores"]["binder_score"],
        "surface_hydrophobicity": interface_metrics["interface_scores"][
            "surface_hydrophobicity"
        ],
        "interface_shape_comp": interface_metrics["interface_scores"]["interface_sc"],
        "interface_packstat": interface_metrics["interface_scores"][
            "interface_packstat"
        ],
        "interface_dG": interface_metrics["interface_scores"]["interface_dG"],
        "interface_dSASA": interface_metrics["interface_scores"]["interface_dSASA"],
        "interface_dG_SASA_ratio": interface_metrics["interface_scores"][
            "interface_dG_SASA_ratio"
        ],
        "interface_fraction": interface_metrics["interface_scores"][
            "interface_fraction"
        ],
        "interface_hydrophobicity": interface_metrics["interface_scores"][
            "interface_hydrophobicity"
        ],
        "interface_nres": interface_metrics["interface_scores"]["interface_nres"],
        "interface_hbonds": interface_metrics["interface_scores"][
            "interface_interface_hbonds"
        ],
        "interface_hbond_percentage": interface_metrics["interface_scores"][
            "interface_hbond_percentage"
        ],
        "interface_delta_unsat_hbonds": interface_metrics["interface_scores"][
            "interface_delta_unsat_hbonds"
        ],
        "interface_delta_unsat_hbonds_percentage": interface_metrics[
            "interface_scores"
        ]["interface_delta_unsat_hbonds_percentage"],
        "hydrophobic_patches_binder": len(hydrophobic_patches_binder),
        "hydrophobic_patches_struct": len(hydrophobic_patches_struct),
        "sap_score": sap_score,
        "cdr_sap": cdr_sap,
        "cdr3_hotspot_contacts": cdr3_hotspot_contacts,
        "cdr_hotspot_contacts": cdr_hotspot_contacts,
        # derived confidence
        "pdockq2": pdockq_metrics["pDockQ2"],
        "ipsae_pdockq2": None if ipsae is None else ipsae["pdockq2"],
        "lis_lis": lis_metrics["lis"],
        "lis_lia": lis_metrics["lia"],
        # secondary structure + framework metrics
        "percent_interface_cdr": percent_interface_is_cdr[0],
        "percent_interface_cdr3": percent_interface_is_cdr[1],
        "alpha_interface": ss_content["alpha_i"],
        "beta_interface": ss_content["beta_i"],
        "loops_interface": ss_content["loops_i"],
        "alpha_all": ss_content["alpha_"],
        "beta_all": ss_content["beta_"],
        "loops_all": ss_content["loops_"],
        "sc_rmsd": binder_rmsd,
        "n_framework_mutations": n_framework_mutations,
        # large logs
        "framework_mutations": framework_mutations,
        "interface_AA": interface_metrics["interface_AA"],
        "interface_residues": interface_metrics["interface_residues"],
        "ss_content": ss_content,
        # ablm log-likelihood (iglm or ablang)
        "lm_ll": lm_ll,
    }

    # round floats to 4 decimals for compactness
    metrics = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
    }
    return metrics


def evaluate_filters(
    filter_set: dict, filter_metrics: dict
) -> Tuple[bool, Dict[str, bool]]:
    """
    Evaluate metrics against quality filters (operators: <, <=, >, >=, ==, =).

    Args:
        filter_set: {metric_name: {"value": threshold, "operator": op}}
        filter_metrics: Calculated metrics dict

    Returns:
        Tuple[bool, Dict[str, bool]]: (all_passed, individual_results)
    """
    filter_results = {}
    if not filter_set:
        return False, {"empty_filter_set_filter": False}

    for filter_name, filter_config in filter_set.items():
        if filter_name not in filter_metrics:
            print(f"Warning: Filter '{filter_name}' not found in metrics, skipping")
            filter_results[f"{filter_name}_filter"] = False
            continue

        metric_value = filter_metrics[filter_name]
        threshold = filter_config["value"]
        operator = filter_config["operator"]

        # SCIENCE_FIX_NANINF: None/NaN/Inf → fail-closed (never fabricate / never pass)
        import math as _math
        _bad = metric_value is None
        if not _bad:
            try:
                _fv = float(metric_value)
                if _math.isnan(_fv) or _math.isinf(_fv):
                    _bad = True
            except (TypeError, ValueError):
                _bad = True
        if _bad:
            # Fail-closed: missing/non-finite metric means predictor did not
            # produce usable data. Silently passing would let unchecked designs
            # through. Loudly FAIL (status is failure/pending, not pass).
            print(
                f"\n\n[FILTER ERROR] Metric '{filter_name}' is None/NaN/Inf — cannot "
                f"evaluate filter ({operator} {threshold}). FAILING this filter "
                f"(fail-closed; not a pass). Value={metric_value!r}\n\n",
                flush=True,
            )
            passed = False
        elif operator == "<":
            passed = metric_value < threshold
        elif operator == "<=":
            passed = metric_value <= threshold
        elif operator == ">":
            passed = metric_value > threshold
        elif operator == ">=":
            passed = metric_value >= threshold
        elif operator == "==":
            passed = metric_value == threshold
        elif operator == "=":
            passed = metric_value == threshold
        else:
            print(f"Warning: Unknown operator '{operator}' for filter '{filter_name}'")
            passed = False

        filter_results[f"{filter_name}_filter"] = passed

    # All filters must pass
    all_passed = all(filter_results.values())

    return all_passed, filter_results


def get_framework_mutations(
    trajectory_sequence: str,
    framework_sequence: str,
    cdr_positions: Sequence[int],
) -> Tuple[int, List[str]]:
    """
    Identify mutations outside CDR regions (format: 'A123B').

    Args:
        trajectory_sequence: Final designed sequence
        framework_sequence: Original reference sequence
        cdr_positions: CDR positions (0-indexed)

    Returns:
        Tuple[int, List[str]]: (count, mutation_list)
    """
    framework_mutations = []
    for i, (seq, framework) in enumerate(zip(trajectory_sequence, framework_sequence)):
        if seq != framework and i not in cdr_positions:
            framework_mutations.append(f"{framework}{i + 1}{seq}")
    return len(framework_mutations), framework_mutations


def is_binder_near_hotspot(
    target_contacts: Sequence[int],
    target_hotspots: Sequence[int],
    binder_contacts: Sequence[int],
    cdr_positions: Sequence[int],
    cdr3: Sequence[int],
    min_hotspot_contacts: int = 3,
) -> Union[Tuple[bool, int, int], Tuple[Set[int], Set[int]]]:
    """Check whether the binder interface is near target hotspot residues.

    Args:
        target_contacts: Target residue indices at the interface (1-indexed).
        target_hotspots: Target hotspot residue indices (1-indexed).
        binder_contacts: Binder residue indices at the interface (1-indexed).
        cdr_positions: Binder CDR residue indices (1-indexed).
        cdr3: Binder CDR3 residue indices (1-indexed).
        return_bool: If True, return summary booleans/counts; otherwise, return
            the sets of binder residues contacting hotspots for CDR3 and all CDRs.

    Returns:
        If return_bool is True:
            (has_min_cdr_contacts, num_cdr3_hotspot_contacts, num_cdr_hotspot_contacts)
        Else:
            (cdr3_hotspot_contact_set, cdr_hotspot_contact_set)
    """
    cdr_hotspot_contacts = []
    cdr3_hotspot_contacts = []
    for i, tc in enumerate(target_contacts):
        if tc in target_hotspots:
            if binder_contacts[i] in cdr_positions:
                cdr_hotspot_contacts.append(binder_contacts[i])
            if binder_contacts[i] in cdr3:
                cdr3_hotspot_contacts.append(binder_contacts[i])
    cdr_hotspot_contacts = set(cdr_hotspot_contacts)
    cdr3_hotspot_contacts = set(cdr3_hotspot_contacts)
    accept_critera = len(cdr_hotspot_contacts) >= min_hotspot_contacts

    return accept_critera, len(cdr3_hotspot_contacts), len(cdr_hotspot_contacts)


def run_structure_prediction(
    trajectory_sequence: str,
    target_sequence: str,
    target_chain: str,
    binder_chain: str,
    structures_directory,
    design_name: str,
    run_settings: dict,
    target_len: int,
    hotspot_residue = None,
    select_mode: str = "best",
    af3_seed_size: int = 5,
    h3_positions = None,
) -> Tuple[str, dict, Optional[dict]]:
    """
    Run AF3 or Chai structure prediction for antibody-target complex.

    Args:
        trajectory_sequence: Designed antibody sequence
        target_sequence: Target protein sequence
        target_chain: Target chain ID
        binder_chain: Binder chain ID
        structures_directory: Output directory
        design_name: Design identifier
        run_settings: Config with model choice and parameters

    Returns:
        Tuple[str, dict]: (pdb_path, confidence_metrics)
    """
    af3_seed = [int(x) for x in np.random.randint(0, 999999, size=af3_seed_size)]
    ipsae = None
    if run_settings["structure_model"] == "af3":
        external_pdb, external_metrics, ipsae = af3.run_af3(
            trajectory_sequence,
            target_sequence,
            target_chain,
            structures_directory,
            design_name,
            af3_seed,
            run_settings,
            binder_chain=binder_chain,
            msa_mode=run_settings["msa_mode"],
            select_mode=select_mode,
        )
    elif run_settings["structure_model"] == "chai":

        # Use h3_positions computed by run_filters (PR #67 3-way branch
        # correctly handles nb / VH-first scFv / VL-first scFv). The old
        # hardcoded slice mis-sliced VL-first scFv runs (landed on H1/H2).
        # cdr3_idx passed to chai must be 1-indexed (PDB residue numbers,
        # matching the chai.restraints template format like "L13", "D108").
        # h3_positions is 0-indexed, hence the +1.
        if h3_positions is not None:
            cdr3_idx = h3_positions[len(h3_positions)//2] + 1
        else:
            cdr3_idx = run_settings["cdr_positions"][run_settings["cdr_lengths"][0] + run_settings["cdr_lengths"][1]:]
            cdr3_idx = cdr3_idx[len(cdr3_idx)//2] + 1

        external_pdb, external_metrics = chai.run_chai(
            trajectory_sequence,
            gettempdir(),
            structures_directory,
            run_settings["starting_pdb_complex"],
            target_chain,
            seed=af3_seed[0],
            cdr3_idx = cdr3_idx,
            hotspot_residue = hotspot_residue,
            binder_chain=binder_chain,
            target_len=target_len,
            num_trunk_recycles=run_settings.get("chai_num_trunk_recycles", 3),
            num_diffn_timesteps=run_settings.get("chai_num_diffn_timesteps", 200),
            use_esm_embeddings=run_settings.get("chai_use_esm_embeddings", True),
        )
        # SCIENCE_FIX_CHAI_IPSAE
        # Compute real ipSAE/pDockQ2 for Chai (no placeholders). Fail loud if unavailable.
        try:
            import sys as _sys
            _sys.path.insert(0, "/AIRvePFS/dair/sulixian-data/mmdesign_like_vhh/overlay")
            from chai_ipsae import compute_ipsae_pdockq2
            import numpy as _np
            # SCIENCE_FIX_IPSAE_TOKEN_VALIDATE: real Chai token dims; never broadcast scalar/fabricate
            import sys as _sys2
            _sys2.path.insert(0, "/AIRvePFS/dair/sulixian-data/mmdesign_like_vhh/overlay")
            from chai_token_metrics import prepare_pae_plddt_for_ipsae
            _tgt_chains = [c for c in str(target_chain).split(",") if c]
            _plddt_src = external_metrics.get("plddt_per_token")
            if _plddt_src is None:
                _plddt_src = external_metrics.get("plddt_per_residue")
            if _plddt_src is None:
                _plddt_src = external_metrics.get("plddt")  # may be scalar mean → omitted
            _pae_arr, _plddt_arr, _shape_ev = prepare_pae_plddt_for_ipsae(
                external_metrics.get("pae_matrix"),
                _plddt_src,
                external_pdb,
                binder_chain=binder_chain,
                target_chains=_tgt_chains,
            )
            print(f"Chai ipSAE shape evidence: {_shape_ev}")
            if _shape_ev.get("plddt_status") != "ok_per_token_eq_res":
                raise ValueError(f"per-token pLDDT unavailable: {_shape_ev.get('plddt_status')}")
            ipsae = compute_ipsae_pdockq2(
                external_pdb, _pae_arr,
                plddt=_plddt_arr,
                binder_chain=binder_chain,
                target_chains=_tgt_chains,
                work_dir=str(structures_directory),
            )
            ipsae["shape_evidence"] = _shape_ev
        except Exception as _ipsae_err:
            print(f"Chai ipSAE/pDockQ2 computation FAILED: {_ipsae_err}")
            ipsae = None  # evaluate_filters must reject None — do not fabricate scores
    elif run_settings["structure_model"] == "protenix":
        external_pdb, external_metrics, ipsae = protenix.run_protenix(
            trajectory_sequence,
            target_sequence,
            target_chain,
            structures_directory,
            design_name,
            af3_seed,
            run_settings,
            binder_chain=binder_chain,
            msa_mode=run_settings["msa_mode"],
            select_mode=select_mode,
        )
    else:
        raise ValueError(
            f"Structure model {run_settings['structure_model']} not supported, select either af3, chai, or protenix"
        )

    return external_pdb, external_metrics, ipsae


def compute_hotspot_proximity(
    external_relaxed_pdb: str,
    target_settings: dict,
    binder_chain: str,
    target_chain: str,
    one_indexed_cdr_positions: Sequence[int],
    cdr3: Sequence[int],
    distance_threshold: float = 5.3,
    contact_distance: float = 6.0,
    min_hotspot_contacts: int = 3,
) -> Tuple[bool, int, int]:
    """
    Compute CDR contacts with target hotspot residues (5.3Å threshold, ≥3 contacts required).

    Args:
        external_relaxed_pdb: Relaxed complex PDB path
        target_settings: Config with hotspot definitions
        binder_chain: Binder chain ID
        target_chain: Target chain ID
        one_indexed_cdr_positions: All CDR positions (1-indexed)
        cdr3: CDR3 positions (1-indexed)

    Returns:
        Tuple[bool, int, int]: (near_hotspot, cdr3_contacts, cdr_contacts)
    """
    # Default values when no hotspot specification is provided
    binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = True, 0, 0
    cdr3_hotspot_contacts_ch = 0
    cdr_hotspot_contacts_ch = 0
    if str(target_settings.get("target_hotspots", "")).strip():
        positions = target_settings.get("hotspot_output_positions")
        if not positions:
            raise ValueError("hotspot_output_positions required; raw input numbers cannot score output")
        binder_near_hotspot = []
        for ch, target_hotspots in positions.items():
            if ch not in target_chain.split(",") or ch == binder_chain or not target_hotspots:
                raise ValueError(f"invalid hotspot target chain {ch}")

            hotspot_region = pyrosetta_utils.find_nearby_residues_from_pdb(
                external_relaxed_pdb,
                target_hotspots,
                distance_threshold=distance_threshold,
                chain=ch,
            )

            contacts = pyrosetta_utils.get_residue_contacts(
                external_relaxed_pdb, ch, binder_chain, contact_distance
            )
            contacts_per_chain = np.array(list(contacts.keys()))

            if contacts_per_chain.size == 0:
                binder_near_chain_ht, cdr3_hotspot_contacts_ch, cdr_hotspot_contacts_ch = (False, 0, 0)
            else:
                try:
                    binder_near_chain_ht, cdr3_hotspot_contacts_ch, cdr_hotspot_contacts_ch = (
                        is_binder_near_hotspot(
                            contacts_per_chain[:, 0],
                            hotspot_region,
                            contacts_per_chain[:, 1],
                            one_indexed_cdr_positions,
                            cdr3,
                            min_hotspot_contacts=min_hotspot_contacts,
                        )
                    )
                except Exception as _hs_err:
                    raise RuntimeError(f"hotspot proximity failed on chain {ch}: {_hs_err}") from _hs_err

            binder_near_hotspot.append(binder_near_chain_ht)
            cdr3_hotspot_contacts += cdr3_hotspot_contacts_ch
            cdr_hotspot_contacts += cdr_hotspot_contacts_ch
        binder_near_hotspot = any(binder_near_hotspot)  # pass if any specified hotspot chain contacts

    return binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts


def compute_pdockq_and_lis(
    external_pdb: str,
    external_metrics: dict,
    binder_chain: str,
    ipsae: Optional[dict] = None,
) -> Tuple[dict, dict, dict]:
    """
    Compute docking quality metrics.

    pDockQ2 and LIS now come exclusively from the upstream ``ipsae`` tool
    (single scalar each), avoiding the chain-keying bug in the old
    pDockQ.pDockQ2 per-chain aggregation that produced wrong values for
    ≥3-chain complexes. The pDockQ2 module is still called once to obtain
    the per-residue ``ifpae_norm`` / ``ifplddt`` arrays which feed the
    interface-PAE and interface-pLDDT metrics (i_pae / i_plddt) — these
    arrays are not affected by the chain-key bug.

    LIA is no longer computed (ipsae does not produce it); set to None
    so any filter on lis_lia will fail-loud rather than silently pass.

    Args:
        external_pdb: Complex PDB path
        external_metrics: Metrics with PAE matrix
        binder_chain: Binder chain id (kept for signature compatibility;
            no longer used in selection now that ipsae provides the scalar)
        ipsae: ipsae output dict with keys {ipsae, pdockq2, LIS}; None if
            the ipsae tool failed or full_data was unavailable.

    Returns:
        Tuple[dict, dict, dict]: (pdockq_metrics, lis_metrics, pDockQ2_out)
    """
    external_pae = external_metrics["pae_matrix"]
    # pDockQ2_out (DataFrame) is still needed for ifpae_norm / ifplddt;
    # discard the buggy chain_specific_pdockq2 dict entirely.
    pDockQ2_out, _ = pDockQ.pDockQ2(external_pdb, external_pae)

    # Use ipsae's pDockQ2 scalar directly. None when ipsae unavailable so
    # downstream filter evaluation fails-loud rather than silently passes.
    pdockq_metrics = {
        "pDockQ2": ipsae["pdockq2"] if ipsae is not None else None,
    }
    lis_metrics = {
        "lis": ipsae.get("LIS") if ipsae is not None else None,
        "lia": None,  # ipsae does not compute LIA; no longer derived locally
    }

    return pdockq_metrics, lis_metrics, pDockQ2_out


def get_iglm_ll(
    sequence,
    chain_token="[HEAVY]",
    species_token="[CAMEL]",
    vh_first=True,
    vh_len=0,
    vl_len=0,
):
    """
    Calculate antibody sequence log-likelihood using IgLM language model.

    Attribution: Shuai, R. W., Ruffolo, J. A., & Gray, J. J. (2023). IgLM: Infilling
    language modeling for antibody sequence design. Cell Systems, 14(11), 979-989.
    License: JHU Academic Software License (non-commercial use). Commercial inquiries: awichma2@jhu.edu
    Source: https://github.com/Graylab/IgLM

    Args:
        sequence: Antibody amino acid sequence
        chain_token: "[HEAVY]" or "[LIGHT]"
        species_token: "[HUMAN]", "[CAMEL]", etc.
        vh_first: Heavy chain first in scFv sequence
        vh_len: Heavy chain length (0 for nanobodies)
        vl_len: Light chain length (0 for nanobodies)

    Returns:
        float: Log-likelihood score (higher = more natural)
    """

    # Initialize the model
    model = IgLM()

    # Compute the log likelihood, depending on nanobody or scfv
    if vl_len and vh_len:
        if vh_first:
            log_likelihood_h = model.log_likelihood(
                sequence[:vh_len], "[HEAVY]", species_token
            )
            log_likelihood_l = model.log_likelihood(
                sequence[-vl_len:], "[LIGHT]", species_token
            )
        else:
            log_likelihood_l = model.log_likelihood(
                sequence[:vl_len], "[LIGHT]", species_token
            )
            log_likelihood_h = model.log_likelihood(
                sequence[-vh_len:], "[HEAVY]", species_token
            )
        log_likelihood = log_likelihood_h + log_likelihood_l
    else:
        log_likelihood = model.log_likelihood(sequence, chain_token, species_token)

    return log_likelihood


def get_ablang_ll(
    sequence,
    vh_first=True,
    vh_len=None,
    vl_len=None,
    ablm_temp=0.6,
):
    """
    Calculate antibody sequence pseudo-log-likelihood using AbLang (MLM scoring).

    Each residue is masked once and scored against the full bidirectional context,
    giving the true MLM pseudolikelihood. Uses AbLang1 for VHH and AbLang2 for scFv.

    Args:
        sequence: Antibody amino acid sequence (full scFv including linker, or VHH)
        vh_first: Heavy chain first in scFv sequence
        vh_len: Heavy chain length (None for nanobodies)
        vl_len: Light chain length (None for nanobodies)
        ablm_temp: Unused (kept for API compatibility)

    Returns:
        float: Pseudo-log-likelihood score (higher = more natural)
    """
    is_scfv = bool(vh_len and vl_len)
    model = CustomAbLang(
        is_scfv=is_scfv,
        vh_first=vh_first,
        vh_len=vh_len,
        vl_len=vl_len,
    )
    return model.compute_pll(sequence)
