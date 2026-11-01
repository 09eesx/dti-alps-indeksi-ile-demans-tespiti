"""
run_pipeline.py
-----------------
End-to-end DTI-ALPS pipeline:
  1. Preprocess        (denoise/unring, optional eddy/motion correction)
  2. Tensor fitting     (FA/MD/AD/RD, Dxx/Dyy/Dzz, eigen-info, purity map)
  3. Registration        + two reference ROI strategies (atlas-based, fixed-sphere)
  4. Purity-based multi-center ROI strategy (N centers x radius-mm spheres) -- novel
  5. ALPS-index per strategy + ensemble agreement check
  6. Results -> results.csv

Example:
    python run_pipeline.py \\
        --dwi sub01_dwi.nii.gz --bval sub01.bval --bvec sub01.bvec \\
        --template-fa  $FSLDIR/data/atlases/JHU/JHU-ICBM-FA-1mm.nii.gz \\
        --template-labels $FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz \\
        --output-dir out/sub01
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ants

from preprocess import denoise_and_unring, eddy_correct
from tensor_fit import fit_tensor, save_tensor_maps
from registration_roi import (
    register_to_template,
    warp_atlas_labels_to_native,
    warp_sphere_rois_to_native,
)
from purity_roi import find_multi_center_roi, save_multicenter_roi, AXIS_Z, AXIS_Y
from alps_ensemble import alps_from_masks, alps_from_multi_center, compare_strategies


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full DTI-ALPS pipeline with 3 ROI strategies.")
    p.add_argument("--dwi", type=Path, required=True)
    p.add_argument("--bval", type=Path, required=True)
    p.add_argument("--bvec", type=Path, required=True)
    p.add_argument("--template-fa", type=Path, required=True)
    p.add_argument("--template-labels", type=Path, required=True,
                    help="JHU-ICBM-DTI-81 white-matter label atlas, SAME space as --template-fa")
    p.add_argument("--output-dir", type=Path, default=Path("alps_output"))
    p.add_argument("--skip-preprocess", action="store_true",
                    help="Skip denoise/unring/eddy (use if data is already preprocessed)")
    p.add_argument("--eddy-ref-volume", type=int, default=0)
    p.add_argument("--n-centers", type=int, default=5,
                    help="Number of purity-based ROI centers per region/hemisphere")
    p.add_argument("--roi-radius-mm", type=float, default=5.0,
                    help="Radius (mm) of each purity-based ROI sphere")
    p.add_argument("--median-radius", type=int, default=3,
                    help="Median filter radius for tensor-fit brain masking")
    p.add_argument("--numpass", type=int, default=2,
                    help="Number of median_otsu passes for tensor fitting")
    p.add_argument("--autocrop", action="store_true",
                    help="Enable autocrop in tensor-fit brain masking")
    p.add_argument("--save-rgb-fa", action="store_true",
                    help="Also save RGB FA map in the tensor output folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Preprocessing ---------------------------------------------------
    dwi_for_fit = args.dwi
    bval_for_fit = args.bval
    bvec_for_fit = args.bvec
    if not args.skip_preprocess:
        denoise_dir = out / "preproc" / "01_denoise_unring"
        eddy_dir = out / "preproc" / "02_eddy"

        dwi_for_fit = denoise_and_unring(args.dwi, args.bval, args.bvec, denoise_dir)
        if dwi_for_fit.name == "dwi_denoised_unring.nii.gz":
            bval_for_fit = denoise_dir / "dwi_denoised_unring.bval"
            bvec_for_fit = denoise_dir / "dwi_denoised_unring.bvec"
        else:
            bval_for_fit = denoise_dir / "dwi_denoised.bval"
            bvec_for_fit = denoise_dir / "dwi_denoised.bvec"

        dwi_for_fit, bval_for_fit, bvec_for_fit = eddy_correct(
            dwi_for_fit,
            bval_for_fit,
            bvec_for_fit,
            args.eddy_ref_volume,
            eddy_dir,
        )
    else:
        print("[INFO] --skip-preprocess set: using input DWI as-is.")

    # ---- 2. Tensor fitting ---------------------------------------------------
    tm = fit_tensor(
        dwi_for_fit,
        bval_for_fit,
        bvec_for_fit,
        median_radius=args.median_radius,
        numpass=args.numpass,
        autocrop=args.autocrop,
    )
    save_tensor_maps(tm, out / "tensor", save_rgb_fa=args.save_rgb_fa)

    # ---- 3. Registration + reference ROI strategies --------------------------
    subject_fa_ants = ants.image_read(str(out / "tensor" / "fa.nii.gz"))
    template_fa_ants = ants.image_read(str(args.template_fa))
    reg = register_to_template(out / "tensor" / "fa.nii.gz", args.template_fa)

    atlas_masks = warp_atlas_labels_to_native(reg, args.template_labels, subject_fa_ants)
    sphere_masks = warp_sphere_rois_to_native(reg, template_fa_ants, subject_fa_ants, radius_mm=2.5)

    # ---- 4. Purity-based multi-center ROI (novel strategy) --------------------
    voxel_size = tm.header.get_zooms()[:3]
    multi_center_detail = {}
    purity_dir = out / "purity_rois_native"
    subject_fa_path = out / "tensor" / "fa.nii.gz"

    for side in ("L", "R"):
        proj_search = atlas_masks[f"SCR_{side}"]    # anatomically constrained search window
        assoc_search = atlas_masks[f"SLF_{side}"]

        proj_rois = find_multi_center_roi(
            tm.cl, tm.evecs, AXIS_Z, proj_search, voxel_size,
            n_centers=args.n_centers, radius_mm=args.roi_radius_mm,
        )
        assoc_rois = find_multi_center_roi(
            tm.cl, tm.evecs, AXIS_Y, assoc_search, voxel_size,
            n_centers=args.n_centers, radius_mm=args.roi_radius_mm,
        )
        multi_center_detail[side] = (proj_rois, assoc_rois)

        save_multicenter_roi(
            proj_rois,
            reference_nifti=subject_fa_path,
            out_dir=purity_dir,
            prefix=f"{side}_projection",
        )
        save_multicenter_roi(
            assoc_rois,
            reference_nifti=subject_fa_path,
            out_dir=purity_dir,
            prefix=f"{side}_association",
        )

    # ---- 5. ALPS-index per strategy --------------------------------------------
    alps_values = {}

    for side in ("L", "R"):
        alps_values[f"atlas_{side}"] = alps_from_masks(
            tm.Dxx, tm.Dyy, tm.Dzz, atlas_masks[f"SCR_{side}"], atlas_masks[f"SLF_{side}"]
        )
        alps_values[f"sphere_{side}"] = alps_from_masks(
            tm.Dxx, tm.Dyy, tm.Dzz, sphere_masks[f"SCR_{side}"], sphere_masks[f"SLF_{side}"]
        )
        proj_rois, assoc_rois = multi_center_detail[side]
        mc_result = alps_from_multi_center(tm.Dxx, tm.Dyy, tm.Dzz, proj_rois, assoc_rois)
        alps_values[f"purity_{side}"] = mc_result.pooled_alps
        alps_values[f"purity_{side}_std"] = mc_result.std_across_pairs

    # ---- 6. Ensemble agreement (per side, across the 3 strategies) -------------
    ensemble = {}
    for side in ("L", "R"):
        ensemble[side] = compare_strategies({
            "atlas": alps_values[f"atlas_{side}"],
            "sphere": alps_values[f"sphere_{side}"],
            "purity": alps_values[f"purity_{side}"],
        })

    # ---- 7. Write results.csv ---------------------------------------------------
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "side", "atlas_alps", "sphere_alps", "purity_alps",
            "purity_std_across_pairs", "ensemble_mean",
            "ensemble_cv", "ensemble_flag",
        ])
        for side in ("L", "R"):
            e = ensemble[side]
            writer.writerow([
                side,
                alps_values[f"atlas_{side}"],
                alps_values[f"sphere_{side}"],
                alps_values[f"purity_{side}"],
                alps_values[f"purity_{side}_std"],
                e.mean, e.coefficient_of_variation, e.flag,
            ])

    print(f"[OK] Pipeline finished. Results written to {csv_path}")


if __name__ == "__main__":
    main()
