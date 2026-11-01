"""
apply_orthogonality_check.py
-----------------------------
Runs the angle-based ROI QC mechanism (orthogonality_check.py), validated in
test_orthogonality_check.py with synthetic tensors, on this subject's real
DTI data (dti_metrics/) instead of colour-based (RGB-FA) visual inspection.

The literature (Taoka et al., 2017; Liu/Barisano et al., 2023) places the
DTI-ALPS ROIs at a SINGLE axial level, where the projection fibers,
association fibers and perivascular spaces intersect near the body of the
lateral ventricle -- not anywhere along the full SCR/SLF white-matter tracts
(which span many centimeters superior-inferior). To avoid the candidate
search wandering off to anatomically implausible ("external") locations
elsewhere in the tract, the search window for each region is restricted to
a narrow slab of slices (--z-tol-slices) around that literature-standard
level, anchored on the existing fixed-sphere ROI (roi_outputs/sphere_rois_native/,
warped from the Liu/Barisano template coordinates).

For each hemisphere:
  1. Find up to N candidate 5mm projection-fiber ROI centers (axis ~ Z) inside
     the SCR (corona radiata) atlas label, restricted to the anchor slab, and
     up to N candidate association-fiber ROI centers (axis ~ Y) inside the
     SLF atlas label, same restriction.
  2. Check every (projection candidate, association candidate) pair with
     check_roi_orthogonality(): angle between their dominant tensor
     orientations must be close to 90 degrees (within --tol-deg), not
     necessarily exactly 90.
  3. Keep only the passing pairs. Save the surviving ("masked") ROI points as
     NIfTI masks + report their voxel/mm coordinates and angles in a CSV.

Usage:
    python apply_orthogonality_check.py
    python apply_orthogonality_check.py --tol-deg 20 --n-centers 8 --z-tol-slices 2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np

from orthogonality_check import check_roi_orthogonality
from purity_roi import find_multi_center_roi, AXIS_Y, AXIS_Z, MultiCenterROI

SIDES = ("L", "R")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor-dir", type=Path, default=Path("dti_metrics"))
    p.add_argument("--atlas-roi-dir", type=Path, default=Path("roi_outputs/atlas_rois_native"))
    p.add_argument("--anchor-roi-dir", type=Path, default=Path("roi_outputs/sphere_rois_native"),
                    help="Literature-standard fixed-point ROI (single level) used to anchor "
                         "the search window and stop it from drifting to other parts of the tract.")
    p.add_argument("--output-dir", type=Path, default=Path("roi_outputs/orthogonality_checked"))
    p.add_argument("--n-centers", type=int, default=5,
                    help="Candidate ROI centers per region/hemisphere before filtering.")
    p.add_argument("--roi-radius-mm", type=float, default=5.0)
    p.add_argument("--tol-deg", type=float, default=30.0,
                    help="Allowed deviation from 90 degrees for a pair to pass.")
    p.add_argument("--z-tol-slices", type=int, default=2,
                    help="How many slices above/below the anchor ROI's level the candidate "
                         "search is allowed to explore. Keeps the search at the literature's "
                         "single lateral-ventricle level instead of the whole SCR/SLF tract.")
    return p.parse_args()


def restrict_to_anchor_level(atlas_mask: np.ndarray, anchor_mask: np.ndarray, z_tol_slices: int) -> np.ndarray:
    """Intersect `atlas_mask` with a narrow slab of slices around the
    z-centroid of `anchor_mask` (the literature-standard fixed-sphere ROI),
    so the candidate search stays at the correct single axial level instead
    of ranging across the entire white-matter tract.
    """
    anchor_voxels = np.argwhere(anchor_mask)
    if anchor_voxels.size == 0:
        raise ValueError("Anchor ROI mask is empty -- cannot determine the reference level.")
    anchor_z = anchor_voxels[:, 2].mean()

    z_lo = int(round(anchor_z - z_tol_slices))
    z_hi = int(round(anchor_z + z_tol_slices))
    band = np.zeros_like(atlas_mask, dtype=bool)
    band[:, :, z_lo:z_hi + 1] = True

    return atlas_mask & band


def voxel_to_mm(affine: np.ndarray, voxel) -> np.ndarray:
    return (affine @ np.array([*voxel, 1.0], dtype=np.float64))[:3]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fa_img = nib.load(str(args.tensor_dir / "fa.nii.gz"))
    affine = fa_img.affine
    header = fa_img.header
    voxel_size = header.get_zooms()[:3]

    cl = nib.load(str(args.tensor_dir / "cl_purity.nii.gz")).get_fdata(dtype=np.float32)
    evecs = nib.load(str(args.tensor_dir / "evecs.nii.gz")).get_fdata(dtype=np.float32)

    atlas_masks = {
        name: nib.load(str(args.atlas_roi_dir / f"{name}.nii.gz")).get_fdata() > 0
        for name in ("SCR_L", "SCR_R", "SLF_L", "SLF_R")
    }
    anchor_masks = {
        name: nib.load(str(args.anchor_roi_dir / f"{name}.nii.gz")).get_fdata() > 0
        for name in ("SCR_L", "SCR_R", "SLF_L", "SLF_R")
    }
    search_masks = {
        name: restrict_to_anchor_level(atlas_masks[name], anchor_masks[name], args.z_tol_slices)
        for name in atlas_masks
    }
    for name, m in search_masks.items():
        print(f"[INFO] {name}: search window restricted from {int(atlas_masks[name].sum())} "
              f"to {int(m.sum())} voxels (+/-{args.z_tol_slices} slices around the standard level)")

    csv_rows = []

    for side in SIDES:
        print(f"\n[INFO] --- Side {side} ---")
        proj_rois = find_multi_center_roi(
            cl, evecs, AXIS_Z, search_masks[f"SCR_{side}"], voxel_size,
            n_centers=args.n_centers, radius_mm=args.roi_radius_mm,
        )
        assoc_rois = find_multi_center_roi(
            cl, evecs, AXIS_Y, search_masks[f"SLF_{side}"], voxel_size,
            n_centers=args.n_centers, radius_mm=args.roi_radius_mm,
        )
        print(f"[INFO] {len(proj_rois.centers)} projection candidates, "
              f"{len(assoc_rois.centers)} association candidates")

        passing_proj_idx = set()
        passing_assoc_idx = set()

        for i, proj_center in enumerate(proj_rois.centers):
            for j, assoc_center in enumerate(assoc_rois.centers):
                result = check_roi_orthogonality(
                    evecs, proj_rois.masks[i], assoc_rois.masks[j], tol_deg=args.tol_deg
                )
                if result.passed:
                    passing_proj_idx.add(i)
                    passing_assoc_idx.add(j)

                proj_mm = voxel_to_mm(affine, proj_center)
                assoc_mm = voxel_to_mm(affine, assoc_center)
                csv_rows.append({
                    "side": side,
                    "proj_idx": i, "assoc_idx": j,
                    "proj_voxel": proj_center, "assoc_voxel": assoc_center,
                    "proj_mm": tuple(round(v, 1) for v in proj_mm),
                    "assoc_mm": tuple(round(v, 1) for v in assoc_mm),
                    "angle_deg": round(result.angle_deg, 2),
                    "deviation_from_90_deg": round(result.deviation_from_90_deg, 2),
                    "passed": result.passed,
                })
                status = "PASS" if result.passed else "reject"
                print(f"  [{status}] proj#{i}{tuple(proj_center)} vs assoc#{j}{tuple(assoc_center)}: "
                      f"angle={result.angle_deg:6.2f} deg (dev from 90 = {result.deviation_from_90_deg:5.2f})")

        n_pairs = len(proj_rois.centers) * len(assoc_rois.centers)
        n_passed = sum(1 for r in csv_rows if r["side"] == side and r["passed"])
        print(f"[INFO] Side {side}: {n_passed}/{n_pairs} pairs passed "
              f"(tol={args.tol_deg} deg). Surviving centers: "
              f"{len(passing_proj_idx)} projection, {len(passing_assoc_idx)} association")

        # Save only the ROI points that survived the orthogonality check.
        _save_kept(proj_rois, passing_proj_idx, fa_img, args.output_dir, f"{side}_projection_kept")
        _save_kept(assoc_rois, passing_assoc_idx, fa_img, args.output_dir, f"{side}_association_kept")

    csv_path = args.output_dir / "orthogonality_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[OK] Report written to {csv_path}")
    print(f"[OK] Masked ROI points written to {args.output_dir}")


def _save_kept(rois: MultiCenterROI, keep_idx: set, ref_img, out_dir: Path, prefix: str) -> None:
    if not keep_idx:
        print(f"[WARN] No surviving ROI points for {prefix} -- writing empty mask.")
        empty = np.zeros_like(rois.union_mask, dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty, ref_img.affine, ref_img.header), str(out_dir / f"{prefix}.nii.gz"))
        return

    kept_mask = np.zeros_like(rois.union_mask, dtype=bool)
    for idx in keep_idx:
        kept_mask |= rois.masks[idx]

    nib.save(
        nib.Nifti1Image(kept_mask.astype(np.uint8), ref_img.affine, ref_img.header),
        str(out_dir / f"{prefix}.nii.gz"),
    )


if __name__ == "__main__":
    main()
