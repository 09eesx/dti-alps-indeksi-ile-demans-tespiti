"""
find_alternative_orthogonal_rois.py
-------------------------------------
The other direction of orthogonality_check.py: instead of confirming the
known/literature-standard DTI-ALPS ROI (SCR/SLF at the lateral-ventricle
level, roi_outputs/*_rois_native/), this searches the REST of the white
matter -- explicitly excluding the SCR/SLF atlas footprint and a buffer zone
around the standard fixed-sphere ROI -- for OTHER 5mm ROI pairs that pass
the full three-way (triad) orthogonality check: projection-vs-association,
projection-vs-perpendicular(left-right), and association-vs-perpendicular
must ALL fall in [--low-deg, --high-deg] (default 78-90 degrees).

This is a specificity probe: if such triads turn up all over the brain, the
angle criterion alone is not a strong enough anatomical signature and needs
the atlas prior to mean anything; if they are rare/only found in expected
white-matter crossing zones, that supports the mechanism being meaningful
rather than incidental.

Usage:
    python find_alternative_orthogonal_rois.py
    python find_alternative_orthogonal_rois.py --fa-threshold 0.25 --n-centers 8
    python find_alternative_orthogonal_rois.py --low-deg 75
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from orthogonality_check import check_triad_orthogonality
from purity_roi import find_multi_center_roi, sphere_mask, AXIS_Y, AXIS_Z, MultiCenterROI

SIDES = ("L", "R")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor-dir", type=Path, default=Path("dti_metrics"))
    p.add_argument("--atlas-roi-dir", type=Path, default=Path("roi_outputs/atlas_rois_native"))
    p.add_argument("--anchor-roi-dir", type=Path, default=Path("roi_outputs/sphere_rois_native"),
                    help="Known/standard fixed-point ROI -- excluded (plus a buffer) from the search.")
    p.add_argument("--output-dir", type=Path, default=Path("roi_outputs/orthogonality_checked_alternative_triad"))
    p.add_argument("--fa-threshold", type=float, default=0.2,
                    help="Minimum FA to count a voxel as white matter.")
    p.add_argument("--anchor-exclude-mm", type=float, default=15.0,
                    help="Radius (mm) around the standard ROI's centroid to exclude, "
                         "on top of excluding the whole SCR/SLF atlas footprint.")
    p.add_argument("--n-centers", type=int, default=8,
                    help="Candidate ROI centers per region/hemisphere before filtering.")
    p.add_argument("--roi-radius-mm", type=float, default=5.0)
    p.add_argument("--min-sep-mm", type=float, default=6.0)
    p.add_argument("--low-deg", type=float, default=78.0,
                    help="Lower bound (degrees) for all three pairwise triad angles.")
    p.add_argument("--high-deg", type=float, default=90.0,
                    help="Upper bound (degrees) for all three pairwise triad angles.")
    return p.parse_args()


def voxel_to_mm(affine: np.ndarray, voxel) -> np.ndarray:
    return (affine @ np.array([*voxel, 1.0], dtype=np.float64))[:3]


def anchor_exclusion_sphere(shape, anchor_mask: np.ndarray, voxel_size, radius_mm: float) -> np.ndarray:
    centroid = np.argwhere(anchor_mask).mean(axis=0)
    radius_vox = np.array(radius_mm) / np.array(voxel_size)
    return sphere_mask(shape, centroid, radius_vox)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fa_img = nib.load(str(args.tensor_dir / "fa.nii.gz"))
    affine = fa_img.affine
    header = fa_img.header
    voxel_size = header.get_zooms()[:3]
    shape = fa_img.shape

    fa = fa_img.get_fdata(dtype=np.float32)
    cl = nib.load(str(args.tensor_dir / "cl_purity.nii.gz")).get_fdata(dtype=np.float32)
    evecs = nib.load(str(args.tensor_dir / "evecs.nii.gz")).get_fdata(dtype=np.float32)
    brain_mask = nib.load(str(args.tensor_dir / "brain_mask.nii.gz")).get_fdata() > 0

    wm_mask = brain_mask & (fa > args.fa_threshold)

    xs = np.where(brain_mask.any(axis=(1, 2)))[0]
    midline_x = (xs.min() + xs.max()) / 2.0
    print(f"[INFO] White matter voxels (fa>{args.fa_threshold}): {int(wm_mask.sum())}")
    print(f"[INFO] Estimated brain midline x = {midline_x:.1f} "
          f"(L = x > midline, R = x <= midline, matches existing atlas convention)")

    atlas_masks = {
        name: nib.load(str(args.atlas_roi_dir / f"{name}.nii.gz")).get_fdata() > 0
        for name in ("SCR_L", "SCR_R", "SLF_L", "SLF_R")
    }
    anchor_masks = {
        name: nib.load(str(args.anchor_roi_dir / f"{name}.nii.gz")).get_fdata() > 0
        for name in ("SCR_L", "SCR_R", "SLF_L", "SLF_R")
    }

    csv_rows = []

    for side in SIDES:
        print(f"\n[INFO] --- Side {side} ---")
        hemisphere_mask = (
            np.arange(shape[0])[:, None, None] > midline_x
            if side == "L" else
            np.arange(shape[0])[:, None, None] <= midline_x
        )
        hemisphere_mask = np.broadcast_to(hemisphere_mask, shape)

        known_footprint = atlas_masks[f"SCR_{side}"] | atlas_masks[f"SLF_{side}"]
        anchor_buffer = (
            anchor_exclusion_sphere(shape, anchor_masks[f"SCR_{side}"], voxel_size, args.anchor_exclude_mm)
            | anchor_exclusion_sphere(shape, anchor_masks[f"SLF_{side}"], voxel_size, args.anchor_exclude_mm)
        )
        exclude_mask = known_footprint | anchor_buffer

        base_search = wm_mask & hemisphere_mask & ~exclude_mask
        print(f"[INFO] Search space excluding known SCR/SLF footprint + "
              f"{args.anchor_exclude_mm}mm anchor buffer: {int(base_search.sum())} voxels")

        proj_rois = find_multi_center_roi(
            cl, evecs, AXIS_Z, base_search, voxel_size,
            n_centers=args.n_centers, radius_mm=args.roi_radius_mm, min_sep_mm=args.min_sep_mm,
        )
        # keep association candidates spatially distinct from projection candidates
        assoc_search = base_search & ~proj_rois.union_mask
        assoc_rois = find_multi_center_roi(
            cl, evecs, AXIS_Y, assoc_search, voxel_size,
            n_centers=args.n_centers, radius_mm=args.roi_radius_mm, min_sep_mm=args.min_sep_mm,
        )
        print(f"[INFO] {len(proj_rois.centers)} projection-like candidates, "
              f"{len(assoc_rois.centers)} association-like candidates (outside known ROI)")

        anchor_scr_centroid = np.argwhere(anchor_masks[f"SCR_{side}"]).mean(axis=0)
        anchor_slf_centroid = np.argwhere(anchor_masks[f"SLF_{side}"]).mean(axis=0)

        passing_proj_idx = set()
        passing_assoc_idx = set()

        for i, proj_center in enumerate(proj_rois.centers):
            for j, assoc_center in enumerate(assoc_rois.centers):
                result = check_triad_orthogonality(
                    evecs, proj_rois.masks[i], assoc_rois.masks[j],
                    low_deg=args.low_deg, high_deg=args.high_deg,
                )
                if result.passed:
                    passing_proj_idx.add(i)
                    passing_assoc_idx.add(j)

                proj_mm = voxel_to_mm(affine, proj_center)
                assoc_mm = voxel_to_mm(affine, assoc_center)
                dist_from_known_mm = min(
                    np.linalg.norm((np.array(proj_center) - anchor_scr_centroid) * voxel_size),
                    np.linalg.norm((np.array(assoc_center) - anchor_slf_centroid) * voxel_size),
                )
                csv_rows.append({
                    "side": side,
                    "proj_idx": i, "assoc_idx": j,
                    "proj_voxel": proj_center, "assoc_voxel": assoc_center,
                    "proj_mm": tuple(round(v, 1) for v in proj_mm),
                    "assoc_mm": tuple(round(v, 1) for v in assoc_mm),
                    "angle_proj_assoc_deg": round(result.angle_proj_assoc_deg, 2),
                    "angle_proj_perp_deg": round(result.angle_proj_perp_deg, 2),
                    "angle_assoc_perp_deg": round(result.angle_assoc_perp_deg, 2),
                    "max_deviation_from_90_deg": round(result.max_deviation_from_90_deg, 2),
                    "min_dist_from_known_roi_mm": round(dist_from_known_mm, 1),
                    "passed": result.passed,
                })
                status = "PASS" if result.passed else "reject"
                print(f"  [{status}] proj#{i}{tuple(proj_center)} vs assoc#{j}{tuple(assoc_center)}: "
                      f"P-A={result.angle_proj_assoc_deg:6.2f} P-X={result.angle_proj_perp_deg:6.2f} "
                      f"A-X={result.angle_assoc_perp_deg:6.2f} (max dev={result.max_deviation_from_90_deg:5.2f}), "
                      f"~{dist_from_known_mm:.0f}mm from known ROI")

        n_pairs = len(proj_rois.centers) * len(assoc_rois.centers)
        n_passed = sum(1 for r in csv_rows if r["side"] == side and r["passed"])
        print(f"[INFO] Side {side}: {n_passed}/{n_pairs} alternative pairs passed "
              f"(all 3 angles in [{args.low_deg}, {args.high_deg}] deg)")

        _save_kept(proj_rois, passing_proj_idx, fa_img, args.output_dir, f"{side}_projection_alt_kept")
        _save_kept(assoc_rois, passing_assoc_idx, fa_img, args.output_dir, f"{side}_association_alt_kept")

    if csv_rows:
        csv_path = args.output_dir / "alternative_triad_report.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[OK] Report written to {csv_path}")
    else:
        print("\n[WARN] No candidates found on either side -- nothing to report.")

    print(f"[OK] Alternative ROI masks written to {args.output_dir}")


def _save_kept(rois: MultiCenterROI, keep_idx: set, ref_img, out_dir: Path, prefix: str) -> None:
    if not keep_idx:
        print(f"[WARN] No surviving alternative ROI points for {prefix} -- writing empty mask.")
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
