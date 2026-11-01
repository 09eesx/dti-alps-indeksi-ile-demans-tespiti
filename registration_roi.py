from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import ants
import nibabel as nib
import numpy as np


JHU_LABELS = {
    "SCR_R": 25,
    "SCR_L": 26,
    "SLF_R": 41,
    "SLF_L": 42,
}

SPHERE_VOXEL_CENTERS = {
    "SCR_L": (116, 110, 99),
    "SCR_R": (64, 110, 99),
    "SLF_L": (128, 110, 99),
    "SLF_R": (52, 110, 99),
}


def register_to_template(subject_fa: Path, template_fa: Path) -> dict:
    print(f"[INFO] Registering {subject_fa.name} -> {template_fa.name} (SyN)")
    fixed = ants.image_read(str(template_fa))
    moving = ants.image_read(str(subject_fa))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
    return reg


def make_sphere_mask_template_space(template_img, center_vox, radius_mm: float):
    arr = template_img.numpy()
    zooms = np.array(template_img.spacing, dtype=np.float32)
    radius_vox = radius_mm / zooms
    grid = np.indices(arr.shape)
    dist2 = sum(((grid[i] - center_vox[i]) / radius_vox[i]) ** 2 for i in range(3))
    mask = (dist2 <= 1.0).astype(np.float32)
    return template_img.new_image_like(mask)


def warp_atlas_labels_to_native(
    reg: dict,
    jhu_labels_template: Path,
    subject_fa_ants,
) -> Dict[str, np.ndarray]:
    labels_img = ants.image_read(str(jhu_labels_template))
    labels_native = ants.apply_transforms(
        fixed=subject_fa_ants,
        moving=labels_img,
        transformlist=reg["invtransforms"],
        interpolator="genericLabel",
    )
    arr = labels_native.numpy()
    return {name: (arr == idx) for name, idx in JHU_LABELS.items()}


def warp_sphere_rois_to_native(
    reg: dict,
    template_fa_ants,
    subject_fa_ants,
    radius_mm: float = 2.5,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, center in SPHERE_VOXEL_CENTERS.items():
        sphere_template = make_sphere_mask_template_space(
            template_fa_ants, center, radius_mm
        )
        sphere_native = ants.apply_transforms(
            fixed=subject_fa_ants,
            moving=sphere_template,
            transformlist=reg["invtransforms"],
            interpolator="nearestNeighbor",
        )
        out[name] = sphere_native.numpy() > 0.5
    return out


def save_mask(mask: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path) -> None:
    data = mask.astype(np.uint8)
    nib.save(nib.Nifti1Image(data, reference_img.affine, reference_img.header), str(out_path))


def save_all_masks(
    atlas_masks: Dict[str, np.ndarray],
    sphere_masks: Dict[str, np.ndarray],
    subject_fa: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(str(subject_fa))

    atlas_dir = out_dir / "atlas_rois_native"
    sphere_dir = out_dir / "sphere_rois_native"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    sphere_dir.mkdir(parents=True, exist_ok=True)

    for name, mask in atlas_masks.items():
        save_mask(mask, ref_img, atlas_dir / f"{name}.nii.gz")

    for name, mask in sphere_masks.items():
        save_mask(mask, ref_img, sphere_dir / f"{name}.nii.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register subject FA to template and warp ALPS ROIs to native space."
    )
    parser.add_argument(
        "--subject-fa",
        type=Path,
        default=Path("dti_metrics/fa.nii.gz"),
        help="Subject FA map in native space.",
    )
    parser.add_argument(
        "--template-fa",
        type=Path,
        required=True,
        help="Template FA image, e.g. JHU/MNI FA template.",
    )
    parser.add_argument(
        "--jhu-labels",
        type=Path,
        required=True,
        help="JHU ICBM-DTI-81 white-matter labels image in template space.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("roi_outputs"),
        help="Directory to save native-space ROI masks.",
    )
    parser.add_argument(
        "--sphere-radius-mm",
        type=float,
        default=2.5,
        help="Radius of fixed sphere ROIs in mm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reg = register_to_template(args.subject_fa, args.template_fa)

    template_fa_ants = ants.image_read(str(args.template_fa))
    subject_fa_ants = ants.image_read(str(args.subject_fa))

    print("[INFO] Warping atlas ROIs to native space")
    atlas_masks = warp_atlas_labels_to_native(
        reg=reg,
        jhu_labels_template=args.jhu_labels,
        subject_fa_ants=subject_fa_ants,
    )

    print("[INFO] Warping sphere ROIs to native space")
    sphere_masks = warp_sphere_rois_to_native(
        reg=reg,
        template_fa_ants=template_fa_ants,
        subject_fa_ants=subject_fa_ants,
        radius_mm=args.sphere_radius_mm,
    )

    save_all_masks(
        atlas_masks=atlas_masks,
        sphere_masks=sphere_masks,
        subject_fa=args.subject_fa,
        out_dir=args.output_dir,
    )

    print("[OK] ROI warping finished")
    print(f"[OK] Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()