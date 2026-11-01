from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import (
    TensorModel,
    color_fa,
    fractional_anisotropy,
)
from dipy.segment.mask import median_otsu


@dataclass
class TensorMaps:
    affine: np.ndarray
    header: object
    mask: np.ndarray
    fa: np.ndarray
    md: np.ndarray
    ad: np.ndarray
    rd: np.ndarray
    dxx: np.ndarray
    dyy: np.ndarray
    dzz: np.ndarray
    dxy: np.ndarray
    dxz: np.ndarray
    dyz: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray
    cl: np.ndarray

    @property
    def Dxx(self) -> np.ndarray:
        return self.dxx

    @property
    def Dyy(self) -> np.ndarray:
        return self.dyy

    @property
    def Dzz(self) -> np.ndarray:
        return self.dzz

    @property
    def Dxy(self) -> np.ndarray:
        return self.dxy

    @property
    def Dxz(self) -> np.ndarray:
        return self.dxz

    @property
    def Dyz(self) -> np.ndarray:
        return self.dyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit diffusion tensor and export maps required for ALPS."
    )
    parser.add_argument(
        "--dwi",
        type=Path,
        default=Path("nifti_output/axial_dti/axial_dti.nii.gz"),
        help="Input 4D DWI NIfTI file.",
    )
    parser.add_argument(
        "--bval",
        type=Path,
        default=Path("nifti_output/axial_dti/axial_dti.bval"),
        help="Input bval file.",
    )
    parser.add_argument(
        "--bvec",
        type=Path,
        default=Path("nifti_output/axial_dti/axial_dti.bvec"),
        help="Input bvec file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dti_metrics"),
        help="Directory for output maps.",
    )
    parser.add_argument(
        "--median-radius",
        type=int,
        default=3,
        help="Median filter radius for median_otsu.",
    )
    parser.add_argument(
        "--numpass",
        type=int,
        default=2,
        help="Number of passes for median_otsu.",
    )
    parser.add_argument(
        "--autocrop",
        action="store_true",
        help="Enable autocrop in median_otsu.",
    )
    parser.add_argument(
        "--save-rgb-fa",
        action="store_true",
        help="Also save RGB FA map.",
    )
    return parser.parse_args()


def clean_array(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def save_nifti(data: np.ndarray, affine: np.ndarray, header, out_path: Path) -> None:
    new_header = header.copy()
    img = nib.Nifti1Image(data, affine, header=new_header)
    nib.save(img, str(out_path))


def fit_tensor(
    dwi_path: Path,
    bval_path: Path,
    bvec_path: Path,
    median_radius: int,
    numpass: int,
    autocrop: bool,
) -> TensorMaps:
    print(f"[INFO] Loading DWI: {dwi_path}")
    img = nib.load(str(dwi_path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    if data.ndim != 4:
        raise ValueError(f"Expected 4D DWI, got shape {data.shape}")

    print(f"[INFO] DWI shape: {data.shape}")

    print("[INFO] Loading gradients")
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    if data.shape[3] != len(bvals):
        raise ValueError(
            f"Volume count ({data.shape[3]}) != number of bvals ({len(bvals)})"
        )

    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    print("[INFO] Brain extraction (median_otsu)")
    masked_data, mask = median_otsu(
        data,
        vol_idx=np.where(bvals <= 50)[0],
        median_radius=median_radius,
        numpass=numpass,
        autocrop=autocrop,
    )

    print("[INFO] Fitting diffusion tensor model")
    tenfit = TensorModel(gtab).fit(masked_data, mask=mask)

    evals = tenfit.evals
    evecs = tenfit.evecs
    quad = tenfit.quadratic_form

    print("[INFO] Computing tensor-derived maps")
    fa = clean_array(fractional_anisotropy(evals))
    md = clean_array(np.mean(evals, axis=-1))
    ad = clean_array(evals[..., 0])
    rd = clean_array(np.mean(evals[..., 1:3], axis=-1))

    dxx = clean_array(quad[..., 0, 0])
    dyy = clean_array(quad[..., 1, 1])
    dzz = clean_array(quad[..., 2, 2])
    dxy = clean_array(quad[..., 0, 1])
    dxz = clean_array(quad[..., 0, 2])
    dyz = clean_array(quad[..., 1, 2])

    l1 = evals[..., 0]
    l2 = evals[..., 1]
    l3 = evals[..., 2]
    denom = l1 + l2 + l3
    cl = np.divide(l1 - l2, denom, out=np.zeros_like(denom), where=denom > 1e-9)
    cl = clean_array(cl)

    return TensorMaps(
        affine=affine,
        header=header,
        mask=mask,
        fa=fa,
        md=md,
        ad=ad,
        rd=rd,
        dxx=dxx,
        dyy=dyy,
        dzz=dzz,
        dxy=dxy,
        dxz=dxz,
        dyz=dyz,
        evals=clean_array(evals),
        evecs=clean_array(evecs),
        cl=cl,
    )


def save_tensor_maps(tm: TensorMaps, out_dir: Path, save_rgb_fa: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(tm.fa, tm.affine, tm.header, out_dir / "fa.nii.gz")
    save_nifti(tm.md, tm.affine, tm.header, out_dir / "md.nii.gz")
    save_nifti(tm.ad, tm.affine, tm.header, out_dir / "ad.nii.gz")
    save_nifti(tm.rd, tm.affine, tm.header, out_dir / "rd.nii.gz")

    save_nifti(tm.dxx, tm.affine, tm.header, out_dir / "Dxx.nii.gz")
    save_nifti(tm.dyy, tm.affine, tm.header, out_dir / "Dyy.nii.gz")
    save_nifti(tm.dzz, tm.affine, tm.header, out_dir / "Dzz.nii.gz")
    save_nifti(tm.dxy, tm.affine, tm.header, out_dir / "Dxy.nii.gz")
    save_nifti(tm.dxz, tm.affine, tm.header, out_dir / "Dxz.nii.gz")
    save_nifti(tm.dyz, tm.affine, tm.header, out_dir / "Dyz.nii.gz")

    save_nifti(tm.cl, tm.affine, tm.header, out_dir / "cl_purity.nii.gz")
    save_nifti(tm.mask.astype(np.uint8), tm.affine, tm.header, out_dir / "brain_mask.nii.gz")

    save_nifti(tm.evals, tm.affine, tm.header, out_dir / "evals.nii.gz")
    save_nifti(tm.evecs, tm.affine, tm.header, out_dir / "evecs.nii.gz")

    if save_rgb_fa:
        rgb = color_fa(tm.fa, tm.evecs)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        save_nifti(rgb, tm.affine, tm.header, out_dir / "rgb_fa.nii.gz")

    print(f"[OK] Tensor maps saved to {out_dir}")


def main() -> None:
    args = parse_args()

    tm = fit_tensor(
        dwi_path=args.dwi,
        bval_path=args.bval,
        bvec_path=args.bvec,
        median_radius=args.median_radius,
        numpass=args.numpass,
        autocrop=args.autocrop,
    )
    save_tensor_maps(tm, args.output_dir, args.save_rgb_fa)

    mask = tm.mask.astype(bool)
    print("[OK] Finished")
    print(f"[OK] Output folder: {args.output_dir}")
    print(f"[OK] Mean FA in mask: {tm.fa[mask].mean():.4f}")
    print(f"[OK] Mean MD in mask: {tm.md[mask].mean():.6f}")
    print(f"[OK] Mean AD in mask: {tm.ad[mask].mean():.6f}")
    print(f"[OK] Mean RD in mask: {tm.rd[mask].mean():.6f}")
    print(f"[OK] Mean Dxx in mask: {tm.dxx[mask].mean():.6f}")
    print(f"[OK] Mean Dyy in mask: {tm.dyy[mask].mean():.6f}")
    print(f"[OK] Mean Dzz in mask: {tm.dzz[mask].mean():.6f}")
    print(f"[OK] Mean Dxy in mask: {tm.dxy[mask].mean():.6f}")
    print(f"[OK] Mean Dxz in mask: {tm.dxz[mask].mean():.6f}")
    print(f"[OK] Mean Dyz in mask: {tm.dyz[mask].mean():.6f}")
    print(f"[OK] Mean cl in mask: {tm.cl[mask].mean():.4f}")


if __name__ == "__main__":
    main()
