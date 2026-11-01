from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def infer_grad_paths(dwi_path: Path, bval: Path | None, bvec: Path | None) -> tuple[Path, Path]:
    if bval is not None and bvec is not None:
        return bval, bvec

    stem = dwi_path.name
    if stem.endswith(".nii.gz"):
        base = dwi_path.with_name(stem[:-7])
    elif stem.endswith(".nii"):
        base = dwi_path.with_name(stem[:-4])
    else:
        base = dwi_path

    guessed_bval = base.with_suffix(".bval")
    guessed_bvec = base.with_suffix(".bvec")

    if not guessed_bval.exists():
        raise FileNotFoundError(f"bval file not found: {guessed_bval}")
    if not guessed_bvec.exists():
        raise FileNotFoundError(f"bvec file not found: {guessed_bvec}")

    return guessed_bval, guessed_bvec


def copy_if_needed(src: Path, dst: Path) -> None:
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def denoise_and_unring(
    dwi_path: Path,
    bval_path: Path,
    bvec_path: Path,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    denoised = out_dir / "dwi_denoised.nii.gz"
    unringed = out_dir / "dwi_denoised_unring.nii.gz"

    if has_tool("dwidenoise") and has_tool("mrdegibbs"):
        print("[INFO] MRtrix3 found -> using dwidenoise + mrdegibbs")
        subprocess.run(["dwidenoise", str(dwi_path), str(denoised)], check=True)
        subprocess.run(["mrdegibbs", str(denoised), str(unringed)], check=True)
        copy_if_needed(bval_path, out_dir / "dwi_denoised_unring.bval")
        copy_if_needed(bvec_path, out_dir / "dwi_denoised_unring.bvec")
        return unringed

    print("[WARN] MRtrix3 not found on PATH.")
    print("[WARN] Falling back to DIPY Patch2Self denoising (no unringing step).")

    from dipy.denoise.patch2self import patch2self
    from dipy.io.gradients import read_bvals_bvecs

    img = nib.load(str(dwi_path))
    data = img.get_fdata(dtype=np.float32)
    bvals, _ = read_bvals_bvecs(str(bval_path), str(bvec_path))

    denoised_data = patch2self(data, bvals, model="ols", shift_intensity=True)
    nib.save(
        nib.Nifti1Image(denoised_data.astype(np.float32), img.affine, img.header),
        str(denoised),
    )

    copy_if_needed(bval_path, out_dir / "dwi_denoised.bval")
    copy_if_needed(bvec_path, out_dir / "dwi_denoised.bvec")
    return denoised


def eddy_correct(
    dwi_path: Path,
    bval_path: Path,
    bvec_path: Path,
    ref_volume_index: int,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / "dwi_eddy_corrected"
    out_nii = out_dir / "dwi_eddy_corrected.nii.gz"
    out_bval = out_dir / "dwi_eddy_corrected.bval"
    out_bvec = out_dir / "dwi_eddy_corrected.bvec"

    if not has_tool("eddy_correct"):
        print("[WARN] FSL eddy_correct not found on PATH.")
        print("[WARN] Skipping eddy/motion correction.")
        print("[WARN] ALPS-index will be computed without eddy-current/motion correction.")
        return dwi_path, bval_path, bvec_path

    print(f"[INFO] Running FSL eddy_correct (reference volume index={ref_volume_index})")
    subprocess.run(
        ["eddy_correct", str(dwi_path), str(out_prefix), str(ref_volume_index), "trilinear"],
        check=True,
    )

    copy_if_needed(bval_path, out_bval)
    copy_if_needed(bvec_path, out_bvec)
    return out_nii, out_bval, out_bvec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess DWI with denoising, unringing, and optional eddy correction."
    )
    parser.add_argument(
        "--dwi",
        type=Path,
        default=Path("nifti_output/axial_dti/axial_dti.nii.gz"),
        help="Input DWI NIfTI.",
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
        default=Path("preprocessed_dwi"),
        help="Output directory.",
    )
    parser.add_argument(
        "--ref-volume-index",
        type=int,
        default=0,
        help="Reference volume index for eddy_correct.",
    )
    parser.add_argument(
        "--skip-eddy",
        action="store_true",
        help="Skip eddy/motion correction even if FSL is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dwi_path = args.dwi.resolve()
    bval_path, bvec_path = infer_grad_paths(dwi_path, args.bval, args.bvec)

    preprocess_dir = args.output_dir.resolve()
    denoise_dir = preprocess_dir / "01_denoise_unring"
    eddy_dir = preprocess_dir / "02_eddy"

    processed_dwi = denoise_and_unring(
        dwi_path=dwi_path,
        bval_path=bval_path,
        bvec_path=bvec_path,
        out_dir=denoise_dir,
    )

    if processed_dwi.name == "dwi_denoised_unring.nii.gz":
        processed_bval = denoise_dir / "dwi_denoised_unring.bval"
        processed_bvec = denoise_dir / "dwi_denoised_unring.bvec"
    else:
        processed_bval = denoise_dir / "dwi_denoised.bval"
        processed_bvec = denoise_dir / "dwi_denoised.bvec"

    if args.skip_eddy:
        final_dwi = processed_dwi
        final_bval = processed_bval
        final_bvec = processed_bvec
        print("[INFO] Eddy correction skipped by user.")
    else:
        final_dwi, final_bval, final_bvec = eddy_correct(
            dwi_path=processed_dwi,
            bval_path=processed_bval,
            bvec_path=processed_bvec,
            ref_volume_index=args.ref_volume_index,
            out_dir=eddy_dir,
        )

    print("[OK] Preprocessing finished")
    print(f"[OK] Final DWI : {final_dwi}")
    print(f"[OK] Final bval: {final_bval}")
    print(f"[OK] Final bvec: {final_bvec}")


if __name__ == "__main__":
    main()