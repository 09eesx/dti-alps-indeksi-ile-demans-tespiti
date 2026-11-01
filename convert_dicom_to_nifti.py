from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SERIES_CONFIG = {
    "axial_dti": {
        "input_dir": "Axial_DTI",
        "output_name": "axial_dti",
    },
    "sag_ir_spgr": {
        "input_dir": "Sag_IR-SPGR",
        "output_name": "sag_ir_spgr",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected DICOM series to NIfTI using dcm2niix."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root that contains Axial_DTI and Sag_IR-SPGR folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nifti_output"),
        help="Directory where NIfTI files will be written.",
    )
    parser.add_argument(
        "--dcm2niix",
        type=Path,
        default=None,
        help="Optional full path to dcm2niix executable.",
    )
    parser.add_argument(
        "--compress",
        choices=("y", "n"),
        default="y",
        help="Use gzip compression for NIfTI output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def find_dcm2niix(user_path: Path | None) -> str:
    if user_path:
        if user_path.is_file():
            return str(user_path)
        raise FileNotFoundError(f"dcm2niix bulunamadi: {user_path}")

    auto = shutil.which("dcm2niix")
    if auto:
        return auto

    common_locations = [
        Path(r"C:\Program Files\dcm2niix\dcm2niix.exe"),
        Path(r"C:\Program Files\MRIcroGL\dcm2niix.exe"),
        Path(r"C:\Program Files\dcm2niix\bin\dcm2niix.exe"),
    ]
    for candidate in common_locations:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "dcm2niix PATH'te bulunamadi. "
        "--dcm2niix ile exe yolunu ver veya dcm2niix kur."
    )


def find_leaf_dicom_dir(series_root: Path) -> Path:
    dicom_files = sorted(series_root.rglob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"DICOM dosyasi bulunamadi: {series_root}")

    parent_dirs = {dicom_file.parent for dicom_file in dicom_files}
    if len(parent_dirs) == 1:
        return parent_dirs.pop()

    # Birden fazla alt klasor varsa en fazla DICOM iceren klasoru sec.
    counts = []
    for parent_dir in parent_dirs:
        count = sum(1 for _ in parent_dir.glob("*.dcm"))
        counts.append((count, parent_dir))
    counts.sort(reverse=True)
    return counts[0][1]


def run_conversion(
    dcm2niix: str,
    input_dir: Path,
    output_dir: Path,
    output_name: str,
    compress: str,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        dcm2niix,
        "-z",
        compress,
        "-b",
        "y",
        "-o",
        str(output_dir),
        "-f",
        output_name,
        str(input_dir),
    ]

    print(f"\n[INFO] Input : {input_dir}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Komut : {' '.join(command)}")

    if dry_run:
        return

    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_root = (root / args.output).resolve() if not args.output.is_absolute() else args.output

    try:
        dcm2niix = find_dcm2niix(args.dcm2niix)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    for series_key, config in SERIES_CONFIG.items():
        series_root = root / config["input_dir"]
        if not series_root.exists():
            print(f"[WARN] Klasor bulunamadi, atlandi: {series_root}")
            continue

        try:
            dicom_dir = find_leaf_dicom_dir(series_root)
            run_conversion(
                dcm2niix=dcm2niix,
                input_dir=dicom_dir,
                output_dir=output_root / series_key,
                output_name=config["output_name"],
                compress=args.compress,
                dry_run=args.dry_run,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"[ERROR] {series_key}: {exc}")
            return 1

    print("\n[OK] Donusum tamamlandi.")
    print(f"[OK] Cikti klasoru: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
