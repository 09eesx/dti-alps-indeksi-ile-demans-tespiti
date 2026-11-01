# DTI-ALPS Pipeline (Python) — Multi-Strategy ROI Comparison

A from-scratch, fully-Python pipeline to compute the DTI-ALPS index (Taoka et al.,
2017), built for *learning* the method end-to-end and for *validating* it against
the established reference pipeline, [gbarisano/alps](https://github.com/gbarisano/alps).

Unlike most public implementations (which use a single, fixed ROI), this pipeline
computes the ALPS-index with **three independent ROI strategies** and reports how
much they agree:

| Strategy | How the ROI is chosen | Reference |
|---|---|---|
| `atlas` | JHU-ICBM-DTI-81 white-matter labels, warped into native space via registration | Yokota et al., 2019 |
| `sphere` | Fixed-coordinate spheres in template space, warped into native space | Liu, Barisano et al., 2023 (gbarisano/alps) |
| `purity` | **Novel**: 5 (configurable) local maxima of a tensor-orthogonality/"purity" score, found *inside* the atlas search window, each a 5mm sphere, pooled + averaged | This project |

## Why a third strategy?

ROI placement is the documented "Achilles' heel" of DTI-ALPS (Taoka et al., 2024).
A single fixed ROI (atlas or sphere) can land on a partial-volume voxel mixing two
fiber populations. The `purity` strategy instead looks, voxel by voxel, for places
where the diffusion tensor's principal eigenvector is cleanly aligned with the
expected axis (z for projection fibers, y for association fibers) AND has high
linear anisotropy (Westin et al., 1997) — i.e. genuinely "unmixed" diffusion. It
then averages across 5 such spots instead of trusting one, and reports the
**spread across those 5 spots** as a built-in uncertainty estimate.

The pipeline also reports a **coefficient of variation across all three
strategies**. High disagreement between methods is flagged (`LOW_AGREEMENT`) —
this is information most pipelines silently discard.

## Pipeline steps

1. **Preprocess** (`preprocess.py`) — denoise + Gibbs-unring (MRtrix3 if available,
   else DIPY Patch2Self fallback), then eddy-current/motion correction (FSL
   `eddy_correct` if available, otherwise skipped with a warning).
2. **Tensor fitting** (`tensor_fit.py`) — DIPY `TensorModel`, brain mask
   (`median_otsu`), exports FA/MD/AD/RD, Dxx/Dyy/Dzz, eigenvalues/eigenvectors,
   and the linear-anisotropy ("purity") map.
3. **Registration + reference ROIs** (`registration_roi.py`) — ANTsPy SyN
   registration of the subject's FA map to a template (JHU-ICBM-FA-1mm by
   default); warps the JHU white-matter atlas labels AND the fixed-sphere ROIs
   from template space into the subject's native space.
4. **Purity-based multi-center ROI** (`purity_roi.py`) — the novel piece: finds
   N (default 5) well-separated, anatomically-constrained local maxima of the
   orthogonality/purity score per region/hemisphere, builds small spheres
   around them.
5. **ALPS-index + ensemble comparison** (`alps_ensemble.py`) — applies the
   Taoka formula per strategy, computes the multi-center pooled estimate + its
   uncertainty, and the cross-strategy agreement.
6. **Orchestration** (`run_pipeline.py`) — runs 1-5 end to end, writes
   `results.csv` with every strategy's L/R ALPS-index plus the ensemble flag.

## Requirements

```
pip install -r requirements.txt
```

External (optional but recommended for full validity):
- **MRtrix3** (`dwidenoise`, `mrdegibbs`) — denoising/unringing
- **FSL** (`eddy_correct`, or for best results the full `eddy`/`eddy_cpu` from
  the reference pipeline) — eddy-current/motion correction
- A template: `$FSLDIR/data/atlases/JHU/JHU-ICBM-FA-1mm.nii.gz` and
  `$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz`

If MRtrix3/FSL are missing, the pipeline still runs (with documented fallbacks/
skips) — useful for learning, but report this explicitly when comparing numbers
against the reference pipeline.

## Usage

```bash
python run_pipeline.py \
    --dwi sub01_dwi.nii.gz --bval sub01.bval --bvec sub01.bvec \
    --template-fa $FSLDIR/data/atlases/JHU/JHU-ICBM-FA-1mm.nii.gz \
    --template-labels $FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz \
    --output-dir out/sub01
```

Output: `out/sub01/results.csv` with columns
`side, atlas_alps, sphere_alps, purity_alps, purity_std_across_pairs,
ensemble_mean, ensemble_cv, ensemble_flag`.

## Validation strategy

Run the same subject through both this pipeline and `alps.sh`
(gbarisano/alps) and compare:
- `atlas_alps` / `sphere_alps` here vs. `alps.csv` from `alps.sh` (should be
  close — these use equivalent ROI definitions, so differences mostly reflect
  preprocessing/registration choices)
- whether `purity_alps` tracks group differences (e.g. age, disease) at least
  as well as the reference, with a tighter (lower-std) estimate

## References

- Taoka T, et al. *Evaluation of glymphatic system activity with the diffusion
  MR technique: DTI-ALPS in Alzheimer's disease cases.* Jpn J Radiol. 2017.
- Taoka T, et al. *DTI-ALPS: Revisiting the Meaning and Significance of the
  Method.* Magn Reson Med Sci. 2024.
- Liu X, Barisano G, et al. *Cross-Vendor Test-Retest Validation of DTI-ALPS.*
  Aging and Disease. 2023. — github.com/gbarisano/alps
- Yokota H, et al. *Diagnostic performance of glymphatic system evaluation
  using DTI in iNPH and mimickers.* Curr Gerontol Geriatr Res. 2019.
- Westin CF, et al. *Geometrical diffusion measures for MRI from tensor basis
  analysis.* ISMRM. 1997.

## Known limitations (be upfront about these in your write-up)

- Without FSL `eddy`/MRtrix3 installed, preprocessing falls back to simpler/partial
  alternatives — note this explicitly when reporting numbers.
- `eddy_correct` does not rotate b-vectors (a real, if minor, limitation vs. full
  `eddy`).
- The purity-based ROI search is still anchored inside the atlas search window —
  it refines *where within* the anatomical region to sample, it does not replace
  anatomical prior knowledge entirely.
---

## Arka plan (Türkçe)

**DTI (Diffusion Tensor Imaging)**, beyin dokusundaki su moleküllerinin hareketini ölçen bir MR tekniğidir. Suyun beyaz madde liflerinin yönünde daha serbest, dik yönde daha kısıtlı hareket etmesi (**anizotropi**) sayesinde lif yönelimi ve mikroyapı bütünlüğü hakkında bilgi verir.

**DTI-ALPS** (*Analysis along the Perivascular Space*), perivasküler boşluklar boyunca su difüzyonunu ölçerek glenfatik sistem aktivitesini dolaylı olarak değerlendiren bir indekstir. Projeksiyon liflerinin (z ekseni) ve asosiasyon liflerinin (y ekseni) olduğu bölgelerde, perivasküler boşluklara paralel (x ekseni) difüzyonun, lif difüzyonuna oranı alınır (Taoka vd., 2017).

Literatürde ALPS indeksi Alzheimer, Parkinson, multipl skleroz ve normal basınçlı hidrosefali gibi durumlarda çalışılmıştır. Bu repo bir tanı aracı değildir; yöntemi uçtan uca öğrenmek ve referans hat ile doğrulamak için yazılmış bir araştırma kodudur.

### DTI indeksleri nasıl hesaplanır?

1. **Difüzyon tensörü:** Farklı yönlerde alınan difüzyon ağırlıklı görüntülerden, her voksel için difüzyonu üç boyutta tanımlayan bir tensör (3×3 matris) kestirilir.
2. **Skaler haritalar:** Tensörün özdeğerlerinden FA, MD, AD ve RD haritaları türetilir. Bunlar ALPS indeksinin girdileridir, indeksin kendisi değildir.
3. **Görselleştirme:** Haritalar renkli haritalar veya grafikler olarak görüntülenir.

![image](https://github.com/09eesx/dti-alps-indeksi-ile-demans-tespiti/assets/123479483/b960f1d2-fb16-41b9-9329-f1c8453bc3eb)

![image](https://github.com/09eesx/dti-alps-indeksi-ile-demans-tespiti/assets/123479483/9d013f93-e838-41f4-884d-afc4c78c890b)

![image](https://github.com/09eesx/dti-alps-indeksi-ile-demans-tespiti/assets/123479483/4f79a92d-d440-45e7-bb08-a3fcef2a6807)

## Veri hakkında

Bu repoda **görüntü verisi yoktur**. Geliştirme sırasında kullanılan DICOM/NIfTI dosyaları ADNI veri kullanım koşulları altındadır ve yeniden dağıtılamaz. Hattı çalıştırmak için kendi verinize erişiminiz olmalıdır; `.gitignore` bu dosyaların yanlışlıkla commit edilmesini engeller.
