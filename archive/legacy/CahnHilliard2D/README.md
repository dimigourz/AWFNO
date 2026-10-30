# Cahn–Hilliard 2D

Source notebooks and executed experiment records for the Cahn–Hilliard benchmark in *Alternating Windowed Fourier Neural Operators for Evolution Equations*.

## Notebooks

- `cahn_hillard_final.ipynb` — original sample-size sweep: TB-AWFNO, AWFNO, WFNO, and full-trajectory FNO; Table 2.
- `Cahn_Hilliard_2D_Baselines_v3_FP32FFT.ipynb` — F-FNO, CNO, and U-Net comparison.
- `Cahn_Hilliard_2D_UNet_AR_Only_Colab.ipynb` — separate autoregressive U-Net study.
- `ablations_cahn_hilard.ipynb` — reviewer experiments; uses the accompanying Python driver.
- `AWFNO_H10_Cahn_Hilliard_K10_R2_Colab.ipynb` and `AWFNO_H10_Cahn_Hilliard_K10_R3_Colab.ipynb` — ten-frame-history variants.
- `UNO2D_H10_Cahn_Hilliard_Official_Protocol_Colab.ipynb` — U-NO with ten observed frames.

The `archive/` directory preserves earlier competitive-baseline notebook versions. The notebooks are kept as exported from Google Colab, including their saved outputs.

## Reviewer drivers

- `cahn_hilliard2d_reviewer_ablations_colab.py`
- `cahn_hilliard2d_reviewer_ablations_with_tb_colab.py`

## Data and reproduction

The experiments load `cahn_hilliard_train_grid_32x32.pt` from Google Drive; the dataset and trained checkpoints are **not** committed here. Google Drive paths and saved checkpoint locations in the notebooks reflect the original execution environment. Reproducing training requires the same data, dependency versions, and a GPU. The H10 experiments use additional observed frames and should not be compared directly against single-initial-frame protocols without acknowledging that difference.
