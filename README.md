# Alternating Windowed Fourier Neural Operators (AWFNO)

Research code and numerical experiments for learning the evolution operators of nonlinear partial differential equations.

**Current benchmarks:** 1D viscous Burgers and 2D annular Boussinesq (HyTeG scalar-field benchmark). Each PDE has its own numerical-experiment folder.

## Experiments

| Directory | Contents |
| --- | --- |
| [`Burgers1D/01_AWFNO_H10_K5/`](Burgers1D/01_AWFNO_H10_K5/) | AWFNO with 10 observed input states and a 5-state forecast window. |
| [`Burgers1D/02_UNO_H10/`](Burgers1D/02_UNO_H10/) | U-NO-inspired 1D encoder–decoder adapted to a 10-state history, rolled forward recursively. |
| [`Burgers1D/03_Architecture_Ablations/`](Burgers1D/03_Architecture_Ablations/) | Controlled component, local/Fourier ordering, and radius/modes/window sensitivities. Includes Python experiment engine and Colab driver. |
| [`Burgers1D/04_Competitive_Baselines/`](Burgers1D/04_Competitive_Baselines/) | F-FNO, autoregressive U-Net, and direct CNO baseline implementations. |

## PDEBench dataset

All Burgers 1D notebooks are configured for the PDEBench **viscous Burgers** training file with viscosity `nu = 0.001`:

**[`1D_Burgers_Sols_Nu0.001.hdf5` — download exact HDF5 from PDEBench DaRUS](https://darus.uni-stuttgart.de/api/access/datafile/268190)** (~7.7 GB).

- **Official dataset DOI/catalog:** [PDEBench Datasets — DOI:10.18419/DARUS-2986](https://doi.org/10.18419/DARUS-2986)
- **PDEBench Hugging Face listing:** [`pdebench/Burgers`](https://huggingface.co/datasets/pdebench/Burgers/tree/main) (select `1D_Burgers_Sols_Nu0.001.hdf5`, **not** `Nu0.01`).
- **DaRUS current-revision file checksum (MD5):** `44cb784d5a07aa2b1c864cabdcf625f9`. The dataset archive has older versions with a different checksum; verify your local file if reproducing these experiments.

Download in Colab or another Linux environment, with sufficient disk space:

```bash
wget -O 1D_Burgers_Sols_Nu0.001.hdf5 \
  "https://darus.uni-stuttgart.de/api/access/datafile/268190"
md5sum 1D_Burgers_Sols_Nu0.001.hdf5
```

Copy or mount the dataset where the notebooks expect it, or update `DATA_PATH` to the downloaded file. The HDF5 dataset is not committed to GitHub.

## Boussinesq annulus experiments

[**Boussinesq2D/ — one-run Colab notebook, original benchmark code, and numerical documentation**](Boussinesq2D/)

This folder contains the original four-model sample-size study, the three-seed controlled architectural experiments, and a **separate competitive-baseline Colab** for FourierFlow/Cahn–Hilliard-style F-FNO, a CNO simplified-reference backbone, a four-level U-Net, and the actual `neuralop.models.UNO` library architecture. All four use an adapted K=30 annular prediction head. The new baselines have **not** been trained on the production dataset.

## Numerical comparison: Burgers H10 sample-size sweep

Reported **mean absolute physical L2** errors (not normalized relative L2). Individual model runs, rather than multi-seed mean ± confidence interval.

| Training trajectories | AWFNO-H10-K5, full | U-NO-H10, full | AWFNO, final | U-NO, final |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 1.682620 | 1.370799 | 0.140898 | 0.095506 |
| 50 | 0.873756 | 0.849425 | 0.054573 | 0.052009 |
| 100 | 0.589395 | 0.632981 | 0.035648 | 0.036892 |
| 200 | 0.324680 | 0.403800 | 0.020628 | 0.024316 |
| 400 | 0.170492 | 0.205577 | 0.010623 | 0.014145 |
| 800 | 0.115609 | 0.155182 | 0.006596 | 0.009959 |

The AWFNO configuration contains **128,453 trainable tensor elements** and the U-NO adaptation **2,830,001**. The models differ in output chunking, training schedule, model capacity, and computational workload. These results **are not parameter- or compute-matched** and should not be interpreted as a controlled architecture-only effect. U-NO is a research adaptation, not an official U-NO result published by its original authors.

## Reproducibility and scope

- The notebooks use **Google Colab** and HDF5 Burgers data on **Google Drive**; configure the paths for your environment.
- The HDF5 datasets, learned weights/checkpoints, and full experiment-output directories are not included.
- Source and markdown notebook cells are retained; executed output cells have been cleared for GitHub.
- The `Burgers1D/03_Architecture_Ablations/` directory includes both the Colab launcher and the companion `burgers1d_reviewer_ablations_colab.py` script.
- Comparisons between notebooks require checking dataset split, exact history length, rollout horizon, loss/metric implementation, checkpoint selection, hardware, epochs, and random seeds. Some baselines use **H1** (initial-state-only); those runs must not be presented as directly information-matched to **H10** runs.
- This archive records the supplied research code and numerical summaries; it has not been independently rerun as part of repository publication.

**Research status:** Experimental source archive for an ongoing manuscript revision; results and documentation may change.
