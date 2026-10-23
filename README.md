# Alternating Windowed Fourier Neural Operators (AWFNO)

Research code and numerical experiments for learning the evolution operators of nonlinear partial differential equations.

**Current benchmarks:** 1D viscous Burgers, 2D annular Boussinesq (HyTeG scalar-field benchmark), and 2D Kolmogorov Navier–Stokes vorticity. Each PDE has its own numerical-experiment folder.

## Navier–Stokes Kolmogorov flow (JCP)

[**Corrected 101-frame, five-model comparative and controlled ablation Colab**](NavierStokes2D/Kolmogorov_NS_JCP_Original_Five_Model_Colab.ipynb) — independently compares TB-AWFNO, AWFNO, standard WFNO, real-parameter-matched WFNO, and native full-trajectory FNO at 25/50/100/200 training trajectories, then runs validation-only ordering/component/K/mode/radius/depth/temporal-basis ablations. [Full benchmark protocol and output table paths](NavierStokes2D/README.md). Original verified dataset and earlier 22-run exploration remain unchanged; **do not conflate old exploratory validation errors with the new five-model test table**.

### Additional Re=500 reference-oriented baselines

[**Open F-FNO / CNO / U-Net / UNO-H10 directly in Google Colab**](https://colab.research.google.com/github/dimigourz/AWFNO/blob/colab/navier-stokes-re500-extra-baselines-20261023/notebooks/NavierStokes_Re500_FFNO_CNO_UNet_UNO_H10_Colab.ipynb)

The companion notebook adds **16 independent fits**: four model families at N = 25, 50, 100 and 200, on the verified 101-frame, 128×128, Re=500 dataset. The original Navier–Stokes comparison and ablation suite are **unchanged**.

| Baseline | Temporal formulation | Observed test frames |
| --- | --- | --- |
| 24-layer F-FNO | One-step autoregressive | Initial frame only |
| CNO | Direct-time, initial state + target time | Initial frame only |
| U-Net | One-step autoregressive | Initial frame only |
| UNO-H10 | Seven-block spectral U-shaped network; 40-step differentiable training rollout | First 10 frames |

**Fairness:** UNO-H10 has nine additional exact observations; its scores must not be treated as equal-input evidence of architectural superiority. These are reference-oriented **adaptations**, not untouched author-provided implementations. Models retain their own temporal training formulations, so compute and optimizer-step budgets also differ.

The notebook includes readable Python source, model-shape/gradient smoke tests, resumable Drive checkpoints, validation-only model selection, and test-set evaluation **only after all 16 fits finish**. It saves separate validation/test CSVs, per-model parameter and timing summaries, physical L² and H¹ diagnostics, and field plots. Merged figures are conditional on the original locked test table being present. **No new numerical test results have been verified here.** See the [Navier–Stokes experiment guide](NavierStokes2D/README.md) for dataset provenance, split and the original suite.

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

**Boussinesq dataset:** [`hyteg_annulus_c_grid_64x128_samples500.pt` (Google Drive, individual file)](https://drive.google.com/file/d/1ESx5j66waqVytDR01h0fw_7mmHOZYS1b/view?usp=drive_link). The file owner must set **General access → Anyone with the link → Viewer** for readers to open/download it. Public anonymous access has not been independently verified.

The primary competitive suite is [**Native_Method_Baselines_2D_Colab.ipynb**](Boussinesq2D/Native_Method_Baselines_2D_Colab.ipynb): **F-FNO and U-Net one-step autoregressive**, **CNO direct-time**, and **U-NO-H10 history-to-one-step**. U-NO-H10 consumes ten observed fields, so its results are **not information-matched** to initial-state-only operators. The earlier **completed K30 adaptation** experiment ([separate K30 notebook](Boussinesq2D/Competitive_Baselines_2D_Colab.ipynb)) evaluates different backbones using **AWFNO's K30 forecasting protocol**, not the baselines' own temporal formulations; do not conflate those results with the native comparison. The native-method **N25 runs are complete**. The updated Colab now targets **N=50,100,200 only**, using GPU-measured batch-size autotuning and checkpointed early stopping to reduce computational cost; the N25 fixed-batch results remain labelled as a different optimization configuration.

**Boussinesq setup and troubleshooting:** see the [experiment README](Boussinesq2D/README.md#reproducing-the-native-baseline-experiments-colab). The first native U-Net shape test exposed a factory argument error (`width=32` instead of `channels=32`); this has been corrected in [`reference_architectures_2d.py`](Boussinesq2D/reference_architectures_2d.py). Refresh the downloaded source in Colab before rerunning the smoke tests.

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
