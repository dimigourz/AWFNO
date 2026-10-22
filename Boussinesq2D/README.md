# Annular Boussinesq / HyTeG — AWFNO numerical experiments

## Colab execution

**[Open the one-run notebook in Google Colab](https://colab.research.google.com/github/dimigourz/AWFNO/blob/main/Boussinesq2D/Boussinesq2D_JCP_OneRun_Colab.ipynb)**

1. Select a GPU runtime.
2. Put the dataset at **MyDrive/anus/hyteg_annulus_c_grid_64x128_samples500.pt** or edit DATASET in the notebook.
3. Select **Runtime → Run all**. The default executes the original sample-size sweep followed by all controlled reviewer experiments.
4. Read results and checkpoints under **MyDrive/anus/Boussinesq_JCP**. Re-executing with FORCE_RETRAIN=False reuses finished configurations.

The notebook downloads the readable Python experiment engine directly from this repository and verifies its SHA256 before writing it to /content/annulus2d_jcp_experiments.py. No opaque embedded archive or additional script upload is required. Readable standalone implementation: [annulus2d_jcp_experiments.py](annulus2d_jcp_experiments.py). [Original benchmark source](annulus2d_benchmark_original.py).

## Dataset and field

This is the **HyTeG annular convection** scalar-field benchmark; it is not the PDEBench 1D Burgers dataset. Expected tensor c_grid (or Y_output) has shape (N,T,Nr,Ntheta,1), with r_grid and theta_grid. We model only scalar c_grid, **not** velocity or pressure.

Original settings: polar grid 64×128; K=30; width=32; four layers; modes=(16,24); local radius=2; 150 epochs. Fixed split of 400 train pool, 30 validation, 70 test trajectories. Five training-set sizes: 25, 50, 100, 200, 400.

## Burgers-style progress and reporting

The **original** sample-size suite now trains and evaluates one training-set size at a time. After each `N_train = 25, 50, 100, 200, 400` experiment, its test metrics are printed and appended to a cumulative **CSV and JSON**; sample-size plots are updated immediately. The output remains usable if a later run fails or the Colab runtime disconnects.

Recorded for **TB-AWFNO / AWFNO / WFNO / FNO**: full/final absolute physical L2 mean, standard deviation and median; trainable parameter count and real scalar degrees of freedom; training time; total wall-clock inference over the held-out test set, amortized inference per trajectory, and peak GPU memory during full-test rollout; low/high azimuthal-frequency L2 and high-gradient-region diagnostics. Full-trajectory and final-time sample-size plots, loss curves, time-dependent errors and annulus comparisons are retained. Selected plots are previewed in Colab, while all saved figures remain available on Drive.

**Timings are not directly comparable to Burgers notebook median per-trajectory timings:** this notebook reports total inference time for the complete Boussinesq test set plus the amortized time per trajectory.

The Colab launcher now streams training output live, saves `original_stdout.log` and `rebuttal_stdout.log`, and prints a heartbeat when there has been no output. Reopen the GitHub Colab link to obtain this version; an already running kernel will continue to run its previously loaded script.

## Next comparison: F-FNO, CNO-style, U-Net and U-NO-style

**[Open competitive baseline Colab](https://colab.research.google.com/github/dimigourz/AWFNO/blob/main/Boussinesq2D/Competitive_Baselines_2D_Colab.ipynb)** · [Read the complete Python source](competitive_baselines_2d.py)

The four networks are *separate implementations*: (i) factorized 1D spectral convolutions along the radial and azimuthal axes with residual blocks (F-FNO), (ii) a filtered, multiresolution convolutional encoder-decoder with upsample–nonlinearity–low-pass stages (CNO-style), (iii) a conventional convolutional U-Net, and (iv) a U-shaped spectral-operator encoder-decoder with skip connections (U-NO-style).

**Protocol held fixed:** identical original dataset `hyteg_annulus_c_grid_64x128_samples500.pt`, disjoint train/val/test partition **400/30/70**, sample-size sweep `25,50,100,200,400`, one observed scalar field, predictions of the next **K=30** frames, autoregressive 100-step rollout from true initial state only, 150 epochs, training-only Gaussian normalizers, absolute physical annular L2 metrics. For these new baselines, the training recipe is common: AdamW + cosine annealing, MSE with zero auxiliary final loss, gradient clipping at norm 1 and best validation-window-L2 checkpoint. The original AWFNO sample-size results are imported only for plotting, not retrained.

**Numerical outputs:** an incremental `competitive_2d_K30/results.csv` + JSON; model checkpoints; loss/validation curves; full/final mean, SD, median; error versus time; frequency- and sharp-region errors; test-case CSVs; best/worst annulus visualizations; train and inference seconds; trainable tensor elements and real scalar degrees of freedom; allocated peak inference GPU memory; optional full-test rollout arrays. These are saved independently from the original and ablation folders. `BASELINE_SEEDS=1234` by default, with additional seeds configurable. The Colab notebook runs forward/backward smoke tests before training and can disconnect after finishing.

**Attribution and limits:** These implementations are **research adaptations**, not verbatim code releases by the original researchers. F-FNO follows the central factorized Fourier design of [Tran et al. (ICLR 2023)](https://openreview.net/forum?id=tmIiMPl4IPa); the **CNO-style** network uses alias-reducing spatial filtering inspired by [Raonić et al. (NeurIPS 2023)](https://arxiv.org/abs/2302.01178), but does **not** establish the exact continuous-discrete equivalence of the authors' CNO; and the **U-NO-style** network is a multiresolution spectral U architecture inspired by [Rahman, Ross & Azizzadenesheli (2022)](https://arxiv.org/abs/2204.11127). The U-Net is a standard discretization-dependent image-to-image baseline. **Do not describe CNO-style or U-NO-style results as the original implementations without verifying against their reference repositories.** There is no parameter- or compute-matched claim across the four different models.

These training runs have **not** been executed on the production HyTeG dataset by ChatGPT; no numerical performance is asserted until the Colab runs complete.

## Implemented experimental coverage

| Suite | Methods / diagnostics |
| --- | --- |
| Original sample-size | TB-AWFNO, AWFNO, WFNO, full-trajectory FNO |
| Ordering | LGL, LLG, GLL; three random seeds |
| Components | All-local LLL and all-global GGG |
| Matching | AWFNO vs width-adjusted WFNO |
| Sensitivity | K = 1, 10, 30; radius = 1, 2, 3; Fourier modes = (8,12), (16,24), (24,32) |
| Temporal basis | Basis-decoder ablation |
| Diagnostics | Annulus-weighted absolute physical full/final L2, low/high azimuthal-frequency errors, top-gradient-region error, train/rollout times, parameter count and peak GPU memory |

Controlled runs use N=400 with seeds 1234,1235,1236 and write trial JSON plus results.csv. The notebook produces results_aggregated_by_seed.csv. The original sample-size sweep is **not** replicated across multiple seeds.

**Separate competitive implementations now available above:** F-FNO, U-Net, CNO-style and U-NO-style. The original one-run ablation notebook does not execute these baselines automatically. CNO and U-NO reference-implementation parity and numerical results remain outstanding.

The production dataset is not included and was unavailable to rerun in this environment; the controlled driver passed a synthetic annulus smoke test. The original FNO dependency is pinned to neuraloperator==2.0.0. Check metric definitions, exact temporal conditioning, and hardware when making quantitative cross-method comparisons.
