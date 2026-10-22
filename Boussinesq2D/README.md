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

## Method-faithful competitive experiments (primary JCP comparison)

**[Launch NEW native-method Colab](https://colab.research.google.com/github/dimigourz/AWFNO/blob/main/Boussinesq2D/Native_Method_Baselines_2D_Colab.ipynb)** · [Native training and evaluation source](native_method_baselines_2d.py) · [Architectures](reference_architectures_2d.py)

This suite corrects a **material protocol error** in the earlier K30 comparisons: windowed prediction is the AWFNO contribution, not the baseline methods' standard temporal formulations.

| Model | Established prediction formulation | Training examples/trajectory | True observations required at test |
|---|---|---:|---:|
| **AWFNO** | *Proposed method*: windowed, **K=30** | 3 window starts | 1 initial field |
| **F-FNO** | **One-step** recurrent map u(t)->u(t+dt) (FourierFlow/Cahn–Hilliard) | 100 adjacent pairs | 1 initial field |
| **U-Net** | **One-step** recurrent map (Cahn–Hilliard/Burgers) | 100 adjacent pairs | 1 initial field |
| **CNO** | **Direct-time** operator (u0,t)->u(t), queried at 100 times (Cahn–Hilliard) | 100 time-query pairs | 1 initial field |
| **U-NO-H10** | **Ten-frame history -> next frame**, autoregressive (our earlier Burgers UNO-H10) | 91 history/target pairs | **10 initial fields** |

*IMPORTANT INFORMATION MATCHING:* U-NO-H10 has additional observed input fields u1,...,u9 unavailable to the other models, and cannot be ranked as an initial-state-only competitor. **Its full-trajectory L2 is deliberately omitted**, since those observed frames must not be counted as prediction. All five models can be evaluated on a common post-t10 interval, but even this does not equalize input information. Its final-time metric is presented explicitly as **H10 conditioned**, and should be in a distinct table/panel.

The native models use the original 64×128×101 HyTeG dataset, disjoint 400/30/70 trajectory partitions, and the **original annulus physical absolute-L2 quadrature**, but **NOT** the K30 windows used by AWFNO. Like the Cahn–Hilliard baselines, these models optimize pairwise or time-query targets and use model-specific batch sizes, learning rates, and weight decay. They use 200 epochs by default, validation every ten epochs, best-val checkpoint selection, checkpoint/resume every ten epochs, and **float32 FFT for F-FNO**.

**Training cost:** At N=400, the F-FNO one-step method has 40,000 train pairs/epoch, batch size 8, and 200 epochs, or **one million optimizer updates**, versus 5,700 updates for the earlier K30 architecture-adaptation test. Start with N=25 and confirm learning curves and memory before the full sweep.

This new study saves to **`MyDrive/anus/Boussinesq_JCP/native_reference_2d/`**, independent of all prior runs. The earlier [K30 comparison Colab](Competitive_Baselines_2D_Colab.ipynb) and its completed sample-size sweep under `competitive_2d_K30/` are retained **only as K30 protocol adaptations**, not as evidence of each baseline's established temporal formulation. The native-method baseline suite is new; its numerical accuracy has **not yet been established**. A completed K30 run is not a completed native-method run.

### Reproducing the native baseline experiments (Colab)

1. Open [Native_Method_Baselines_2D_Colab.ipynb](https://colab.research.google.com/github/dimigourz/AWFNO/blob/main/Boussinesq2D/Native_Method_Baselines_2D_Colab.ipynb), attach an **NVIDIA GPU** runtime, and mount Drive. Dataset expected at `/content/gdrive/MyDrive/anus/hyteg_annulus_c_grid_64x128_samples500.pt`; adjust `DATASET` if your Drive uses another location. Drive already being mounted is harmless; `torch.cuda.is_available()` must be true.
2. In the configuration cell select `MODELS='FFNO,CNO,UNet,UNO'`, `SAMPLE_SIZES='25'`, `SEEDS='1234'`, `EPOCHS=200`, and `SMOKE_TEST=True` for the initial verification. Each selected model is smoke-tested independently for output shape `(B,64,128)`, finite forward values and finite backpropagation gradients. **Do not launch the expensive training sweep until every smoke test passes.**
3. Rerun the **source-download cell**, then the **architecture-smoke-test cell** whenever the GitHub Python source changes, or reopen the latest Colab notebook. The smoke-test cell now also refreshes `/content/reference_architectures_2d.py` with cache-bypass and verifies the native U-Net constructor before spawning test subprocesses. Merely editing GitHub does **not** update files already copied into an active Colab runtime.
4. Start the training cell after smoke tests pass. Training logs stream to Colab and to `native_reference_2d/native_training_stdout.log`. Each completed run writes `result.json`, plots and checkpoints under `native_reference_2d/runs/<model>/N<size>_seed<seed>_<config-hash>/`. The aggregate table is `native_reference_2d/results.csv`. Checkpoints are saved periodically for restart/resumption if the exact configuration matches; leave `FORCE_RETRAIN=False` to reuse existing runs. Start at `N_train=25` because native temporal protocols are much more expensive than K30 adaptations.

**Known U-Net smoke-test constructor fix (22 October 2026).** The first native-method version failed before training with

```text
TypeError: UNet2D.__init__() got an unexpected keyword argument 'width'
```

`UNet2D` has signature `UNet2D(k, r, theta, channels=32)`, so the native factory in [`reference_architectures_2d.py`](reference_architectures_2d.py) was corrected from `UNet2D(1, r, theta, width=32)` to **`UNet2D(1, r, theta, channels=32)`**. The [native Colab](Native_Method_Baselines_2D_Colab.ipynb) also refreshes this module and checks the keyword before attempting training. If your traceback still shows `width=32`, you are executing a **stale local script**; rerun the source-download cell and smoke tests. **The constructor correction has been committed but a successful full native smoke test and training run have not yet been confirmed.**

**Cost and comparability notes:** F-FNO uses 24 spectral/feed-forward layers, one-step targets, 200 epochs and batch size 8; at `N_train=400` this corresponds to `400*100/8*200 = 1,000,000` optimizer updates. U-NO-H10 uses **ten observed initial frames** and its full-trajectory error is intentionally undefined; compare its final-time and post-`t_{10}` errors separately, not directly in an initial-state-only ranking. All models retain the same held-out test trajectories and physical annular error quadrature, but **their training problems and amounts of observed initial data are not identical**.

## Modern competitive models — reference-oriented, Cahn–Hilliard-verified backbones

**[Launch updated Colab](https://colab.research.google.com/github/dimigourz/AWFNO/blob/main/Boussinesq2D/Competitive_Baselines_2D_Colab.ipynb)** · [Training and evaluation driver](competitive_baselines_2d.py) · [Readable reference-style architectures](reference_architectures_2d.py)

The first draft of this Boussinesq baseline comparison contained loose architectural approximations. They **have been replaced** to track the implementations previously used for Burgers and Cahn–Hilliard more closely:

| Model | Architecture and provenance | Differences from published task |
|---|---|---|
| **F-FNO** | Adapted from our Cahn–Hilliard `FFNO2D_AR` / [FourierFlow](https://github.com/alasdairtran/fourierflow): 24 factorized spectral/feed-forward layers, spectral weights **shared across depth**, residual backcasts, width 64, 16 modes per axis, 4× feed-forward expansion; float32 FFT. | Predicts the **next 30 fields** rather than a single next field; single-state input and the same 30-frame rollout as AWFNO. |
| **CNO** | Port of the **official-simplified CNO2d** architecture used in our Cahn–Hilliard baseline, with bicubic antialiased LeakyReLU blocks, 3 multiresolution layers, per-level residual subnets and a residual neck, with multiscale skip paths. See the [official CNO repository](https://github.com/camlab-ethz/ConvolutionalNeuralOperator). | Adapted to rectangular 64×128 polar coordinates and a **30-frame window output head**, rather than Cahn–Hilliard's time-query direct CNO. Uses periodic angular and nonperiodic radial padding. Not the original NeurIPS 2023 classic CNO. |
| **U-Net** | Four-level convolutional encoder–decoder with group norm, transposed upconvolutions, skip connections; follows our Cahn–Hilliard U-Net baseline. | 30-frame output, polar coordinate/angle encoding and radial/azimuthal boundary behavior. |
| **U-NO** | Imports **`neuralop.models.UNO`** from `neuraloperator==2.0.0`, with a 5-level down/up Fourier configuration and horizontal skips. See the [library UNO reference](https://neuraloperator.github.io/dev/modules/generated/neuralop.models.UNO.html). | Adds polar coordinate features and 30-frame output head. |

**Fixed annular comparison protocol:** exact original dataset and training/validation/test trajectories (400/30/70); sample sizes 25–400; `Nr×Ntheta×Nt=64×128×101`; **K=30**, window stride 30, no test teacher forcing; 150 epochs, MSE, AdamW + cosine schedule, clip norm 1 for new models; best validation-window physical-L2 checkpoint. All reported test errors use the original absolute, annulus-weighted (L^2) quadrature. The original AWFNO/FNO/WFNO rows are loaded from previous checkpoint evaluations, not retrained.

**Design caveat:** Holding **K=30** makes the mapping and training target identical to AWFNO, but the original Burgers/Cahn–Hilliard implementations used different temporal protocols (e.g. single-step F-FNO and direct-time CNO). Thus this is a **reference-architecture, same-task comparison**, **not an exact reproduction of the authors' benchmark training protocols**. The 30-channel prediction head is an intentional experimental adaptation, not a claim of source-identical code.

**Outputs:** `MyDrive/anus/Boussinesq_JCP/competitive_2d_K30/` with resumed checkpoints; aggregate JSON/CSV; full/final mean, median, SD; test-case errors; per-time error and annulus figures; spectral/sharp-region diagnostics; parameter/real-DOF counts; training and inference timing; peak GPU allocation. It also includes combined charts with the *previously computed* AWFNO/FNO/WFNO values.

The **K30 adaptation** Colab installs `neuraloperator==2.0.0` and provides independent forward/backward architecture smoke checks; its complete N=25–400, seed-1234 sample-size sweep was run by the user. This does **not** validate the separate native-method suite above. The first native-method smoke test exposed the U-Net constructor error documented earlier; the corrected code still requires a successful rerun.

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

**Two distinct competitive suites are available above:** the **executed K30 architecture-adaptation comparison** and the **new, as-yet unvalidated native temporal-protocol comparison**. The original one-run ablation notebook does not execute either suite automatically. CNO and U-NO are research adaptations of the documented backbone/information protocols; do not claim exact reproduction of original authors' full training configurations.

The production dataset is not included and was unavailable to rerun in this environment; the controlled driver passed a synthetic annulus smoke test. The original FNO dependency is pinned to neuraloperator==2.0.0. Check metric definitions, exact temporal conditioning, and hardware when making quantitative cross-method comparisons.
