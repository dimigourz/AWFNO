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

## Competitive baselines — original temporal prediction protocols

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

The native models use the original 64×128×101 HyTeG dataset, disjoint 400/30/70 trajectory partitions, and the **original annulus physical absolute-L2 quadrature**, but **NOT** the K30 windows used by AWFNO. Like the Cahn–Hilliard baselines, these models optimize pairwise or time-query targets and use model-specific batch sizes, learning rates, and weight decay. The original N25 baselines were trained for 200 epochs with fixed small batches. The accelerated N50/100/200 experiments have a **maximum** of 200 epochs, validation every ten epochs, checkpointing every ten epochs, best-validation selection and configurable early stopping, with float32 FFT for F-FNO.

**Training cost:** At N=200, the F-FNO one-step method has 20,000 train pairs/epoch; the previous fixed batch of 8 and 200 epochs would require 500,000 optimizer updates. The accelerated version benchmarks batches 8/16/32/64 and selects the best measured throughput subject to a 75%-of-VRAM *allocated-memory* safety budget (not a requirement to fill the entire GPU). It also stops if validation has no meaningful improvement for 40 epochs after at least 80 epochs. The actual number of updates and epochs is recorded per run.

This new study saves to **`MyDrive/anus/Boussinesq_JCP/native_reference_2d/`**, independent of all prior runs. The earlier [K30 comparison Colab](Competitive_Baselines_2D_Colab.ipynb) and its completed sample-size sweep under `competitive_2d_K30/` are retained **only as K30 protocol adaptations**, not as evidence of each baseline's established temporal formulation. The original native-method N25 runs **have completed**, with independent prediction-protocol evaluation. N50, N100 and N200 are the remaining accelerated runs. The completed K30 experiment is separate and not interchangeable with native-method results.

### Reproducing the native baseline experiments (Colab)

1. Open [Native_Method_Baselines_2D_Colab.ipynb](https://colab.research.google.com/github/dimigourz/AWFNO/blob/main/Boussinesq2D/Native_Method_Baselines_2D_Colab.ipynb), attach an **NVIDIA GPU** runtime, and mount Drive. Dataset expected at `/content/gdrive/MyDrive/anus/hyteg_annulus_c_grid_64x128_samples500.pt`; adjust `DATASET` if your Drive uses another location. Drive already being mounted is harmless; `torch.cuda.is_available()` must be true.
2. In the configuration cell select `MODELS='FFNO,CNO,UNet,UNO'`, `SAMPLE_SIZES='50,100,200'`, `SEEDS='1234'`, `EPOCHS=200`, `AUTOTUNE_BATCH=True`, `GPU_MEMORY_FRACTION=0.75`, `EARLY_STOP_PATIENCE=40`, `EARLY_STOP_MIN_EPOCHS=80` and `SMOKE_TEST=True`. N25 is already completed and will not be retrained by default. Each selected model is smoke-tested independently for output shape `(B,64,128)`, finite forward values and finite backpropagation gradients. **Do not launch the expensive training sweep until every smoke test passes.**
3. Rerun the **source-download cell**, then the **architecture-smoke-test cell** whenever the GitHub Python source changes, or reopen the latest Colab notebook. The smoke-test cell now also refreshes `/content/reference_architectures_2d.py` with cache-bypass and verifies the native U-Net constructor before spawning test subprocesses. Merely editing GitHub does **not** update files already copied into an active Colab runtime.
4. Start the training cell after smoke tests pass. Training logs stream to Colab and to `native_reference_2d/native_training_stdout.log`. Each completed run writes `result.json`, plots and checkpoints under `native_reference_2d/runs/<model>/N<size>_seed<seed>_<config-hash>/`. The aggregate table is `native_reference_2d/results.csv`. Checkpoints are saved periodically for restart/resumption if the exact configuration matches; leave `FORCE_RETRAIN=False` to reuse existing runs. The notebook executes **N50, N100 and N200** in sequence; it preserves prior N25 result rows when writing `results.csv`. Different effective batch sizes and stopping epochs are recorded in each new run and should not be presented as a strictly matched optimizer-budget comparison.

**Known U-Net smoke-test constructor fix (22 October 2026).** The first native-method version failed before training with

```text
TypeError: UNet2D.__init__() got an unexpected keyword argument 'width'
```

`UNet2D` has signature `UNet2D(k, r, theta, channels=32)`, so the native factory in [`reference_architectures_2d.py`](reference_architectures_2d.py) was corrected from `UNet2D(1, r, theta, width=32)` to **`UNet2D(1, r, theta, channels=32)`**. The [native Colab](Native_Method_Baselines_2D_Colab.ipynb) also refreshes this module and checks the keyword before attempting training. If your traceback still shows `width=32`, you are executing a **stale local script**; rerun the source-download cell and smoke tests. **The native-method N25 numerical runs have now completed successfully; the subsequent batch-autotuned driver is newly modified and should be smoke-tested before the N50–200 sweep.**

**Time and comparability notes:** Based on measured N25 runtime and linear scaling, N50+N100+N200 would take approximately **17.9 hours** if using the original small batches for all four baselines at 200 epochs. GPU-aware batch tuning and early stopping should reduce the workload, but the achieved speedup cannot be known before execution; the driver logs the actual seconds/epoch and a conservative remaining-time projection. Original N25 fixed-batch and new N50–200 autotuned runs have **different optimization settings**, requiring explicit disclosure or a later N25 rerun under identical settings before a fully controlled sample-size learning curve is claimed. U-NO-H10 uses **ten observed initial frames** and its full-trajectory error is intentionally undefined; compare its final-time and post-t10 errors separately, not directly in a one-initial-state ranking.

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
