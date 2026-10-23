# 2D Kolmogorov–Navier–Stokes — corrected JCP numerical study

**Primary Colab:** [Corrected original-model comparison + reviewer ablations](Kolmogorov_NS_JCP_Original_Five_Model_Colab.ipynb)

**Standalone implementation:** [kolmogorov_jcp_model_comparison.py](kolmogorov_jcp_model_comparison.py)

## Verified dataset

- Google Drive: `MyDrive/Kolmogorov_NS_Re500_start0.25_horizon0.25_dt0.0025_v1/kolmogorov_dataset.pt`
- SHA-256: `7030bc43e6331cdf472182ae0ed4f2684c18da8d336ca9ed92c37baf0bfc8a06`
- Vorticity tensor: `(300,128,128,101)`, 2π-periodic torus, Re=500.
- Post-spin-up forecast absolute time `[0.25,0.5]`, sample spacing `0.0025`.
- Disjoint fixed train/val/test: `200/30/70`.

The original 22-run [AWFNO_Re500_101Frames_FULL_ABLATIONS_Colab.ipynb](AWFNO_Re500_101Frames_FULL_ABLATIONS_Colab.ipynb) is an **earlier exploratory study**, not the final architecture/protocol comparison. Its outputs are never read or reused by the new study.

## Main Burgers/Boussinesq-style model comparison

| Name | Architecture/prediction task | Temporal conditioning |
| --- | --- | --- |
| TB-AWFNO | 4-layer LGL; projection onto training-derived temporal basis (rank 12) | one initial field → 20-frame window → autoregressive |
| AWFNO | 4-layer LGL alternating local–Fourier–local | one initial field → 20-frame window → autoregressive |
| WFNO | 4-layer conventional spectral + pointwise skip | one initial field → 20-frame window → autoregressive |
| WFNO-PM | WFNO with nearest latent width in real trainable scalar degrees of freedom | same K20 window as AWFNO |
| FNO | maintained `neuraloperator==2.0.0` FNO | one initial field → all 100 future fields, **not** windowed |
| Persistence | initial vorticity held constant | no training |

Main training sizes: **25, 50, 100, 200**. Dataset fixed across every method and N; no method has access to true intermediate test frames. The direct-full FNO has a different training protocol and cannot be interpreted as a parameter- or update-matched K20 baseline.

The parameter-matched comparison uses **real scalar degrees of freedom**, treating complex Fourier weights as two real trainable scalar values. Because width is integral, it reports the residual mismatch instead of implying exact equality.

The reference settings are K=20, window stride=20, M=(16,16), local radius=2, depth=4, width=32 (except WFNO-PM), cosine AdamW optimization; max 120 epochs by default. Epoch optimizer and best-model checkpoints are written to Drive every five epochs. Training data only define the temporal SVD basis; the full eigensystem is cached and reused for ablations. Test results are not used for checkpoint selection.

### Original-model result files

New outputs are isolated under `MyDrive/AWFNO_NS_Re500_JCP_FIVE_MODELS_v1/`.

- `results/sample_size_sweep_kolmogorov_absolute_L2.csv`: publication-style **wide** table.
- `results/results_long.csv`: **long** table by method and sample size, with full/final physical absolute `L²` mean/std/median, parameter counts, scalar DOF, training seconds, inference seconds on 70 held-out cases, per-trajectory time and peak GPU allocation.
- `plots/all_models_full_l2_mean_vs_n.png` and `plots/all_models_final_l2_mean_vs_n.png`.
- `plots/*_l2_per_time.png`, `plots/*_fields_ch0/`: time error, true vorticity, predicted vorticity and absolute error at times corresponding to indices 0,25,50,75,100.

The physical absolute error is integrated with area element `dx dy` on `[0,2π)²` and time-trapezoid weights at `dt=0.0025`. The tabulated **full** statistic is the trajectory-wise `L²(Ω×[0,0.25])` norm averaged across test cases, *not* an unweighted per-frame MSE. The final statistic is the spatial `L²(Ω)` norm at the last time.

### Corrected controlled reviewer experiments

Under `controlled_corrected_v1/`:

- Layer ordering: LGL/GLL/LLG with equal parameter count.
- Component controls: LLL (no Fourier) / GGG (no local).
- WFNO vs WFNO-PM architecture controls.
- K = 2,5,10,20; Fourier modes = 8,16,24,32; local radius = 1,2,3; depth = 2,4,6.
- TB basis ranks 6,12,18; repeat seeds 1235,1236 for LGL/GLL.

These are one-factor-at-a-time comparisons, not a full factorial grid. Component ablations do not have matched parameter counts; the table always lists both the real scalar DOF and the architecture. Reviewers' ablations are **validation-only**, kept separate from the original-model 70-case test comparison. Each run writes a checkpoint, trajectory error curves, 3×5 true/predicted/error vorticity panels and a JSON result; aggregated outputs are `results.csv` and `ALL_corrected_ablation_results.png`.

## Checkpoint recovery: CNO / U-Net / UNO-H10 (F-FNO paused)

[**One-click Colab: resume saved models and generate incremental comparison tables and figures**](https://colab.research.google.com/github/dimigourz/AWFNO/blob/colab/navier-stokes-re500-extra-baselines-20261023/notebooks/NavierStokes_Re500_Resume_NoFFNO_Colab.ipynb)

The recovery notebook reads the **same** training checkpoints at `MyDrive/AWFNO_NS_Re500_101_NATIVE_BASELINES_H10_v3/runs/`, skips complete fits, resumes partial runs using the unchanged v3 config hashes and training hyperparameters, and excludes F-FNO from training/test (its source and saved checkpoints remain intact). First it prints the complete cached validation table and evaluates only currently completed CNO, U-Net and UNO-H10 models on the 70 held-out test trajectories, without retraining. Then it resumes remaining model/N combinations and saves physical L², common forecast-only t10–t100 L², relative H¹, frequency-region error, timing, trajectories and truth/prediction/error panels. A fresh, separate output folder `RECOVERY_COMPARISON_NO_FFNO_v1/` prevents overwriting the original locked test plan. **Incremental testing does not imply all N have finished; do not tune using test scores.** UNO-H10 sees ten true initial frames. Run on GPU and use *Runtime → Run all*, then rerun to resume after Colab time limits. Avoid running the old and new notebooks simultaneously against the same Drive checkpoint folders.
## Additional competitive neural-operator baselines

[**Run the F-FNO / CNO / U-Net / UNO-H10 extension in Colab**](https://colab.research.google.com/github/dimigourz/AWFNO/blob/colab/navier-stokes-re500-extra-baselines-20261023/notebooks/NavierStokes_Re500_FFNO_CNO_UNet_UNO_H10_Colab.ipynb)

This **separate, readable notebook** preserves the original five-model and ablation code and adds 16 fits for N = 25, 50, 100, 200. F-FNO (24-layer) and U-Net use one-step autoregressive forecasting; CNO uses direct-time queries from the initial state; UNO-H10 uses the first ten true frames and a 40-step differentiable training rollout. Consequently, **UNO-H10 is not information-matched to the initial-state-only methods**. The adaptations follow previously used Burgers/Cahn–Hilliard architecture families; they are not official untouched reference training scripts.

Open on a GPU runtime, verify the mounted Drive `DATASET` path, and select **Runtime → Run all**. Code is present in plain Python `%%writefile` cells. The new run folder is `MyDrive/AWFNO_NS_Re500_101_NATIVE_BASELINES_H10_v3/`, separate from the original results. Rerun with unchanged settings to resume checkpoints. `validation_selection.csv` is written as configurations finish; `new_baselines_test_results.csv` and `ALL_MODELS_COMPARISON.csv` appear after all fits have passed validation selection and the locked test is evaluated. Combined comparisons depend on the original publication table being present. The full added GPU sweep has not been verified here.

## Running safely

Open the primary Colab above on a GPU runtime. The notebook installs the pinned native FNO package, verifies generation metadata, retrieves readable source pinned to a Git commit, compiles the source, and runs forward/backward smoke tests for all five architectures **before starting training**. The Colab streams unbuffered progress, prints heartbeat while a subprocess is alive, and archives full stdout logs to Drive. It may take multiple GPU sessions to complete; rerun the training cells to resume rather than restarting or regenerating PDE data.

**Current execution status:** generated code and notebook have not run against the user's Google Drive dataset from this environment. No new numerical results should be claimed until Colab smoke tests and GPU training actually finish.
