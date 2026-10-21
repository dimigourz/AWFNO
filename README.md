# Alternating Windowed Fourier Neural Operators (AWFNO)

Research code and numerical experiments for learning the evolution operators of nonlinear partial differential equations.

**Current archive:** 1D viscous Burgers experiments, including AWFNO and competitive baselines, controlled architecture/order ablations, sensitivity sweeps, and sample-size studies. Other PDE benchmarks will be added as separate folders.

## Experiments

| Directory | Contents |
| --- | --- |
| [`Burgers1D/01_AWFNO_H10_K5/`](Burgers1D/01_AWFNO_H10_K5/) | AWFNO with 10 observed input states and a 5-state forecast window. |
| [`Burgers1D/02_UNO_H10/`](Burgers1D/02_UNO_H10/) | U-NO-inspired 1D encoder–decoder adapted to a 10-state history, rolled forward recursively. |
| [`Burgers1D/03_Architecture_Ablations/`](Burgers1D/03_Architecture_Ablations/) | Controlled component, local/Fourier ordering, and radius/modes/window sensitivities. Includes Python experiment engine and Colab driver. |
| [`Burgers1D/04_Competitive_Baselines/`](Burgers1D/04_Competitive_Baselines/) | F-FNO, autoregressive U-Net, and direct CNO baseline implementations. |

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
