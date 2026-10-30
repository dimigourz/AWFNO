# AWFNO — Alternating Windowed Fourier Neural Operators

Publication-oriented source code for four evolution-equation benchmarks. **Research code is preserved unchanged** in this organizational revision; earlier material and original file names are retained in [archive/legacy](archive/legacy).

| Benchmark | Main model | Baselines | Ablations | H10 reference |
|---|---|---|---|---|
| Burgers 1D | [awfno_h10.ipynb](experiments/burgers_1d/awfno_h10.ipynb) | [baselines.ipynb](experiments/burgers_1d/baselines.ipynb) | [ablations.py](experiments/burgers_1d/ablations.py) | [uno_h10.ipynb](experiments/burgers_1d/uno_h10.ipynb) |
| Cahn–Hilliard 2D | [awfno.ipynb](experiments/cahn_hilliard_2d/awfno.ipynb) | [baselines.ipynb](experiments/cahn_hilliard_2d/baselines.ipynb) | [ablations.py](experiments/cahn_hilliard_2d/ablations.py) | [uno_h10.ipynb](experiments/cahn_hilliard_2d/uno_h10.ipynb) |
| Annular Boussinesq 2D | [awfno.py](experiments/boussinesq_2d/awfno.py) | [baselines.py](experiments/boussinesq_2d/baselines.py) | [original analysis](archive/legacy/Boussinesq2D/Boussinesq2D_JCP_OneRun_Colab.ipynb) | [uno_h10.ipynb](experiments/boussinesq_2d/uno_h10.ipynb) |
| Kolmogorov Navier–Stokes 2D | [awfno.py](experiments/navier_stokes_2d/awfno.py) | [main five-model notebook](experiments/navier_stokes_2d/awfno.ipynb) | [ablations.py](experiments/navier_stokes_2d/ablations.py) | [uno_h10.ipynb](experiments/navier_stokes_2d/uno_h10.ipynb) |

## Naming convention

Each benchmark uses `awfno`, `baselines`, `ablations`, and `uno_h10` where applicable. The `h10` suffix means ten observed input snapshots; it is **not information-matched** to initial-condition-only (H1) models. The source notebooks under `experiments/` may retain Colab download references to the historical file names: check [MIGRATION_NOTES.md](MIGRATION_NOTES.md) before running them.

AWFNO is evaluated for finite-horizon recursive forecasting; the paper does not establish a mathematical stability guarantee or extrapolation beyond the training regime. Metrics, training protocols and observed history must be specified per comparison.

**Experimental status:** The 70-epoch U-NO notebooks are newly prepared training studies; their outputs are not claimed as completed. No GPU experiments were rerun for this folder reorganization.
