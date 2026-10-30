# Running the public experiments

**Reproducibility status:** The source files are available and the main Colab download/import paths have been repaired. The complete suite has NOT been independently executed on a clean Colab GPU session, so the project is not currently certified as an end-to-end reproducibility package.

## Prerequisites

- A Python runtime with a compatible PyTorch/CUDA installation, NumPy, Matplotlib, pandas, h5py, and packages imported by the individual experiment.
- The maintained neural-operator implementation used by reference FNO/U-NO comparisons is `neuraloperator==2.0.0` where specified by the original launcher.
- Google Colab with GPU for the resource-intensive experiments, Google Drive for datasets and checkpoints.
- Dataset access and a documented method to populate the originally configured `DATA_PATH`.

## Dataset identifiers

| Folder | Original dataset | Resolution and split |
|---|---|---|
| `experiments/burgers_1d/` | PDEBench `1D_Burgers_Sols_Nu0.001.hdf5` | downsampled 512 spatial nodes; 50 validation, 100 test |
| `experiments/cahn_hilliard_2d/` | `cahn_hilliard_train_grid_32x32.pt` | 32×32 trajectories; inspect notebook for precise splits |
| `experiments/boussinesq_2d/` | `hyteg_annulus_c_grid_64x128_samples500.pt` | 400/30/70; 64×128×101 |
| `experiments/navier_stokes_2d/` | `kolmogorov_dataset.pt` and matching `metadata.json` | 200/30/70; 128×128×101; SHA256 7030bc43e6331cdf472182ae0ed4f2684c18da8d336ca9ed92c37baf0bfc8a06 |

The source repo does not distribute most datasets. For the Burgers source, see the [PDEBench DaRUS dataset](https://doi.org/10.18419/DARUS-2986). The other data should be released to an accessible data repository with checksums and preferably a persistent DOI before making a public reproduction guarantee.

## Entry points

- Burgers: `experiments/burgers_1d/awfno_h10.ipynb`, `baselines.ipynb`, `ablations.ipynb`, `uno_h10.ipynb`
- Cahn–Hilliard: `experiments/cahn_hilliard_2d/awfno.ipynb`, `baselines.ipynb`, `ablations.ipynb`, `uno_h10.ipynb`
- Boussinesq: `experiments/boussinesq_2d/awfno.py` (needs configured `DATA_PATH`, `OUTPUT_ROOT`, and `EXPERIMENT_SUITE=original`), `baselines.ipynb`, `uno_h10.ipynb`
- Navier–Stokes: `experiments/navier_stokes_2d/awfno.ipynb`, `ablations.ipynb`, `uno_h10.ipynb`

Each notebook may need the dataset at its configured Google Drive path. Source filenames are normalized in GitHub, but the notebooks may copy them to their historical local module filenames for import compatibility.

## Caveats before manuscript publication

1. **Do not** present the seven-block 70-epoch U-NO-H10 notebooks as completed until their runs finish.
2. Distinguish H10 (10 observed frames) from H1. Do not claim an information-matched comparison.
3. Preserve exact historical version and optimizer configuration for each numerical result.
4. A static source audit cannot rule out GPU out-of-memory errors, changes in dependencies, dataset mismatches, or numerical instability.
5. The `archive/legacy/` directory intentionally holds historical experiments and may include notebooks with hardcoded old GitHub URLs. It is **not a supported entry point** for the publication workflow.
6. A full, independently verified reproduction requires a fresh GPU test of every supported entry point and access to all input datasets, fixed package versions, training checkpoints/seed information and reported result files.
