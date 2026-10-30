# One-dimensional Burgers benchmarks

The following four folders organize the current experimental suite:

1. [AWFNO-H10-K5 sample-size study](01_AWFNO_H10_K5/)
2. [U-NO-H10 sample-size study](02_UNO_H10/)
3. [Architecture and ordering ablations, parameter-matched WFNO, sensitivity sweeps](03_Architecture_Ablations/)
4. [F-FNO, U-Net and CNO competitive baselines](04_Competitive_Baselines/)

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

See the [repository README](../README.md) for numerical summaries and comparability limitations.

Inputs are external HDF5 data from Google Drive. Notebooks contain source and markdown, with executed outputs removed; datasets, checkpoints and full results files are not included.
