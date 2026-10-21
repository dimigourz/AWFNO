# Controlled architecture, ordering and sensitivity experiments

**Both files are included:**

- [`Burgers_ablation_driver.ipynb`](Burgers_ablation_driver.ipynb): Google Colab launcher, configuration and commands.
- [`burgers1d_reviewer_ablations_colab.py`](burgers1d_reviewer_ablations_colab.py): complete Python implementation invoked by the driver.

The script supports `REVIEWER_SUITE=core`, `ordering`, `sensitivity` and `all`. The core suite compares LFL, local-only, Fourier-only, standard WFNO and parameter-matched WFNO. The ordering suite compares LFL/LLF/FLL, with the same operator types and parameter count under different ordering. The sensitivity suite varies local radius (`1,3,5`), Fourier modes (`8,16,32`) and window length (`1,5,10`).

For Colab, upload the Python script when prompted, mount Drive, configure `DATA_PATH` and `OUTPUT_ROOT`, and run the notebook's chosen suite.

Training seeds and epochs are configured by environment variables; sweeps that alter radius/modes/window sizes are **not** parameter matched. Check source for exact metric definitions.
