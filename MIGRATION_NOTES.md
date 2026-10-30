# Migration notes — read before running notebooks

This revision **does not rewrite the history** or alter the original scientific implementation: every original tracked file was copied to `archive/legacy/<original-path>` and selected source files were promoted to consistent `experiments/<pde>/` paths. Historical saved notebook outputs remain in archive.

**Important:** Some original Colab launchers dynamically download Python modules using historical GitHub paths. After moving the source files, their hardcoded URLs may no longer point to the active source. Until those launchers are adapted and smoke-tested, use the original launchers in `archive/legacy/` (which retain all original paths inside the archive but whose download URLs may also need updates). The repository reorganization is not an endorsement that all notebooks are immediately runnable.

For Boussinesq, `baselines.py` is the native-temporal-protocol driver, while the older K30-adapted comparator `competitive_baselines_2d.py` remains archived. Code relying on Python imports `annulus2d_jcp_experiments` and `reference_architectures_2d` needs explicit filename aliasing or a module import update; renaming a file alone does not do this safely.

The original public repository and all of its past commits remain accessible while its visibility is public. GitHub's visibility control is not available through the connected tools.
