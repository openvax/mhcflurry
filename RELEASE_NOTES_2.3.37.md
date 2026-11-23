# MHCflurry 2.3.37

Harmonized data directory resolution, offline mode, and downloads CLI with `datacache>=1.34.0` (incremental step toward #484):

- **Shared cache resolution with zero-break legacy preservation**: `mhcflurry.downloads.resolve_data_location()` uses `datacache.resolve_cache_dir("mhcflurry", "MHCFLURRY_DATA_DIR", legacy=(user_data_dir("mhcflurry", version="4"),))`. Existing `MHCFLURRY_DOWNLOADS_DIR`, `MHCFLURRY_DATA_DIR`, and populated legacy `user_data_dir("mhcflurry", version="4")` directories continue to work without re-downloading; new installs honor `OPENVAX_DATA_CACHE/mhcflurry` and fall back to the shared platform cache.
- **`mhcflurry downloads dirs [--json]`**: strictly read-only inspection of the resolved data root, active release directory, installed releases, disk usage, and offline state.
- **`mhcflurry downloads prune [--yes] [--keep-release RELEASE] [--json]`**: dry-run-by-default cleanup of non-current downloaded release directories (only deletes with `--yes`, and never touches custom unversioned `MHCFLURRY_DOWNLOADS_DIR` overrides).
- **Offline mode and progress policy**: added `--offline` to `mhcflurry downloads fetch` (also honoring `OPENVAX_OFFLINE` / `DATACACHE_OFFLINE`, failing fast with `datacache.OfflineError` before creating directories while keeping `--already-downloaded-dir` and cache hits working), wired progress bars to `datacache.should_show_progress("auto")`, and added `--json` to `mhcflurry downloads path`.

Model weights, catalogue identities, and predictions are unchanged.
