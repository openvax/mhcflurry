# Release notes and changelog

MHCflurry separates **code versions** (published on PyPI) from **weight
catalogue releases** (such as `2.3.0` and `2.2.0`; see {doc}`model_downloads`).
Full per-version notes live in the repository's
[`release-notes/` directory](https://github.com/openvax/mhcflurry/tree/master/release-notes)
and on [GitHub Releases](https://github.com/openvax/mhcflurry/releases).

## 2.5 series (current)

- **[2.5.2](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.2.md)**:
  Adds a torch-free fast path for `--list-supported-alleles` and
  `--list-supported-peptide-lengths` in `mhcflurry predict` and
  `mhcflurry predict-scan`, suppresses duplicate `WARNING` logs when
  `Class1AffinityPredictor` raises under `throw=True` (#443), and sets explicit
  default `timeout=60`s and `max_retries=3` in `install_download` (#462).
  Existing weights and predictions are unchanged.
- **[2.5.1](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.1.md)**:
  Migrates `mhcflurry downloads fetch` onto `datacache>=1.37.0`
  (`install_archive` / `inspect_archive`) with pinned `2.3.0` SHA-256 checksums
  and byte sizes, `--force` refresh/repair, `downloads info --verify`,
  read-only legacy directory preservation, and `portable_names` support for
  historical POSIX tarballs (#484). Existing weights and predictions are
  unchanged.
- **[2.5.0](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.0.md)**:
  Adds `mhcseqs`-backed Class I binding-groove pseudosequence extraction
  (`mhcflurry.pseudosequences`, `Class1AffinityPredictor.add_allele_from_sequence`,
  `extend_alleles_from_mhcseqs`, and `mhcflurry pseudosequences extract|build`),
  delegating to `mhcseqs >= 2.9.0` IMGT G-DOMAIN alignment with groove-insertion
  and no-op-mutation diagnostic warnings. Includes zero-shot cross-species
  validation benchmarks and figures ({doc}`cross_species_validation`). Existing
  `2.3.0` weights and predictions are unchanged.

## 2.4 series

- **2.4.1**: Documentation navigation cleanup (moves internal training-recipe
  notes under {doc}`maintainers`, adds a public API overview to {doc}`api`,
  adds this release-notes page, and documents `OPENVAX_DATA_CACHE` /
  `OPENVAX_OFFLINE` in {doc}`configuration`), plus `datacache 1.35.0` test
  compatibility.
- **[2.4.0](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.4.0.md)**:
  Removes deprecated Keras-era Python API aliases, legacy top-level
  `mhcflurry.*_command` module wrappers, duplicate `mhcflurry` evaluation
  shortcuts, and deprecated hyphenated training/selection/calibration console
  scripts. Keeps `2.3.0` model weights and predictions unchanged.

## 2.3 series

- **[2.3.38](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.3.38.md)**:
  Harmonizes the data cache and downloads CLI on `datacache>=1.34.0`
  (`OPENVAX_DATA_CACHE`, `OPENVAX_OFFLINE`, `mhcflurry downloads dirs`,
  `mhcflurry downloads prune`, and `mhcflurry downloads path --json`) while
  preserving existing legacy cache directories.
- **[2.3.0](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.3.0.md)**:
  PyTorch-native rewrite, updated `2.3.0` presentation/affinity/processing
  weights (see {doc}`release_model_evaluation`), automatic GPU/CPU workload
  planning, unified `mhcflurry` CLI, and compact percentile calibration.

## Earlier releases

- **2.2.0 / 2.1.x / 2.0.x**: MHCflurry 2.0 pan-allele binding and antigen
  processing models (weights remain available via `--model-release 2.2.0`).
