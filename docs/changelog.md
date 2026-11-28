# Release notes and changelog

MHCflurry separates **code versions** (published on PyPI) from **weight
catalogue releases** (such as `2.3.0` and `2.2.0`; see {doc}`model_downloads`).
Full per-version notes live in the repository's
[`release-notes/` directory](https://github.com/openvax/mhcflurry/tree/master/release-notes)
and on [GitHub Releases](https://github.com/openvax/mhcflurry/releases).

## 2.5 series (current)

- **[2.5.18](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.18.md)**:
  Documentation clarity and organization pass: links and explains companion
  packages ([`mhcseqs`](https://github.com/pirl-unc/mhcseqs),
  [`mhcgnomes`](https://github.com/pirl-unc/mhcgnomes),
  [`hitlist`](https://github.com/pirl-unc/hitlist)) on first mention, adds a
  plain-English glossary table to {doc}`whitepaper_25x_cross_species` and its
  rebuilt companion PDF, refactors {doc}`cross_species_validation` into a
  concise how-to guide without figure duplication, tucks whitepapers, canine
  audits, and dated studies under **Advanced topics → Experiment log and
  whitepapers** ({doc}`experiments`) to keep {doc}`index` focused on everyday
  usage, and updates the processing fallback note in {doc}`configuration`. Zero
  changes to code, `2.3.0` weights, or predictions.
- **[2.5.17](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.17.md)**:
  Publishes the unified MHCflurry 2.5 technical whitepaper
  ({doc}`whitepaper_25x_cross_species`) and its companion typeset vector PDF
  (`mhcflurry_25x_cross_species_whitepaper.pdf`, generated reproducibly via
  `docs/build_whitepaper_pdf.py`) synthesizing all `2.5.x` cross-species Class I
  affinity/presentation benchmarks, audited Cattle `BoLA-I` / Dog `DLA-88`
  validation, training parity, and exploratory Class I → Class II transfer
  studies; updates `.github/workflows/docker.yml` and {doc}`docker` to publish
  multi-platform `amd64`/`arm64` images automatically to
  `ghcr.io/openvax/mhcflurry` while gating Docker Hub push on credentials
  (`#498`, `#467`). Zero changes to runtime code, `2.3.0` weights, or
  predictions.
- **[2.5.16](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.16.md)**:
  Reliability, streaming, and performance fixes (`#175`, `#290`, `#374`,
  `#411`, `#440`): adds streaming `--chunk-size` to `mhcflurry predict-scan`
  to bound peak memory on large FASTA/CSV scans, speeds up `FlankingEncoding`
  by `~14x` via a direct 256-entry ASCII byte lookup table with bit-identical
  output, records explicit `sampler` and separate `candidate_representation`
  metadata in processing preparation provenance, enforces current download
  stanza completeness in `test/test_model_catalogue.py`, and aligns the
  `worker_pool` implementation note with the shared-memory slot-assignment
  code. Zero changes to `2.3.0` weights or predictions.
- **[2.5.15](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.15.md)**:
  Removes Biopython (`#501`): migrates the three remaining `Bio.SeqIO` scripts
  in `downloads-generation/` to `mhcflurry.fasta.FastaParser`, drops
  `biopython` from `downloads-generation/data_references/requirements.txt`,
  syncs `requirements.txt` with `pyproject.toml`, and adds regression tests.
  Zero changes to runtime prediction code, `2.3.0` weights, or predictions.
- **[2.5.14](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.14.md)**:
  Repository hygiene and experiment-script organization: deletes 7 dead
  unreferenced one-off campaign scripts and 8 legacy Mount Sinai LSF/TensorFlow
  `.lsf` fossils from `downloads-generation/`, quarantines 15 reproducible
  `2.3.0`/`2.5.0` experiment scripts into `scripts/experiments/` with an
  indexed provenance `README.md`, and fixes stale CLI documentation in script
  READMEs. Zero changes to runtime code, `2.3.0` weights, or predictions.
- **[2.5.13](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.13.md)**:
  Publishes {doc}`experiment_25x_cattle_dog_hitlist` in {doc}`experiments`
  (cross-referenced from {doc}`dla_validation`): a three-table `2.3.0`
  training-overlap audit and strictly peptide-disjoint evaluation of `21,332`
  flanked Cattle `BoLA-I` MS ligands (PXD008151; `0.902–0.960` Presentation
  `+F` AUROC with ultra-few-shot allele ablations on `4–12` training rows) plus
  confirmation that IEDB `DLA-88*501:01`/`508:01` MS cohorts are `100%`
  in-sample in `2.3.0` (`#490`). Also fixes the `#471` CI `classify` checkout
  (`fetch-depth: 0` + `filter: blob:none` in `.github/workflows/ci.yml`) after
  GitHub Actions' falsy-`0` ternary expression was caught live on PR `#607`.
  Existing `2.3.0` weights and predictions are unchanged.
- **[2.5.12](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.12.md)**:
  CI and GitHub Pages infrastructure modernization (`#471`, `#359`): adds
  fail-safe change classification (`scripts/classify_ci_changes.py`) in
  `.github/workflows/ci.yml` so docs/container-only PRs run lint, packaging,
  and CLI checks in ~2 minutes instead of repeating the 4-Python ML matrix, and
  migrates `.github/workflows/docs.yml` off legacy `gh-pages` branch publishing
  to official GitHub Actions Pages (`configure-pages@v6`,
  `upload-pages-artifact@v5`, `deploy-pages@v5`) while preserving
  `previous-versions/` and eliminating Node 20 deprecation warnings. Existing
  `2.3.0` weights and predictions are unchanged.
- **[2.5.11](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.11.md)**:
  Codebase cleanup and modularization (`#506`, `#419`, partial `#513`): splits
  legacy Keras config/weight loading out of `class1_neural_network.py` into
  `mhcflurry/keras_model_config.py`, breaks `fit()` and
  `fit_streaming_batches()` into standalone-tested phase helpers in
  `mhcflurry/class1_training.py`, removes the deprecated dense-vector encoding
  remnants (`FlankingEncoding.vector_encode`/`encode`,
  `EncodableSequences.variable_length_to_fixed_length_vector_encoding`, and
  `Class1NeuralNetworkModel(peptide_input_is_indices=False)`), and drops the
  undeclared optional `pyarrow` CSV engine fallback in pan-allele training.
  Verified bit-identical (`179/179` arrays, `max_abs_diff=0.0`) against `2.5.10`
  across released `2.0.0`/`2.2.0`/`2.3.0` models and deterministic training.
- **[2.5.10](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.10.md)**:
  CLI UX polish across prediction, downloads, training/selection/calibration,
  and `pseudosequences` (`#424`–`#428`): adds `--float-format` and documents
  units on `predict`/`predict-scan`, eliminates duplicated flank columns and
  inconsistent processing-only scan columns, silences noisy default repr/plan/
  unused-key logs, replaces routine user-error tracebacks with one-line CLI
  errors, fixes train/select/calibrate `--help` usage synopses and stray
  `SIGUSR1` banners, and fixes `downloads --verbose`/`--quiet` and subcommand
  help. Existing `2.3.0` weights and predictions are unchanged.
- **[2.5.9](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.9.md)**:
  Adds first-class `mhcgnomes.AccessionedAllele` support in
  `mhcflurry/common.py` so all `2,884` sequence-backed non-model vertebrate
  Class I alleles from `mhcseqs 2.14.0` (e.g.,
  `TriakisScyllium-UAA*uniprot:O46817`, `Saha*uniprot:G3V1L5 class I`) flow
  cleanly through `normalize_allele_name`,
  `build_pseudosequences_from_mhcseqs(normalize_alleles=True)` (`26,156`
  alleles across `223` canonical vertebrate species with `0` unparsed rows),
  and `Class1AffinityPredictor`; publishes the speculative pan-class Class I →
  Class II transfer experiment against `NetMHCIIpan-4.3` in
  {doc}`experiment_25x_pan_class_ii` ({doc}`experiments`); and bumps
  `mhcgnomes>=3.87.2` / optional `mhcseqs>=2.14.0`. Existing `2.3.0` weights
  and predictions are unchanged.
- **[2.5.8](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.8.md)**:
  Adds the published {doc}`experiments` log and full `2.5.x` cross-species
  `hitlist` + IEDB validation experiment report
  ({doc}`experiment_25x_cross_species_hitlist`) with a new 16-cohort
  groove-specificity & training-parity synthesis figure, and defensively guards
  `build_pseudosequences_from_mhcseqs(normalize_alleles=True)` against bare
  UniProt accessions being misparsed as MHC alleles (`pirl-unc/mhcgnomes#306`).
  Existing `2.3.0` weights and predictions are unchanged.
- **[2.5.7](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.7.md)**:
  Adds `--lock-existing` / `lock_existing=True` to freeze baseline alleles
  verbatim, switches `downloads-generation/allele_sequences/GENERATE.sh` to
  `mhcflurry pseudosequences build --lock-existing` with a hard 0-diff parity
  gate on all trained alleles (`verify_trained_allele_parity.py`), adds the
  `hitlist` zero-shot Presentation-vs-Affinity validation figures to
  {doc}`cross_species_validation`, and bumps `mhcgnomes>=3.79.0` /
  `mhcseqs>=2.12.5`. Existing `2.3.0` weights and predictions are unchanged.
- **[2.5.6](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.6.md)**:
  Fixes `Class1NeuralNetwork` batch normalization & merging (`#564`, `#565`),
  `robust_mean` (`#566`), percentile-rank calibration & empty-input prediction
  (`#567`–`#569`, `#577`), `Class1ProcessingModel` / `predict_sequences` /
  `ProcessingProteome` (`#572`, `#575`, `#578`), and `FastaParser` (`#579`).
  Existing `2.3.0` weights and predictions are unchanged.
- **[2.5.5](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.5.md)**:
  Fixes training-loop, `RandomNegativePeptides`, `ppv_at_n`,
  `compare-models`, allele-specific model selection, and `predict` /
  `predict-scan` / `downloads` CLI edge cases (`#588`–`#595`). Existing
  `2.3.0` weights and predictions are unchanged.
- **[2.5.4](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.4.md)**:
  Fixes local parallelism, GPU/batch auto-sizing, `torch.compile`, and cluster
  retry bugs (`#580`–`#587`), accepts registered custom non-`mhcgnomes` allele
  keys in `AlleleKeyResolver` (`#597`), and guards `mhcseqs >= 2.12.2`
  fragment statuses (`status="short"`, `alpha1_only`, `alpha2_only`,
  `fragment_fallback`). Existing `2.3.0` weights and predictions are unchanged.
- **[2.5.3](https://github.com/openvax/mhcflurry/blob/master/release-notes/2.5.3.md)**:
  Fixes `2.5.x` regressions (`#550`–`#555`), NumPy 2 / Pandas 3 compatibility
  crashes (`#556`–`#559`), and core affinity/processing/presentation bugs
  (`#560`–`#563`, `#570`, `#571`, `#573`, `#574`, `#576`). Existing `2.3.0`
  weights and predictions are unchanged.
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
