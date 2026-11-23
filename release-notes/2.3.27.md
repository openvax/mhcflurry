# MHCflurry 2.3.27

File hashing for provenance records now has one implementation (#528).
Digests are unchanged, so recorded provenance stays valid, and predictions are
unchanged.

- `mhcflurry.experiment_archive.sha256_file` is the single public function. It
  uses `hashlib.file_digest` and no longer takes a `chunk_size` argument.
- `mhcflurry.training_provenance.file_sha256` is removed; import `sha256_file`
  from `mhcflurry.experiment_archive`.
- `mhcflurry.experiment_archive.fingerprint_directory` is new. It was defined
  in `scripts/training/compose_processing_ensemble.py`, and other scripts
  imported it from there, which only worked with `scripts/training` on
  `sys.path`.
- Three private copies in the package and six in `scripts/training` are
  deleted in favour of the shared function.

`scripts/release/validate_release_provenance.py` keeps its own copy: the
release scripts run it with a bare `python3`, and it imports only the standard
library.
