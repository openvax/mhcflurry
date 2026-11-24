# MHCflurry 2.3.14

* Add `mhcflurry eval allele-capabilities` to distinguish model support,
  bundled training evidence, and unknown empirical validation/calibration.
  Reports bind canonical allele identities and model-input sequence hashes to
  the full model file tree. Symlinks inside a bundle are rejected explicitly
  to prevent components from being omitted. Existing prediction behavior is
  unchanged.
* Add `mhcflurry eval dla` and a reproducible report for the original observation
  sheets from Kaabinejadian et al. 2026. Human-host DLA observations and canine
  tumors remain separate. Historical training overlap and incomplete lineage
  prevent held-out performance claims; exploratory scores are labeled as such.
