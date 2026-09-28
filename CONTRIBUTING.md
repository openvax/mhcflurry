# Contributing to MHCflurry

We would love your help in making MHCflurry a useful resource for the community. No contribution is too small, and we especially appreciate usability improvements like better documentation, tutorials, tests, or code cleanup.

## Project scope
We hope MHCflurry will grow to include **reference implementations for state-of-the-art approaches for T cell epitope prediction**. This includes pan-allele MHC I and II prediction and closely related tasks such as prediction of antigen processing and immunogenicity. It does not include tasks such as B cell (antibody) epitope prediction, prediction of TCR/pMHC interactions, or downstream tasks such as cancer vaccine design. All committed code to MHCflurry should be suitable for regular research use by practioners. This likely means that new models will require a benchmark evaluation with a publication or preprint before they can be accepted.

If you are contemplating a large contribution, such as the addition of a new predictive model, it probably makes sense to reach out on the Github issue tracker (or email us at hello@openvax.org) to discuss and coordinate the work.

## Making a contribution
All contributions can be made as pull requests on Github. One of the core developers will review your contribution. As needed the core contributors will also make releases and submit to PyPI.

A few other guidelines:

 * Any generated resource, such as trained models, must be associated with a `GENERATE.sh` script in [downloads-generation](https://github.com/openvax/mhcflurry/tree/master/downloads-generation). Running this script with no arguments should fully reproduce the generated result. Reproducability of MHCflurry trained models and related data (such as curated training data, allele sequences, etc.) is key to allowing others to build upon and improve our work.
 * MHCflurry supports Python 3.10+ on Linux and OS X. We can't guarantee support for Windows. If you are having trouble running MHCflurry on Windows we would appreciate contributions that help us address this.
 * All functions should be documented using [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) and associated with unit tests.
 * Bugfixes should be accompanied with test that illustrates the bug when feasible.
 * Contributions are licensed under Apache 2.0
 * Please adhere to our [code of conduct](https://github.com/openvax/mhcflurry/blob/master/code-of-conduct.md).

Working on your first Pull Request? One resource that may be helpful is [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

## Development checks

Run the same checks as CI before opening a pull request:

```shell
./lint.sh
python -m pytest test/
```

CI uses Ruff 0.16.0 and the checked-in `.ruff.toml`. The selected rules focus
on syntax, undefined names, mutable defaults, unsafe loop closures, environment
default types, and stale suppressions. Formatting and broad modernization are
kept out of the release gate so they can be reviewed separately from scientific
or prediction-affecting changes.

## Publish a package release

1. Update `mhcflurry/version.py` and the matching `RELEASE_NOTES_<version>.md`.
   Verify the version against PyPI, including development builds. Model
   training versions are provenance and must not be rewritten to match a tag.
2. Run lint, the complete tests, documentation HTML/doctests and a wheel/source
   distribution build. Merge the reviewed PR after GitHub CI passes.
3. Create the release tag on that tested commit and publish a GitHub release
   with the corresponding release notes. `.github/workflows/release.yml`
   builds and uploads to PyPI when the release is **published**; pushing a tag
   alone does not publish a package.
4. Verify the PyPI version and install the wheel in a clean environment.

Model assets are separate. Follow `scripts/release/README.md` to validate,
package, checksum and upload them, then update `mhcflurry/downloads.yml` with
the actual URLs. Validate the downloaded archives and prediction outputs
before changing the default model release. Publication requires maintainer
or user authorization; a review request alone does not authorize a release.
