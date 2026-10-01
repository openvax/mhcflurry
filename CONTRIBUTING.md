# Contributing to MHCflurry

We would love your help in making MHCflurry a useful resource for the community. No contribution is too small, and we especially appreciate usability improvements like better documentation, tutorials, tests, or code cleanup.

## Project scope
We hope MHCflurry will grow to include **reference implementations for state-of-the-art approaches for T cell epitope prediction**. This includes pan-allele MHC I and II prediction and closely related tasks such as prediction of antigen processing and immunogenicity. It does not include tasks such as B cell (antibody) epitope prediction, prediction of TCR/pMHC interactions, or downstream tasks such as cancer vaccine design. All committed code to MHCflurry should be suitable for regular research use by practitioners. This likely means that new models will require a benchmark evaluation with a publication or preprint before they can be accepted.

If you are contemplating a large contribution, such as the addition of a new predictive model, it probably makes sense to reach out on the GitHub issue tracker (or email us at hello@openvax.org) to discuss and coordinate the work.

## Making a contribution
All contributions can be made as pull requests on GitHub. One of the core developers will review your contribution. As needed the core contributors will also make releases and submit to PyPI.

A few other guidelines:

 * Generated resources must include their generation commands, input/source hashes, configuration, random seed and dependency versions. Historical generators live in `downloads-generation/`; the maintained training and release workflows are documented in [scripts/training/README.md](scripts/training/README.md) and [scripts/release/README.md](scripts/release/README.md). Record provenance sufficient to rerun the workflow; exact numerical reproduction across hardware or library versions is not guaranteed.
 * MHCflurry supports Python 3.10+ on Linux and macOS. We can't guarantee support for Windows. If you are having trouble running MHCflurry on Windows we would appreciate contributions that help us address this.
 * All functions should be documented using [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) and associated with unit tests.
 * Bugfixes should be accompanied by a test that illustrates the bug when feasible.
 * Contributions are licensed under Apache 2.0
 * Please adhere to our [code of conduct](https://github.com/openvax/mhcflurry/blob/master/code-of-conduct.md).

Working on your first Pull Request? One resource that may be helpful is [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

## Development checks

Run lint and the tests before opening a pull request. The fast tier needs only
the default presentation bundle:

```shell
./lint.sh
python -m pytest -q test -m "not slow and not downloads"
```

CI runs the full suite, `python -m pytest test/`, with the `2.2.0` download
catalogue pinned and several bundles fetched. The fixtures are listed under
"Full verification" in [docs/testing.md](docs/testing.md). On macOS, use
`python -m pytest` rather than the `pytest` script so PyTorch can see MPS.

CI uses Ruff 0.16.0 and the checked-in `.ruff.toml`. The selected rules focus
on syntax, undefined names, mutable defaults, unsafe loop closures, environment
default types, and stale suppressions. Formatting and broad modernization are
kept out of the release gate so they can be reviewed separately from scientific
or prediction-affecting changes.

## Publish a package release

Publishing requires maintainer authorization; a review or release-preparation
request does not authorize it. Versions and tags are bare `MAJOR.MINOR.PATCH`
(for example `2.3.10`, with no `v`). The Docker workflow rejects a tag that does
not match `mhcflurry/version.py`.

1. **Choose the version.** Bump `mhcflurry/version.py` and add
   `RELEASE_NOTES_<version>.md`. The version must be unused on PyPI, including
   pre-release and development builds, and not already taken on `master` or by
   another open pull request. Model training versions are provenance and must
   not be rewritten to match a tag.
2. **Verify locally.** Run `./lint.sh`, the full test suite (see Development
   checks), the strict documentation build with doctests
   (`cd docs && make generate html doctest SPHINXOPTS="-W --keep-going"`), and
   `python -m build -sw`. Run the build from outside the checkout root if a local
   `build/` directory exists there; it shadows the `build` package.
3. **Merge.** Merge the reviewed pull request with a merge commit after GitHub
   CI passes on its final commit. If `master` moved after CI ran, update the
   branch and wait for CI again, so the merged tree is the one that was tested.
   Merging deploys the documentation site.
4. **Allow the tag to deploy.** The `release` environment that uploads to PyPI
   accepts only tags on an explicit allow-list. A repository admin adds the new
   tag before publishing, under Settings → Environments → release, or with:

   ```shell
   gh api -X POST repos/openvax/mhcflurry/environments/release/deployment-branch-policies \
       -f name=<version> -f type=tag
   ```

5. **Publish the GitHub release.** Tag the merge commit and publish a release
   titled `MHCflurry <version>` whose notes are the release-notes file:

   ```shell
   gh release create <version> --target <merge-commit-sha> \
       --title "MHCflurry <version>" --notes-file RELEASE_NOTES_<version>.md
   ```

   Publishing the release, not pushing a tag, starts
   `.github/workflows/release.yml` (build, then PyPI upload by trusted
   publishing) and `.github/workflows/docker.yml`. If the PyPI job fails with
   "not allowed to deploy to release due to environment protection rules", the
   tag is missing from the allow-list: add it as in step 4, then re-run only the
   failed job with `gh run rerun <run-id> --failed`.
6. **Verify PyPI.** Confirm the version on PyPI, then install the published
   wheel in a fresh environment and run a prediction.
7. **Verify Docker.** Each architecture's image must pass the offline check in
   `docker/smoke.py` before anything is pushed. Publishing then needs the
   repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` with push
   access. The workflow pushes `openvax/mhcflurry:<version>` for `linux/amd64`
   and `linux/arm64`, and moves `latest` only when `<version>` is GitHub's
   latest release. If publishing fails, fix the cause and re-run for the same
   tag with `gh workflow run docker.yml -f tag=<version>`; do not create a new
   code version.
8. **Update Bioconda.** Bioconda is maintained upstream and does not publish
   when our GitHub release does. Submit the new version to the
   [Bioconda recipe](https://github.com/bioconda/bioconda-recipes/tree/master/recipes/mhcflurry)
   with the SHA-256 of the source distribution downloaded from PyPI, not a
   local build. Keep dependencies and console entry points aligned with
   `setup.py`. Every dependency must be available as a Conda package; add
   missing dependencies to conda-forge or Bioconda before the MHCflurry recipe
   can build. Verify the package in the channel after upstream CI and review
   complete.

If PyPI, Docker Hub or Bioconda did not publish, say so on the release's
tracking issue with a link to the failing run. Do not describe the release as
complete.

Model assets are separate. Follow `scripts/release/README.md` to validate,
package, checksum and upload them, then update `mhcflurry/downloads.yml` with
the actual URLs. Validate the downloaded archives and prediction outputs
before changing the default model release. Publishing model assets also
requires maintainer authorization.
