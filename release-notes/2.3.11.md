# MHCflurry 2.3.11

This documentation-only release corrects the contributor guide. Library
behavior, model weights and predictions are unchanged; default weights remain
2.3.0.

- The package release procedure in `CONTRIBUTING.md` now matches the release
  workflows. It covers bare `MAJOR.MINOR.PATCH` tags, checking that a version
  is not already taken on `master` or by an open pull request, and keeping the
  merged tree identical to the tested one. It adds the `release` environment's
  per-tag allow-list, which a repository admin must update before PyPI accepts
  an upload (#481), and the exact commands for publishing, re-running a
  rejected PyPI job, and re-running Docker publication. Bioconda checksums
  must come from the PyPI source distribution.
- The development checks distinguish the fast test tier, which needs only the
  presentation bundle, from the full suite that CI runs with the pinned
  `2.2.0` catalogue.
