# MHCflurry 2.3.20

Documentation only.

The README now has a Python quick start with its real output, a separate
`Install` section, and a `Command line` section, matching the section names
the sibling openvax libraries use. The Docker instructions moved from the
README to a new documentation page, `docs/docker.md`, linked from the "Start
here" list.

The dead `readthedocs.yml` was removed. It is a v1 Read the Docs config
pointing at `docs/environment.yml`, which was deleted in c4b65e5a; the
documentation builds and publishes through `.github/workflows/docs.yml`
(openvax/mhcflurry#418).

The prediction models and numerical prediction behavior are unchanged.
