# Maintainer workflows

This section collects repository and release material that ordinary prediction
users do not need.

## Contributing and verification

- Read the [contribution guide](https://github.com/openvax/mhcflurry/blob/master/CONTRIBUTING.md)
  before opening a pull request.
- Use {doc}`testing` for the fast local loop and full verification commands.
- Use {doc}`development` for internal API and compatibility-shim conventions.
- Use {doc}`orchestrator` when changing worker processes, hardware planning, or
  training data residency.

## Training and model releases

Maintained operational documentation lives next to the scripts it describes:

- [Training pipeline](https://github.com/openvax/mhcflurry/tree/master/scripts/training)
- [Release, synchronization, and deployment](https://github.com/openvax/mhcflurry/tree/master/scripts/release)
- [Generated model and data bundles](https://github.com/openvax/mhcflurry/tree/master/downloads-generation)

The public release entry point is:

```shell
mhcflurry train pan-allele-release --help
```

It coordinates training, evaluation, plots, remote artifact synchronization,
and optional deployment. Deployment is never enabled by default.

## Historical material

The 2023 retraining notebook audit is retained as a maintainer record, not as a
current workflow. Use {doc}`evaluation` and the release guide above for current
commands and artifact contracts.
