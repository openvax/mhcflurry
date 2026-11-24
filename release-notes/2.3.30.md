# MHCflurry 2.3.30

Model and dataset downloads now extract into a temporary sibling directory,
write the source receipt there, and publish the completed directory with one
rename (#476). Failed extraction or receipt writing leaves the destination
absent so the same fetch can be retried. Per-bundle filesystem locks serialize
concurrent installers. Existing installations, including unreceipted custom
bundles, are preserved. Model weights, catalogue identities and predictions
are unchanged.

The lock uses filelock, now an explicit dependency (also required by PyTorch).
