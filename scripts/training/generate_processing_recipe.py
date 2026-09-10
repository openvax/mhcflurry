"""The paired 64-fit processing optimizer/initialization/batch factorial."""

from copy import deepcopy

from generate_processing_cleavage_boundaries import build_kernel_conditions


def build_processing_recipe_conditions():
    """Return 32 family-specific recipes, each evaluated on two paired folds."""
    anchors = [(grid[0], axes) for _, grid, axes in build_kernel_conditions()
               if axes["kernel_width"] == 11]
    records = []
    for optimizer, implementation in (("adam", "keras"), ("rmsprop", "pytorch")):
        for method in ("none", "orthogonal", "lsuv_pre", "lsuv_post"):
            for batch in (512, 1024):
                for base, axes in anchors:
                    hp = deepcopy(base)
                    hp.update(optimizer=optimizer, optimizer_implementation=implementation,
                              initialization_method=method, initialization_batch_size=512,
                              minibatch_size=batch, learning_rate=0.001,
                              convolutional_kernel_l1_l2=[0.0, 0.0],
                              save_all_checkpoints=True, restore_best_weights=True)
                    name = "%s__%s_%s__%s__mb%d" % (
                        axes["family"], optimizer, implementation, method, batch)
                    records.append((name, [hp], {
                        **axes, "fold_count": 2, "network_count": 2,
                        "initialization": method, "optimizer": optimizer,
                        "optimizer_implementation": implementation, "batch_size": batch}))
    return records


def processing_candidate_hyperparameters(kernel_width=13, optimizer="rmsprop"):
    """Return the explicit shortlisted processing recipe, not a public default.

    The optimizer/width must still pass confirmation and presentation gates.
    Ranking checkpoints use inner validation only; patience remains loss-based.
    """
    if kernel_width not in (11, 13, 15) or optimizer not in ("adam", "rmsprop"):
        raise ValueError("Candidate requires width 11/13/15 and adam/rmsprop")
    base = next(grid[0] for _, grid, axes in build_kernel_conditions()
                if axes["family"] == "legacy_5aa" and axes["kernel_width"] == kernel_width)
    hp = deepcopy(base)
    hp.update(optimizer=optimizer, optimizer_implementation="keras" if optimizer == "adam" else "pytorch",
              initialization_method="none", minibatch_size=512, learning_rate=0.001,
              convolutional_kernel_l1_l2=[0.0, 0.0], save_all_checkpoints=True,
              restore_best_weights=True, monitor_validation_ranking=True,
              checkpoint_metric="val_macro_ap")
    return hp


def build_processing_confirmation_conditions():
    """Six recipes / 24 paired fits; three retained policies per trajectory."""
    records = []
    for width in (11, 13, 15):
        for optimizer in ("adam", "rmsprop"):
            hp = processing_candidate_hyperparameters(width, optimizer)
            name = "legacy_5aa__%s_%s__k%02d" % (optimizer, hp["optimizer_implementation"], width)
            records.append((name, [hp], {
                "family": "legacy_5aa", "kernel_width": width, "fold_count": 4, "network_count": 4,
                "optimizer": optimizer, "optimizer_implementation": hp["optimizer_implementation"],
                "batch_size": 512, "initialization": "none", "checkpoint_policy": "best_ap",
                "retained_policies": ["best", "best_ap", "terminal"], "stopping_metric": "val_loss"}))
    return records


CONFIRMED_PROCESSING_CANDIDATE = {
    "condition": "legacy_5aa__rmsprop_pytorch__k13",
    "kernel_width": 13,
    "optimizer": "rmsprop",
    "checkpoint_policy": "best_ap",
    "control": "legacy_5aa__adam_keras__k11 / best",
    "experiment": "processing-ranking-confirmation-20260909",
    "runplz_run_id": "a8889cc8dfbb4f4390d4ceb153ad81e1",
    "source_commit": "9778c6256b7e99d98238dd19b648a34bfc8f5d9e",
    "decision_sha256": "5529a81d35defa8939784cabc96f7f077c8918f059a13fee11fb78fafdb29921",
    "release_accepted": False,
}


def confirmed_processing_candidate_hyperparameters():
    """Return the recipe promoted by the 2026-09-10 ranking confirmation.

    Development selection only: four paired fits per condition on 37 held-out
    samples, gated against Adam/Keras width-11 best-loss weights by
    ``mhcflurry eval processing-confirmation-analysis``. It is not a public
    default, a release weight set, a compact-ensemble selection or a
    presentation result; those gates remain separate.
    """
    return processing_candidate_hyperparameters(
        CONFIRMED_PROCESSING_CANDIDATE["kernel_width"], CONFIRMED_PROCESSING_CANDIDATE["optimizer"])


def main(argv=None):
    """Write the confirmed candidate as a one-architecture hyperparameter grid."""
    import argparse
    import sys
    import yaml
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--confirmed-candidate", action="store_true",
                        help="Emit confirmed_processing_candidate_hyperparameters() as YAML on stdout.")
    args = parser.parse_args(argv)
    if not args.confirmed_candidate:
        parser.error("Pass --confirmed-candidate")
    sys.stdout.write(yaml.safe_dump([confirmed_processing_candidate_hyperparameters()], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
