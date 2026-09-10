#!/usr/bin/env bash
# Shared, machine-launcher-independent release recipe presets.

apply_final_230_candidate_recipe() {
    MHCFLURRY_RELEASE_RECIPE=final-2.3.0-candidate
    AFFINITY_MINIBATCH_SIZE=1024
    AFFINITY_OPTIMIZER_IMPLEMENTATION=pytorch
    AFFINITY_LSUV_TARGET=pre_activation
    AFFINITY_INIT=glorot_uniform
    PROCESSING_MINIBATCH_SIZE=512
    PROCESSING_WITH_FLANKS_OPTIMIZER_IMPLEMENTATION=pytorch
    PROCESSING_WITH_FLANKS_INIT=kaiming_uniform_fan_in
    PROCESSING_VARIANTS="with_flanks no_flank short_flanks"
    PROCESSING_MODES="with_flanks,no_flank,short_flanks"
    PROCESSING_SHORT_FLANK_BOUNDARY_RADIUS=5
    PRESENTATION_PROCESSING_WITH_FLANKS_KIND=short_flanks
}

apply_final_230_candidate_v2_recipe() {
    # v1 plus the 2026-09-10 confirmed 5-aa ranking candidate: one legacy
    # architecture (native RMSprop, width 13, inner-best-AP checkpoints) per
    # fold instead of the 128-architecture Glorot/Keras-Adam grid. The 15-aa
    # grid is not a presentation input; it is deferred unless
    # FINAL_230_V2_PROCESSING_VARIANTS adds it back for a release build.
    apply_final_230_candidate_recipe
    MHCFLURRY_RELEASE_RECIPE=final-2.3.0-candidate-v2
    PROCESSING_SHORT_FLANKS_HYPERPARAMETERS=confirmed-ranking-candidate
    PROCESSING_VARIANTS="${FINAL_230_V2_PROCESSING_VARIANTS:-no_flank short_flanks}"
    PROCESSING_MODES="${PROCESSING_VARIANTS// /,}"
}
