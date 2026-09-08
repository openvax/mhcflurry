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
