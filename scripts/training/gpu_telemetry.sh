#!/usr/bin/env bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sourced by release training scripts. Writes a durable nvidia-smi CSV while a
# stage is running so GPU utilization can be analyzed after remote teardown.

GPU_TELEMETRY_PID="${GPU_TELEMETRY_PID:-}"

start_gpu_telemetry() {
    local gpu_log="$1"
    local interval="${2:-${MHCFLURRY_GPU_TELEMETRY_SECONDS:-30}}"

    stop_gpu_telemetry

    if [ "${MHCFLURRY_GPU_TELEMETRY:-1}" = "0" ]; then
        printf >&2 '[gpu_telemetry] disabled; not writing %s\n' "$gpu_log"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf >&2 '[gpu_telemetry] nvidia-smi missing; not writing %s\n' "$gpu_log"
        return 0
    fi

    mkdir -p "$(dirname "$gpu_log")"
    (
        echo "timestamp,gpu_index,util_percent,mem_used_mib,mem_total_mib"
        while :; do
            ts=$(date +%s)
            nvidia-smi \
                --query-gpu=index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null \
              | awk -F, -v ts="$ts" '
                    {
                        for (i = 1; i <= NF; i++) {
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                        }
                        print ts "," $1 "," $2 "," $3 "," $4
                    }
                ' || true
            sleep "$interval"
        done
    ) > "$gpu_log" 2>/dev/null &
    GPU_TELEMETRY_PID=$!
    printf >&2 \
        '[gpu_telemetry] started pid=%s interval_seconds=%s log=%s\n' \
        "$GPU_TELEMETRY_PID" "$interval" "$gpu_log"
}

stop_gpu_telemetry() {
    if [ -n "${GPU_TELEMETRY_PID:-}" ]; then
        kill "$GPU_TELEMETRY_PID" 2>/dev/null || true
        wait "$GPU_TELEMETRY_PID" 2>/dev/null || true
        printf >&2 '[gpu_telemetry] stopped pid=%s\n' "$GPU_TELEMETRY_PID"
        GPU_TELEMETRY_PID=""
    fi
}
