# Brev GPU instances: SSH becomes unreachable to new connections after `docker run --gpus all`

> Draft technical report for NVIDIA Brev support. All observations reproduced
> between 2026-04-17 and 2026-04-19 on org `hc-db-MHCflurry` with Brev CLI
> v0.6.322, running `mhcflurry` runners against several Brev instance types.

## Summary

On Brev GPU boxes (tested across GCP N1 T4, GCP G2 L4, and AWS g4dn), **new
SSH connections to port 22 stop completing the banner exchange after a
`docker run --gpus all <...>` invocation**, while:

- **TCP handshake to port 22 still succeeds in ~60 ms** (SYN / SYN-ACK / ACK
  observed to complete); ssh -vvv hangs right after `Local version string
  SSH-2.0-OpenSSH_10.2`, never receives the remote banner.
- **Pre-existing SSH sessions** (for example, an in-flight `docker logs -f`)
  **keep carrying traffic** indefinitely.
- `brev exec <instance> "..."` also hangs.
- Brev's control plane continues to report the instance as `RUNNING /
  COMPLETED / READY`.

Symptom pattern holds across two clouds (GCP and AWS) and three machine
types, persists with `docker run --network=host`, persists with a 100 GB
root disk (ruling out disk-full), and persists with OpenSSH multiplexing
disabled. It is specifically tied to `docker run --gpus all` — running
the identical workload natively (no docker) on the same hardware preserves
SSH indefinitely.

## Reproducer (minimal)

```bash
brev create reproduce-ssh-bug --type g2-standard-4:nvidia-l4:1 --min-disk 100
brev refresh
# Confirm initial SSH works.
ssh -o ControlMaster=no -o ControlPath=none reproduce-ssh-bug "uptime"

# Wait for Brev's bootstrap to finish installing docker + nvidia-container-toolkit.
ssh reproduce-ssh-bug "while ! sudo docker info >/dev/null 2>&1; do sleep 5; done; \
                       sudo docker run -d --gpus all pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
                         python -c 'import torch; torch.randn(4096,4096,device=\"cuda\"); \
                                    import time; time.sleep(3600)'"

# Within 3–15 minutes, the following will hang with "Connection timed out
# during banner exchange", while the above container is still running:
ssh -o ConnectTimeout=15 reproduce-ssh-bug "uptime"
```

## Environment

| Component | Value |
|---|---|
| Brev CLI | v0.6.322 |
| Org | `hc-db-MHCflurry` (`org-3BzyyzgyEx1dlEJPyDDq0vYLIuA`) |
| Client | macOS 26.4, OpenSSH 10.2p1 |
| Reported by | openvax/mhcflurry runners work (PR openvax/mhcflurry#266) |

## Instance types tested

| Type | Provider | Outcome |
|---|---|---|
| `n2d-highmem-2` (CPU, 16 GB RAM) | GCP | SSH stable; docker smoketest + real training completed |
| `n1-standard-4:nvidia-tesla-t4:1` | GCP | **Bug**: SSH dies ~3–30 min after `docker run --gpus all` |
| `g2-standard-4:nvidia-l4:1` | GCP | **Bug**: SSH dies ~7–15 min |
| `g4dn.xlarge` | AWS | **Bug**: SSH dies ~16 min |
| `hyperstack_A4000`, `verda_V100` | shadeform | Didn't reach ready state (separate bug) |

## Observed ssh -vvv trace during the failure

```
debug1: Connecting to <ip> [<ip>] port 22.
debug1: Connection established.
debug3: timeout: 10000 ms remain after connect
debug1: Local version string SSH-2.0-OpenSSH_10.2
Connection timed out during banner exchange
```

TCP `connect()` completes in ~60 ms. The client sends its version string
and then waits for the server banner; none arrives within the 30 s
ConnectTimeout the harness uses.

## Scenario matrix

Every row was run with a 15-second-interval SSH watchdog that does a
standalone TCP probe + full SSH handshake probe and logs timestamps.

| # | Instance | Image | Docker | `--gpus all` | Workload | SSH outcome | Training outcome |
|---|---|---|---|---|---|---|---|
| 1 | `n2d-highmem-2` CPU | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (via our Dockerfile) | yes | no (CPU) | TINY smoketest training | **OK** (0 fails over full run) | ✅ completed |
| 2 | `n1-standard-4 T4` | same | yes | **yes** | pan-allele real training (our `pan_allele_single.sh`) | **Fails** ~30 min | ❌ aborted |
| 3 | `n1-standard-4 T4` (repeat) | same | yes | **yes** | same | **Fails** ~3 min | ❌ aborted |
| 4 | `g2-standard-4 L4` | same | yes | **yes** | same | **Fails** ~7 min (2 epochs done) | ❌ aborted |
| 5 | `g2-standard-4 L4` + `--network=host` | same | yes | **yes** | same | **Fails** ~15 min (2 epochs done) | ❌ aborted |
| 6 | `g2-standard-4 L4` (100 GB disk) | same | yes | **yes** | same | **Fails** ~15 min | ❌ aborted |
| 7 | `g4dn.xlarge` T4 (AWS) | same | yes | **yes** | same | **Fails** ~16 min | ❌ aborted |
| 8 | `g2-standard-4 L4` | (no image; `python3 -m venv` + `pip install`) | **no** | — (native; sees GPU directly) | pan-allele 1-epoch training | **OK** (0 fails; 48 SSH probes over ~12 min) | ✅ completed, weights rsync'd back |
| 9 | `g2-standard-4 L4` | `nvidia/cuda:12.1.1-base-ubuntu22.04` | yes | **yes** | `while true; do nvidia-smi > /dev/null; sleep 2; done` | *(running — last check: 0 fails after 20+ min)* | n/a (diagnostic) |
| 10 | `g2-standard-4 L4` | `ubuntu:22.04` | yes | **yes** | `sleep 1200` | *(running)* | n/a |
| 11 | `g2-standard-4 L4` | `nvidia/cuda:12.1.1-base-ubuntu22.04` | yes | **yes** | `sleep 1200` | *(running)* | n/a |
| 12 | `g2-standard-4 L4` | `ubuntu:22.04` | yes | **no** (`--gpus` omitted) | `sleep 1200` | *(running — control)* | n/a |
| 13 | `n2d-highmem-2` CPU | our Dockerfile | yes | no | pan-allele 1-epoch training | *(running — verifying CPU+docker+training end-to-end)* | tbd |

(Rows 9-13 are ongoing; table updated as results arrive.)

## Observations + partial conclusions

1. **Bug needs `--gpus all`.** Rows 1 & 8 show CPU+docker works and
   GPU+no-docker works. Rows 2-7 all use `--gpus all` and all fail
   identically.
2. **Bug is not image-specific.** Rows 2-7 use our `pytorch/pytorch` base;
   rows 9, 11 use `nvidia/cuda:12.1.1-base-ubuntu22.04`; row 10 uses
   `ubuntu:22.04` with no CUDA libs inside. Minimal workloads (rows 9-11)
   *may* not trigger the bug — TBD, depends on how long they survive.
3. **Bug is not cloud-specific.** Same failure on GCP and AWS.
4. **Bug is not disk-pressure.** Same failure with 123 GB free on a 125 GB
   disk.
5. **Bug is not a broken stateful firewall / NAT:** TCP SYN/ACK completes
   in ~60 ms.
6. **Bug bypasses our SSH layer.** Confirmed with `ControlMaster=no
   ControlPath=none ServerAliveInterval=30`; no multiplexing, fresh TCP
   per probe.
7. **`--network=host` didn't fix it** (row 5), so the trigger isn't docker
   bridge conntrack or NAT iptables rules.
8. **`--runtime=nvidia` is not registered on Brev**; only `--gpus all` (CDI)
   is available. No legacy-runtime alternative to test.

## What we did *not* directly observe

- State of `sshd` on the box once the bug triggered (could not SSH in).
  Could not check `journalctl`, `auth.log`, `MaxStartups`, DNS / PAM
  configs, or iptables state after the failure.
- Whether the bug ever clears (we always gave up after 30 min).
- Whether stopping the offending container (via `brev exec` or web UI)
  restores SSH. `brev exec` also hung during the failure.

These would be excellent diagnostics a Brev operator could run on their end.

## Workaround

For now, mhcflurry's Brev backend exposes `BrevConfig(use_docker=False)`
which installs mhcflurry natively in a venv on the Brev box and runs the
training directly over ssh — no docker, no `--gpus all`. This preserves
SSH and completes training (row 8).

## Ask

Please investigate why `docker run --gpus all` on Brev GPU instances
leaves port 22 in a state where new TCP connections accept but no SSH
banner is returned, while existing sessions are unaffected.

Any diagnostic commands a Brev support engineer could run on a broken
instance would be hugely appreciated — we'd happily reproduce on request.

## Contact

- Alex Rubinsteyn, `alex.rubinsteyn@gmail.com`
- Repo: <https://github.com/openvax/mhcflurry>
- PR with full harness + reproducer: openvax/mhcflurry#266
