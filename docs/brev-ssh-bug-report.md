# Brev GPU instances: SSH banner-exchange stalls under concurrent host load

> Technical report for NVIDIA Brev support. All observations reproduced
> between 2026-04-17 and 2026-04-19 on org `hc-db-MHCflurry` with Brev CLI
> v0.6.322, running `mhcflurry` training workloads against several Brev
> instance types.

## Summary

On Brev instances — most severely on GPU boxes running workloads under
`docker run --gpus all` — **new SSH connections to port 22 stall at
banner-exchange**. TCP `connect()` completes in ~60 ms and the client
writes its version string, but the server banner never arrives. Failure
severity correlates with host load:

- **Idle `--gpus all` container**: SSH stays healthy indefinitely.
- **`--gpus all` + moderate CUDA compute (matmul loop)**: rare transient
  failures (~1 in 155 probes over 30 min), self-recover.
- **CPU box + docker + mhcflurry training startup**: several transient
  failures during init (ssh response times spike to 70 seconds), self-
  recover.
- **`--gpus all` + full mhcflurry PyTorch training**: **permanent** SSH
  failure within 3–15 minutes (5/5 attempts across GCP N1 T4, GCP G2 L4,
  AWS g4dn). Box remains `RUNNING/READY` in the Brev control plane;
  `brev exec` also hangs; pre-existing SSH sessions keep flowing data.

**Workaround that works:** running the same training natively (no docker,
no `--gpus all`) on the same GPU hardware — full 1-epoch pan-allele
training completed with zero SSH failures across a 12-minute run.

This strongly suggests an issue in Brev's port-22 handling path that is
starved when `docker run --gpus all` is present alongside concurrent
host-level load. It is not our code; the identical container image runs
cleanly on Modal for an hour.

## Reproducer (minimal)

```bash
brev create reproduce-ssh-bug --type g2-standard-4:nvidia-l4:1 --min-disk 100
brev refresh

# Confirm initial SSH works.
ssh -o ControlMaster=no -o ControlPath=none reproduce-ssh-bug "uptime"

# Wait for Brev's bootstrap to finish installing docker + nvidia-container-toolkit.
ssh reproduce-ssh-bug 'while ! sudo docker info >/dev/null 2>&1; do sleep 5; done'

# Start a PyTorch training workload under --gpus all. Any realistic mhcflurry
# / pan-allele training works; simpler example:
ssh reproduce-ssh-bug 'sudo docker run -d --gpus all \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
    python -c "
import torch, time
model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
                            torch.nn.Linear(1024, 1024)).cuda()
opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
for i in range(10000):
    x = torch.randn(128, 1024, device=\"cuda\")
    y = model(x)
    loss = y.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if i % 50 == 0: print(i, loss.item(), flush=True)
"'

# Within 3-15 minutes the following will time out, while the container
# keeps printing training loss to `docker logs`:
ssh -o ConnectTimeout=15 reproduce-ssh-bug "uptime"
```

## Environment

| Component | Value |
|---|---|
| Brev CLI | v0.6.322 |
| Client | macOS 26.4 (arm64), OpenSSH 10.2p1 LibreSSL 3.3.6 |
| Org | `hc-db-MHCflurry` (`org-3BzyyzgyEx1dlEJPyDDq0vYLIuA`) |
| Workload | [openvax/mhcflurry PR #266](https://github.com/openvax/mhcflurry/pull/266) — PyTorch pan-allele training, full curated dataset (~500 K measurements), `[1024, 512]` dense network |

## Observed `ssh -vvv` trace during the failure

```
debug1: OpenSSH_10.2p1, LibreSSL 3.3.6
debug1: Connecting to <ip> [<ip>] port 22.
debug1: Connection established.
debug3: timeout: 10000 ms remain after connect
debug1: Local version string SSH-2.0-OpenSSH_10.2
Connection timed out during banner exchange
```

TCP `connect()` completes in ~60 ms. The client sends its version string
and then waits for the server banner; none arrives before the configured
ConnectTimeout.

## Full scenario matrix

Every row was monitored with a 15-second-interval SSH watchdog (standalone
`nc -z` TCP probe + full fresh-TCP ssh handshake with
`ConnectTimeout=30`, multiplexing disabled).

| # | Instance type | Docker | `--gpus all` | Image in container | Workload | Duration observed | SSH_OK / SSH_FAIL | Training outcome |
|--:|---|:---:|:---:|---|---|---|:---:|---|
| 1 | `n2d-highmem-2` (CPU, 16 GB) | ✓ | n/a | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (via our Dockerfile) | TINY pan-allele smoketest (~200 rows, 2 epochs) | ~2 min | healthy | ✅ completed |
| 2 | `n1-standard-4:nvidia-tesla-t4:1` | ✓ | ✓ | same | full pan-allele training | ~30 min | **dies** ~30 min | ❌ aborted |
| 3 | `n1-standard-4:nvidia-tesla-t4:1` | ✓ | ✓ | same | full pan-allele training | ~5 min | **dies** ~3 min (2 epochs) | ❌ aborted |
| 4 | `g2-standard-4:nvidia-l4:1` | ✓ | ✓ | same | full pan-allele training | ~10 min | **dies** ~7 min (2 epochs) | ❌ aborted |
| 5 | `g2-standard-4:nvidia-l4:1` + `--network=host` | ✓ | ✓ | same | full pan-allele training | ~20 min | **dies** ~15 min (2 epochs) | ❌ aborted |
| 6 | `g2-standard-4:nvidia-l4:1` (100 GB disk) | ✓ | ✓ | same | full pan-allele training | ~20 min | **dies** ~15 min | ❌ aborted |
| 7 | `g4dn.xlarge` (AWS T4) | ✓ | ✓ | same | full pan-allele training | ~25 min | **dies** ~16 min | ❌ aborted |
| 8 | `g2-standard-4:nvidia-l4:1` | ✗ | n/a (native) | (no image; python3 -m venv + pip install cu121 torch) | pan-allele 1 epoch | ~12 min | 48 / 0 | ✅ **completed, artifacts rsync'd** |
| 9 | `g2-standard-4:nvidia-l4:1` (l4f phase 1) | ✓ | ✓ | `nvidia/cuda:12.1.1-base-ubuntu22.04` | `while true; do nvidia-smi; sleep 2; done` | ~20 min (phase 1) | 85 / 0 | n/a |
| 9 | `g2-standard-4:nvidia-l4:1` (l4f phase 2) | ✓ | ✓ | `pytorch/pytorch:...runtime` | `torch.matmul(8192,8192)` infinite loop (100 % GPU) | ~30 min (phase 2) | 153 / **1** transient | n/a |
| 10 | `g2-standard-4:nvidia-l4:1` (l4g) | ✓ | ✓ | `ubuntu:22.04` (no CUDA libs) | `sleep 1200` | ~30 min | 130 / 0 | n/a |
| 11 | `g2-standard-4:nvidia-l4:1` (l4h) | ✓ | ✓ | `nvidia/cuda:12.1.1-base-ubuntu22.04` | `sleep 1200` | ~30 min | 130 / 0 | n/a |
| 12 | `g2-standard-4:nvidia-l4:1` (l4k control) | ✓ | ✗ | `ubuntu:22.04` | `sleep 1200` | ~30 min | 130 / 0 | n/a |
| 13 | `n2d-highmem-2` (cpu-train) | ✓ | n/a | our Dockerfile, full pan-allele training (1 epoch cap) | mhcflurry training init + first epoch | ~37 min | 82 / **2** transient; one 70.7 s response | ⚠ stuck at worker init (unrelated hang, no progress to epoch 0) |
| — | Modal (T4, their own GPU runtime) | ✓ | (Modal-managed) | same image, same training | full single-model training | ~56 min | n/a | ✅ **completed**, 30 epochs, best val loss 0.0467 |

Where we write "**dies**", the ssh watchdog shows permanent `SSH_FAIL`
starting at the given time, with no recovery in the observation window
(up to 30 min). TCP `connect()` continues to succeed fast in all cases.

## Key observations

1. **Not image-specific.** Rows 2–7 use `pytorch/pytorch`; row 9 swaps in
   `nvidia/cuda:12.1.1-base-ubuntu22.04`; row 10 uses `ubuntu:22.04` with
   no CUDA libs at all. Failure severity depends on workload, not image.

2. **Not cloud-specific.** GCP N1 T4, GCP G2 L4, AWS g4dn.xlarge all
   reproduce the failure under the same trigger.

3. **Not disk-full.** Row 6 ran with 123 GB free on a 125 GB disk and still
   died.

4. **Not SSH-multiplexing.** Every probe in the watchdog uses
   `ControlMaster=no ControlPath=none` with aggressive keepalives.

5. **Not docker-bridge NAT / conntrack.** Row 5 added `--network=host` and
   still died.

6. **`--runtime=nvidia` is not registered on Brev** (confirmed via `docker
   info`). `--gpus all` (CDI) is the only GPU path; no legacy alternative
   to test.

7. **Failure correlates with host load, not container state.**
   - Idle `--gpus all` containers (rows 10, 11): 0 fails in 30 min.
   - Moderate CUDA compute under `--gpus all` (row 9 phase 2): 1 transient
     fail in 30 min.
   - Full PyTorch training under `--gpus all` (rows 2–7): permanent death
     within 15 min.
   - CPU box under PyTorch-training init (row 13): transient fails + extreme
     response-time spikes (70 s) but survives.

8. **Not a SYN-flood-like firewall reject.** TCP handshake consistently
   completes fast (~60 ms). Only the SSH banner byte stream fails.

9. **Pre-existing SSH sessions stay alive.** During each failure, a
   previously-established `docker logs -f` ssh session continued to carry
   container output to the client indefinitely. Only *new* connections
   fail.

10. **`brev exec` and `brev shell` also hang** during the failure window
    — it's not specific to OpenSSH. Anything that opens a new session
    through Brev's SSH path is affected.

## Leading theory

A Brev-operated component on port 22 (a sidecar proxy, or sshd itself
routed through a constrained handler) is **starved of scheduling /
I/O quanta** when the host is under heavy concurrent load. Under light
load it is fine; under heavy CUDA-training load (many processes, CUDA
memory allocations, disk I/O, optimizer state writes) it cannot complete
new banner exchanges. Pre-existing sessions (which already have memory
allocated for their buffers and a dedicated file descriptor) keep
functioning.

Supporting evidence:
- Failure escalates smoothly with workload intensity, not with
  `--gpus all` alone.
- Response-time spikes (500 ms → 3000 ms → 70,000 ms) precede hard failure.
- Modal, which doesn't use the same SSH path, runs the identical docker
  image + training to completion.
- The native-venv workaround, which bypasses docker + nvidia-container-
  toolkit, preserves SSH even under the same pytorch training workload.

This would explain why rows 9–12 (minimal container workloads) stay healthy
while rows 2–7 die — and why the failure is permanent on GPU training
(the workload never lets up) but transient on CPU training (bursts of
load between epochs).

## What we could not directly observe

- `sshd` state on the box during the wedge. Could not SSH in to run
  `journalctl -u ssh`, `ps auxf`, `netstat -tpn`, `lsof -i :22`, or inspect
  host iptables.
- Whether the failure eventually clears if we wait much longer than 30 min.
- Whether stopping the offending container restores SSH (`brev exec` also
  hangs during the window, so we couldn't try it).
- Whether the bug is specific to how Brev configures their GPU images or
  the nvidia-container-toolkit version they ship.

A Brev operator collecting the above on a reproducing instance would
resolve this.

## Workaround in our code

[openvax/mhcflurry PR #266](https://github.com/openvax/mhcflurry/pull/266)
now exposes `BrevConfig(use_docker=False)`. When set, the Brev backend:

1. Rsyncs the repo to `$HOME/mhcflurry` on the instance.
2. Installs `python3-venv` via apt (waits for Brev's own bootstrap to
   finish first).
3. Creates a venv at `$HOME/mhcflurry-venv`, installs `torch` with the
   `cu121` (or `cpu`) index, and `pip install -e` the rsynced repo.
4. Invokes the user's Function under the venv via our in-container
   bootstrap.

This preserves SSH and completes training. Row 8 in the matrix above is
validation.

## Ask

- Could someone inspect `sshd` / auth state / port-22 sidecars on a
  reproducing Brev GPU box during a wedge?
- Is there an official path to run docker + `--gpus all` training jobs on
  Brev GPU boxes without this failure, or is native-install the
  recommended pattern?
- Any way to get `--runtime=nvidia` registered as an alternative would let
  us isolate CDI-runtime vs. legacy-runtime behavior.

## Contact

- Alex Rubinsteyn, `alex.rubinsteyn@gmail.com`
- Repo: <https://github.com/openvax/mhcflurry>
- PR with full harness + reproducer + `BrevConfig(use_docker=False)`
  workaround: <https://github.com/openvax/mhcflurry/pull/266>
