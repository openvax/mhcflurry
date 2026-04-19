"""Backend credential/config slots.

Defaults rely on each tool's native auth (`brev login`, `~/.modal.toml`).
The `api_key_env` hook exists for a future REST fallback but is unused now.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BrevConfig:
    instance_type: str = "a100-40gb"
    auto_create: bool = False
    api_key_env: str = "BREV_API_KEY"   # env var name for REST fallback; optional
    auth: str = "cli"                    # "cli" (default) or "rest" (not implemented)
    # Provisioning mode:
    # - "vm" (default): `brev create` provisions a full VM with Brev's
    #   sidecar stack (grafana, influxdb, jupyter, cloudflared). User code
    #   runs inside `docker run --gpus all ...` on top of that.
    # - "container": `brev create --mode container --container-image <base>`
    #   where `<base>` comes from Image.from_registry(...). The box *is*
    #   the user's image; our backend skips docker entirely and runs the
    #   declared apt/pip layer ops inline over ssh. Lighter host footprint
    #   and sidesteps the `docker run --gpus all` path that historically
    #   wedged SSH on Brev GPU boxes (see docs/brev-ssh-bug-report.md).
    mode: str = "vm"
    # Legacy escape hatch for VM mode — skip docker, install the training
    # environment natively (apt + python3-venv + pip) and run the user's job
    # directly over ssh. Kept for boxes where mode="container" isn't an
    # option (different provider / legacy flow).
    use_docker: bool = True


@dataclass(frozen=True)
class ModalConfig:
    # Modal reads ~/.modal.toml; nothing to configure here today.
    # Placeholder so the slot exists symmetrically with BrevConfig.
    app_prefix: Optional[str] = None
