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
    # When False, skip docker entirely on the Brev box: install the training
    # environment natively (apt + python3-venv + pip) and run the user's job
    # directly over ssh. The Function's `image=` is ignored — the repo is
    # rsync'd and `pip install -e .` runs inside the venv. Needed because
    # `docker run --gpus all` kills SSH on Brev GPU boxes after a few min
    # (nvidia-container-toolkit interaction, likely a Brev-side bug).
    use_docker: bool = True


@dataclass(frozen=True)
class ModalConfig:
    # Modal reads ~/.modal.toml; nothing to configure here today.
    # Placeholder so the slot exists symmetrically with BrevConfig.
    app_prefix: Optional[str] = None
