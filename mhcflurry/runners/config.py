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


@dataclass(frozen=True)
class ModalConfig:
    # Modal reads ~/.modal.toml; nothing to configure here today.
    # Placeholder so the slot exists symmetrically with BrevConfig.
    app_prefix: Optional[str] = None
