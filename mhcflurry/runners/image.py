"""Image: thin wrapper around a Dockerfile path.

Minimal by design — all real image construction happens via the Dockerfile
itself. We do not expose .pip_install / .run_commands / layer APIs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Image:
    dockerfile: str
    context: Optional[str] = None

    @classmethod
    def from_dockerfile(cls, dockerfile: str, context: Optional[str] = None) -> "Image":
        return cls(dockerfile=dockerfile, context=context)

    def resolve(self, repo_root: Path) -> tuple[Path, Path]:
        """Return (dockerfile_path, build_context_path), both absolute."""
        df = (repo_root / self.dockerfile).resolve()
        if not df.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {df}")
        ctx = (repo_root / (self.context or ".")).resolve()
        return df, ctx
