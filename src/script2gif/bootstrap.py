from dataclasses import dataclass

from script2gif.infrastructure.app_paths import AppPaths
from script2gif.infrastructure.logging_config import configure_logging


@dataclass
class BootstrapResult:
    paths: AppPaths


def bootstrap() -> BootstrapResult:
    paths = AppPaths.discover()
    paths.ensure()
    configure_logging(paths.log_dir)
    return BootstrapResult(paths=paths)
