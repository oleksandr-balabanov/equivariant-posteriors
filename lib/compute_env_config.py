from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

_TERM_ENDC = "\033[0m"
_TERM_GREEN = "\033[92m"
_TERM_WARNING = "\033[93m"


@dataclass
class Paths:
    checkpoints: Path = field(default=Path("./checkpoints"))
    locks: Path = field(default=Path("./locks"))
    distributed_requests: Path = field(default=Path("./distributed_requests"))
    artifacts: Path = field(default=Path("./artifacts"))
    datasets: Path = field(default=Path("./datasets"))

    def __post_init__(self):
        # super().__init__(*args, **kwargs)

        for key in self.__dict__.keys():
            if not isinstance(self.__dict__[key], Path):
                self.__dict__[key] = Path(self.__dict__[key])

    def __str__(self):
        ret = []
        for key, path in self.__dict__.items():
            ret.append(
                f"{_TERM_GREEN}[Paths] {key}: {path} ({path.absolute()}){_TERM_ENDC}"
            )

        return "\n" + "\n".join(ret)


@dataclass
class ComputeEnvironment:
    paths: Paths = field(default_factory=lambda: Paths())
    postgres_host: str = "localhost"
    postgres_port: int = 5432
