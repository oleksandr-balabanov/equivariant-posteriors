import traceback

from lib.compute_env_config import ComputeEnvironment


_current_env = None

_TERM_ENDC = "\033[0m"
_TERM_GREEN = "\033[92m"
_TERM_WARNING = "\033[93m"


def env():
    global _current_env
    if _current_env is None:
        try:
            import env

            _current_env = env.env
        except Exception:
            print(
                f"{_TERM_WARNING}[Compute environment] Could not load env.py: \n{traceback.format_exc()}{_TERM_ENDC}"
            )
            print(f"{_TERM_WARNING}[Compute environment] Using defaults{_TERM_ENDC}")
            _current_env = ComputeEnvironment()

        for key, value in _current_env.__dict__.items():
            print(f"{_TERM_GREEN}[Compute environment] {key}: {str(value)}{_TERM_ENDC}")

    return _current_env


env()

# def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)

#     for key in self.__dict__.keys():
#         if self.
