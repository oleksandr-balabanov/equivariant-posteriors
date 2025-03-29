from datetime import timedelta
import os
import torch


def get_rank() -> int:
    try:
        return int(os.environ["LOCAL_RANK"])
    except KeyError:
        return 0
    except ValueError:
        return 0


_DEVICE = None


def ddp_setup(backend=None) -> str:
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    global _DEVICE
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if _DEVICE is not None:
        return _DEVICE
    if "TORCH_DEVICE" in os.environ:
        device = os.environ["TORCH_DEVICE"]
        print(f"Using device {device}")
        _DEVICE = device
        return device

    if "EP_TORCHRUN" not in os.environ:
        _DEVICE = "cpu"
        return "cpu"

    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"

    torch.distributed.init_process_group(
        backend=backend, init_method="env://", timeout=timedelta(seconds=300)
    )
    device_id = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(device_id)
        _DEVICE = device_id
        return device_id
    if backend == "gloo":
        _DEVICE = "cpu"
        return "cpu"

    raise Exception("Unsupported backend")
