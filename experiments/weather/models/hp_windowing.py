import math

import torch


def window_partition(x: torch.Tensor, window_size):
    """
    Args:
        x: (B, D, N, C)
        window_size (int,int): Must be a power of 2 in the healpy grid.

    Returns:
        windows: (num_windows*B, window_size_d * window_size_hp , C)
    """
    # assert that window_size is a power of 2
    # assert (math.log(window_size) / math.log(2)) % 1 == 0

    B, D, N, C = x.shape
    window_size_d, window_size_hp = window_size
    x = x.view(
        B, D // window_size_d, window_size_d, N // window_size_hp, window_size_hp, C
    )
    # B, D//wd, wd, N//whp, whp, c
    # 0  1      2   3       4    5
    # =>
    # B, D//wd, N//whp, wd, whp, c
    # 0  1      3       2   4    5
    x = x.permute(0, 1, 3, 2, 4, 5)
    windows = x.contiguous().view(-1, window_size_d * window_size_hp, C)
    return windows


def window_reverse(windows, window_size, D, N):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Must be a power of 2 in the healpy grid
        N (int): Number of pixels in the healpy grid

    Returns:
        x: (B, N, C)
    """
    # assert that window_size is a power of 2
    # assert (math.log(window_size) / math.log(2)) % 1 == 0
    window_size_d, window_size_hp = window_size

    B = int(windows.shape[0] / (D * N // (window_size_hp * window_size_d)))
    x = windows.view(
        B, D // window_size_d, N // window_size_hp, window_size_d, window_size_hp, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.contiguous().view(B, D, N, -1)
    return x


def get_nest_win_idcs(window_size):
    """Returns a sqrt(window_size) x sqrt(window_size) tensor with the indices from the nested index
    scheme"""
    result = torch.zeros(
        (int(window_size**0.5), int(window_size**0.5)), dtype=torch.int64
    )

    def fill_quadrant(idx, x, y, size):
        if size == 2:
            result[x, y + 1] = idx
            result[x, y] = idx + 1
            result[x + 1, y + 1] = idx + 2
            result[x + 1, y] = idx + 3
        else:
            fill_quadrant(idx, x, y + size // 2, size // 2)
            fill_quadrant(idx + size**2 // 4, x, y, size // 2)
            fill_quadrant(
                idx + 2 * (size**2 // 4), x + size // 2, y + size // 2, size // 2
            )
            fill_quadrant(idx + 3 * (size**2 // 4), x + size // 2, y, size // 2)

    fill_quadrant(0, 0, 0, int(window_size**0.5))

    return result
