import numpy as np
import torch
import torch.nn.functional as F

def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        starts[-1] = len(spect) - (chunk_size - border_size)
    chunks = [
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    
    return chunks, starts

def zeropad(spect: torch.Tensor, left: int = 0, right: int = 0):
    if left == 0 and right == 0:
        return spect
    else:
        return F.pad(spect, (0, 0, left, right), "constant", 0)