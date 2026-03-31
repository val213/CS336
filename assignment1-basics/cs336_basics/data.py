import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    # dataset: (n_samples, context_length)
    n_samples = dataset.shape[0]
    idx = np.random.randint(0, n_samples - context_length, size=(batch_size,))
    # idx 是一组起始位置，形状 (batch_size,)                                                                                                                                                       
    # 对每个起始位置 i，取 dataset[i : i+context_length]            
    x = np.stack([dataset[i : i + context_length] for i in idx])
    y = np.stack([dataset[i+1 : i + context_length + 1] for i in idx])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
