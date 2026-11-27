import numpy as np
import numpy.typing as npt
import torch


def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样语言模型训练 batch。
    """

    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if context_length <= 0:
        raise ValueError("context_length must be positive")

    # y 需要比 x 多读取一个 token，
    # 因此合法的起点最大为 len(dataset) - context_length - 1
    max_start = len(dataset) - context_length

    if max_start <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens"
        )

    # 随机生成 batch_size 个起点
    start_positions = np.random.randint(
        low=0,
        high=max_start,
        size=batch_size,
    )

    input_sequences = []
    label_sequences = []

    for start in start_positions:
        # 输入：
        # dataset[start : start + context_length]
        x = dataset[
            start : start + context_length
        ]

        # 标签整体向右错位一位：
        # dataset[start + 1 : start + context_length + 1]
        y = dataset[
            start + 1 : start + context_length + 1
        ]

        input_sequences.append(x)
        label_sequences.append(y)

    # 先转换为 LongTensor，再放到指定设备
    inputs = torch.tensor(
        np.stack(input_sequences),
        dtype=torch.long,
        device=device,
    )

    labels = torch.tensor(
        np.stack(label_sequences),
        dtype=torch.long,
        device=device,
    )

    return inputs, labels