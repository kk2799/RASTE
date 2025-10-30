import numpy as np
import torch
import torch.nn.functional as F

def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
    """
    将频谱图张量按时间维度切分成多个重叠的片段。
    
    处理流程:
    1. 生成起始位置列表,每个片段之间重叠border_size帧
    2. 对于第一个和最后一个片段,分别在开头和结尾填充border_size帧
    3. 如果avoid_short_end为True,则调整最后一个片段的起始位置,避免过短
    4. 如果输入长度小于chunk_size,则忽略avoid_short_end,直接返回一个较短的片段
    
    参数:
        spect (torch.Tensor): 输入频谱图张量,形状为(时间 x 频率)
        chunk_size (int): 每个片段的长度
        border_size (int): 片段之间的重叠帧数,默认为6
        avoid_short_end (bool): 是否避免最后一个片段过短,默认为True
    
    返回:
        tuple: (chunks, starts)
            - chunks: 切分后的片段列表
            - starts: 每个片段的起始位置列表
    """
    # 生成起始和结束索引
    # 步长为chunk_size - 2*border_size,确保相邻片段有重叠区域
    # 从-border_size开始以处理开头部分
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    
    # 如果需要避免最后一个片段过短且音频长度足够长
    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        # 将最后一个索引移到合适位置,使最后一个片段大小合适
        starts[-1] = len(spect) - (chunk_size - border_size)
        
    # 生成切分后的片段列表
    chunks = [
        # 对每个片段:
        # 1. 提取有效范围内的数据 (使用max/min避免越界)
        # 2. 使用zeropad填充开头和结尾
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),  # 开头填充
            right=max(0, min(border_size, start + chunk_size - len(spect))),  # 结尾填充
        )
        for start in starts
    ]
    
    return chunks, starts

def zeropad(spect: torch.Tensor, left: int = 0, right: int = 0):
    """
    对频谱图进行零填充
    
    处理流程:
    1. 检查是否需要填充
    2. 在时间维度上进行填充
    
    参数:
        spect: 输入频谱图 (time x bins)
        left: 左侧填充帧数
        right: 右侧填充帧数
    
    返回:
        填充后的频谱图
    """
    # 如果不需要填充,直接返回原始频谱图
    if left == 0 and right == 0:
        return spect
    else:
        # 使用F.pad进行填充:
        # - (0, 0): 频率维度不填充
        # - (left, right): 时间维度在左右分别填充指定帧数
        # - "constant", 0: 使用常数0进行填充
        return F.pad(spect, (0, 0, left, right), "constant", 0)