import os
import torch
import torch.nn.functional as F

def center_crop(x, padding):
    """移除中心填充
    
    Args:
        x: 输入张量 (B, C, H, W) 或 (B, H, W, C)
        padding: (padding_left, padding_right, padding_top, padding_bottom)
    
    Returns:
        裁剪后的张量
    """
    if padding == (0, 0, 0, 0):
        return x
    
    padding_left, padding_right, padding_top, padding_bottom = padding
    
    # 计算裁剪区域
    if x.dim() == 4:
        # (B, C, H, W) 格式
        _, _, h, w = x.shape
        h_start = padding_top
        h_end = h - padding_bottom
        w_start = padding_left
        w_end = w - padding_right
        return x[:, :, h_start:h_end, w_start:w_end]
    elif x.dim() == 3:
        # (B, H, W) 格式
        _, h, w = x.shape
        h_start = padding_top
        h_end = h - padding_bottom
        w_start = padding_left
        w_end = w - padding_right
        return x[:, h_start:h_end, w_start:w_end]
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            pass



def get_output(output, task, p=None, label=None, semseg_save_train_class=True):
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)), dim=3)

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)
    
    elif task in {'scene'}:
        _, output = torch.max(output, dim=1)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def to_cuda(batch):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) == list:
        return [to_cuda(v) for v in batch]
    else:
        return batch

# From PyTorch internals
import collections.abc as container_abcs
from itertools import repeat
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)