import torch
import torchac


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device="cpu")
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.0)
    return cdf_with_0


def get_uniform_pmf(batch_size, alphabet_size):
    p = 1.0 / alphabet_size
    return torch.full((batch_size, alphabet_size), p, dtype=torch.float32)



def encode_uniform_to_bits(indices, alphabet_size):
    """
    将均匀分布的索引编码为比特字符串
    
    Args:
        indices (torch.Tensor): 要编码的索引张量
        alphabet_size (int): 符号表大小
        
    Returns:
        bytes: 编码后的比特字符串
    """
    # 创建均匀分布的CDF
    batch_size = indices.shape[0]
    pmf = get_uniform_pmf(batch_size, alphabet_size)
    cdf = pmf_to_cdf(pmf)
    
    # 使用算术编码
    bitstream = torchac.encode_float_cdf(
        cdf_float=cdf,
        sym=indices.to(dtype=torch.int16).cpu(),
        check_input_bounds=True,
    )
    
    return bitstream


def decode_uniform_from_bits(bitstream, batch_size, alphabet_size):
    """
    从比特字符串解码均匀分布的索引
    
    Args:
        bitstream (bytes): 编码后的比特字符串
        alphabet_size (int): 符号表大小
        num_symbols (int): 要解码的符号数量
        
    Returns:
        torch.Tensor: 解码后的索引张量
    """
    # 创建均匀分布的CDF
    pmf = get_uniform_pmf(batch_size, alphabet_size)
    cdf = pmf_to_cdf(pmf)
    
    # 使用算术解码
    indices = torchac.decode_float_cdf(cdf, bitstream)
    
    return indices


class ArithmeticCoder:
    @staticmethod
    def compress(idx, cdf, bin_path):
        """
        使用给定的 cdf 对 idx 进行算术编码，并写入 bin_path。
        idx: torch.Tensor，整数索引
        cdf: torch.Tensor，float cdf
        bin_path: str，输出二进制文件路径
        """
        wb_stream = torchac.encode_float_cdf(
            cdf_float=cdf,
            sym=idx.to(dtype=torch.int16).cpu(),
            check_input_bounds=True,
        )
        with open(bin_path, "wb") as f:
            f.write(wb_stream)

    @staticmethod
    def decompress(cdf, bin_path):
        """
        从 bin_path 读取二进制流，并用给定的 cdf 解码，返回索引 tensor。
        cdf: torch.Tensor，float cdf
        bin_path: str，输入二进制文件路径
        return: torch.Tensor
        """
        with open(bin_path, "rb") as f:
            rb_stream = f.read()
        index_out = torchac.decode_float_cdf(cdf, rb_stream)
        return index_out 


class UniformArithmeticCoder(ArithmeticCoder):
    """
    均匀分布算术编码器
    专门用于处理均匀分布的符号进行算术编码和解码
    """
    
    def __init__(self, alphabet_size):
        """
        初始化均匀算术编码器
        
        Args:
            alphabet_size (int): 符号表大小，即可能的符号数量
        """
        self.alphabet_size = alphabet_size
        self._create_uniform_cdf()
    
    def _create_uniform_cdf(self):
        """
        创建均匀分布的CDF
        """
        # 创建均匀概率分布
        pmf = torch.ones(self.alphabet_size) / self.alphabet_size
        # 转换为CDF
        self.cdf = pmf_to_cdf(pmf.unsqueeze(0))  # 添加batch维度
    
    def compress_uniform(self, idx, bin_path):
        """
        对均匀分布的索引进行压缩
        
        Args:
            idx (torch.Tensor): 要压缩的索引张量
            bin_path (str): 输出二进制文件路径
        """
        # 确保索引在有效范围内
        if torch.any(idx >= self.alphabet_size) or torch.any(idx < 0):
            raise ValueError(f"索引必须在 [0, {self.alphabet_size-1}] 范围内")
        
        # 使用父类的压缩方法
        return self.compress(idx, self.cdf, bin_path)
    
    def decompress_uniform(self, bin_path, num_symbols=None):
        """
        从二进制文件解压缩均匀分布的索引
        
        Args:
            bin_path (str): 输入二进制文件路径
            num_symbols (int, optional): 要解压缩的符号数量，如果为None则解压缩所有
            
        Returns:
            torch.Tensor: 解压缩后的索引张量
        """
        # 使用父类的解压缩方法
        return self.decompress(self.cdf, bin_path)
    
    def compress_batch(self, idx_batch, bin_path):
        """
        批量压缩均匀分布的索引
        
        Args:
            idx_batch (torch.Tensor): 批量索引张量，形状为 (batch_size, ...)
            bin_path (str): 输出二进制文件路径
        """
        # 展平批次维度
        original_shape = idx_batch.shape
        idx_flat = idx_batch.flatten()
        
        # 压缩
        self.compress_uniform(idx_flat, bin_path)
        
        return original_shape
    
    def decompress_batch(self, bin_path, original_shape):
        """
        批量解压缩均匀分布的索引
        
        Args:
            bin_path (str): 输入二进制文件路径
            original_shape (tuple): 原始张量形状
            
        Returns:
            torch.Tensor: 解压缩后的索引张量，恢复原始形状
        """
        # 解压缩
        idx_flat = self.decompress_uniform(bin_path)
        
        # 恢复原始形状
        return idx_flat.reshape(original_shape)
    
    def get_compression_ratio(self, original_size, compressed_size):
        """
        计算压缩比
        
        Args:
            original_size (int): 原始数据大小（字节）
            compressed_size (int): 压缩后数据大小（字节）
            
        Returns:
            float: 压缩比
        """
        return original_size / compressed_size
    
    def estimate_entropy(self):
        """
        估计均匀分布的熵
        
        Returns:
            float: 熵值（比特/符号）
        """
        return torch.log2(torch.tensor(self.alphabet_size, dtype=torch.float32)).item() 