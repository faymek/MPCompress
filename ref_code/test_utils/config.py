# 指标2后处理设置
# 测试时的批量大小（每次处理一张图像）
BATCH_SIZE = 1  # 固定为1，因为每张图像可能有不同的分割
# 数据加载时的工作线程数
NUM_WORKERS = 4
# 滑动窗口参数
PATCH_SIZE = 512
STRIDE = 256  # 50% 重叠

# 指标2抽帧设置
IMG_FORMAT  = "png"       # 建议无损
THREADS     = 4
FRAME_PREFIX = "frame"    # 保留以防后续扩展
# 支持的 MP4 扩展名（大小写均可）
VIDEO_EXTS = (".mp4", ".m4v")  