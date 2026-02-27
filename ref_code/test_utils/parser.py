import argparse

def parse_label_conversion_args():
    parser = argparse.ArgumentParser(description='将3D BBOX标注转换为2D BBOX标注')
    parser.add_argument('--input_folder', type=str, required=True, help='原标签')
    parser.add_argument('--output_folder', type=str, required=True, help='转换后标签')
    args = parser.parse_args()
    return args

def parse_coco_converter_args():
    parser = argparse.ArgumentParser(description='将json标注转换为coco格式的pt标注')
    parser.add_argument('--input_folder', type=str, required=True, help='2d标签')
    parser.add_argument('--output_folder', type=str, required=True, help='coco标签')
    args = parser.parse_args()
    return args

def parse_image_to_video_args():
    parser = argparse.ArgumentParser(description='将图片序列转换为mp4视频')
    parser.add_argument('--input_folder', type=str, required=True, help='图片序列路径')
    parser.add_argument('--output_folder', type=str, required=True, help='视频存储路径')
    args = parser.parse_args()
    return args

def parse_pre_args():
    parser = argparse.ArgumentParser(description='前处理：将图片经过前处理器后，按一定帧率保存为视频')
    parser.add_argument('--metric', type=int, default=2, choices=[0, 2, 3, 4], help='执行的指标, 0代表不使用前处理')
    parser.add_argument('--input_path', type=str, required=True, help='输入图片或视频目录')
    parser.add_argument('--output_path', type=str, required=True, help='输出视频文件/图片文件夹路径')
    parser.add_argument('--fps', type=int, default=15, help='视频帧率 (默认15)')
    parser.add_argument('--model_path', type=str, required=True, help='前处理模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (默认cuda)')

    args = parser.parse_args()
    return args

def parse_compress_args():
    parser = argparse.ArgumentParser(description='视频H.265压缩工具')
    parser.add_argument('--metric', type=int, default=2, choices=[2, 3, 4], help='执行的指标')
    parser.add_argument('--input', help='输入视频文件路径')
    parser.add_argument('-o', '--output', help='输出视频文件路')
    parser.add_argument('-r', '--ratio', type=float, default=0.5, 
                       help='压缩比例（默认0.5，即压缩至原码率的一半）')
    parser.add_argument('--ffmpeg_path', default='ffmpeg', help='使用的ffmpeg路径（默认使用环境自带的ffmpeg）')
    
    args = parser.parse_args()
    return args

def parse_extract_args():
    parser = argparse.ArgumentParser(description='抽帧（指标2）')
    parser.add_argument('--metric', type=int, default=2, choices=[2, 3, 4], help='执行的指标')
    parser.add_argument('--interval', type=int, default=30, help='抽帧间隔')
    parser.add_argument('--input_path', type=str, required=True, help='输入路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径')
    parser.add_argument('--ffmpeg_path', default='ffmpeg', help='使用的ffmpeg路径（默认使用环境自带的ffmpeg）')

    args = parser.parse_args()
    return args

def parse_post_args():
    parser = argparse.ArgumentParser(description='后处理：将视频按帧拆分经过后处理器。后处理后经过检测器输出结果')
    parser.add_argument('--input_path', type=str, required=True, help='输入路径')
    parser.add_argument('--feat_path', type=str, default=None, help='特征路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径')
    parser.add_argument('--model_path', type=str, required=True, help='后处理模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (默认cuda)')
    parser.add_argument('--metric', type=int, default=2, help='后处理模式，对应指标（默认指标2）')
    args = parser.parse_args()
    return args


def parse_calcu_metric_args():
    parser = argparse.ArgumentParser(description='后处理：将视频按帧拆分经过后处理器。后处理后经过检测器输出结果')
    parser.add_argument('--frame_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='标签目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (默认cuda)')
    args = parser.parse_args()
    return args