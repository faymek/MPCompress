import subprocess
import os
import json


def get_video_bitrate(video_path):
    """
    使用ffprobe获取视频码率信息
    
    Args:
        video_path (str): 视频文件路径
    
    Returns:
        int: 码率值（单位：bps），如果获取失败返回None
    """
    try:
        # 使用ffprobe获取详细的视频信息
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # 首先尝试从format中获取整体码率
        bitrate = None
        if 'format' in data and 'bit_rate' in data['format']:
            bitrate = int(data['format']['bit_rate'])
            print(f"从format获取到码率: {bitrate} bps ({bitrate/1000:.1f} kbps)")
        
        # 如果format中没有码率信息，从视频流中获取
        if bitrate is None:
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    if 'bit_rate' in stream:
                        bitrate = int(stream['bit_rate'])
                        print(f"从视频流获取到码率: {bitrate} bps ({bitrate/1000:.1f} kbps)")
                        break
        
        # 如果还是没有码率信息，尝试计算
        if bitrate is None:
            duration = float(data['format'].get('duration', 0))
            size = int(data['format'].get('size', 0))
            if duration > 0 and size > 0:
                bitrate = int((size * 8) / duration)
                print(f"通过文件大小和时长计算得到码率: {bitrate} bps ({bitrate/1000:.1f} kbps)")
        
        return bitrate
        
    except subprocess.CalledProcessError as e:
        print(f"ffprobe执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"解析ffprobe输出失败: {e}")
        return None
    except Exception as e:
        print(f"获取视频码率时发生错误: {e}")
        return None


def compress_video_h265(input_path, output_path, target_bitrate, ffmpeg_path='ffmpeg'):
    """
    使用H.265编码压缩视频
    
    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        target_bitrate (int): 目标码率（bps）
    
    Returns:
        bool: 压缩是否成功
    """
    try:
        # 转换码率单位为kbps
        target_bitrate_kbps = target_bitrate // 1000
        
        print(f"开始压缩视频...")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print(f"目标码率: {target_bitrate_kbps} kbps")
        
        # ffmpeg H.265压缩命令
        cmd = [
            ffmpeg_path,
            '-i', input_path,
            '-c:v', 'libx265',           # 使用H.265编码器
            '-b:v', f'{target_bitrate_kbps}k',  # 设置视频码率
            # '-preset', 'medium',         # 编码预设（可选：ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow）
            '-y',                        # 覆盖输出文件
            output_path
        ]
        
        print("执行命令:")
        print(" ".join(cmd))
        print("-" * 50)
        
        # 执行压缩
        result = subprocess.run(cmd, check=True)
        
        print("-" * 50)
        print("视频压缩完成！")
        
        # 检查输出文件是否存在
        if os.path.exists(output_path):
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            compression_ratio = (1 - output_size / input_size) * 100
            
            print(f"原始文件大小: {input_size / (1024*1024):.2f} MB")
            print(f"压缩后大小: {output_size / (1024*1024):.2f} MB")
            print(f"压缩率: {compression_ratio:.1f}%")
            
            return True
        else:
            print("错误: 输出文件未生成")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg压缩失败: {e}")
        return False
    except Exception as e:
        print(f"压缩过程中发生错误: {e}")
        return False