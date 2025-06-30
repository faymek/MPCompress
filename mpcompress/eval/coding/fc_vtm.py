import os
import numpy as np
import subprocess as subp
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt 
import json
import time
from omegaconf import OmegaConf
from tqdm import tqdm

def truncation(feat, trun_low, trun_high):
    trun_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(trun_low, list):
        for idx in range(len(trun_low)):
            trun_feat[:,idx,:,:] = np.clip(feat[:,idx,:,:], trun_low[idx], trun_high[idx])
    else:
        trun_feat = np.clip(feat, trun_low, trun_high)
    
    return trun_feat


def load_quantization_points(file_path: str or list[str]):
    """
    Load quantization points from a file or a list of files.
    
    Parameters:
        file_path (Union[str, List[str]]): Path to load the quantization points from.
            Can be a single file path (str) or a list of file paths (List[str]).
    
    Returns:
        Union[numpy.ndarray, List[numpy.ndarray]]: Loaded quantization points. If `file_path`
            is a single path, returns a single numpy.ndarray. If `file_path` is a list of paths,
            returns a list of numpy.ndarray.
    """
    def load_file(path):
        with open(path, 'r') as f:
            quantization_points = np.array(json.load(f))
        # print(f"Quantization points loaded from {path}")
        return quantization_points

    if isinstance(file_path, list):
        # Load quantization points from each file in the list
        return [load_file(path) for path in file_path]
    elif isinstance(file_path, str):
        # Load quantization points from a single file
        return load_file(file_path)
    else:
        raise ValueError("file_path must be a string or a list of strings.")


def uniform_quantization(feat, min_v, max_v, bit_depth):
    quant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
            quant_feat[:,idx,:,:] = ((feat[:,idx,:,:]-min_v[idx]) * scale)
    else:
        scale = ((2**bit_depth) -1) / (max_v - min_v)
        quant_feat = ((feat-min_v) * scale)

    quant_feat = quant_feat.astype(np.uint16) if bit_depth>8 else quant_feat.astype(np.uint8)
    return quant_feat

def uniform_dequantization(feat, min_v, max_v, bit_depth):
    feat = feat.astype(np.float32)
    dequant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
            dequant_feat[:,idx,:,:] = feat[:,idx,:,:] / scale + min_v[idx]
    else:
        scale = ((2**bit_depth) -1) / (max_v - min_v)
        dequant_feat = feat / scale + min_v
    return dequant_feat

def nonlinear_quantization(data, quantization_points, bit_depth):
    """
    Apply quantization to data using a single or multiple sets of quantization points.
    
    Parameters:
        data (numpy.ndarray): Original floating-point array with shape (N, C, H, W).
        quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]): 
            A single numpy array of quantization points or a list of numpy arrays,
            one for each channel (C).
    
    Returns:
        numpy.ndarray: Quantized integer array with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        num_levels = len(quantization_points)
        data_flat = data.flatten()
        quantized_data_flat = np.digitize(data_flat, quantization_points) - 1
        quantized_data_flat = np.clip(quantized_data_flat, 0, num_levels - 1)
        quantized_data = quantized_data_flat.reshape(data.shape)
    elif isinstance(quantization_points, list):
        if len(quantization_points) != data.shape[1]:
            raise ValueError("Length of quantization_points list must match the number of channels (C) in data.")
        
        quantized_data = np.zeros_like(data, dtype=int)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            num_levels = len(qp)
            channel_data = data[:, i, :, :]
            channel_data_flat = channel_data.flatten()
            quantized_channel_flat = np.digitize(channel_data_flat, qp) - 1
            quantized_channel_flat = np.clip(quantized_channel_flat, 0, num_levels - 1)
            quantized_data[:, i, :, :] = quantized_channel_flat.reshape(channel_data.shape)
    else:
        raise ValueError("quantization_points must be a numpy array or a list of numpy arrays.")
    
    quantized_data = quantized_data.astype(np.uint16) if bit_depth>8 else quantized_data.astype(np.uint8)
    return quantized_data

def nonlinear_dequantization(quantized_data, quantization_points):
    """
    Dequantize quantized data back to its approximate original floating-point values.
    
    Parameters:
        quantized_data (numpy.ndarray): Quantized integer array with shape (N, C, H, W).
        quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]): 
            A single numpy array of quantization points or a list of numpy arrays,
            one for each channel (C).
    
    Returns:
        numpy.ndarray: Dequantized floating-point array with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        quantization_points = np.sort(quantization_points)  # Ensure points are sorted
        dequantized_data = quantization_points[quantized_data]
    elif isinstance(quantization_points, list):
        if len(quantization_points) != quantized_data.shape[1]:
            raise ValueError("Length of quantization_points list must match the number of channels (C) in quantized_data.")
        
        dequantized_data = np.zeros_like(quantized_data, dtype=np.float32)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            qp = np.sort(qp)  # Ensure points are sorted
            channel_data = quantized_data[:, i, :, :]
            dequantized_data[:, i, :, :] = qp[channel_data]
    else:
        raise ValueError("quantization_points must be a numpy array or a list of numpy arrays.")
    
    # print(dequantized_data.dtype)
    dequantized_data = dequantized_data.astype(np.float32)
    return dequantized_data


def packing(feat, model_type):
    N, C, H, W = feat.shape
    if model_type == 'llama3':
        feat = feat[0,0,:,:]
    elif model_type == 'dinov2':
        feat = feat.transpose(0,2,1,3).reshape(N*H,C*W)
    elif model_type == 'sd3':
        feat = feat.reshape(int(C/4), int(C/4), H, W).transpose(0, 2, 1, 3).reshape(int(C/4*H), int(C/4*W)) 
    return feat

def unpacking(feat, shape, model_type):
    N, C, H, W = shape
    if model_type == 'llama3':
        feat = np.expand_dims(feat, axis=0); feat = np.expand_dims(feat, axis=0)
    elif model_type == 'dinov2':
        feat = feat.reshape(N,H,C,W).transpose(0, 2, 1, 3) 
    elif model_type == 'sd3':
        feat = feat.reshape(int(C/4), H, int(C/4), W).transpose(0,2,1,3).reshape(N,C,H,W)
    return feat


def vtm_encoding(vtm_path,preprocessed_yuv_name, bitstream_name, compress_log_name, wdt, hgt, qp, bit_depth):
    stdout_vtm = open(f"{compress_log_name}", 'w')
    preprocessed_yuv_name = "\"" + preprocessed_yuv_name + "\""
    bitstream_name = "\"" + bitstream_name + "\"" 
    subp.run(f"{vtm_path}/EncoderAppStatic -c {vtm_path}/encoder_intra_vtm.cfg -i {preprocessed_yuv_name} -o \"\" -b {bitstream_name} -q {qp} --ConformanceWindowMode=1 -wdt {wdt} -hgt {hgt} -f 1 -fr 1 --InternalBitDepth={bit_depth} --InputBitDepth={bit_depth} --InputChromaFormat=400 --OutputBitDepth={bit_depth}",
            stdout=stdout_vtm, shell=True)

def vtm_decoding(vtm_path, bitstream_name, decoded_yuv_name, decompress_log_name):
    stdout_vtm = open(f"{decompress_log_name}", 'w')
    bitstream_name = "\"" + bitstream_name + "\"" 
    decoded_yuv_name = "\"" + decoded_yuv_name + "\"" 
    subp.run(f"{vtm_path}/DecoderAppStatic -b {bitstream_name} -o {decoded_yuv_name}", stdout=stdout_vtm, shell=True)

def vtm_pipeline(org_feat_path, test_root, cfg, QP):
    # Set related paths
    preprocessed_yuv_path = f"{test_root}/preprocessed"; os.makedirs(preprocessed_yuv_path, exist_ok=True)
    bitstream_path = f"{test_root}/bitstream/QP{QP}"; os.makedirs(bitstream_path, exist_ok=True)
    decoded_yuv_path = f"{test_root}/decoded/QP{QP}"; os.makedirs(decoded_yuv_path, exist_ok=True)
    postprocessed_feat_path = f"{test_root}/postprocessed/QP{QP}"; os.makedirs(postprocessed_feat_path, exist_ok=True)
    encoding_log_path = f"{test_root}/encoding_log/QP{QP}"; os.makedirs(encoding_log_path, exist_ok=True)
    decoding_log_path = f"{test_root}/decoding_log/QP{QP}"; os.makedirs(decoding_log_path, exist_ok=True)

    feat_names = os.listdir(org_feat_path)
    # feat_names = feat_names[:1]
    for feat_name in tqdm(feat_names, total=len(feat_names), desc="Coding features"):
        # Set related names
        stem = feat_name[:-4]
        org_feat_name = os.path.join(org_feat_path, f"{stem}.npy"); #print(org_feat_name)
        preprocessed_yuv_name = os.path.join(preprocessed_yuv_path, f"{stem}.yuv"); #print(preprocessed_yuv_name)
        bitstream_name = os.path.join(bitstream_path, f"{stem}.bin"); #print(bitstream_name)
        decoded_yuv_name = os.path.join(decoded_yuv_path, f"{stem}.yuv"); #print(decoded_yuv_name)
        postprocessed_feat_name = os.path.join(postprocessed_feat_path, f"{stem}.npy"); #print(postprocessed_feat_name)
        encoding_log_name = os.path.join(encoding_log_path, f"{stem}.txt"); #print(encoding_log_name)
        decoding_log_name = os.path.join(decoding_log_path, f"{stem}.txt"); #print(decoding_log_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        # print(idx, feat_name, N,C,H,W)

        # Truncation
        if cfg.trun_flag is True:
            trun_feat = truncation(org_feat, cfg.trun_low, cfg.trun_high)
        else:
            trun_feat = org_feat

        # Quantization
        quant_feat = uniform_quantization(trun_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)      

        # Packing
        pack_feat = packing(quant_feat, cfg.model_type)
        with open(preprocessed_yuv_name, 'wb') as f:
            pack_feat.tofile(f)

        # VTM encoding
        start = time.time()
        vtm_encoding(cfg.vtm_path, preprocessed_yuv_name, bitstream_name, encoding_log_name, pack_feat.shape[1], pack_feat.shape[0], QP, cfg.bit_depth)
        encoding_time = time.time() - start

        # VTM decoding
        vtm_decoding(cfg.vtm_path, bitstream_name, decoded_yuv_name, decoding_log_name)

        # Load decoded YUV
        decoded_yuv = np.zeros_like(pack_feat)
        with open(decoded_yuv_name, 'rb') as f:
            decoded_yuv = np.fromfile(f, dtype=np.uint16 if cfg.bit_depth==10 else np.uint8) # save converted YUV file to dist 
            decoded_yuv = decoded_yuv.reshape(pack_feat.shape) #(H,W)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, [N,C,H,W], cfg.model_type)

        # Dequantization
        dequant_feat = uniform_dequantization(unpack_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)      

        # Save features
        if cfg.model_type == 'sd3': dequant_feat = dequant_feat.astype(np.float16)
        elif cfg.model_type == 'dinov2': dequant_feat = dequant_feat.astype(np.float32)
        elif cfg.model_type == 'llama3': dequant_feat = dequant_feat.astype(np.float32)
        np.save(postprocessed_feat_name, dequant_feat)
        # print(np.mean((org_feat-dequant_feat)**2), np.mean((trun_feat-dequant_feat)**2), np.mean((quant_feat-unpack_feat)**2))

def vtm_decode_only(org_feat_path, test_path, cfg, QP):
    # Set related paths
    preprocessed_yuv_path = f"{test_path}/preprocessed"; os.makedirs(preprocessed_yuv_path, exist_ok=True)
    bitstream_path = f"{test_path}/bitstream/QP{QP}"; os.makedirs(bitstream_path, exist_ok=True)
    decoded_yuv_path = f"{test_path}/decoded/QP{QP}"; os.makedirs(decoded_yuv_path, exist_ok=True)
    postprocessed_feat_path = f"{test_path}/postprocessed/QP{QP}"; os.makedirs(postprocessed_feat_path, exist_ok=True)
    encoding_log_path = f"{test_path}/encoding_log/QP{QP}"; os.makedirs(encoding_log_path, exist_ok=True)
    decoding_log_path = f"{test_path}/decoding_log/QP{QP}"; os.makedirs(decoding_log_path, exist_ok=True)

    print(bitstream_path)
    feat_names = os.listdir(bitstream_path)
    # feat_names = feat_names[:1]
    for idx, feat_name in enumerate(feat_names):
        # Set related names
        stem = feat_name[:-4]
        org_feat_name = os.path.join(org_feat_path, f"{stem}.npy"); #print(org_feat_name)
        preprocessed_yuv_name = os.path.join(preprocessed_yuv_path, f"{stem}.yuv"); #print(preprocessed_yuv_name)
        bitstream_name = os.path.join(bitstream_path, f"{stem}.bin"); #print(bitstream_name)
        decoded_yuv_name = os.path.join(decoded_yuv_path, f"{stem}.yuv"); #print(decoded_yuv_name)
        postprocessed_feat_name = os.path.join(postprocessed_feat_path, f"{stem}.npy"); #print(postprocessed_feat_name)
        encoding_log_name = os.path.join(encoding_log_path, f"{stem}.log"); #print(encoding_log_name)
        decoding_log_name = os.path.join(decoding_log_path, f"{stem}.log"); #print(decoding_log_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        # print(idx, feat_name, N,C,H,W) 

        # Packing
        pack_feat = packing(org_feat, cfg.model_type)

        # VTM decoding
        vtm_decoding(cfg.vtm_path, bitstream_name, decoded_yuv_name, decoding_log_name)

        # Load decoded YUV
        decoded_yuv = np.zeros_like(pack_feat)
        with open(decoded_yuv_name, 'rb') as f:
            decoded_yuv = np.fromfile(f, dtype=np.uint16 if cfg.bit_depth==10 else np.uint8) # save converted YUV file to dist 
            decoded_yuv = decoded_yuv.reshape(pack_feat.shape) #(H,W)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, [N,C,H,W], cfg.model_type)

        # Dequantization
        dequant_feat = uniform_dequantization(unpack_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)      

        # Save features
        if cfg.model_type == 'sd3': dequant_feat = dequant_feat.astype(np.float16)
        elif cfg.model_type == 'dinov2': dequant_feat = dequant_feat.astype(np.float32)
        elif cfg.model_type == 'llama3': dequant_feat = dequant_feat.astype(np.float32)
        np.save(postprocessed_feat_name, dequant_feat)
        # print(np.mean((org_feat-dequant_feat)**2), np.mean((trun_feat-dequant_feat)**2), np.mean((quant_feat-unpack_feat)**2))

    command = f'rm -rf {decoded_yuv_path}'
    os.system(command)


def get_vtm_fc_config(preset_name):
    # 预设配置
    presets = {
        'llama3_csr': {
            'model_type': 'llama3',
            'task': 'csr', 
            'max_v': 47.75,
            'min_v': -78,
            'trun_high': 5,
            'trun_low': -5
        },
        'dinov2_cls': {
            'model_type': 'dinov2',
            'task': 'cls',
            'max_v': 104.1752,
            'min_v': -552.451,
            'trun_high': 20,
            'trun_low': -20
        },
        'dinov2_seg': {
            'model_type': 'dinov2',
            'task': 'seg',
            'max_v': 103.2168,
            'min_v': -530.9767,
            'trun_high': 20,
            'trun_low': -20
        },
        'dinov2_dpt': {
            'model_type': 'dinov2',
            'task': 'dpt',
            'max_v': [3.2777, 5.0291, 25.0456, 102.0307],
            'min_v': [-2.4246, -26.8908, -323.2952, -504.4310],
            'trun_high': [1,2,10,20],
            'trun_low': [-1,-2,-10,-20]
        },
        'sd3_tti': {
            'model_type': 'sd3',
            'task': 'tti',
            'max_v': 4.668,
            'min_v': -6.176,
            'trun_high': 4.668,
            'trun_low': -6.176
        }
    }

    compression_config = {
        "trun_flag": True,
        "quant_samples": 0,
        "quant_type": 'uniform',
        "bit_depth": 10,
    }

    preset = presets[preset_name]
    cfg = OmegaConf.create(preset)
    cfg.update(compression_config)

    if cfg.trun_flag is False:
        cfg.trun_high = cfg.max_v
        cfg.trun_low = cfg.min_v
        config_str = f"trun0_{cfg.quant_type}{cfg.quant_samples}_bitdepth{cfg.bit_depth}"
    else:   
        config_str = f"trunl{cfg.trun_low}_trunh{cfg.trun_high}_{cfg.quant_type}{cfg.quant_samples}_bitdepth{cfg.bit_depth}"

    cfg.vtm_path = os.path.join(os.path.dirname(__file__), "vtm_baseline")
    cfg.config_str = config_str
    return cfg


def run_vtm_compression(org_feat_path, test_root, cfg, QP):    
    os.makedirs(test_root, exist_ok=True)
    vtm_pipeline(org_feat_path, test_root, cfg, QP)


if __name__ == "__main__":
    org_feat_path = '/home/faymek/MPCompress/data/dataset/ImageNet_val_sel100/feat_provide'
    preset_name = 'dinov2_cls'
    QPs = [42]
    for QP in QPs:
        cfg = get_vtm_fc_config(preset_name)
        test_root = f'/home/faymek/MPCompress/data/test-fc/ImageNet--dinov2_cls/vtm_{cfg.config_str}'
        run_vtm_compression(org_feat_path, test_root, cfg, QP)