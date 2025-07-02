import os
import re
import matplotlib.pyplot as plt
import time

from omegaconf import OmegaConf

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def generate_commands(data_root, prefix, model_name, cfg, trun_high, trun_low, quant_type, samples, bit_depth, arch, lambda_value,):
    
    model_type = cfg.model_type
    task = cfg.task
    
    epochs = cfg.epochs
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    patch_size = cfg.patch_size
    trun_flag = cfg.trun_flag

    # os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    
    if isinstance(trun_low, list):
        trun_low = '[' + ','.join(map(str, trun_low)) + ']'
        trun_high = '[' + ','.join(map(str, trun_high)) + ']'    
    

    train_log_dir = f"{data_root}/train-fc/{prefix}/{model_name}/training_log/{cfg.config_str}/"
    test_log_dir = f"{data_root}/test-fc/{prefix}/{model_name}/encoding_log/{cfg.config_str}/"
    ckpt_dir = f'{data_root}/train-fc/{prefix}/{model_name}/training_models/{cfg.config_str}/'
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(test_log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)

    model_config_str = f"lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
    ckpt_path = f"{ckpt_dir}/{model_config_str}_checkpoint.pth.tar"
    train_log_path = f"{train_log_dir}/train_{model_name}_{model_config_str}.txt"
    test_log_path = f"{test_log_dir}/compress_{model_name}_{model_config_str}.txt"



    # generate train command
    train_command = (
        f"python {cfg.this_path}/train_hyper.py --model {arch} "
        f"-d {data_root}/train-fc/{prefix}/feat "
        f"--lambda {lambda_value} --epochs {epochs} -lr {learning_rate} "
        f"--batch-size {batch_size} --patch-size {patch_size} --cuda --save "
        f"--model_type={model_type} --task={task} "
        f"--trun_flag={cfg.trun_flag} --trun_low={cfg.trun_low} --trun_high={cfg.trun_high} "
        f"--quant_type={cfg.quant_type} --qsamples={cfg.samples} --bit_depth={cfg.bit_depth} "
        f"-mp {ckpt_path}"
        f">{train_log_path} 2>&1"
    )

    # generate eval command
    eval_command = (
        f"python {cfg.this_path}/eval_hyper.py checkpoint "
        f"{data_root}/test-fc/{prefix}/feat "
        f"-a {arch} --cuda -v "
        f"-d {data_root}/test-fc/{prefix}/{model_name}/decoded/{cfg.config_str}/{model_config_str} "
        f"--per-image -p {ckpt_path} "
        f"--model_type={model_type} --task={task} "
        f"--trun_flag={cfg.trun_flag} --trun_low={cfg.trun_low} --trun_high={cfg.trun_high} "
        f"--quant_type={cfg.quant_type} --qsamples={cfg.samples} --bit_depth={cfg.bit_depth} "
        f">{test_log_path} 2>&1"
    )

    return train_command, eval_command

def plot_train_loss(train_log_path, train_config):
    # Read all content
    log_name = os.path.join(train_log_path, train_config+'.txt')
    pdf_name = os.path.join(train_log_path, train_config+'.pdf')
    with open(log_name, 'r') as file:
        file_content = file.read()

    # Extract content that starts with "Test epoch"
    test_epoch_lines = re.findall(r"Test epoch.*", file_content)

    # Init
    losses = []
    mse_losses = []
    bpp_losses = []
    aux_losses = []

    # Extract losses
    for line in test_epoch_lines:
        match = re.search(r"Loss:\s*([\d\.]+)\s*\|\s*MSE\s*loss:\s*([\d\.]+)\s*\|\s*Bpp\s*loss:\s*([\d\.]+)\s*\|\s*Aux\s*loss:\s*([\d\.]+)", line)
        if match:
            losses.append(float(match.group(1)))
            mse_losses.append(float(match.group(2)))
            bpp_losses.append(float(match.group(3)))
            aux_losses.append(float(match.group(4)))

    epochs = range(len(losses))

    plt.figure(figsize=(12,8))

    plt.subplot(221)
    plt.plot(epochs, losses, marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(222)
    plt.plot(epochs, mse_losses, marker='o')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    plt.subplot(223)
    plt.plot(epochs, bpp_losses, marker='o')
    plt.title('Bpp Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Bpp Loss')

    plt.subplot(224)
    plt.plot(epochs, aux_losses, marker='o')
    plt.title('Aux Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Aux Loss')

    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf', bbox_inches='tight')



def hyperprior_train_pipeline(data_root, prefix, cfg, lambda_value):
    arch = 'bmshj2018-hyperprior'
    model_name = "hyperprior"

    if cfg.trun_flag == False: 
        trun_high = cfg.max_v; trun_low = cfg.min_v
    else:
        trun_high = cfg.trun_high; trun_low = cfg.trun_low    

    quant_type = 'uniform'; samples = 0; bit_depth = 1

    train_cmd, eval_cmd = generate_commands(data_root, prefix, model_name, cfg, trun_high, trun_low, quant_type, samples, bit_depth, arch, lambda_value,)

    time_start = time.time()
    print("Train Command:")
    print(train_cmd)
    os.system(train_cmd)
    print('training time: ', time.time() - time_start)

    train_log_path = f"{data_root}/train-fc/{prefix}/{model_name}/training_log/{cfg.config_str}/"
    train_config = f"train_{model_name}_lambda{lambda_value}_epoch{cfg.epochs}_lr{cfg.learning_rate}_bs{cfg.batch_size}_patch{cfg.patch_size.replace(' ', '-')}"
    plot_train_loss(train_log_path, train_config)



def hyperprior_evaluate_pipeline(data_root, prefix, cfg, lambda_value):
    arch = 'bmshj2018-hyperprior'
    model_name = arch.split('-')[-1]; print(model_name)

    if cfg.trun_flag == False: 
        trun_high = cfg.max_v; trun_low = cfg.min_v
    else:
        trun_high = cfg.trun_high; trun_low = cfg.trun_low    

    quant_type = 'uniform'; samples = 0; bit_depth = 1

    train_cmd, eval_cmd = generate_commands(data_root, prefix, model_name, cfg, trun_high, trun_low, quant_type, samples, bit_depth, arch, lambda_value,)

    time_start = time.time()
    print("\nEval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print('encoding time: ', time.time() - time_start)



def get_hyper_fc_config(preset_name):
    # 预设配置
    conf_dir = os.path.join(os.path.dirname(__file__), 'conf')
    conf_path = os.path.join(conf_dir, f'hyper_{preset_name}.yaml')
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"配置文件不存在: {conf_path}")
    cfg = OmegaConf.load(conf_path)


    if cfg.trun_flag is False:
        config_str = f"trun0_{cfg.quant_type}{cfg.quant_samples}_bitdepth{cfg.bit_depth}"
    else:   
        config_str = f"trunl{cfg.trun_low}_trunh{cfg.trun_high}_{cfg.quant_type}{cfg.quant_samples}_bitdepth{cfg.bit_depth}"

    cfg.config_str = config_str
    cfg.this_path = os.path.join(os.path.dirname(__file__))

    return cfg



if __name__ == "__main__":
    data_root = '/home/fz2001/Ant/MPCompress/data'
    preset_name = "dinov2_cls"
    prefix = "ImageNet--dinov2_cls"
    cfg = get_hyper_fc_config(preset_name)
    lambda_value_all = cfg.lambda_value_all  

    for lambda_value in lambda_value_all:    
        hyperprior_train_pipeline(data_root, prefix, cfg, lambda_value)

    for lambda_value in lambda_value_all:    
        hyperprior_evaluate_pipeline(data_root, prefix, cfg, lambda_value)