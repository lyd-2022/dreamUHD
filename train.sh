#!/bin/bash


# wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/vgg16-397923af.pth
# wget -P /root/.cache/torch/hub/checkpoints/LPIPS_v0.1_vgg-a78928a0.pth https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.1_vgg-a78928a0.pth
# mkdir -p /root/.cache/torch/hub/checkpoints
# cp /model/liuyidi/vgg-19/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
# cp /model/liuyidi/vgg-19/LPIPS_v0.1_vgg-a78928a0.pth /root/.cache/torch/hub/checkpoints/LPIPS_v0.1_vgg-a78928a0.pth
# cp /model/liuyidi/vgg-19/alexnet-owt-7be5be79.pth /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
# cp /model/liuyidi/vgg-19/LPIPS_v0.1_alex-df73285e.pth /root/.cache/torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth
# pip3 install PyWavelets wandb packaging lpips opencv-python einops scikit-image torchmetrics omegaconf tensorboard thop lmdb matplotlib timm openai-clip pandas  facexlib sentencepiece future icecream imgaug accelerate addict transformers==4.37.2 pyiqa -i https://pypi.mirrors.ustc.edu.cn/simple/

# pip install pyyaml

# python /code/UHDformer-main/basicsr/train.py -opt /code/UHDformer-main/options/VAE/kl_16_updown.yml
# 定义Python命令的路径
# export WANDB_API_KEY='16b1e63bc4552c39530cdb1fbcf31a36ec311def'

PYTHON_PATH="python /fs-computility/ai4sData/liuyidi/code/dreamUHD"



# 定义配置文件、权重文件和输出目录的路径
CONFIG_PATH="/fs-computility/ai4sData/liuyidi/code/dreamUHD/options/DreamUHD_LL.yml"

COMMON_PATH="/fs-computility/ai4sData/liuyidi/code/DreamUHD/exp"

# 定义权重文件和输出目录的路径，使用COMMON_PATH变量
WEIGHT_PATH="$COMMON_PATH/models/net_g_latest.pth"
OUTPUT_PATH="$COMMON_PATH/result"

# # 训练命令
$PYTHON_PATH/basicsr/train2.py -opt $CONFIG_PATH 
# $PYTHON_PATH/basicsr/train_pgt.py -opt $CONFIG_PATH 

# # # 推理命令
# $PYTHON_PATH/inference_VAE.py \
# $PYTHON_PATH/inference_uhdformer.py \
# --config $CONFIG_PATH \
# --weight $WEIGHT_PATH \
# --output $OUTPUT_PATH 


# $PYTHON_PATH/calculate_psnr_ssim.py \
# --results_path $OUTPUT_PATH 


# srun -p ai4earth --job-name=debug --quotatype=spot --gres=gpu:1 --ntasks-per-node=1 bash /mnt/petrelfs/liuyidi/Code/dreamUHD/train.sh
# srun -p ai4earth --job-name=debug --debug --gres=gpu:1 --ntasks-per-node=1 bash /mnt/petrelfs/liuyidi/Code/dreamUHD/train.sh
# srun -p ai4earth --job-name=debug --gres=gpu:1 --ntasks-per-node=1 bash /mnt/petrelfs/liuyidi/Code/dreamUHD/train.sh

# srun -p ai4earth \
# --job-name=DEVAE-2 --gres=gpu:1 \
# --quotatype=spot --ntasks-per-node=1 \
# --async \
# bash /mnt/petrelfs/liuyidi/Code/dreamUHD/train.sh


# salloc -p ai4earth -n1 -N1 --gres=gpu:4