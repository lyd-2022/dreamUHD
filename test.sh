#!/bin/bash


# wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/vgg16-397923af.pth
# wget -P /root/.cache/torch/hub/checkpoints/LPIPS_v0.1_vgg-a78928a0.pth https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.1_vgg-a78928a0.pth
# mkdir -p /root/.cache/torch/hub/checkpoints
# cp /model/liuyidi/vgg-19/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
# cp /model/liuyidi/vgg-19/LPIPS_v0.1_vgg-a78928a0.pth /root/.cache/torch/hub/checkpoints/LPIPS_v0.1_vgg-a78928a0.pth
# cp /model/liuyidi/vgg-19/alexnet-owt-7be5be79.pth /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
# cp /model/liuyidi/vgg-19/LPIPS_v0.1_alex-df73285e.pth /root/.cache/torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth
# pip3 install PyWavelets packaging lpips opencv-python einops scikit-image torchmetrics omegaconf tensorboard thop lmdb matplotlib timm openai-clip pandas  facexlib sentencepiece future icecream imgaug accelerate addict transformers==4.37.2 pyiqa -i https://pypi.mirrors.ustc.edu.cn/simple/
# # python /code/UHDformer-main/basicsr/train.py -opt /code/UHDformer-main/options/VAE/kl_16_updown.yml
# # # 定义Python命令的路径
# PYTHON_PATH="/fs-computility/ai4sData/liuyidi/code/DreamUHD"


# # srun -p ai4earth --job-name=try --gres=gpu:0 --ntasks-per-node=1 cd /tmp/liuyidi-sftp-src/UHD_LL/testing_set/input
# # 定义配置文件、权重文件和输出目录的路径
# # CONFIG_PATH="/code/UHDformer-main/options/VAE/kl_16.yml"

# EXP_PATH="/mnt/petrelfs/liuyidi/Code/dreamUHD/experiments"
# EXP_NAMES=("IR7")
# # EXP_NAMES = adoenc_restormer
# for EXP_NAME in "${EXP_NAMES[@]}"; do
#     COMMON_PATH="$EXP_PATH/$EXP_NAME"
#     # CONFIG_PATH=$(find "$COMMON_PATH" -type f -name "*.yml")
#     # CONFIG_PATH=$(find "$COMMON_PATH" -maxdepth 1 -type f -name "*.yml")

#     echo "Found YAML files:"
#     echo "$CONFIG_PATH"

#     # 定义权重文件和输出目录的路径，使用COMMON_PATH变量
#     WEIGHT_PATH="$COMMON_PATH/models/net_g_latest.pth"
#     # WEIGHT_PATH='/mnt/petrelfs/liuyidi/Code/dreamUHD/experiments/IRtry3-lr8/models/net_g_12000.pth'
#     OUTPUT_PATH="$COMMON_PATH/result1"
#     # WEIGHT_PATH="/mnt/petrelfs/liuyidi/Code/dreamUHD/experiments/LL/net_g_54000.pth"
#     # OUTPUT_PATH="$COMMON_PATH/result"
    

#     # # 训练命令
#     # $PYTHON_PATH/basicsr/train.py -opt $CONFIG_PATH

#     # # 推理命令
#     # $PYTHON_PATH/inference_VAE.py \
#     $PYTHON_PATH/inference_uhdformer.py \
#     --config $CONFIG_PATH \
#     --input /mnt/petrelfs/liuyidi/data/UHD_deblur/test/input300 \
#     --weight $WEIGHT_PATH \
#     --output $OUTPUT_PATH

#     # # # --test_tile \
#     # $PYTHON_PATH/calculate_psnr_ssim.py \
#     # --gt_path /mnt/petrelfs/liuyidi/data/UHD_LL/testing_set/gt \
#     # --test_log /test2.log \
#     # --results_path $OUTPUT_PATH 
# done

# srun -p ai4earth --job-name=test --quotatype=spot --gres=gpu:1 --ntasks-per-node=1 bash /mnt/petrelfs/liuyidi/Code/dreamUHD/test.sh

# srun -p ai4earth --job-name=test  --gres=gpu:1 --ntasks-per-node=1 bash /mnt/petrelfs/liuyidi/Code/dreamUHD/test.sh


# srun -p ai4earth \
# --job-name=test --gres=gpu:1 \
# --quotatype=spot --ntasks-per-node=1 \
# --async \
# bash /mnt/petrelfs/liuyidi/Code/dreamUHD/test.sh



python /fs-computility/ai4sData/liuyidi/code/dreamUHD/inference.py \
--config /fs-computility/ai4sData/liuyidi/code/dreamUHD/options/DreamUHD_LL.yml \
--input /fs-computility/ai4sData/liuyidi/data/UHD_LL/test/input \
--weight /fs-computility/ai4sData/liuyidi/code/dreamUHD/weight/DreamUHD_lowlight.pth \
--output /fs-computility/ai4sData/liuyidi/code/dreamUHD/exp/LL2_818


python calculate_psnr_ssim.py \
--gt_path /fs-computility/ai4sData/liuyidi/data/UHD_LL/test/gt \
--test_log /test2.log \
--results_path /fs-computility/ai4sData/liuyidi/code/dreamUHD/exp/LL2_818