PYTHON_PATH="python /mnt/petrelfs/liuyidi/Code/dreamUHD"

EXP_PATH="/mnt/petrelfs/liuyidi/Code/dreamUHD/experiments"
EXP_NAMES=("RA_VAE_3_512")
# EXP_NAMES = adoenc_restormer
for EXP_NAME in "${EXP_NAMES[@]}"; do
    COMMON_PATH="$EXP_PATH/$EXP_NAME"
    # CONFIG_PATH=$(find "$COMMON_PATH" -type f -name "*.yml")
    CONFIG_PATH=$(find "$COMMON_PATH" -maxdepth 1 -type f -name "*.yml")

    echo "Found YAML files:"
    echo "$CONFIG_PATH"
    WEIGHT_PATH="$COMMON_PATH/models/net_g_latest.pth"
    # WEIGHT_PATH='/mnt/petrelfs/liuyidi/Code/dreamUHD/experiments/IRtry3-lr8/models/net_g_12000.pth'
    OUTPUT_PATH="$COMMON_PATH/result1"

    $PYTHON_PATH/inference_uhdformer.py \
    --config $CONFIG_PATH \
    --input /mnt/petrelfs/liuyidi/data/UHD_deblur/test/input300 \
    --weight $WEIGHT_PATH \
    --output /mnt/petrelfs/liuyidi/Code/dreamUHD/output/latent_vis/LL/deg_rec \
done