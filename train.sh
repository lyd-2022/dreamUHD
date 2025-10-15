
PYTHON_PATH="python /fs-computility/ai4sData/liuyidi/code/dreamUHD"

CONFIG_PATH="/fs-computility/ai4sData/liuyidi/code/dreamUHD/options/DreamUHD_LL.yml"

COMMON_PATH="/fs-computility/ai4sData/liuyidi/code/DreamUHD/exp"

WEIGHT_PATH="$COMMON_PATH/models/net_g_latest.pth"
OUTPUT_PATH="$COMMON_PATH/result"

$PYTHON_PATH/basicsr/train.py -opt $CONFIG_PATH 
