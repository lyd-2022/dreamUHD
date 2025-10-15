
python /fs-computility/ai4sData/liuyidi/code/dreamUHD/inference.py \
--config /fs-computility/ai4sData/liuyidi/code/dreamUHD/options/DreamUHD_haze.yml \
--input /fs-computility/ai4sData/liuyidi/data/UHD_haze/test/input \
--weight /fs-computility/ai4sData/liuyidi/code/dreamUHD/weight/DreamUHD_haze.pth \
--output /fs-computility/ai4sData/liuyidi/code/dreamUHD/exp/haze_820


python calculate_psnr_ssim.py \
--gt_path /fs-computility/ai4sData/liuyidi/data/UHD_haze/test/gt \
--test_log /test2.log \
--results_path /fs-computility/ai4sData/liuyidi/code/dreamUHD/exp/haze_820