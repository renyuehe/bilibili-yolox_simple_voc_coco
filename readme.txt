/////////////////////////////////////  直接训练方式  //////////////////////////////////////////
########################### coco 训练与测试
右键train 直接运行(高亮以下选项, 注释掉voc):
    experiment-name = r"yolox_s_coco128/pth_dir"
    exp_file == r"exps/example/custom/yolox_s.py"
右键demo 直接运行(高亮以下选项, 注释掉voc):
    experiment-name = "yolox_s_coco128/predict"
    name = yolox-s
    path = "assets/000000000002.jpg"
    exp_file = "exps/example/custom/yolox_s.py"
    ckpt = "./YOLOX_outputs/yolox_voc_s/pth_dir/last_epoch_ckpt.pth"
    VOC_CLASSES 改为 COCO_CLASSES

########################### voc 训练与测试
右键train 直接运行(高亮以下选项, 注释掉coco):
    experiment-name = r"yolox_voc_s/pth_dir"
    exp_file == r"exps/example/yolox_voc/yolox_voc_s.py"
右键demo 直接运行(高亮以下选项, 注释掉coco):
    experiment-name = "yolox_voc_s/predict"
    name = yolox-voc-s
    path = "assets/000009.jpg"
    exp_file = "exps/example/yolox_voc/yolox_voc_s.py"
    ckpt = "./YOLOX_outputs/yolox_voc_s/pth_dir/last_epoch_ckpt.pth"
    COCO_CLASSES 改为 VOC_CLASSES




// 命令行可以不用看
///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////  命令行方式  //////////////////////////////////////////
# 预测一张图片
python tools/demo.py --demo image -n yolox-s -c ./yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
# 预测一个视频
python tools/demo.py --demo video -n yolox-s -c ./yolox_s.pth --path assets/11.mp4  --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu


# 训练自定义的 yolox_voc_s 数据集
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 1 --fp16 -c ./yolox_s.pth
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 1 --fp16
    -d 使用多少张显卡训练
    -b 批次大小
    -fp16 是否开启半精度训练
    -c 检查权重文件


# 测试自己训练的 yolox_voc_s 数据集
python tools/demo.py --demo image -n yolox-voc-s -f exps/example/yolox_voc/yolox_voc_s.py -c ./weights/best_ckpt.pth --path assets/000009.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu


# 训练 coco 数据集
# (但是有问题,ap一直为0) 关注了该网站：https://github.com/Megvii-BaseDetection/YOLOX/issues/504
# 等待更好的解决方案
python tools/train.py -f exps/example/custom/yolox_s.py -d 0 -b 1 --fp16 -o -c ./yolox_s.pth
python tools/train.py -f exps/example/custom/yolox_s.py -d 0 -b 1 --fp16
# 测试 coco 训练效果
python tools/demo.py --demo image -n yolox-s -f exps/example/custom/yolox_s.py -c ./weights/latest_ckpt.pth --path assets/000000000002.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu

