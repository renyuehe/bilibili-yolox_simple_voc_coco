# 补充
import sys, os
sys.path.append(os.path.abspath(os.path.curdir))

# 正文
import argparse
import time
from loguru import logger
import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )

    # parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-expn", "--experiment-name", type=str, default=r"yolox_s_coco128/predict")
    # parser.add_argument("-expn", "--experiment-name", type=str, default=r"yolox_voc_s/predict")


    # parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
    # parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
    parser.add_argument("-n", "--name", type=str, default="yolox-voc-s", help="model name")


    # parser.add_argument("--path", default="assets/000000000002.jpg", help="path to images or video")
    parser.add_argument("--path", default="assets/331.jpg", help="path to images or video")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        # default = None,
        # default="exps/example/custom/yolox_s.py",
        default="exps/example/yolox_voc/yolox_voc_s.py",
        type=str,
        help="pls input your experiment description file",
    )

    # parser.add_argument("-c", "--ckpt", default="./YOLOX_outputs/yolox_s_coco128/pth_dir/last_epoch_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument("-c", "--ckpt", default="./YOLOX_outputs/yolox_voc_s/pth_dir/best_ckpt.pth", type=str, help="ckpt for eval")

    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")

    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    # 如果没有指定 视频path 的情况下, 则选择相机 id, 0为默认的摄像头
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        # cls_names=COCO_CLASSES,
        cls_names=VOC_CLASSES,
        device="cpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)

    # 推理过程
    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()


        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)

            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    # 绘制框,可视化
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    # 输出路径
    args.experiment_name = exp.exp_name if not args.experiment_name else args.experiment_name
    outputdir = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(outputdir, exist_ok=True)

    # 置信度阈值, nms阈值, tsize
    logger.info("Args: {}".format(args))
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 装载模型
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval() # 关闭训练模式
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # 预测
    # predictor = Predictor(model=model, exp=exp, cls_names=COCO_CLASSES, device=args.device)
    predictor = Predictor(model=model, exp=exp, cls_names=VOC_CLASSES, device=args.device)

    # 后续处理
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, outputdir, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, outputdir, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)