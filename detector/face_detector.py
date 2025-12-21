import torch
import cv2
import numpy as np

# Import theo structure Django
from detector.retinaface_torch.data import cfg_mnet, cfg_re50
from detector.retinaface_torch.layers.functions.prior_box import PriorBox
from detector.retinaface_torch.utils.box_utils import decode, decode_landm
from detector.retinaface_torch.utils.nms.py_cpu_nms import py_cpu_nms
from detector.retinaface_torch.models.retinaface import RetinaFace


class RetinaFaceDetector:
    def __init__(
        self,
        model_path="./weights/Resnet50_Final.pth",
        network="resnet50",
        confidence_threshold=0.5,
        nms_threshold=0.4,
        top_k=5000,
        keep_top_k=750,
        vis_thres=0.6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # chọn backbone
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        # Load model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = self.load_model(self.net, model_path)
        self.net.to(self.device)
        self.net.eval()

        # params
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres

    # -------------------------------------------------
    # Utils load model
    # -------------------------------------------------
    def remove_prefix(self, state_dict, prefix):
        """Remove 'module.' prefix nếu tồn tại"""
        return {
            k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }

    def load_model(self, model, pretrained_path):
        print(f"Loading RetinaFace pretrained: {pretrained_path}")

        pretrained_dict = torch.load(
            pretrained_path,
            map_location=lambda storage, loc: storage
        )

        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]

        pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def expand_bbox(self, boxes, img_w, img_h, scale=1.25):
        """
        boxes: (N, 4) [x1, y1, x2, y2]
        return: expanded boxes, vẫn clip trong ảnh
        """
        expanded = boxes.copy()

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        cx = boxes[:, 0] + w / 2
        cy = boxes[:, 1] + h / 2

        new_w = w * scale
        new_h = h * scale

        expanded[:, 0] = cx - new_w / 2
        expanded[:, 1] = cy - new_h / 2
        expanded[:, 2] = cx + new_w / 2
        expanded[:, 3] = cy + new_h / 2

        # clip vào kích thước ảnh
        expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)
        expanded[:, 2] = np.clip(expanded[:, 2], 0, img_w - 1)
        expanded[:, 3] = np.clip(expanded[:, 3], 0, img_h - 1)

        return expanded

    # -------------------------------------------------
    # Face detection
    # -------------------------------------------------
    def detect(self, img):
        """
        img: BGR (numpy array)
        return: dets shape (N, 15)
        """

        img_raw = img.copy()

        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape

        # preprocess
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # forward pass
        loc, conf, landms = self.net(img)

        # PriorBox
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        # decode bounding boxes
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        scale = torch.tensor(
            [im_width, im_height, im_width, im_height]
        ).to(self.device)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        # decode scores
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # decode landmarks
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.tensor([
            im_width, im_height, im_width, im_height,
            im_width, im_height, im_width, im_height,
            im_width, im_height
        ]).to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # filter by confidence
        inds = np.where(scores > self.confidence_threshold)[0]
        if len(inds) == 0:
            return np.zeros((0, 15))

        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # sort scores (descending)
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # NMS
        dets = np.hstack((
            boxes, scores[:, None]
        )).astype(np.float32, copy=False)

        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep][:self.keep_top_k]
        landms = landms[keep][:self.keep_top_k]

        # concat bboxes + landmarks
        dets = np.concatenate((dets, landms), axis=1)
        dets[:, 0:4] = self.expand_bbox(
            dets[:, 0:4],
            img_w=im_width,
            img_h=im_height,
            scale=1.2
        )
        return dets  # shape (N, 15)
