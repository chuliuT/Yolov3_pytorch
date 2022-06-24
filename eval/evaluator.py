import config.yolov3_config_voc as cfg
import os
import shutil
from eval import voc_eval
from utils.datasets import *
import cv2
import numpy as np
from utils.data_augment import *
import torch
from utils.tools import *
from tqdm import tqdm

def img_preprocess(img,target_shape=(416,416)):
    h_org, w_org, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    resize_ratio = min(1.0 * target_shape[0] / w_org, 1.0 * target_shape[1] / h_org)
    resize_w = int(resize_ratio * w_org)
    resize_h = int(resize_ratio * h_org)
    image_resized = cv2.resize(img, (resize_w, resize_h))

    image_paded = np.full((target_shape[1], target_shape[0], 3), 128.0)
    dw = int((target_shape[0] - resize_w) / 2)
    dh = int((target_shape[1] - resize_h) / 2)
    image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
    image = image_paded / 255.0  # normalize to [0, 1]
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image[np.newaxis, ...]).float()


def convert_pred(pred_bbox, test_input_size, org_img_shape, valid_scale):
    """
    预测框进行过滤，去除尺度不合理的框
    """
    pred_coor = xywh2xyxy(pred_bbox[:, :4])
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (1)
    # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
    # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
    # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
    # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (2)将预测的bbox中超出原图的部分裁掉
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    # (3)将无效bbox的coor置为0
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4)去掉不在有效范围内的bbox
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5)将score低于score_threshold的bbox去掉
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > 0.45

    mask = np.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]

    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    keep=py_cpu_nms(bboxes,0.45)
    bboxes=bboxes[keep]
    return bboxes

def get_yolo_bboxes(model,img,test_shape,device):
    org_h,org_w,_=img.shape
    img=img_preprocess(img,test_shape)
    # model.head_l.training=False
    # model.head_m.training=False
    # model.head_s.training=False
    with torch.no_grad():
        _, p_d = model(img.to(device))
    pred_bbox = p_d.squeeze().cpu().numpy()
    bboxes = convert_pred(pred_bbox, test_shape[0], (org_h, org_w), (0,np.inf))
    return bboxes

class Evaluator(object):
    def __init__(self, model):
        self.classes = cfg.DATA["CLASSES"]
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'data', 'results')
        self.val_data_path = os.path.join(cfg.DATA_PATH,'VOCdevkit', 'VOC2007')
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape =  cfg.TEST["TEST_IMG_SIZE"]

        self.model = model
        self.model.eval()
        self.device = torch.device('cuda:1')

    def AP_VOC(self,multi_test=False,flip_test=False):
        img_inds_file=os.path.join(self.val_data_path,'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file,'r') as f:
            lines=f.readlines()
            img_inds=[line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.makedirs(self.pred_result_path,exist_ok=True)

        for img_ind in img_inds:
            img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind + '.jpg')
            img = cv2.imread(img_path)
            bboxes = get_yolo_bboxes(self.model, img, (416, 416),self.device)
            for bbox in bboxes:
                coor=bbox[:4]
                score=bbox[4]
                class_id=int(bbox[5])
                class_name=self.classes[class_id]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)
        return self.cal_aps()

    def get_bbox(self, img, multi_test, flip_test):
        if multi_test:
            test_input_sizes=range(320,640,96)
            bboxes_list=[]
            for test_input_size in test_input_sizes:
                valid_scale=(0,np.inf)
                bboxes_list.append(self.predict(img,test_input_size,valid_scale))
                if flip_test:
                    bboxes_flip=self.predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes=np.row_stack(bboxes_list)
        else:
            bboxes=self.predict(img,self.val_shape,(0,np.inf))

        bboxes=py_cpu_nms(bboxes,self.conf_thresh)

        return bboxes

    def predict(self, img, test_input_size, valid_scale):
        org_img=np.copy(img)
        org_h,org_w,_=org_img.shape
        img=self.get_img_tensor(img,test_input_size).to(self.device)
        self.model.eval()

        with torch.no_grad():
            _,pd=self.model(img)
        pred_bbox=pd.squeeze().cpu().numpy()
        bboxes=self.convert_pred(pred_bbox,test_input_size,(org_h,org_w),valid_scale)
        return bboxes

    def get_img_tensor(self, img, test_input_size):
        img=Resize((test_input_size,test_input_size),correct_box=False)(img,None).transpose(2,0,1)
        return torch.from_numpy(img[np.newaxis, ...]).float()

    def convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        org_h,org_w = org_img_shape
        resize_ratio=min(1.0*test_input_size/org_w,1.0*test_input_size/org_h)

        dw=(test_input_size-resize_ratio*org_w)/2
        dh=(test_input_size-resize_ratio*org_h)/2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        pred_coor=np.concatenate([np.maximum(pred_coor[:,:2],[0,0]),
                                  np.minimum(pred_coor[:,2:],[org_w - 1, org_h - 1])],axis=-1)

        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        classes=np.argmax(pred_prob,axis=-1)
        scores=pred_conf*pred_prob[np.arange(len(pred_coor)),classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes



    def cal_aps(self):
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path, 'ImageSets', 'Main', 'test.txt')
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs



