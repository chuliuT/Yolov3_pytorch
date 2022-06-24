import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from utils.tools import xywh2xyxy,py_cpu_nms
from yolov3 import Yolov3
from yolov3_pretrained import Darknet
import config.yolov3_config_voc as cfg

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

def get_yolo_bboxes(model,img,test_shape):
    org_h,org_w,_=img.shape
    img=img_preprocess(img,test_shape)
    with torch.no_grad():
        _,p_d = model(img)
    pred_bbox = p_d.squeeze().cpu().numpy()
    bboxes = convert_pred(pred_bbox, test_shape[0], (org_h, org_w), (0,np.inf))
    return bboxes



if __name__ == '__main__':
    # net = Yolov3()
    # net.load_darknet_weights('yolov3.weights')
    # net.load_state_dict(torch.load('weights/best.pt',map_location='cuda:0'))
    with open('config/coco.names','r') as f:
        coco_name=f.readlines()

    # net =Darknet(config_path='config/yolov3_ori.cfg')
    # net.load_darknet_weights('yolov3.weights')
    net = Darknet(config_path='config/yolov3.cfg')
    net.load_state_dict(torch.load('weights/best.pt', map_location='cuda:0'))
    net.eval()
    print("load done")
    img=cv2.imread('2007_000836.jpg')
    bboxes=get_yolo_bboxes(net,img,(416,416))
    print(bboxes)
    for item in bboxes:
        box=item[:4]
        prob=item[4]
        cls=item[5]
        # cv2.putText(img,str(round(prob,3))+'|'+str(coco_name[int(cls)]),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.putText(img,str(round(prob,3))+'|'+str(cfg.DATA['CLASSES'][int(cls)]),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
    plt.imshow(img[:,:,::-1])
    plt.show()