# coding=utf-8
# project
DATA_PATH = "/media/fq/disk_8T/tcl_work"
PROJECT_PATH = "/media/fq/disk_8T/tcl_work/Yolov3_pytorch"

DATA = {"CLASSES": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor'],
        "NUM": 20}

# model
# 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326
MODEL = {"ANCHORS": [[(10, 13), (16, 30), (33, 23)],  # Anchors for small obj
                     [(30, 61), (62, 45), (59, 119)],  # Anchors for medium obj
                     [(116, 90), (156, 198), (373, 326)]],  # Anchors for big obj
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3
         }

# train
TRAIN = {
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": False,
    "BATCH_SIZE": 4,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2  # or None
}

# test
TEST = {
    "TEST_IMG_SIZE": 544,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": True,
    "FLIP_TEST": True
}
