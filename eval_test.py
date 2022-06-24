import argparse
from eval.evaluator import Evaluator
from utils.tools import *
# from yolov3 import Yolov3
from yolov3_pretrained import Darknet

class yolov3tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=416,
                 visiual=None,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = torch.device('cuda:1')
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__visiual = visiual
        self.__eval = eval
        self.__classes = cfg.DATA["CLASSES"]

        # self.__model = Yolov3().to(self.__device)
        self.__model = Darknet('config/yolov3.cfg').to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def test(self):
        if self.__eval:
            mAP = 0
            print('*' * 20 + "Validate" + '*' * 20)

            with torch.no_grad():
                APs = Evaluator(self.__model).AP_VOC(self.__multi_scale_test, self.__flip_test)

                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                print('mAP:%g' % (mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weights/best.pt', help='weight file path')
    parser.add_argument('--visiual', type=str, default='./data/test', help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=True, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    opt = parser.parse_args()

    yolov3tester(weight_path=opt.weight_path,
                 gpu_id=opt.gpu_id,
                 eval=opt.eval,
                 visiual=opt.visiual).test()
    # mobilenet
    # aeroplane --> mAP: 0.516254389435098
    # bicycle --> mAP: 0.5168035814668492
    # bird --> mAP: 0.2969817160738938
    # boat --> mAP: 0.26569082424278
    # bottle --> mAP: 0.1720514081260432
    # bus --> mAP: 0.48831343134056415
    # car --> mAP: 0.6223821883411716
    # cat --> mAP: 0.47037661370843675
    # chair --> mAP: 0.2360627876780596
    # cow --> mAP: 0.31901461408289233
    # diningtable --> mAP: 0.39121990322612205
    # dog --> mAP: 0.46363350739726505
    # horse --> mAP: 0.6172647046659617
    # motorbike --> mAP: 0.5905205910468687
    # person --> mAP: 0.592034649069045
    # pottedplant --> mAP: 0.1681534665298724
    # sheep --> mAP: 0.441727714432486
    # sofa --> mAP: 0.3104313770815275
    # train --> mAP: 0.5144954960144708
    # tvmonitor --> mAP: 0.4742977311191586
    # mAP: 0.423386

