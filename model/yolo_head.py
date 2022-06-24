import torch
import torch.nn as nn


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = nC
        self.stride = stride

    def forward(self, x):
        batch_size, num_grid = x.shape[0], x.shape[-1]
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, num_grid, num_grid).permute(0, 3, 4, 1,
                                                                                                   2).contigous()
        x_decode = self.decode(x.clone())
        return x, x_decode

    def decode(self, x):
        batch_size, num_grid = x.shape[:2]
        device = x.device
        stride = self.stride
        anchors = (1.0 * self.anchors).to(device)

        conv_dxdy = x[..., 0:2]
        conv_dwdh = x[..., 2:4]
        conv_conf = x[..., 4:5]
        conv_prob = x[..., 5:]

        y = torch.arange(0, num_grid).unsqueeze(1).repeat(1, num_grid)
        x = torch.arange(0, num_grid).unsqueeze(0).repeat(num_grid, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_conf)
        pred_prob = torch.sigmoid(conv_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, 5 + self.num_classes) if not self.training else pred_bbox
