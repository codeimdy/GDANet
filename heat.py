from PIL import Image
import torchvision
import cv2
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

# from functions import MSDeformAttnFunction
from functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def transs(a):
    # sampling_offsets_before=nn.Linear(256, 8 * 4 * 4 * 2)
    # a = nn.Linear(256, 8 * 4 * 4 * 2)
    a_split = torch.split(a, 1, dim=2)
    # 这一块可以改循环，或concat加1*1卷积
    a1 = torch.add(a_split[0], a_split[1])
    a2 = torch.add(a1, a_split[2])
    a3 = torch.add(a2, a_split[3])
    a4 = torch.add(a3, a_split[4])
    a5 = torch.add(a4, a_split[5])
    a6 = torch.add(a5, a_split[6])
    a7 = torch.add(a6, a_split[7])
    a = torch.concat([a_split[0], a1, a2, a3, a4, a5, a6, a7], dim=2)
    return a


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        Attention 维数 d_model 256
        4个特征层上使用 n_levels 4
        多头的头 n_head 8
        对应每一个参考点会有4个离散的偏移的采样位置 n_points 4
        """

        # todo 在这个类的定义中定义了4个全连接
        # todo 第一个 采样_offsets：头*特征层数*点*2 （对应每一个头/层级/点有一个x, y坐标）
        # todo 第二个 Attention：头*特征层数*点
        # todo 后两个一个是value的投影一个是输出的投影
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 采样点的偏移量 论文中的2MK 就是n_heads*n_points 因为多层特征因此还有*n_levels
        # self.sampling_offsets=transs(a=nn.Linear(d_model, n_heads * n_levels * n_points * 2))
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # 权重矩阵，论文中的MK 就是n_heads*n_points 因为多层特征因此还有*n_levels
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 图2中左下区域的那个Linear
        self.value_proj = nn.Linear(d_model, d_model)
        # 图2中右下区域的那个Linear
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                              self.n_levels,
                                                                                                              self.n_points,
                                                                                                              1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        """
        query是上一层的输出加上位置编码
        :param query                       (N, Length_{query}, C)
        参考点位的坐标
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        encoder是上一层的输出，decoder使用的是encoder的输出 [bs, all hw, 256]
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        4个特征层的宽高
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        各个特征层的起始index的下标
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        [bs, all hw]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        # query 是src+pos，query下面变成了attention_weights
        # input_flatten 是src，input_flatten对应了 V
        # bs, all hw (decoder是300), 256
        # todo 第一个值Len_q的长度是所有宽高的和
        N, Len_q, _ = query.shape
        # [bs, all hw, 256] input_flatten在encoder和decoder阶段，都是all hw的那个维度大小
        # 因为在decoder阶段，他就是encoder的memory
        # todo 第二个值Len_in 如果是在decoder阶段第一个是300，第二个还是宽高的和
        N, Len_in, _ = input_flatten.shape
        # encoder阶段 Len_q和Len_in是相同的
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        # todo 对输入进行第一次投影
        # 对encoder上一层的输出，或者decoder使用的encoder的输出 进行一层全连接变换，channel不变
        # 图2中左下的全连接
        value = self.value_proj(input_flatten)

        if input_padding_mask is not None:
            # 在mask的地方填充0 [bs, all hw, 256]
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # 分成多头，拆分的是最后的256 [bs, all hw, 256] -> [bs, all hw, 8, 32]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # sampling_offsets 是一层全连接
        # like（bs, all hw, 8, 4, 4, 2）8个头，4个特征层，4个采样点，每个采样点2个偏移量坐标（x, y）
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # bjad=sampling_offsets.shape
        sampling_offsets = transs(sampling_offsets)
        # attention_weights 是一层全连接
        # like（bs, all hw, 8, 16）
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        # like （bs, all hw, 8, 4, 4）
        # 经过softmax 保证权重和为1，然后拆分成4 4（4层，4个采样点
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # input_spatial_shapes 换位置，高宽 变成 宽高
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)

            # reference_points [bs, all hw, 4, 2] -> [bs, all hw, 1, 4, 1, 2]
            # sampling_offsets [bs, all hw, 8, 4, 2]
            # offset_normalizer [4, 2] -> [1, 1, 1, 4, 1, 2]
            # like (bs, hw, 8, 4, 4, 2)
            # 采样点加上偏移量 sampling_offsets / offset_normalizer 表示相对偏移量
            # = 真正采样点的位置
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
        output = self.output_proj(output)
        return output






def save_model(model):
    torch.save(obj=model, f='B.pth')

if __name__ == '__main__':
    net = MSDeformAttn()
    save_model(net)
    # model = torch.load(f="A.pth")

if __name__ == '__main__':
    # 图片路径
    img_path = r'/media/hanyong/zgw/ZWD/choose/000000000071.jpg'
    # 给图片进行标准化操作
    img = Image.open(img_path).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, ], [0.5, ])])
    data = transforms(img).unsqueeze(0)
    # 用于加载Pycharm中封装好的网络框架
    # model = torchvision.models.vgg11_bn(pretrained=True)
    # 用于加载1中生成的.pth文件
    model = torch.load(f="B.pth")
    # 打印一下刚刚生成的.pth文件看看他的网络结构
    print(model)
    model.eval()
    #实例化
    net = MSDeformAttn()
    save_model(net)
    features=MSDeformAttn.forward(net,data,net,net,net)
    # model = torch.load(f="A.pth")
    features.retain_grad()
    # t = model.avgpool(features)
    # t = t.reshape(1, -1)
    # output = model.classifier(t)[0]
    # pred = torch.argmax(output).item()
    # pred_class = output[pred]
    #
    # pred_class.backward()
    grads = features.grad

    features = features[0]
    # avg_grads = torch.mean(grads[0], dim=(1, 2))
    # avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
    # features *= avg_grads

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    cv2.imshow('1', superimposed_img)
    cv2.waitKey(0)