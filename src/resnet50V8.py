# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
            eval_mode=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        # v2

        self.fc_class = nn.Linear(512 * block.expansion, 16)


        # num_out_filters = 512
        # self.inplanes = 2048

        # self.proj_weights = torch.nn.Parameter(
        #     torch.ones(2))

        self.proj_weights = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        self.norm1 = nn.BatchNorm1d(2048)  # 使用批量规范化作为示例
        self.proj_weights.requires_grad = True



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x, low_feature=False):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # x.shape = torch.Size([96, 64, 34, 34])
        x = self.layer1(x)

        # feature_low.shape = torch.Size([96, 256* 34* 34])
        # feature_low = x

        x = self.layer2(x)
        # feature_low = x
        x = self.layer3(x)
        feature_low = x
        x = self.layer4(x)

        if self.eval_mode:
            return x
        if low_feature:
            # feature_low.shape = torch.Size([96, 256* 34* 34])
            # 方法一
            feature_low = self.avgpool(feature_low)
            feature_low = torch.flatten(feature_low, 1)
            # 方法二
            # feature_low = feature_low.view(feature_low.size(0), -1)
            # 方法三
            # adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
            # feature_low = adaptive_pool(feature_low).view(feature_low.size(0), -1)  # 大小 (96, 256*8*8)


            # feature_high.shape = torch.Size([96, 2048* 5* 5])
            feature_high = self.avgpool(x)
            feature_high = torch.flatten(feature_high, 1)

            res = []
            proj_layer1 = torch.nn.Linear(feature_low.shape[1], 2048).cuda()
            res.append(proj_layer1(feature_low))

            # proj_layer2 = torch.nn.Linear(feature_high.shape[1], 2048).cuda()
            # print(proj_layer2(feature_high).shape) = torch.Size([96, 2048])
            # res.append(proj_layer2(feature_high))
            res.append(feature_high)

            res = torch.stack(res)

            proj_weights = F.softmax(self.proj_weights, dim=0)
            res = res * proj_weights[:, None, None]
            # print("aaaaaaaaaaaaaaaaaaaa", proj_weights)
            # proj_weights = [0.0134, 0.9866]
            # res[0] = res[0] * proj_weights[0]
            # res[1] = res[1] * proj_weights[1]

            x = res.sum(dim=0)
            # res.shape = torch.Size([96, 2048])
            # Use final norm
            # x = self.norm1(x)
            # afterNorm1 = torch.Size([96, 2048])

        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs, classification=False, low_feature=False):
        # swav
        # len(inputs) = 2
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in inputs]), return_counts=True, )[1], 0)

        if classification:
            self.proj_weights.requires_grad = False

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True), low_feature=False)
            # h = self.predictor
            # _out = h(_out)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # 展平之前torch.Size([96, 2048, 3, 3])
        # output.shape=torch.Size([96, 2048])
        if not classification:
            # 额外分支
            bs_all = len(inputs[0])
            rotation_transform = RandomRotationTransform()
            label_combined = []

            # inputs中包含两个数据增强bs
            images = inputs[1]

            # 拆分高，维度为dim2
            c1, c2 = images.split([16, 16], dim=2)
            # 拆分宽，总体将图片拆分为四块
            f1, f2 = c1.split([16, 16], dim=3)
            f3, f4 = c2.split([16, 16], dim=3)

            # 创建一个新的列表来保存旋转后的图像分片
            rotated_fragments = []
            fragments = [f1, f2, f3, f4]
            for i, fragment in enumerate(fragments):
                # 应用旋转变换并存储旋转的角度
                rotated_fragment, rotation_idces = rotation_transform(fragment)
                rotated_fragments.append(rotated_fragment)
                # 计算综合标签
                base_label = i * 4  # 四个可能的旋转角度
                combined_labels = [base_label + rot_idx for rot_idx in rotation_idces]
                label_combined.extend(combined_labels)

            q_gather = torch.cat(rotated_fragments, dim=0)
            q_gather = self.forward_backbone(q_gather.cuda(), low_feature=False)
            # q_gather现在是一个展平的特征图

            # 分类处理 - 修改为输出16个分类
            q_class = self.fc_class(q_gather)  # 假设fc_class是一个新的全连接层，输出16个分类

            # 存放综合标签
            label_combined = torch.LongTensor(label_combined).cuda()

            # 返回结果，包括分类和综合标签
            return self.forward_head(output), q_class, label_combined


        return self.forward_head(output)


class RandomRotationTransform:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, images):
        rotated_images = []
        rotation_indices = []

        for image in images:
            angle = random.choice(self.angles)
            # 如果image是一个PIL图像，请注释掉下一行
            # image = TF.to_pil_image(image.cpu())  # 如果图像是Tensor，则转换为PIL图像
            rotated_image = TF.rotate(image, angle)
            # rotated_image = TF.to_tensor(rotated_image).cuda()  # 将图像转换回Tensor并移回GPU
            rotated_images.append(rotated_image)
            rotation_indices.append(self.angles.index(angle))

        return torch.stack(rotated_images), rotation_indices


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


def resnet50w5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)
