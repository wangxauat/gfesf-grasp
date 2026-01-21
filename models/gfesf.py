import torch.nn as nn
import torch
from torchsummary import summary
from inference.models.grasp_model import GraspModel
# from MLLAttention import MLLA

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):

    r"""Rotary Positional Embedding.
    """

    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        # 获取输入张量的形状
        *channel_dims, feature_dim = x.shape[1:-1][0], x.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, "Feature dimension must be divisible by 2 * k_max"

        # 生成角度
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims],
                                           indexing='ij')], dim=-1)

        # 计算旋转矩阵的实部和虚部
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)

        return rotations

    def forward(self, x):
        # 生成旋转矩阵
        rotations = self.generate_rotations(x)

        # 将 x 转换为复数形式
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

        # 应用旋转矩阵
        pe_x = torch.view_as_complex(rotations) * x_complex

        # 将结果转换回实数形式并展平最后两个维度
        return torch.view_as_real(pe_x).flatten(-2)


class MLLA(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim=3, input_resolution=[160, 160], num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        # self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qk = nn.Conv2d(dim, dim * 2, 3, padding=1, groups=dim)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        qk = self.qk(x)
        qk = qk.reshape(qk.size(0), qk.size(2) * qk.size(3), qk.size(1))
        x = x.reshape((x.size(0), x.size(2) * x.size(3), x.size(1)))

        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        # self.rope = RoPE(shape=(h, w, self.dim))
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = qk.reshape(b, n, 2, c).permute(2, 0, 1, 3)
        # qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.transpose(2, 1).reshape((b, c, h, w))
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class DilatedConvModule(nn.Module):
    def __init__(self):
        super(DilatedConvModule, self).__init__()

        # 三个并行的空洞卷积
        #self.dilated_conv2 = nn.Conv2d(in_channels=43, out_channels=43, kernel_size=3, padding=2, dilation=2)
        #self.dilated_conv4 = nn.Conv2d(in_channels=43, out_channels=43, kernel_size=3, padding=4, dilation=4)
        #self.dilated_conv6 = nn.Conv2d(in_channels=42, out_channels=42, kernel_size=3, padding=6, dilation=6)
        self.dilated_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=4, dilation=4)
        self.dilated_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=6, dilation=6)
        # self.dilated_conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=12, dilation=12)
        self.mlla = MLLA(128)

    def forward(self, x):
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        N = x.shape[2] * x.shape[3]
        # 进行通道splt
        x1, x2 = torch.split(x, [C // 2, C // 2], dim=1)   #源代码
        #x1, x2, x3 = torch.split(x, [43, 43, 42], dim=1)#消融实验 三分支
        # 2个并行的空洞卷积
        out4 = self.dilated_conv4(x1)   #源代码
        out6 = self.dilated_conv6(x2)   #源代码
        #out2 = self.dilated_conv2(x1)     #三分支
        #out4 = self.dilated_conv4(x2)   #三分支
        #out6 = self.dilated_conv6(x3)   #三分支

        # 通道拼接
        x_dialate = torch.cat(( out4, out6), dim=1)
        #x_dialate = torch.cat((out2, out4, out6), dim=1)
        out = self.mlla(x_dialate)
        return out


class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=out_channels)

    def forward(self, x, y):
        '''整个模块以D(y)为主'''
        '''Forget gate'''
        forget_gate_feature = self.conv(y)
        forget_gate_weight = self.sigmoid(forget_gate_feature)
        '''Update gate'''
        update_gate_feature_tanh = self.tanh(forget_gate_feature)
        update_gate_weight = forget_gate_weight * update_gate_feature_tanh
        '''Output gate'''


        '''最终的输出'''
        x_stage = x * forget_gate_weight + update_gate_weight
        # output_x = x_stage + x

        output_y = self.tanh(self.conv(x_stage)) * forget_gate_weight + y + y

        # return output_x, output_y
        return output_y

#上采样特征融合
class feature_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, dims):
        super(feature_fusion, self).__init__()
        self.up_upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.lstm = LSTM(in_channels=out_channels, out_channels=out_channels)

    def forward(self, high_feateure, low_feateure):
        high_feateure = self.up_upsample(high_feateure)
        high_feateure = self.lstm(low_feateure, high_feateure)
        return high_feateure




class ResNet(GraspModel):

    def __init__(self,
                 input_channels,
                 block,
                 blocks_num,
                 groups=1,
                 width_per_group=64,
                 dropout=False):
        super(ResNet, self).__init__()
        self.in_channel = 32

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv_bn_relu_pool = nn.Sequential(
            nn.Conv2d(input_channels, self.in_channel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 32代表当前bottleneck的输出channel；
        '''layer1'''
        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        '''layer2'''
        # 64代表当前bottleneck的输出channel；
        # stride=2时表示添加short-cut，eg.bottleneck2-4添加short-cut
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        '''layer3'''
        self.layer3 = self._make_layer(block, 128, blocks_num[2], stride=2)

        '''layer4'''
        self.layer4 = self._make_layer(block, 128, blocks_num[3], stride=2)

        '''Global feature enhancement module(GFEM)'''
        self.GFEM = DilatedConvModule()
        # self.raw_ompensation = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=4)
        # )


        self.up3 = feature_fusion(128, 128, 128)
        self.up2 = feature_fusion(128, 128, 128)
        self.up1 = feature_fusion(128, 64, 64)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        #输出部分
        self.pos_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.cos_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.sin_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.width_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=0.1)
        self.dropout_cos = nn.Dropout(p=0.1)
        self.dropout_sin = nn.Dropout(p=0.1)
        self.dropout_wid = nn.Dropout(p=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print('lstm')
        #宽高/2---[2, 32, 112, 112]
        x = self.conv_bn_relu_pool(x)

        '''bottleneck-1'''
        stage1 = self.layer1(x)  #宽高/2---[2, 64, 112, 112]

        '''bottleneck-2'''
        stage2 = self.layer2(stage1)  # 宽高/4---[2, 128, 56, 56]

        '''bottleneck-3'''
        stage3 = self.layer3(stage2)  # 宽高/8---[2, 128, 28, 28]

        '''bottleneck-4'''
        stage4 = self.layer4(stage3)  # 宽高/16---[2, 128, 14, 14]

        '''GFEM'''
        global_feature = self.GFEM(stage4)  #[2, 128, 14, 14]
        # raw_ompensation = self.raw_ompensation(x)
        # [2, 512, 14, 14]
        # global_feature = torch.cat((global_feature, raw_ompensation), dim=1)

        # up_stage4 = self.up4(global_feature, stage4)
        # up_stage3 = self.up3(up_stage4, stage3)
        up_stage3 = self.up3(global_feature, stage3)
        up_stage2 = self.up2(up_stage3, stage2)
        up_stage1 = self.up1(up_stage2, stage1)
        out = self.up0(up_stage1)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(out))
            cos_output = self.cos_output(self.dropout_cos(out))
            sin_output = self.sin_output(self.dropout_sin(out))
            width_output = self.width_output(self.dropout_wid(out))
        else:
            pos_output = self.pos_output(out)
            cos_output = self.cos_output(out)
            sin_output = self.sin_output(out)
            width_output = self.width_output(out)
        return pos_output, cos_output, sin_output, width_output


def RS_model(input_channels):
    return ResNet(input_channels=input_channels, block=BasicBlock, blocks_num=[2, 2, 2, 2])


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = RS_model(input_channels=4).to(device)
    x = torch.randn((2, 4, 448, 448)).to(device)
    output = net(x)

    # Print my_model architecture.
    summary(net, (4, 448, 448))

