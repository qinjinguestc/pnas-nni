from util.prim_ops_set import *
from .fcn import FCNHead
from .base import BaseNet
from util.functional import *
from torch.nn.functional import interpolate


class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_prev_prev, c_prev, c, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=2, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, ops_order='act_weight_norm')

        if cell_type == 'up':
            op_names, idx = zip(*genotype.up)
            concat = genotype.up_concat
        else:
            op_names, idx = zip(*genotype.down)
            concat = genotype.down_concat
        self.dropout_prob = dropout_prob
        self._compile(c, op_names, idx, concat)

    def _compile(self, c, op_names, idx, concat):
        assert len(op_names) == len(idx)
        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, idx):
            op = OPS[name](c, None, affine=True, dp=self.dropout_prob)
            self._ops += [op]
        self._indices = idx

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]

            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]

            h1 = op1(h1)
            h2 = op2(h2)

            # the size of h1 and h2 may be different, so we need interpolate
            if h1.size() != h2.size() :
                _, _, height1, width1 = h1.size()
                _, _, height2, width2 = h2.size()
                if height1 > height2 or width1 > width2:
                    h2 = interpolate(h2, (height1, width1))
                else:
                    h1 = interpolate(h1, (height2, width2))
            s = h1+h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class PNasUnet(BaseNet):
    """Construct a network"""

    def __init__(self, nclass, in_channels, backbone=None, aux=False,
                 c=32, depth=2, dropout_prob=0,
                 genotype=None, double_down_channel=True, bilinear=True):
        # depth = 5
        super(PNasUnet, self).__init__(nclass, aux, backbone, norm_layer=nn.GroupNorm)
        self._depth = depth
        self._double_down_channel = double_down_channel
        self.n_channels = in_channels
        self.n_classes = nclass
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128, 64, bilinear)
        self.up2 = Up(64, 32, bilinear)
        self.outc = OutConv(32, nclass)
        cc = 32
        stem_multiplier = 4
        c_curr = stem_multiplier * cc

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c

        # the stem need a complicate mode
        in_channel = 128
        self.stem0 = ConvOps(in_channel, c_prev_prev, kernel_size=1, ops_order='weight_norm') # origin:in_channel = in_channels
        self.stem1 = ConvOps(in_channel, c_prev, kernel_size=3, stride=2, ops_order='weight_norm')

        assert depth >= 2, 'depth must >= 2'

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()
        down_cs_nfilters = []

        # create the encoder pathway and add to a list
        down_cs_nfilters += [c_prev]
        down_cs_nfilters += [c_prev_prev]
        for i in range(depth):
            c_curr = 2 * c_curr if self._double_down_channel else c_curr  # double the number of filters
            down_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='down', dropout_prob=dropout_prob)
            self.down_cells += [down_cell]
            c_prev_prev, c_prev = c_prev, down_cell._multiplier*c_curr
            down_cs_nfilters += [c_prev]

        # create the decoder pathway and add to a list
        # c_prev = int(256)
        for i in range(depth+1):
            c_prev_prev = down_cs_nfilters[-(i + 2)] # the horizontal prev_prev input channel
            up_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='up',  dropout_prob=dropout_prob)
            self.up_cells += [up_cell]
            c_prev = up_cell._multiplier*c_curr
            # c_prev = c_curr
            c_curr = c_curr // 2 if self._double_down_channel else c_curr  # halve the number of filters

        self.nas_unet_head = ConvOps(cc, nclass, kernel_size=1, ops_order='weight')

        if self.aux:
            self.auxlayer = FCNHead(c_prev, nclass, nn.BatchNorm2d)


    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        # x3 = self.down3(x2)
        _, _, h, w = x.size()
        s0, s1 = self.stem0(x2), self.stem1(x2)

        down_cs = []

        # encoder pathway
        down_cs.append(s0)
        down_cs.append(s1)
        for i, cell in enumerate(self.down_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + down
            s0, s1 = s1, cell(s0, s1)
            down_cs.append(s1)

        # decoder pathway
        uptensor = []
        for i, cell in enumerate(self.up_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + up
            s0 = down_cs[-(i + 2)]  # horizon input
            s1 = cell(s0, s1)
            uptensor.append(s1)

        x = self.up1(s1, x1)
        x = self.up2(x, x0)

        output = self.nas_unet_head(x)
        outputs = []
        outputs.append(output)

        if self.aux:  # use aux header
            auxout = self.auxlayer(x)
            auxout = interpolate(auxout, (h, w), **self._up_kwargs)
            outputs.append(auxout)

        return outputs

def get_pnas_unet(dataset='pascal_voc', **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = PNasUnet(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].IN_CHANNELS,
                 **kwargs)
    return model

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        # in_channel = x1.size()[attention] + x2.size()[attention]
        # out_channel = x1.size()[attention]
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


