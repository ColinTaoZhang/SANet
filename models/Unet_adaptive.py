import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, mask_in_ch, mask_out_ch):
        super(conv_block, self).__init__()

        self.conv1 = AdaConv(in_ch, out_ch, mask_in_ch, mask_out_ch, kernel_size=3, stride=1)
        self.conv2 = AdaConv(out_ch, out_ch, mask_out_ch, mask_out_ch, kernel_size=3, stride=1)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x, mask):
        res = self.conv_residual(x)

        x, mask = self.conv1(x, mask)
        x, mask = self.act(x), self.act(mask)
        x, mask = self.conv2(x, mask)
        x, mask = self.act(x), self.act(mask)

        return x+res, mask

class Ada_Res_U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, mask_in_ch=1, ps=True, noisemap=0):
        super(Ada_Res_U_Net, self).__init__()
        self.restoration = Ada_U_Net(in_ch+noisemap, out_ch, mask_in_ch, ps)
        self.residual = AdaConv(in_ch+noisemap, out_ch, mask_in_ch, 1, kernel_size=3, stride=1)
    
    def forward(self, x, mask):
        x_res, mask_res = self.residual(x, mask)
        out = x_res + self.restoration(x, mask)
        return out

class Ada_U_Net(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, mask_in_ch=1, ps=True):
        super(Ada_U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        m_n1 = 4
        m_filters = [m_n1, m_n1 * 2, m_n1 * 4, m_n1 * 8, m_n1 * 16]
        self.ps = ps
        # if not self.ps:
        #     out_ch = 4
        
        self.Down1 = Down(filters[0], filters[0], m_filters[0], m_filters[0], kernel_size=4, stride=2)
        self.Down2 = Down(filters[1], filters[1], m_filters[1], m_filters[1], kernel_size=4, stride=2)
        self.Down3 = Down(filters[2], filters[2], m_filters[2], m_filters[2], kernel_size=4, stride=2)
        self.Down4 = Down(filters[3], filters[3], m_filters[3], m_filters[3], kernel_size=4, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], mask_in_ch, m_filters[0])
        self.Conv2 = conv_block(filters[0], filters[1], m_filters[0], m_filters[1])
        self.Conv3 = conv_block(filters[1], filters[2], m_filters[1], m_filters[2])
        self.Conv4 = conv_block(filters[2], filters[3], m_filters[2], m_filters[3])
        self.Conv5 = conv_block(filters[3], filters[4], m_filters[3], m_filters[4])

        # self.Up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up5 = Up(filters[4], filters[3], m_filters[4], m_filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3], m_filters[4], m_filters[3])

        # self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up4 = Up(filters[3], filters[2], m_filters[3], m_filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2], m_filters[3], m_filters[2])

        # self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up3 = Up(filters[2], filters[1], m_filters[2], m_filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1], m_filters[2], m_filters[1])

        # self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up2 = Up(filters[1], filters[0], m_filters[1], m_filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0], m_filters[1], m_filters[0])

        if self.ps:
            self.Conv = nn.Conv2d(filters[0], out_ch*4, kernel_size=1, stride=1, padding=0)
            self.Up = nn.PixelShuffle(2)
        else:
            self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mask):

        e1, me1 = self.Conv1(x, mask)

        e2, me2 = self.Down1(e1, me1)
        e2, me2 = self.Conv2(e2, me2)

        e3, me3 = self.Down2(e2, me2)
        e3, me3 = self.Conv3(e3, me3)

        e4, me4 = self.Down3(e3, me3)
        e4, me4 = self.Conv4(e4, me4)

        e5, me5 = self.Down4(e4, me4)
        e5, me5 = self.Conv5(e5, me5)

        d5, md5 = self.Up5(e5, me5)
        d5, md5 = torch.cat((e4, d5), dim=1), torch.cat((me4, md5), dim=1)
        d5, md5 = self.Up_conv5(d5, md5)

        d4, md4 = self.Up4(d5, md5)
        d4, md4 = torch.cat((e3, d4), dim=1), torch.cat((me3, md4), dim=1)
        d4, md4 = self.Up_conv4(d4, md4)

        d3, md3 = self.Up3(d4, md4)
        d3, md3 = torch.cat((e2, d3), dim=1), torch.cat((me2, md3), dim=1)
        d3, md3 = self.Up_conv3(d3, md3)

        d2, md2 = self.Up2(d3, md3)
        d2, md2 = torch.cat((e1, d2), dim=1), torch.cat((me1, md2), dim=1)
        d2, md2 = self.Up_conv2(d2, md2)

        if self.ps:
            out = self.Conv(d2)
            out = self.Up(out)
        else:
            out = self.Conv(d2)

        return out

class Down(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mask_in_ch=1, mask_out_ch=1, kernel_size=4, stride=2):
        super(Down, self).__init__()
        self.mask_transform = nn.Conv2d(mask_in_ch, mask_out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, mask):
        x = self.conv(x)
        mask = self.mask_transform(mask)
        return x, mask

class Up(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mask_in_ch=1, mask_out_ch=1, kernel_size=3, stride=1):
        super(Up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_transform = nn.Conv2d(mask_in_ch, mask_out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, mask):
        x = self.conv(self.up(x))
        mask = self.mask_transform(self.up(mask))
        return x, mask

class AdaConv(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mask_in_ch=1, mask_out_ch=1, kernel_size=5, stride=1, n_feat=6):
        super(AdaConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mask_in_ch = mask_in_ch
        self.mask_out_ch = mask_out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_channels = out_ch
        self.groups = out_ch // self.groups_channels

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], padding=(kernel_size-1)//2, stride=1)

        self.adaptiveweight = nn.Sequential(
            nn.Conv2d(mask_in_ch, n_feat, kernel_size=2*kernel_size-1, stride=stride, padding=(2*kernel_size-1-stride)//2, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(n_feat, kernel_size**2 * self.groups, kernel_size=1, stride=1, padding=0, bias=True))

        self.mask_transform = nn.Conv2d(mask_in_ch, mask_out_ch, kernel_size=2*kernel_size-1, stride=stride, padding=(2*kernel_size-1-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, mask):
        weight = self.adaptiveweight(mask) # b*k^2*h*w
        batch, channel, height, width = weight.size()
        weight = weight.view(batch, self.groups, 1, self.kernel_size**2, height, width)
        x = self.conv(x)
        x = self.unfold(x)
        x = x.view(batch, self.groups, self.groups_channels, self.kernel_size**2, height, width)
        x = (weight * x).sum(dim=3).view(batch, self.out_ch, height, width)

        mask = self.mask_transform(mask)

        return x, mask