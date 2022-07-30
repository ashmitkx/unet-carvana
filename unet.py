import torch as t
from torch import nn
from torchvision.transforms import functional as tvfs


class DoubleConv(nn.Module):  # double convolution, used in UNet
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.down_dconvs = nn.ModuleList()  # double convolution along downward path
        for feature in features:
            self.down_dconvs.append(DoubleConv(in_ch=in_ch, out_ch=feature))
            in_ch = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pooling along downward path

        self.bottleneck = DoubleConv(in_ch=features[-1], out_ch=features[-1] * 2)

        self.up_dconvs = nn.ModuleList()  # double convolution along upward path
        self.trans_convs = nn.ModuleList()  # transposed convolution along upward path
        for feature in reversed(features):
            self.up_dconvs.append(DoubleConv(in_ch=feature * 2, out_ch=feature))
            self.trans_convs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            ),

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        skip_conns = []

        for down_dconv in self.down_dconvs:
            x = down_dconv(x)  # double convolution along downward path
            skip_conns.append(x)  # store output for copy/concat phase
            x = self.pool(x)  # pooling after each downward double convolution

        x = self.bottleneck(x)  # bottleneck layer at base of UNet
        skip_conns = skip_conns[::-1]

        for idx, up_dconv in enumerate(self.up_dconvs):
            trans_conv = self.trans_convs[
                idx]  # transposed convolution before each upward double convolution
            x = trans_conv(x)

            skip_conn = skip_conns[idx]
            if x.shape != skip_conn.shape:  # tensor x may be smaller than the skip_conn tensor, hence resize x be the same
                x = tvfs.resize(x, size=skip_conn.shape[-2:])
            x = t.cat((skip_conn, x), dim=1)  # concat skip conn along channel dimension

            x = up_dconv(x)  # double convolution along upward path

        x = self.final_conv(x)  # final 1x1 convolution
        return x


def test_unet():
    inp = t.randn((3, 3, 163, 163))  # prime no. dimensions
    model = UNet(in_ch=3, out_ch=1)
    preds = model(inp)

    # assert input and output images are of same exact dimensions
    assert preds.shape[-2:] == inp.shape[-2:]
    assert preds.shape[1] == 1  # assert predictions have 1 channel

    print('test passed')
    print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


if __name__ == '__main__':
    test_unet()
