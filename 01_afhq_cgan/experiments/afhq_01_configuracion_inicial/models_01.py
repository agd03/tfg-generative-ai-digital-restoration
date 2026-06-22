import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, k=3, s=1, p=1, norm=True, dropout=0.0):
    layers = [
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )
    ]

    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))

    layers.append(nn.LeakyReLU(0.2, inplace=True))

    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, k=4, s=2, p=1, norm=True, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )
    ]

    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))

    layers.append(nn.ReLU(inplace=True))

    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class InpaintUNet(nn.Module):
    """
    U-Net condicional para inpainting.

    Entrada:
        (B, 4, H, W) = imagen enmascarada RGB + máscara

    Salida:
        (B, 3, H, W) = predicción completa
    """

    def __init__(self, in_ch=4, out_ch=3, base=64, dropout=0.0):
        super().__init__()

        # Encoder
        self.enc1 = conv_block(in_ch, base, norm=False)
        self.enc2 = conv_block(base, base * 2, s=2)
        self.enc3 = conv_block(base * 2, base * 4, s=2)
        self.enc4 = conv_block(base * 4, base * 8, s=2, dropout=dropout)
        self.enc5 = conv_block(base * 8, base * 8, s=2, dropout=dropout)

        # Bottleneck
        self.bot1 = conv_block(base * 8, base * 8, dropout=dropout)
        self.bot2 = conv_block(base * 8, base * 8, norm=False, dropout=dropout)

        # Decoder con skip connections
        self.up5 = deconv_block(base * 8, base * 8, dropout=dropout)
        self.dec5 = conv_block(base * 8 + base * 8, base * 8)

        self.up4 = deconv_block(base * 8, base * 4, dropout=dropout)
        self.dec4 = conv_block(base * 4 + base * 4, base * 4)

        self.up3 = deconv_block(base * 4, base * 2)
        self.dec3 = conv_block(base * 2 + base * 2, base * 2)

        self.up2 = deconv_block(base * 2, base)
        self.dec2 = conv_block(base + base, base)

        self.out = nn.Conv2d(
            base,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bot2(self.bot1(e5))

        d5 = self.up5(b)
        d5 = self.dec5(torch.cat([d5, e4], dim=1))

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.out(d2)
        return self.tanh(out)


class InpaintDiscriminator(nn.Module):
    """
    Discriminador condicional tipo PatchGAN.

    Entrada:
        imagen condicional de 4 canales + imagen completa de 3 canales

    Total:
        7 canales
    """

    def __init__(self, in_ch_cond=4, in_ch_img=3, base=64, dropout=0.0):
        super().__init__()

        in_ch_total = in_ch_cond + in_ch_img

        self.net = nn.Sequential(
            conv_block(in_ch_total, base, k=4, s=2, p=1, norm=False),
            conv_block(base, base * 2, k=4, s=2, p=1),
            conv_block(base * 2, base * 4, k=4, s=2, p=1),
            conv_block(base * 4, base * 8, k=4, s=2, p=1, dropout=dropout),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                base * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, img_cond, img_full):
        x = torch.cat((img_cond, img_full), dim=1)
        return self.net(x)
