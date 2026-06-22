import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (tipo pix2pix):
      Conv64 -> Conv128 -> Conv256 -> Conv512 -> Conv(1)

    - kernel_size=4
    - stride=2 en las primeras capas
    - stride=1 en las capas finales (típico) para obtener una rejilla de decisiones por parches.
    - Normalización: InstanceNorm2d (recomendado en GANs) o BatchNorm2d.

    Devuelve logits (SIN sigmoid): (B, 1, h, w)
    """

    def __init__(
        self,
        in_channels: int = 3,
        norm: str = "instance",  # "instance" | "batch" | "none"
        features=(64, 128, 256, 512),
    ):
        super().__init__()

        def Norm(c: int):
            if norm == "instance":
                return nn.InstanceNorm2d(c, affine=True)
            if norm == "batch":
                return nn.BatchNorm2d(c)
            if norm in ("none", None):
                return nn.Identity()
            raise ValueError(f"norm desconocida: {norm}")

        layers = []

        # Conv64 (sin norm, como en pix2pix/DCGAN)
        layers += [
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Conv128, Conv256 (stride=2)
        in_f = features[0]
        for out_f in features[1:3]:
            layers += [
                nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False),
                Norm(out_f),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_f = out_f

        # Conv512 (stride=1 para "parches" más locales, estilo 70x70)
        layers += [
            nn.Conv2d(in_f, features[3], kernel_size=4, stride=1, padding=1, bias=False),
            Norm(features[3]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_f = features[3]

        # Conv(1) final: logits por parche
        layers += [
            nn.Conv2d(in_f, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,1,h,w) logits


if __name__ == "__main__":
    # Test rápido de formas
    B, C, H, W = 4, 3, 128, 128
    x = torch.randn(B, C, H, W)

    D = PatchGANDiscriminator(in_channels=C, norm="instance")
    y = D(x)

    print("x:", x.shape)
    print("D(x):", y.shape)   # esperado: (B, 1, h, w), p.ej. (4,1,14,14) o similar
