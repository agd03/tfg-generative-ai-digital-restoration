import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminador MLP simple:
      - Entrada: imagen completa (B, 3, H, W)
      - Salida: score de real/fake por imagen (B, 1)
    """
    def __init__(self, img_size=128, in_channels=3):
        super().__init__()

        # Número de características tras aplanar la imagen
        # Por ejemplo: 3 * 128 * 128 = 49152
        self.img_size = img_size
        self.in_channels = in_channels
        flat_dim = in_channels * img_size * img_size

        # Red MLP:
        # Flatten -> Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear -> Sigmoid
        self.net = nn.Sequential(
            nn.Flatten(),                # (B, C, H, W) -> (B, C*H*W)

            nn.Linear(flat_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1),
            nn.Sigmoid()                 # salida en (0,1) = prob. de "real"
        )

    def forward(self, x):
        """
        x: tensor de imágenes, shape (B, 3, H, W)
           H y W deben coincidir con img_size (por defecto 128).
        """
        return self.net(x)  # devuelve (B, 1)

def test():
    B, C, H, W = 4, 3, 128, 128

    # Batch de imágenes simuladas
    imgs = torch.randn(B, C, H, W)

    D = Discriminator(img_size=H, in_channels=C)
    out = D(imgs)

    print("Input shape:", imgs.shape)  # (4, 3, 128, 128)
    print("Output shape:", out.shape)  # (4, 1)

    # Comprobación sencilla
    assert out.shape == (B, 1), "La salida del discriminador debería ser (B,1)"

if __name__ == "__main__":
    test()
