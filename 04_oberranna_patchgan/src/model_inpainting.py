import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Implemento cambios de ChatGPT para adaptar UNET a inpainting
#   - Entrada: img_masked (3 canales) + mask (1 canal) → 4 canales de entrada.
#   - Salida: imagen completa reconstruida → 3 canales.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # Same convolution (h & w se mantienen)
            nn.BatchNorm2d(out_channels),   # aquí aumentamos (o modificamos) la cantidad de features
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # Same convolution (h & w se mantienen)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels=4, out_channels=3, features=[64, 128, 256, 512] # utilizamos mismos tamaños (o num. de features) que en el paper
        ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET   -   utilizamos transposed convolutions para upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2    # doble de features porque añadimos la skip-connection (estoy concatenando los features del downsampling)

                )
            )
            self.ups.append(DoubleConv(feature*2, feature)) # up->2 convs -> up->2 convs -> ...

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)  # in=512, out=1024

        # Final conv (1x1) - no cambia h & w de la imagen
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []   # guardaremos aquí las skip-connections

        for down in self.downs:
            x = down(x) # aplicamos las double convs
            skip_connections.append(x)  # guardamos el resultado para la skip-connection
            x = self.pool(x)    # aplicamos el maxpooling (reducción de h & w a la mitad)

        x = self.bottleneck(x)  # pasamos por el bottleneck
        skip_connections = skip_connections[::-1] # le damos la vuelta para utilizar las skip-connections desde la última (la de menor resolución)

        for idx in range(0, len(self.ups), 2): # step=2 porque cada par de layers es: upsample + double conv (las consideramos como 1 bloque)
            x = self.ups[idx](x)    # aplicamos la transposed convolution (upsampling)
            skip_connection = skip_connections[idx//2] # idx//2 por el step=2 (para acceder a todas las skip-connection en orden)

            if x.shape != skip_connection.shape:    # en caso de que las dimensiones no coincidan (puede pasar por redondeos al hacer down/up-sampling)
                x = TF.resize(x, size=skip_connection.shape[2:])    # no utilizamos el batch size ni el num. de channels, solo h & w

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenamos el upsampling con la skip-connection de su misma resolución (=w&h, doble de features)
            x = self.ups[idx+1](concat_skip)    # aplicamos la double conv a la concatenación

        return self.final_conv(x)   # pasamos por la última conv (1x1) para reducir a out_channels el número de features
    

def test():
    # Parámetros del test
    B, H, W = 2, 160, 160   # batch size, altura, anchura

    # 1) Imagen real simulada (RGB)
    real_img = torch.randn((B, 3, H, W))

    # 2) Máscara binaria simulada (1 = visible, 0 = tapado)
    mask = (torch.rand((B, 1, H, W)) > 0.5).float()

    # 3) Imagen enmascarada
    img_masked = real_img * mask

    # 4) Entrada al modelo: imagen enmascarada + máscara
    x = torch.cat([img_masked, mask], dim=1)   # (B, 4, H, W)

    # 5) Modelo de inpainting: 4 canales de entrada, 3 de salida
    model = UNET(in_channels=4, out_channels=3)

    # 6) Forward
    preds = model(x)

    print("Shape entrada x:     ", x.shape)
    print("Shape imagen real:   ", real_img.shape)
    print("Shape salida modelo: ", preds.shape)

    # 7) Comprobación: la salida debe tener la misma shape que la imagen real
    assert preds.shape == real_img.shape, "La salida NO coincide con la imagen real"

if __name__ == "__main__":
    test()
