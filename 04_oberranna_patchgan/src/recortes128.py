#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from PIL import Image

# En principio no es necesaria
def png_to_rgb(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """
    Convierte una imagen a RGB. Si tiene canal alfa, lo compone sobre un fondo (por defecto blanco),
    para poder guardar en JPG sin problemas.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, bg + (255,))
        composed = Image.alpha_composite(background, rgba)
        return composed.convert("RGB")
    return img.convert("RGB")
##

def tile_image(img: Image.Image, crop_size: int, stride: int, drop_incomplete: bool):
    """
    Genera recortes (left, top, patch) recorriendo en rejilla.
    Si drop_incomplete=True, descarta bordes que no completen crop_size.
    Si drop_incomplete=False, los bordes se rellenan (pad) antes de recortar.
    """
    w, h = img.size

    if drop_incomplete:
        max_x = (w - crop_size)
        max_y = (h - crop_size)
        xs = range(0, max_x + 1, stride) if max_x >= 0 else []
        ys = range(0, max_y + 1, stride) if max_y >= 0 else []
        for top in ys:
            for left in xs:
                yield left, top, img.crop((left, top, left + crop_size, top + crop_size))
    else:
        # Pad a la mínima dimensión necesaria para cubrir rejilla completa
        import math
        out_w = max(crop_size, int(math.ceil((w - crop_size) / stride) * stride + crop_size)) if w >= crop_size else crop_size
        out_h = max(crop_size, int(math.ceil((h - crop_size) / stride) * stride + crop_size)) if h >= crop_size else crop_size
        if (out_w, out_h) != (w, h):
            padded = Image.new("RGB", (out_w, out_h), (255, 255, 255))
            padded.paste(img, (0, 0))
            img = padded
            w, h = img.size

        xs = range(0, w - crop_size + 1, stride)
        ys = range(0, h - crop_size + 1, stride)
        for top in ys:
            for left in xs:
                yield left, top, img.crop((left, top, left + crop_size, top + crop_size))

def process_folder(input_dir: Path, output_dir: Path, crop_size: int, stride: int,
                   quality: int, drop_incomplete: bool):
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".PNG"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not images:
        raise SystemExit(f"No encontré PNGs en: {input_dir}")

    for img_path in images:
        with Image.open(img_path) as im:
            im = png_to_rgb(im)

            base = img_path.stem
            idx = 0
            for left, top, patch in tile_image(im, crop_size, stride, drop_incomplete):
                out_name = f"{base}_x{left}_y{top}_{idx:06d}.jpg"
                out_path = output_dir / out_name
                patch.save(
                    out_path,
                    format="JPEG",
                    quality=quality,
                    subsampling=0,
                    optimize=True
                )
                idx += 1

        print(f"{img_path.name}: {idx} recortes guardados")

def main():
    ap = argparse.ArgumentParser(
        description="Genera recortes (tiles) de tamaño fijo a partir de PNGs y los guarda en JPG."
    )
    ap.add_argument("--input_dir", required=True, help="Carpeta con PNGs 1024x1024")
    ap.add_argument("--output_dir", required=True, help="Carpeta de salida para JPGs")
    ap.add_argument("--crop_size", type=int, default=128, help="Tamaño del recorte (default: 128)")
    ap.add_argument("--stride", type=int, default=128, help="Paso entre recortes (default: 128, no solapado)")
    ap.add_argument("--quality", type=int, default=95, help="Calidad JPG (default: 95)")
    ap.add_argument(
        "--keep_incomplete",
        action="store_true",
        help="Si se activa, NO descarta bordes: rellena (pad) y también genera recortes en bordes."
    )

    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"No existe input_dir: {input_dir}")

    drop_incomplete = not args.keep_incomplete

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        crop_size=args.crop_size,
        stride=args.stride,
        quality=args.quality,
        drop_incomplete=drop_incomplete
    )

if __name__ == "__main__":
    main()
