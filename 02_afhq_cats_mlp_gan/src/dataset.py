import os
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AFHQCatsDataset(Dataset):
    def __init__(self, split="train", size=128, root="data/afhq_cats"):
        super().__init__()
        self.root = root
        os.makedirs(root, exist_ok=True)

        # Si ya está guardado en disco, usamos las imágenes locales
        local_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".jpg", ".png"))]
        if len(local_files) > 0:
            print(f"📂 Cargando {len(local_files)} imágenes locales desde {root}")
            self.images = [Image.open(f).convert("RGB") for f in local_files]
        else:
            print("⬇️ Descargando dataset 'bitmind/AFHQ' de Hugging Face...")
            data = load_dataset("bitmind/AFHQ", split=split)

            # Filtrar solo los gatos y guardar en disco
            cats = [x["image"] for x in data if "cat" in x["filename"].lower()]
            print(f"🐱 Guardando {len(cats)} imágenes de gatos en {root}...")
            for i, img in enumerate(cats):
                img = img.convert("RGB") if not isinstance(img, Image.Image) else img
                img.save(os.path.join(root, f"cat_{i:05d}.jpg"))
            self.images = cats

        # Transformaciones estándar
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img)

# --- Test rápido ---
if __name__ == "__main__":
    ds = AFHQCatsDataset(size=128)
    print(f"Total imágenes de gatos: {len(ds)}")
    print("Tamaño de un tensor:", ds[0].shape)

