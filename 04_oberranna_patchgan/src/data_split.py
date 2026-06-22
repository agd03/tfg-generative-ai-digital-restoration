import os
import json
import glob
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Type
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_image_paths(root: str, exts: Tuple[str, ...] = IMG_EXTS) -> List[str]:
    """
    Devuelve lista ordenada de rutas a imágenes en root.
    Busca extensiones típicas y ordena para determinismo.
    """
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
    paths = sorted(set(paths))
    if len(paths) == 0:
        raise RuntimeError(f"No se encontraron imágenes en {root}")
    return paths


def _validate_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    s = train_frac + val_frac + test_frac
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Las fracciones deben sumar 1.0 (actual: {s})")
    if min(train_frac, val_frac, test_frac) < 0:
        raise ValueError("Las fracciones no pueden ser negativas")

@dataclass
class Split:
    seed: int
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    paths: Optional[List[str]] = None  # opcional: guardar lista completa para auditoría

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "paths": self.paths,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Split":
        return Split(
            seed=int(d["seed"]),
            train_idx=list(map(int, d["train_idx"])),
            val_idx=list(map(int, d["val_idx"])),
            test_idx=list(map(int, d["test_idx"])),
            paths=d.get("paths", None),
        )
    
@dataclass
class DataConfig:
    data_root: str
    batch_size: int = 16

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Split reproducible
    split_seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_filename: str = "split.json"
    save_paths_for_audit: bool = True

    # Dataset params (kwargs)
    dataset_kwargs_train: Dict[str, Any] = field(default_factory=dict)
    dataset_kwargs_val:   Dict[str, Any] = field(default_factory=dict)
    dataset_kwargs_test:  Dict[str, Any] = field(default_factory=dict)

def create_split_indices(
    n: int,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Crea split determinista de índices [0..n-1].
    """
    _validate_fracs(train_frac, val_frac, test_frac)

    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)

    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)
    n_test  = n - n_train - n_val

    # Garantizar mínimo 1 en val/test si su frac > 0 y hay datos suficientes
    if val_frac > 0 and n_val == 0:
        n_val = 1
    if test_frac > 0 and n_test == 0:
        n_test = 1

    # Reajustar train para que la suma sea n
    n_train = n - n_val - n_test

    # Último sanity: si n es demasiado pequeño para cumplir todo, falla explícitamente
    if n_train <= 0:
        raise ValueError(
            f"Dataset demasiado pequeño (n={n}) para crear train/val/test no vacíos "
            f"con fracs train={train_frac}, val={val_frac}, test={test_frac}."
        )

    train_idx = idxs[:n_train].tolist()
    val_idx = idxs[n_train:n_train + n_val].tolist()
    test_idx = idxs[n_train + n_val:].tolist()

    # sanity
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(test_idx) == n_test

    return train_idx, val_idx, test_idx


def load_or_create_split(
    paths: List[str],
    run_dir: str,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    filename: str = "split.json",
    save_paths_for_audit: bool = True,
) -> Split:
    """
    Si existe run_dir/split.json lo carga (reproducibilidad).
    Si no existe, lo crea, lo guarda y lo devuelve.
    """
    os.makedirs(run_dir, exist_ok=True)
    split_path = os.path.join(run_dir, filename)

    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            d: Dict[str, Any] = json.load(f)
        split = Split.from_dict(d)

        # Auditoría básica: comprobar tamaño si guardamos paths
        if split.paths is not None:
            if len(split.paths) != len(paths):
                raise RuntimeError(
                    "El número de imágenes ha cambiado desde que se creó el split. "
                    f"Antes: {len(split.paths)}, ahora: {len(paths)}. "
                    "Esto rompe reproducibilidad."
                )
            # Opcional: comprobar igualdad exacta (más estricto)
            if split.paths != paths:
                raise RuntimeError("La lista de archivos ha cambiado desde que se creó el split.")
        return split

    # crear split nuevo
    train_idx, val_idx, test_idx = create_split_indices(
        n=len(paths),
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    split = Split(
        seed=seed,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        paths=paths if save_paths_for_audit else None,
    )

    with open(split_path, "w") as f:
        json.dump(split.to_dict(), f, indent=2)

    return split


def make_dataloaders(
    cfg: DataConfig,
    run_dir: str,
    dataset_cls: Type[Dataset],
) -> Tuple[DataLoader, DataLoader, DataLoader, Split]:
    """
    Construye train/val/test DataLoaders de forma reproducible:
    - Lista imágenes
    - Carga o crea split.json en run_dir
    - Crea datasets (train con augment, eval sin augment típicamente)
    - Envuelve en Subset usando los índices del split
    - Devuelve DataLoaders + objeto Split
    """
    _validate_fracs(cfg.train_frac, cfg.val_frac, cfg.test_frac)

    paths = list_image_paths(cfg.data_root)

    split = load_or_create_split(
        paths=paths,
        run_dir=run_dir,
        seed=cfg.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        filename=cfg.split_filename,
        save_paths_for_audit=cfg.save_paths_for_audit,
    )

    # Datasets base (mismo root)
    dataset_train = dataset_cls(root=cfg.data_root, **cfg.dataset_kwargs_train)
    dataset_val  = dataset_cls(root=cfg.data_root, **cfg.dataset_kwargs_val)
    dataset_test  = dataset_cls(root=cfg.data_root, **cfg.dataset_kwargs_test)

    # Subsets
    train_set = Subset(dataset_train, split.train_idx)
    val_set   = Subset(dataset_val,  split.val_idx)
    test_set  = Subset(dataset_test,  split.test_idx)

    pw = cfg.persistent_workers if cfg.num_workers > 0 else False

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=pw
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=pw
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=pw
    )

    return train_loader, val_loader, test_loader, split
