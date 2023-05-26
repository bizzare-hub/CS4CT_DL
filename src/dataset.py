from typing import List, Callable
from pathlib import Path

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import albumentations as A

from src.utils import read_dicom


class TomographyDataset(Dataset):
    def __init__(
        self,
        img_size: int,
        main_dir: str,
        dirs: List[str],
        transforms: List[Callable]
    ):
        self.img_size = img_size
        self.main_dir = Path(main_dir)
        self.dirs = dirs
        self.pipeline = A.Compose(transforms)
        
        self.load_annotations()
    
    def __len__(self):
        return len(self._img_paths)
    
    def load_annotations(self) -> None:
        def load_images(dir_):
            dir_path = self.main_dir / "train" / dir_
            return dir_path.glob("*.dcm")
        
        self._img_paths = []
        for dir_name in self.dirs:
            self._img_paths.extend(load_images(dir_name))
        self._seg_paths = [
            str(p.with_suffix('.png')).replace('train', 'train_seg') 
            for p in self._img_paths
        ]

        bad_indices = []
        for i, sp in enumerate(self._seg_paths):
            if not Path(sp).exists():
                bad_indices.append(i)
            else:
                s = np.array(Image.open(sp)) / 255.0
                if (s.shape[0] != 512) or (np.sum(s) < s.size * 0.1):
                    bad_indices.append(i)

        self._img_paths = [p for p in self._img_paths if p not in bad_indices]
        self._seg_paths = [p for p in self._seg_paths if p not in bad_indices]

        print(f"Removed {len(bad_indices)} samples. Total: {len(self._img_paths)}")
    
    def __getitem__(self, idx):
        img_path = self._img_paths[idx]
        seg_path = self._seg_paths[idx]

        try:
            image = read_dicom(img_path, normalize=True)
        except:
            image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        segmap = np.array(Image.open(seg_path))
        segmap = (segmap / 255.0).astype(np.float32)

        data = self.pipeline(image=image, mask=segmap)
        image, mask = data["image"], data["mask"]

        return image, mask
