import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import torch

class GrandEstDataset(Dataset):
    def __init__(self, data_dir, years, patch_size=64):
        self.data_dir = Path(data_dir)
        self.years = years
        self.patch_size = patch_size
        self.samples = self._index_samples()

    def _index_samples(self):
        samples = []
        patch_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        for p_dir in patch_dirs:
            for year in self.years[1:]:
                samples.append({
                    "patch_id": p_dir.name,
                    "target_year": year,
                    "prev_year": year - 1,
                    "path": p_dir
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        p_dir = meta["path"]
        target_year = meta["target_year"]
        prev_year = meta["prev_year"]

        embedding_prev = np.load(p_dir / f"embedding_{prev_year}.npy") # [C, H, W] or pooled [D]
        embedding_change = np.load(p_dir / f"embedding_change_{target_year}_{prev_year}.npy")
        weather = np.load(p_dir / f"weather_{prev_year}.npy")
        label = np.load(p_dir / f"disturbance_{target_year}.npy")

        return {
            "embedding_prev": torch.tensor(embedding_prev, dtype=torch.float32),
            "embedding_change": torch.tensor(embedding_change, dtype=torch.float32),
            "weather": torch.tensor(weather, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

class ForestDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, patch_size: int = 64, years: list = [2022, 2023, 2024]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.years = years

    def setup(self, stage=None):
        # Split logic into train/val/test datasets here based on spatial blocks or random split
        full_dataset = GrandEstDataset(self.data_dir, self.years, self.patch_size)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
