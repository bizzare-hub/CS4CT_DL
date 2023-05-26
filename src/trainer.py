from typing import Dict
from pathlib import Path

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchmetrics import (
    MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)

from tqdm import tqdm


class Trainer:
    
    _train_dataset: Dataset = None
    _val_dataset: Dataset = None

    def __init__(
        self, 
        model: nn.Module,
        loader_config: Dict, 
        log_every_n_steps: int = 10,
        ckpt_dir: str = '',
        device: str = "cpu"
    ):
        self._model = model.to(device)

        self._loader_config = loader_config
        self._log_every_n_steps = log_every_n_steps

        self._ckpt_dir = Path(ckpt_dir)
        self._epoch: int = 0

        self.device = device

        self._compile()
    
    @property
    def train_dataset(self):
        return self._train_dataset
  
    @property
    def val_dataset(self):
        return self._val_dataset
    
    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset

    @val_dataset.setter
    def val_dataset(self, dataset):
        self._val_dataset = dataset
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, **self._loader_config)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=False, **self._loader_config)          

    def _compile(self) -> None:
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=1e-4, weight_decay=1e-3)
        self._scheduler = None

        self._mse_loss = nn.MSELoss()
        self._masked_l1_loss = nn.L1Loss()
        
        metrics = MetricCollection([
            PeakSignalNoiseRatio(), StructuralSimilarityIndexMeasure()
        ]).to(self.device)

        self._train_metrics = metrics.clone()
        self._val_metrics = metrics.clone(prefix='val_')

    def _save_checkpoint(self, epoch) -> None:
        checkpoint = {}

        checkpoint["state_dict"] = self._model.state_dict()
        checkpoint["optimizer"] = self._optimizer.state_dict()
        checkpoint["epoch"] = epoch

        self._ckpt_dir.mkdir(parents=False, exist_ok=True)

        torch.save(checkpoint, self._ckpt_dir / f"model_{epoch}.pt")
    
    @staticmethod
    def _update_metrics(metric, out, tgt) -> None:
        if metric is not None:
            metric.update(out, tgt)
        
    @staticmethod
    def _compute_metrics(metric, result = {}):
        if metric is not None:
            result.update({key: round(val.item(), 4) for key, val in metric.compute().items()})
        
        return result
    
    def training_step(self, batch, batch_idx):
        self._optimizer.zero_grad()

        inputs, segmaps = batch
        inputs, segmaps = inputs.to(self.device), segmaps.to(self.device)
        segmaps = segmaps[:, None]

        outputs, sinograms, rec_sinograms = self._model(inputs)
        
        masked_inputs = inputs[segmaps.to(bool)]
        masked_outputs = outputs[segmaps.to(bool)]

        sinogram_loss = self._mse_loss(rec_sinograms, sinograms)
        image_loss = self._masked_l1_loss(masked_outputs, masked_inputs)
        
        loss = 0.0 * sinogram_loss + image_loss
        loss.backward()

        self._optimizer.step()

        self._update_metrics(self._train_metrics, outputs, inputs)

        return loss
    
    def training_epoch(self, dataloader, epoch, history) -> None:
        self._model.train()

        loader = tqdm(
            enumerate(dataloader), desc=f"Epoch {epoch}",
            total=len(dataloader), leave=False
        )
        
        loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        for batch_idx, batch in loader:
            batch_loss = self.training_step(batch, batch_idx)

            loss += batch_loss.detach()
        
            if (batch_idx + 1) % self._log_every_n_steps == 0:
                global_loss = loss / batch_idx
                
                results = {"loss": global_loss.item()}
                self._compute_metrics(self._train_metrics, results)

                for key, val in results.items():
                    history[key].append(val)
                
                loader.set_postfix(results)
            loader.update(1)
        loader.close()

    def validation_step(self, batch, batch_idx):
        inputs, segmaps = batch
        inputs, segmaps = inputs.to(self.device), segmaps.to(self.device)
        segmaps = segmaps[:, None]

        with torch.no_grad():
            outputs, sinograms, rec_sinograms = self._model(inputs)

        masked_inputs = inputs[segmaps.to(bool)]
        masked_outputs = outputs[segmaps.to(bool)]

        sinogram_loss = self._mse_loss(rec_sinograms, sinograms)
        image_loss = self._masked_l1_loss(masked_outputs, masked_inputs)
        
        loss = 0.0 * sinogram_loss + image_loss

        self._update_metrics(self._val_metrics, outputs, inputs)
        
        return loss
    
    def validation_epoch(self, dataloader, epoch, history) -> None:
        self._model.eval()

        loader = tqdm(
            enumerate(dataloader), desc=f"Validation",
            total=len(dataloader), leave=False
        )

        loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        for batch_idx, batch in loader:
            batch_loss = self.validation_step(batch, batch_idx)

            loss += batch_loss.detach()

            loader.update(1)

        global_loss = loss / len(dataloader)

        if self._scheduler is not None:
            self._scheduler.step(global_loss.item())
        
        results = {"val_loss": global_loss.item()}
        self._compute_metrics(self._val_metrics, results)

        msg = "Validation metrics: "
        for key, val in results.items():
            history[key].append(val)
            msg += f"{key}: {val:.4f} "
        print(msg)

        loader.close()
    
    def fit(self, n_epochs: int = 1):
        history = defaultdict(list)

        train_dataloader = self.train_dataloader()
        val_dataloader = self.val_dataloader()

        for epoch in range(self._epoch, n_epochs):
            self._train_metrics.reset()
            self._val_metrics.reset()

            self.training_epoch(train_dataloader, epoch, history)
            self.validation_epoch(val_dataloader, epoch, history)

            self._save_checkpoint(epoch)
    
        return history
