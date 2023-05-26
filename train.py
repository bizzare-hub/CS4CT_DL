from pathlib import Path

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models import RecModel, UNetSEResNext50
from src.dataset import TomographyDataset
from src.trainer import Trainer

from segmentation_models_pytorch import Unet


device = "cuda:1" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    data_dir = Path("/home/orogov/smbmount/a_galichin/Skoltech_HW/BioMed")

    patients = sorted((data_dir / "train").iterdir())
    patients = list(map(lambda p: p.name, patients))

    n_total = len(patients)
    n_val_patients = 15
    n_test_patients = 15
    n_train_patients = n_total - n_val_patients - n_test_patients

    indices = np.cumsum([n_train_patients, n_val_patients, n_test_patients])

    np.random.shuffle(patients)

    train_patients = patients[:indices[0]]
    val_patients = patients[indices[0]:indices[1]]
    test_patients = patients[indices[1]:indices[2]]

    img_size = 320
    transforms = [A.Resize(img_size, img_size), ToTensorV2()]

    train_dataset = TomographyDataset(img_size, data_dir, train_patients, transforms)
    val_dataset = TomographyDataset(img_size, data_dir, val_patients, transforms)
    test_dataset = TomographyDataset(img_size, data_dir, test_patients, transforms)

    # Define model
    model = RecModel(
        model_cls=UNetSEResNext50,
        proj_kwargs={
            "resolution": img_size,
            "n_projections": 320,
            "source_distance": 980.,
            "det_distance": 980.,
            "det_spacing": 2.58
        },
        subsample_factor=4
    )
    print(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    trainer = Trainer(
        model=model,
        loader_config={
            "batch_size": 8,
            "num_workers": 8
        },
        ckpt_dir="v2_checkpoints",
        device=device
    )

    trainer.train_dataset = train_dataset
    trainer.val_dataset = val_dataset

    history = trainer.fit(n_epochs=2)

    with open("v2_history.pkl", "wb") as handle:
        pickle.dump(history, handle)

    # test

    model = trainer._model
    model.eval()

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
    for batch in test_dataloader:
        inputs, segmaps = batch
        inputs, segmaps = inputs.to(device), segmaps.to(device)
        segmaps = segmaps[:, None]

        with torch.no_grad():
            outputs, sinograms, rec_sinograms = model(inputs)
        
        psnr.update(outputs, inputs)
        ssim.update(outputs, inputs)
    
    print(f"PSNR: {psnr.compute().item():.4f}; SSIM: {ssim.compute().item():.4f}")
