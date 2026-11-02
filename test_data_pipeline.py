# test_data_pipeline.py (at project root)
from src.data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

for images, labels in train_loader:
    print(f"✓ Batch loaded: {images.shape}")
    break