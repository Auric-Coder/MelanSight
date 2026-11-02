# Data Loading Pipeline Documentation

## Overview

The data loading pipeline is the **fourth stage** of the ML workflow. It bridges the gap between your prepared data splits and the model training process by efficiently loading images, applying transforms, and creating batches.

**Purpose:** Transform static CSV files and image files into streaming batches of tensors ready for GPU training.

---

## What is Data Loading?

Data loading is the process of:
1. **Reading** image paths from CSV files
2. **Loading** images from disk (on-demand, not all at once)
3. **Transforming** images (augmentation for training, preprocessing for validation)
4. **Batching** multiple samples together
5. **Streaming** batches to the model during training

### Why This Matters

**Without proper data loading:**
```
❌ Load all 10,000 images into RAM (10GB+ memory)
❌ Process images one by one (slow)
❌ No augmentation (overfitting)
❌ Complex manual batching (error-prone)
```

**With proper data loading:**
```
✅ Load only current batch (16 images = ~50MB)
✅ Parallel loading with multiple workers
✅ Automatic augmentation
✅ Clean batched tensors ready for model
```

---

## Architecture: Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Dataset (Individual Samples)                     │
│  - Reads CSV file                                            │
│  - Returns one (image, label) pair at a time              │
│  - Lazy loading: images loaded on-demand                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ calls __getitem__(idx)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Transforms (Image Processing)                      │
│  - Resize to 224×224                                        │
│  - Augmentation (training only)                            │
│  - Convert to tensor (0-255 → 0-1)                        │
│  - Normalize with ImageNet stats                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ returns processed tensor
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: DataLoader (Batch Creation)                      │
│  - Groups samples into batches                              │
│  - Shuffles (training only)                                │
│  - Parallel loading (num_workers)                          │
│  - Returns: (B, 3, 224, 224), (B,) tensors                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ yields batches
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Training Loop                                        │
│  - Receives batched tensors                                │
│  - Forward pass → Loss → Backward pass                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Dataset (`dataset.py`)

### Purpose

The Dataset class is a PyTorch abstraction that defines:
- **What** a single data sample looks like
- **How** to load it from storage
- **Where** to find the data

### Implementation

```python
class SkinLesionDataset(Dataset):
    """Custom dataset for HAM10000 skin lesion images."""
    
    def __init__(self, csv_path, data_dir='.', transform=None):
        # Load metadata CSV
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Create class mappings
        self.class_to_idx = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
            'mel': 4, 'nv': 5, 'vasc': 6
        }
    
    def __len__(self):
        # Total number of samples
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load ONE sample
        row = self.df.iloc[idx]
        
        # Resolve relative path
        image_path = self.data_dir / row['image_path']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get label
        label = self.class_to_idx[row['dx']]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Key Features

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **Lazy Loading** | Images loaded in `__getitem__` | Memory efficient |
| **Relative Paths** | `data/raw/...` not absolute | Portable across machines |
| **Path Resolution** | `data_dir / image_path` | Works from any directory |
| **Class Mapping** | String → Integer index | Required for PyTorch |
| **Transform Support** | Optional transform parameter | Flexible augmentation |

### Class Mapping

```
String Label    Integer Index    Full Name
─────────────────────────────────────────────────────
akiec          0                Actinic keratoses
bcc            1                Basal cell carcinoma
bkl            2                Benign keratosis-like
df             3                Dermatofibroma
mel            4                Melanoma
nv             5                Melanocytic nevi
vasc           6                Vascular lesions
```

**Why this order?** Alphabetical consistency across all model runs.

---

## Component 2: Transforms (`transforms.py`)

### Purpose

Transforms define how raw PIL images become normalized tensors suitable for neural networks.

### Training Transforms (With Augmentation)

```python
train_transform = transforms.Compose([
    # 1. Resize to standard size
    transforms.Resize((224, 224)),
    
    # 2. Random horizontal flip (50% probability)
    # Why: Lesions can appear on either body side
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 3. Random rotation (±15 degrees)
    # Why: Camera orientation varies
    transforms.RandomRotation(degrees=15),
    
    # 4. Color jitter (brightness/contrast)
    # Why: Lighting conditions vary
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    
    # 5. Convert to tensor [0, 255] → [0.0, 1.0]
    transforms.ToTensor(),
    
    # 6. Normalize with ImageNet statistics
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Validation/Test Transforms (No Augmentation)

```python
val_transform = transforms.Compose([
    # Resize only
    transforms.Resize((224, 224)),
    
    # Convert to tensor
    transforms.ToTensor(),
    
    # Same normalization as training
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Why Different Transforms?

| Aspect | Training | Validation/Test |
|--------|----------|---------------|
| **Randomness** | ✅ Yes | ❌ No |
| **Purpose** | Increase diversity | Consistent evaluation |
| **Augmentations** | Flip, rotate, jitter | None |
| **Same Normalization** | ✅ Required | ✅ Required |

**Critical:** Validation must use same normalization as training!

### ImageNet Normalization

```python
mean = [0.485, 0.456, 0.406]  # Red, Green, Blue channel means
std = [0.229, 0.224, 0.225]   # Red, Green, Blue channel stds
```

**Why ImageNet stats?**
- Our models are pretrained on ImageNet
- They expect inputs normalized the same way
- Using different stats = worse performance

**Result:**
- Input pixels: [0, 255]
- After ToTensor: [0.0, 1.0]
- After Normalize: approximately [-2.0, 2.0]

---

## Component 3: DataLoader (`dataloader.py`)

### Purpose

DataLoader takes individual samples from Dataset and efficiently creates batches for training.

### Implementation

```python
# Training DataLoader
train_loader = DataLoader(
    train_dataset,          # Source dataset
    batch_size=16,          # Samples per batch
    shuffle=True,           # Randomize order each epoch
    num_workers=4,          # Parallel loading threads
    pin_memory=True,        # Faster GPU transfer
    drop_last=False,        # Keep incomplete batches
)

# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,          # Fixed order for evaluation
    num_workers=4,
    pin_memory=True,
)
```

### Key Parameters

| Parameter | Training | Val/Test | Why |
|-----------|----------|----------|-----|
| **batch_size** | 16 or 32 | Same | GPU memory dependent |
| **shuffle** | ✅ Yes | ❌ No | Prevent learning order |
| **num_workers** | 4 (or 0) | 4 (or 0) | Parallel loading |
| **pin_memory** | ✅ Yes | ✅ Yes | Faster GPU transfer |

### Why Shuffle Training Data?

```
Without shuffle:
Epoch 1: [nv, nv, nv, mel, mel...]  # Same order
Epoch 2: [nv, nv, nv, mel, mel...]  # Same order!
→ Model memorizes sequence

With shuffle:
Epoch 1: [mel, nv, bkl, nv, mel...]  # Random
Epoch 2: [nv, mel, nv, bkl, mel...]  # Different!
→ Model learns actual patterns
```

### Parallel Loading with Workers

```python
# Without workers (num_workers=0):
Time: Load img → Process → Load img → Process...
      ↑_________↑ Sequential

# With 4 workers (num_workers=4):
Time: Load imgs 1-4 (parallel) → Process → Load imgs 5-8 (parallel)
      ↑________________________↑ 2-3x faster!
```

**Windows Note:** Use `num_workers=0` to avoid multiprocessing issues.

---

## Data Flow Example

### Step-by-Step Through Pipeline

```python
# 1. Create Dataset
dataset = SkinLesionDataset(
    csv_path='data/splits/train.csv',
    transform=train_transform
)

# 2. Create DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Get one batch
for images, labels in loader:
    print(images.shape)   # (2, 3, 224, 224)
    print(labels.shape)   # (2,)
    break
```

**Inside the batch:**
```
Batch shape: (2, 3, 224, 224)
             │  │  └─── Height
             │  └────── Width
             └───────── Channels (RGB)
             
Labels: tensor([5, 2])  # nv=5, bkl=2
```

**Memory usage:**
- 2 images × 3 channels × 224 × 224 pixels × 4 bytes ≈ **1.2 MB**
- vs. loading all 10,000 images ≈ **6 GB**

---

## Complete Usage Example

### Training Setup

```python
from src.data import get_dataloaders

# Create all loaders
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=Path('.'),
    batch_size=16,
    num_workers=4,  # Use 0 on Windows
)

# Training loop
for epoch in range(10):
    # Training phase
    model.train()
    for images, labels in train_loader:
        # images: (16, 3, 224, 224)
        # labels: (16,)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # Calculate metrics...
```

### Inference Setup

```python
from src.data import SkinLesionDataset, get_val_transforms
from torch.utils.data import DataLoader

# Single image prediction
dataset = SkinLesionDataset(
    csv_path='data/splits/test.csv',
    transform=get_val_transforms()
)
loader = DataLoader(dataset, batch_size=1)

for image, label in loader:
    prediction = model(image)
    break
```

---

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'src'"

**Cause:** Running Python file directly without package context.

**Solutions:**
```bash
# Option 1: Use -m flag
uv run python -m src.data.dataloader

# Option 2: Run from project root
uv run python test_script.py

# Option 3: Add to path in script
import sys
sys.path.insert(0, '.')
from src.data import get_dataloaders
```

### Issue 2: "FileNotFoundError: Image not found"

**Cause:** Relative paths not resolved correctly.

**Fix:**
```python
# Always provide data_dir
dataset = SkinLesionDataset(
    csv_path='data/splits/train.csv',
    data_dir=Path('.')  # Project root
)
```

### Issue 3: "Out of Memory"

**Cause:** Batch size too large for GPU memory.

**Fix:**
```python
# Reduce batch size
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=8,  # Instead of 16 or 32
)
```

### Issue 4: Slow Data Loading

**Cause:** num_workers=0 or slow disk.

**Fix:**
```python
# Linux/Mac: Use 4 workers
train_loader = DataLoader(..., num_workers=4)

# Windows: Use 0 workers (multiprocessing issues)
train_loader = DataLoader(..., num_workers=0)
```

### Issue 5: Validation Metrics Vary Between Runs

**Cause:** Validation shuffle not disabled.

**Fix:**
```python
# Ensure shuffle=False for val/test
val_loader = DataLoader(val_dataset, shuffle=False)
```

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Dataset Creation                                  │
│  - Load CSV with image paths                                │
│  - Resolve relative paths                                   │
│  - Create class mappings                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Transform Selection                               │
│  - Training: Augmentation enabled                           │
│  - Validation: No augmentation                            │
│  - Both: ImageNet normalization                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: DataLoader Configuration                          │
│  - batch_size: 16 or 32                                     │
│  - shuffle: True (train) / False (val/test)               │
│  - num_workers: 4 (Linux/Mac) / 0 (Windows)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Batch Generation                                  │
│  - Load images in parallel                                  │
│  - Apply transforms                                         │
│  - Stack into tensors                                       │
│  - Yield: (B, 3, 224, 224), (B,)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Training Loop                                     │
│  - Feed batches to model                                    │
│  - Compute loss                                             │
│  - Backpropagate                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/data/__init__.py` | ~150 | Package exports and convenience functions |
| `src/data/dataset.py` | ~350 | Dataset class with lazy loading |
| `src/data/transforms.py` | ~250 | Training/validation transforms |
| `src/data/dataloader.py` | ~230 | DataLoader creation with proper settings |

---

## Summary

| Aspect | Our Implementation |
|--------|-------------------|
| **Dataset** | Lazy loading, relative paths, class mapping |
| **Transforms** | Training: augmentation, Val/Test: deterministic |
| **DataLoader** | Batching, shuffling, parallel loading |
| **Normalization** | ImageNet stats for pretrained compatibility |
| **Batch Size** | 16 (default), configurable |
| **Workers** | 4 (Linux/Mac), 0 (Windows) |

---

## Next Steps

After data loading, proceed to **Stage 5: Model Implementation**

Build:
- `src/models/resnet18.py`
- `src/models/vgg16.py`
- `src/models/efficientnet_b0.py`

See: `README.md` for full pipeline

---

## References

- PyTorch Dataset: https://pytorch.org/docs/stable/data.html
- Torchvision Transforms: https://pytorch.org/vision/stable/transforms.html
- DataLoader Best Practices: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Image Normalization: https://pytorch.org/vision/stable/models.html
