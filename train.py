from src.data.dataset import SkinLesionDataset
dataset = SkinLesionDataset('data/splits/train.csv')
image, label = dataset[0]
print(image.shape)  # (3, 224, 224)
print(label)        # 0 (akiec)