import albumentations as A

def augmentation(nillformat):
    transform = A.Compose([
        A.RandomRotate90(),
        A.RandomCrop(120, 120),
        A.HorizontalFlip(p=0.5),
        A.RandomContrast(p=0.5),
        A.RandomBrightnessContrast(p=0.5)
    ])