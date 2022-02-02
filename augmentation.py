import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor


class BaseAugmentation:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


class Augmentation:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size = 224):
        self.transform = A.Compose([
            A.Resize(height = size, width = size, p = 1),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.RandomRotate90(p = 0.5),

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit = 0.2,
                                     sat_shift_limit = 0.2,
                                     val_shift_limit = 0.2,
                                     p = 0.3),
                A.RandomBrightnessContrast(brightness_limit = 0.2,
                                           contrast_limit = 0.1,
                                           p = 0.3)
            ], p = 0.2),

            A.OneOf([
                A.Blur(p = 0.5),
                A.GaussianBlur(p = 0.5),
                A.Sharpen(p = 0.5)
            ], p = 0.3),

            A.Normalize(mean = mean, std = std),
            ToTensorV2()
        ])