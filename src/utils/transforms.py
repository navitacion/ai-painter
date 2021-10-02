from abc import ABCMeta
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class BaseTransform(metaclass=ABCMeta):
    def __init__(self):
        self.transform = None

    def __call__(self, img, phase='train'):
        transformed = self.transform[phase](image=img)['image']

        return transformed


class ImageTransform(BaseTransform):
    def __init__(self, cfg):
        super(ImageTransform, self).__init__()

        try:
            transform_train_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_train).items()]
            transform_train_list.append(ToTensorV2())
        except:
            transform_train_list = [ToTensorV2()]

        try:
            transform_test_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_test).items()]
            transform_test_list.append(ToTensorV2())
        except:
            transform_test_list = [ToTensorV2()]

        self.transform = {
            'train': albu.Compose(transform_train_list, p=1.0),
            'test': albu.Compose(transform_test_list, p=1.0),
        }