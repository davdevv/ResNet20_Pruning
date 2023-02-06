import abc
import torch
import torchvision
import torchvision.transforms as transforms
from .augmentation import cutout


class Cifar(abc.ABC):
    def __init__(self, batch_size, name, autoaugment, use_cutout):

        self.batch_size = batch_size
        self._train_transform = [transforms.RandomCrop(size=(32, 32), padding=4)]

        if autoaugment:
            self._train_transform.append(transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10))

        self._image_mean = (0.49139968, 0.48215841, 0.44653091)
        self._image_std = (0.24703223, 0.24348513, 0.26158784)
        self.num_classes = None
        self._train_transform += [transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                  transforms.Normalize(self._image_mean, self._image_std)]
        self._batch_augmentation = None
        if use_cutout:
            self._batch_augmentation = cutout

        self._train_transform = transforms.Compose(self._train_transform)
        self._test_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(self._image_mean, self._image_std)])


class Cifar10(Cifar):
    def __init__(self, batch_size, autoaugment, use_cutout, threads=2):
        super().__init__(batch_size, 'cifar10', autoaugment, use_cutout)
        self.num_classes = 10
        self._dataset_name = 'cifar10'

        train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=self._train_transform)
        test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=self._test_transform)
        self.train_size = len(train)
        self.test_size = len(test)

        self.train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                 num_workers=threads)
        self.test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                                num_workers=threads)
