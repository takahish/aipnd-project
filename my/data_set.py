import torch
from torchvision import datasets, transforms, models


class Transform(object):
    """Transform class.
    """
    def __init__(self):
        """Initialize Transform
        """
        self.__train_transform = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.__test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def train_transform(self):
        """Getter for train_transform.
        """
        return self.__train_transform

    @property
    def test_transform(self):
        """Getter for test_transform.
        """
        return self.__test_transform


class DataSet(object):
    """DataSet class.
    """
    def __init__(self, data_directory, transform, batch_size=32):
        """Initialize DataSet.
        """
        train_dir = data_directory + '/train'
        valid_dir = data_directory + '/valid'
        test_dir = data_directory + '/test'

        self.__train_data_set = datasets.ImageFolder(root=train_dir, transform=transform.train_transform)
        self.__valid_data_set = datasets.ImageFolder(root=valid_dir, transform=transform.test_transform)
        self.__test_data_set = datasets.ImageFolder(root=test_dir, transform=transform.test_transform)

        self.__train_dataloader = torch.utils.data.DataLoader(self.__train_data_set, batch_size=batch_size, shuffle=True)
        self.__valid_dataloader = torch.utils.data.DataLoader(self.__valid_data_set, batch_size=batch_size, shuffle=True)
        self.__test_dataloader = torch.utils.data.DataLoader(self.__test_data_set, batch_size=batch_size, shuffle=True)

    @property
    def train_data_set(self):
        """Getter for train_data_set.
        """
        return self.__train_data_set

    @property
    def valid_data_set(self):
        """Getter for valid_data_set.
        """
        return self.__valid_data_set

    @property
    def test_data_set(self):
        """Getter for test_data_set.
        """
        return self.__test_data_set

    @property
    def train_dataloader(self):
        """Getter for train_dataloader.
        """
        return self.__train_dataloader

    @property
    def valid_dataloader(self):
        """Getter for valid_dataloader.
        """
        return self.__valid_dataloader

    @property
    def test_dataloader(self):
        """Getter for test_dataloader.
        """
        return self.__test_dataloader
