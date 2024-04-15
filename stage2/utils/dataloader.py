import os
import cv2 as cv
import numpy as np

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Sampler

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

class NSTSampler(Sampler):

    def __init__(self, data_source):
        assert isinstance(data_source, Dataset) or isinstance(data_source, datasets.ImageFolder)
        self.data_source = data_source
        self.subset_size = len(data_source)

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size

def load_data(training_config):

    transform_list = [transforms.Resize(training_config['image_size']),
                      transforms.CenterCrop(training_config['image_size']),
                      transforms.ToTensor()]

    transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = NSTSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)
    return train_loader