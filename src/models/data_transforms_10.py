import torch
from torch.utils.data import dataset
import torchvision
from torchvision import transforms
import numpy as np


def example():
    # Example Transform
    dataset = torchvision.datasets.MNIST(
        root="./data", transform=torchvision.transforms.ToTensor()
    )


class WineDataset(dataset.Dataset):
    """WineDataset that will be enhanced by a transform."""

    def __init__(self, transform=None):
        """data loading"""

        xy = np.loadtxt(
            "wine.csv",
            delimiter=",",
            dtype=np.float32,
            skiprows=1,
        )
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        # Reason for putting the 0 in a list like[0]:
        # it will make the shape be (n_samples, 1)
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        """ """

        sample = self.x[index], self.y[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


def main():
    dataset = WineDataset(transform=ToTensor())
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features )

    composed = torchvision.transforms.Compose(
        [
            ToTensor(),
            MulTransform(10),
        ]
    )

    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    print(features)
    print(type(features), type(labels))

if __name__ == "__main__":
    main()
