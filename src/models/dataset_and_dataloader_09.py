from functools import total_ordering
import math

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
Definitions:

epoch: 1 forward and backward pass of ALL training samples
batch_size: number of trainign samples in one forward and backward pass
number of iterations: number of passes, each using [batch_size] number of samples

Example:
100 samples, batch_size=20, -> 5 iterations for 1 epoch
"""


def old_training_loop():
    """Aim of this lecture: Use Dataset and DataLoader from pytorch.

    So far:
    """
    total_batches = 100
    data = np.loadtxt("wine.csv")
    for epoch in range(1000):
        for i in range(total_batches):
            x_batch, y_batch = None, None
            x_batch, y_batch

    """Now we want to use Dataset and DataLoader from pytorch"""


class WineDataset(Dataset):
    def __init__(self):
        """data loading"""

        xy = np.loadtxt(
            "wine.csv",
            delimiter=",",
            dtype=np.float32,
            skiprows=1,
        )
        self.x = torch.from_numpy(xy[:, 1:])
        # Reason for putting the 0 in a list like[0]:
        # it will make the shape be (n_samples, 1)
        self.y = torch.from_numpy(xy[:, [0]])

        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        """ """
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def main():
    """ """
    batch_size = 4
    dataset = WineDataset()
    first_data = dataset[0]
    features, labels = first_data
    # will print two row vectors
    print(features, labels)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2, # could use loading data faster
    )
    dataiter = iter(dataloader)
    data = dataiter.next()
    features, labels = data
    print(features, labels)

    # training_loop
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / batch_size)
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # foward, backward, update weights would normally go here!
            if (i + 1) % 5 == 0:
                print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, {inputs.shape}")
                # should give sth like this:
                # epoch 1/2, step 5/45, inputs torch.Size([4,13])


    # some built in datasets
    # torchvision.datasets.MNIST
    # torchvision.datasets.FashionMNIST
    # torchvision.datasets.CIFAR100
    # torchvision.datasets.CIFAR10
   #  torchvision.datasets.CocoDetection


if __name__ == "__main__":
    main()
