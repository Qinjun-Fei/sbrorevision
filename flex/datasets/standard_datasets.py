"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split
from flex.common import utils
from flex.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
def emnist(out_dir: str = ".", split="digits", include_authors=False):
    from scipy.io import loadmat

    if split == "digits":
        url, filename, md5 = (
            utils.EMNIST_DIGITS_URL,
            utils.EMNIST_DIGITS_FILE,
            utils.EMNIST_DIGITS_MD5,
        )
    elif split == "letters":
        url, filename, md5 = (
            utils.EMNIST_LETTERS_URL,
            utils.EMNIST_LETTERS_FILE,
            utils.EMNIST_LETTERS_MD5,
        )
    else:
        raise ValueError(
            f"Unknown split: {split}. Available splits are 'digits' and 'letters'."
        )
    mnist_files = utils.download_dataset(
        url, filename, md5, out_dir=out_dir, extract=False, output=True
    )
    dataset = loadmat(mnist_files)["dataset"]
    train_writers = dataset["train"][0, 0]["writers"][0, 0]
    train_data = np.reshape(
        dataset["train"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
    )
    train_labels = np.squeeze(dataset["train"][0, 0]["labels"][0, 0])
    if include_authors:
        train_labels = np.asarray(
            [(label, train_writers[i][0]) for i, label in enumerate(train_labels)],
            dtype=np.int64,
        )

    test_writers = dataset["test"][0, 0]["writers"][0, 0]
    test_data = np.reshape(
        dataset["test"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
    )
    test_labels = np.squeeze(dataset["test"][0, 0]["labels"][0, 0])
    if include_authors:
        test_labels = np.asarray(
            [(label, test_writers[i][0]) for i, label in enumerate(test_labels)],
            dtype=np.int64,
        )
    train_data_object = Dataset.from_array(train_data, train_labels)
    test_data_object = Dataset.from_array(test_data, test_labels)
    return train_data_object, test_data_object

def shakespeare(out_dir: str = ".", include_actors=False):
    import json

    shakespeare_files = utils.download_dataset(
        utils.SHAKESPEARE_URL,
        utils.SHAKESPEARE_FILE,
        utils.SHAKESPEARE_MD5,
        out_dir=out_dir,
        extract=True,
        output=True,
    )
    train_files = filter(
        lambda n: "train" in n and n.endswith(".json"), shakespeare_files
    )
    train_x = []
    train_y = []
    for f in train_files:
        with open(f) as json_file:
            train_data = json.load(json_file)
        for user_id in train_data["users"]:
            node_ds = train_data["user_data"][user_id]
            if include_actors:
                train_y += [(y, user_id) for y in node_ds["y"]]
            else:
                train_y += node_ds["y"]
            train_x += node_ds["x"]
    test_files = filter(
        lambda n: "test" in n and n.endswith(".json"), shakespeare_files
    )
    test_x = []
    test_y = []
    for f in test_files:
        with open(f) as json_file:
            test_data = json.load(json_file)
        for user_id in test_data["users"]:
            node_ds = test_data["user_data"][user_id]
            if include_actors:
                test_y += [(y, user_id) for y in node_ds["y"]]
            else:
                test_y += node_ds["y"]
            test_x += node_ds["x"]

    return Dataset.from_array(train_x, train_y), Dataset.from_array(test_x, test_y)


def emnist_letters(out_dir: str = ".", training_size = 10000, validation_size = 2000, test_size = None, return_train_unused = False):
    from scipy.io import loadmat
    url, filename, md5 = (
        utils.EMNIST_LETTERS_URL,
        utils.EMNIST_LETTERS_FILE,
        utils.EMNIST_LETTERS_MD5,
    )
    mnist_files = utils.download_dataset(
        url, filename, md5, out_dir=out_dir, extract=False, output=True
    )
    dataset = loadmat(mnist_files)["dataset"]
    train_data = np.reshape(
        dataset["train"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
    )
    train_labels = np.squeeze(dataset["train"][0, 0]["labels"][0, 0])
    test_data = np.reshape(
        dataset["test"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
    )
    test_labels = np.squeeze(dataset["test"][0, 0]["labels"][0, 0])
    train_labels = train_labels - 1
    test_labels = test_labels - 1


    from sklearn.utils import shuffle
    train_data, train_labels = shuffle(train_data, train_labels)
    sampled_train_data = train_data[:training_size]
    sampled_train_labels = train_labels[:training_size]


    test_data, test_labels = shuffle(test_data, test_labels)

    val_data, val_labels = test_data[:validation_size], test_labels[:validation_size]

    if test_size is None:
        test_data, test_labels = test_data[validation_size:], test_labels[validation_size:]
    else:
        test_data, test_labels = test_data[validation_size:validation_size + test_size], test_labels[
                                                                                         validation_size:validation_size + test_size]

    # Apply transform to the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    sampled_train_data = np.array([transform(img) for img in sampled_train_data])
    val_data = np.array([transform(img) for img in val_data])
    test_data = np.array([transform(img) for img in test_data])

    train_data_object = Dataset.from_array(sampled_train_data, sampled_train_labels)
    test_data_object = Dataset.from_array(test_data, test_labels)
    val_data_object = Dataset.from_array(val_data, val_labels)

    if return_train_unused:
        unused_train_data = train_data[training_size:]
        unused_train_labels = train_labels[training_size:]
        unused_train_data = np.array([transform(img) for img in unused_train_data])
        unused_train_object = Dataset.from_array(unused_train_data, unused_train_labels)
        return train_data_object, test_data_object, val_data_object, unused_train_object
    else:
        return train_data_object, test_data_object, val_data_object

def cifar10(out_dir: str = ".", training_size = 10000, validation_size = 2000, test_size = None,return_train_unused = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(out_dir, train=True, download=True,transform=transform)
    test_dataset = datasets.CIFAR10(out_dir, train=False, download=True,transform=transform)

    if training_size is not None:
        train_dataset, unused_dataset = random_split(train_dataset, [training_size, len(train_dataset) - training_size])
    if test_size is None:
        val_dataset, test_dataset = random_split(test_dataset, [validation_size, len(test_dataset) - validation_size])
    else:
        val_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
    train_data_object = Dataset.from_torchvision_dataset(train_dataset)
    test_data_object = Dataset.from_torchvision_dataset(test_dataset)
    val_data_object = Dataset.from_torchvision_dataset(val_dataset)
    if return_train_unused:
        unused_data_object = Dataset.from_torchvision_dataset(unused_dataset)
        return train_data_object, test_data_object, val_data_object, unused_data_object
    else:
        return train_data_object, test_data_object, val_data_object

def svhn(out_dir: str = ".", training_size = 10000, validation_size = 2000, test_size = None, return_train_unused = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.SVHN(out_dir, split="train", download=True, transform=transform)
    test_dataset = datasets.SVHN(out_dir, split="test", download=True, transform=transform)
    if training_size is not None:
        train_dataset, unused_dataset = random_split(train_dataset, [training_size, len(train_dataset) - training_size])
    if test_size is None:
        val_dataset, test_dataset = random_split(test_dataset, [validation_size, len(test_dataset) - validation_size])
    else:
        val_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
    train_data_object = Dataset.from_torchvision_dataset(train_dataset)
    test_data_object = Dataset.from_torchvision_dataset(test_dataset)
    val_data_object = Dataset.from_torchvision_dataset(val_dataset)
    if return_train_unused:
        unused_data_object = Dataset.from_torchvision_dataset(unused_dataset)
        return train_data_object, test_data_object, val_data_object, unused_data_object
    else:
        return train_data_object, test_data_object, val_data_object

def fashionmnist(out_dir: str = ".", training_size = 10000, validation_size = 2000, test_size = None, return_train_unused = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(out_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(out_dir, train=False, download=True, transform=transform)
    if training_size is not None:
        train_dataset, unused_dataset = random_split(train_dataset, [training_size, len(train_dataset) - training_size])
    if test_size is None:
        val_dataset, test_dataset = random_split(test_dataset, [validation_size, len(test_dataset) - validation_size])
    else:
        val_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
    train_data_object = Dataset.from_torchvision_dataset(train_dataset)
    test_data_object = Dataset.from_torchvision_dataset(test_dataset)
    val_data_object = Dataset.from_torchvision_dataset(val_dataset)
    if return_train_unused:
        unused_data_object = Dataset.from_torchvision_dataset(unused_dataset)
        return train_data_object, test_data_object, val_data_object, unused_data_object
    else:
        return train_data_object, test_data_object, val_data_object
