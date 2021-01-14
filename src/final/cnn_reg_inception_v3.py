# %%
from __future__ import print_function

from cnn_reg import ShodouDataset, create_file_list, create_time_dict, train, test

from cnn_reg_search import get_model

import os

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

# %%


def main():
    epochs = 50

    torch.manual_seed(1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 4}
    test_kwargs = {'batch_size': 1}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    rand_deg_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=20)
    ])

    augment_transform_list = [rand_deg_transform] * 4

    time_dict = create_time_dict()

    train_list = ["sho", "dou", "shodou"]
    test_list = ["sho", "dou", "shodou"]
    model_name = "inception_v3"
    model = get_model(model_name, device)

    for train_target, test_target in product(train_list, test_list):
        print(f"Model: {model_name}")
        print(f"train: {train_target}")
        print(f"test: {test_target}")

        train_filelist, test_filelist = create_file_list(
            train_target=train_target, test_target=test_target)
        train_dataset = ShodouDataset(img_file_list=train_filelist, time_dict=time_dict,
                                      transform=transform, is_train=True, augment_transform_list=augment_transform_list)

        test_dataset = ShodouDataset(img_file_list=test_filelist, time_dict=time_dict,
                                     transform=transform, is_train=False, augment_transform_list=augment_transform_list)

        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        mean_error_list = []
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            mean_error = test(model, device, test_loader)
            mean_error_list.append(float(mean_error.float()))
            scheduler.step()
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        t = np.arange(1, epoch + 1)
        plt.xlim(1, epochs)
        plt.ylim(0, 10000)
        plt.xlabel("Epochs")
        plt.ylabel("MAE (ms)")
        plt.plot(t, mean_error_list)
        plt.title(f"Model: {model_name} Train:{train_target} Test:{test_target}")
        plt.savefig(f"{output_dir}/{model_name}_{train_target}_{test_target}.pdf")
        plt.clf()
        best_mae = min(mean_error_list)
        with open(f"{output_dir}/{model_name}_{train_target}_{test_target}.log", "w") as f:
            f.write(f"Best MAE: {best_mae}")
        torch.save(model.state_dict(), f"{model_name}_{train_target}_{test_target}.pt")


if __name__ == '__main__':
    main()
