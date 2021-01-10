# %%
from __future__ import print_function

from cnn_reg import Net, ShodouDataset, create_file_list, create_time_dict, train, test

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models

# %%


def main():
    train_target = "sho"
    test_target = "sho"
    epochs = 50

    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(args.seed)
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

    augment_transform_list = [rand_deg_transform] * 5
    # augment_transform_list.append(horizontal_flip_transform)
    # augment_transform_list.append(vertical_flip_transform)

    train_filelist, test_filelist = create_file_list(
        train_target=train_target, test_target=test_target)

    time_dict = create_time_dict()

    train_dataset = ShodouDataset(img_file_list=train_filelist, time_dict=time_dict,
                                  transform=transform, is_train=True, augment_transform_list=augment_transform_list)

    test_dataset = ShodouDataset(img_file_list=test_filelist, time_dict=time_dict,
                                 transform=transform, is_train=False, augment_transform_list=augment_transform_list)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model_list = []

    # Normal CNN
    model = Net().to(device)
    model_list.append({"name": "normal_cnn", "model": model})

    # Alexnet
    model = models.alexnet().to(device)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "alexnet", "model": model})

    # Resnet18
    model = models.resnet18().to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "resnet18", "model": model})

    # Resnet34
    model = models.resnet34().to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "resnet34", "model": model})

    # Resnet50
    model = models.resnet50().to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "resnet50", "model": model})

    # Resnet101
    model = models.resnet101().to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "resnet101", "model": model})

    # Resnet101
    model = models.resnet152().to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "resnet152", "model": model})

    # Googlenet
    model = models.googlenet(aux_logits=False).to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "googlenet", "model": model})

    # Inception_v3
    model = models.inception_v3(aux_logits=False).to(device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)
    model_list.append({"name": "inception_v3", "model": model})

    for model_and_name in model_list:
        print(f"Model: {model_and_name['name']}")
        optimizer = optim.Adadelta(model_and_name["model"].parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        mean_error_list = []
        for epoch in range(1, epochs + 1):
            train(model_and_name["model"], device, train_loader, optimizer, epoch)
            mean_error = test(model_and_name["model"], device, test_loader)
            mean_error_list.append(float(mean_error.float()))
            scheduler.step()
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        t = np.arange(1, epoch + 1)
        plt.xlim(1, epochs)
        plt.ylim(0, 5000)
        plt.xlabel("Epochs")
        plt.ylabel("MAE (ms)")
        plt.plot(t, mean_error_list)
        plt.title(f"{model_and_name['name']} mean abs error")
        plt.savefig(f"{output_dir}/{model_and_name['name']}.pdf")
        plt.clf()
        with open(f"{output_dir}/{model_and_name['name']}.log", "w") as f:
            f.write(f"Best MAE: {min(mean_error_list)}")
        torch.save(model.state_dict(), f"{output_dir}/{model_and_name['name']}.pt")


if __name__ == '__main__':
    main()
