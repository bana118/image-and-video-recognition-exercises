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


def get_model(name, device):
    if name == "normal_cnn":
        model = Net().to(device)
        return model
    elif name == "alexnet":
        model = models.alexnet().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg11":
        model = models.vgg11().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg13":
        model = models.vgg13().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg16":
        model = models.vgg16().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg19":
        model = models.vgg19().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg11_bn":
        model = models.vgg11_bn().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg13_bn":
        model = models.vgg13_bn().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg16_bn":
        model = models.vgg16_bn().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "vgg19_bn":
        model = models.vgg19_bn().to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "densenet121":
        model = models.densenet121().to(device)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "densenet161":
        model = models.densenet161().to(device)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "densenet169":
        model = models.densenet169().to(device)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "densenet201":
        model = models.densenet201().to(device)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnet18":
        model = models.resnet18().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnet34":
        model = models.resnet34().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnet50":
        model = models.resnet50().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnet101":
        model = models.resnet101().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnet152":
        model = models.resnet152().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "googlenet":
        model = models.googlenet(aux_logits=False).to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "inception_v3":
        model = models.inception_v3(aux_logits=False).to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "shufflenet_v2_x1_5":
        model = models.shufflenet_v2_x1_5().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "shufflenet_v2_x2_0":
        model = models.shufflenet_v2_x2_0().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2().to(device)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnext50_32x4d":
        model = models.resnext50_32x4d().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "resnext101_32x8d":
        model = models.resnext101_32x8d().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "wide_resnet50_2":
        model = models.wide_resnet50_2().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "wide_resnet101_2":
        model = models.wide_resnet101_2().to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "mnasnet0_5":
        model = models.mnasnet0_5().to(device)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "mnasnet0_75":
        model = models.mnasnet0_75().to(device)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "mnasnet1_0":
        model = models.mnasnet1_0().to(device)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1).to(device)
        return model
    elif name == "mnasnet1_3":
        model = models.mnasnet1_3().to(device)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1).to(device)
        return model

# %%


def main():
    train_target = "sho"
    test_target = "sho"
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

    train_filelist, test_filelist = create_file_list(
        train_target=train_target, test_target=test_target)

    time_dict = create_time_dict()

    train_dataset = ShodouDataset(img_file_list=train_filelist, time_dict=time_dict,
                                  transform=transform, is_train=True, augment_transform_list=augment_transform_list)

    test_dataset = ShodouDataset(img_file_list=test_filelist, time_dict=time_dict,
                                 transform=transform, is_train=False, augment_transform_list=augment_transform_list)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model_name_list = ["normal_cnn", "alexnet", "vgg11", "vgg13", "vgg16", "vgg19",
                       "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                       "densenet121", "densenet161", "densenet169", "densenet201",
                       "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                       "googlenet", "inception_v3",
                       "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
                       "mobilenet_v2", "resnext50_32x4d", "resnext101_32x8d",
                       "wide_resnet50_2", "wide_resnet101_2",
                       "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]

    best_mae_list = []
    for model_name in model_name_list:
        print(f"Model: {model_name}")
        model = get_model(model_name, device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        mean_error_list = []
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            mean_error = test(model, device, test_loader)
            mean_error_list.append(float(mean_error.float()))
            scheduler.step()
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        t = np.arange(1, epoch + 1)
        plt.xlim(1, epochs)
        plt.ylim(0, 5000)
        plt.xlabel("Epochs")
        plt.ylabel("MAE (ms)")
        plt.plot(t, mean_error_list)
        plt.title(f"{model_name} mean abs error")
        plt.savefig(f"{output_dir}/{model_name}.pdf")
        plt.clf()
        best_mae_list.append(min(mean_error_list))
        with open(f"{output_dir}/{model_name}.log", "w") as f:
            f.write(f"Best MAE: {min(mean_error_list)}")
        torch.save(model.state_dict(), f"{output_dir}/{model_name}.pt")
    best_model = model_name_list[best_mae_list.index(min(best_mae_list))]
    with open(f"{output_dir}/00best_model.log", "w") as f:
        f.write(f"Best Model: {best_model}, MAE: {min(best_mae_list)}")


if __name__ == '__main__':
    main()
