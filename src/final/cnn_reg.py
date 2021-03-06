# %%
from __future__ import print_function

import os
import csv
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models


# %%

class ShodouDataset(torch.utils.data.Dataset):
    def __init__(self, img_file_list, time_dict, is_train, transform=None, augment_transform_list=[]):
        self.img_file_list = img_file_list
        self.transform = transform
        self.time_dict = time_dict
        self.is_train = is_train
        self.augment_transform_list = augment_transform_list

    def __len__(self):
        """
        画像の枚数を返す
        """
        if self.is_train:
            inflate = len(self.augment_transform_list) + 1
            return inflate * len(self.img_file_list)
        else:
            return len(self.img_file_list)

    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        if self.is_train:
            inflate = len(self.augment_transform_list) + 1
            i = index // inflate
            res = index % inflate
            img_path = self.img_file_list[i]
            img = Image.open(img_path).convert("RGB")

            img_transformed = self.augment_transform_list[res](img) if res < len(
                self.augment_transform_list) else self.transform(img)

            img_id = img_path.split("/")[-1].split(".")[0]
            write_time = self.time_dict[img_id]

            return img_transformed, write_time
        else:
            # 指定したindexの画像を読み込む
            img_path = self.img_file_list[index]
            img = Image.open(img_path).convert("RGB")

            # 画像の前処理を実施
            img_transformed = self.transform(img)

            img_id = img_path.split("/")[-1].split(".")[0]

            write_time = self.time_dict[img_id]

            return img_transformed, write_time


# %%

# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7938, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# %%


def create_file_list(train_target, test_target):
    sho_unicode = "U66F8"
    dou_unicode = "U9053"

    renamed_data_dir_path = os.path.join(
        os.path.dirname(__file__), "renamed_data")

    sho_train_file_list = []
    sho_valid_file_list = []
    dou_train_file_list = []
    dou_valid_file_list = []

    sho_file_list = os.listdir(os.path.join(
        renamed_data_dir_path, sho_unicode))
    dou_file_list = os.listdir(os.path.join(
        renamed_data_dir_path, dou_unicode))

    # TODO ランダム化
    sho_num_data = 20
    sho_num_split = int(sho_num_data * 0.8)
    dou_num_data = 20
    dou_num_split = int(dou_num_data * 0.8)

    sho_train_file_list += [os.path.join(renamed_data_dir_path, sho_unicode,
                                         file_name).replace("\\", "/") for file_name in sho_file_list[:sho_num_split]]
    sho_train_file_list += [os.path.join(renamed_data_dir_path, sho_unicode, file_name).replace("\\", "/")
                            for file_name in sho_file_list[sho_num_data:sho_num_data + sho_num_split]]
    sho_valid_file_list += [os.path.join(renamed_data_dir_path, sho_unicode, file_name).replace("\\", "/")
                            for file_name in sho_file_list[sho_num_split:sho_num_data]]
    sho_valid_file_list += [os.path.join(renamed_data_dir_path, sho_unicode, file_name).replace("\\", "/")
                            for file_name in sho_file_list[sho_num_data + sho_num_split:]]

    dou_train_file_list += [os.path.join(renamed_data_dir_path, dou_unicode,
                                         file_name).replace("\\", "/") for file_name in dou_file_list[:dou_num_split]]
    dou_train_file_list += [os.path.join(renamed_data_dir_path, dou_unicode, file_name).replace("\\", "/")
                            for file_name in dou_file_list[dou_num_data:dou_num_data + dou_num_split]]
    dou_valid_file_list += [os.path.join(renamed_data_dir_path, dou_unicode, file_name).replace("\\", "/")
                            for file_name in dou_file_list[dou_num_split:dou_num_data]]
    dou_valid_file_list += [os.path.join(renamed_data_dir_path, dou_unicode, file_name).replace("\\", "/")
                            for file_name in dou_file_list[dou_num_data + dou_num_split:]]

    shodou_train_file_list = sho_train_file_list + dou_train_file_list
    shodou_valid_file_list = sho_valid_file_list + dou_valid_file_list

    # TODO 場合分け
    if train_target == "sho":
        if test_target == "sho":
            return sho_train_file_list, sho_valid_file_list
        elif test_target == "dou":
            return sho_train_file_list, dou_valid_file_list
        else:
            return sho_train_file_list, shodou_valid_file_list
    elif train_target == "dou":
        if test_target == "sho":
            return dou_train_file_list, sho_valid_file_list
        elif test_target == "dou":
            return dou_train_file_list, dou_valid_file_list
        else:
            return dou_train_file_list, shodou_valid_file_list
    else:
        if test_target == "sho":
            return shodou_train_file_list, sho_valid_file_list
        elif test_target == "dou":
            return shodou_train_file_list, dou_valid_file_list
        else:
            return shodou_train_file_list, shodou_valid_file_list
# %%


def create_time_dict():
    sho_csv_dir_path = os.path.join(
        os.path.dirname(__file__), "renamed_data", "csv", "U66F8.csv")
    dou_csv_dir_path = os.path.join(
        os.path.dirname(__file__), "renamed_data", "csv", "U9053.csv")
    time_dict = {}
    with open(sho_csv_dir_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            time_dict[row[0]] = int(row[1])
    with open(dou_csv_dir_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            time_dict[row[0]] = int(row[1])

    return time_dict
# %%


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# %%

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    mean_error = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output, target.float().view(-1, 1))
            # sum up batch loss
            test_loss += F.mse_loss(output, target.float().view(-1, 1), reduction='sum').item()
            mean_error += abs(output - target.float().view(-1, 1))
            # get the index of the max log-probability
            # pred = output.argmax(keepdim=True)
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    mean_error /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}. Mean Error: {:.4f})\n'.format(test_loss, float(mean_error)))
    return mean_error

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

    # model = Net().to(device)

    # Alexnet
    # model = models.alexnet().to(device)
    # print(model)
    # num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_features, 1).to(device)

    # Resnet18
    # model = models.resnet18().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Resnet34
    # model = models.resnet34().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Resnet50
    # model = models.resnet50().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Resnet101
    # model = models.resnet101().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Resnet152
    # model = models.resnet152().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Googlenet
    # model = models.googlenet(aux_logits=False).to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Inception_v3
    model = models.inception_v3(aux_logits=False).to(device)
    print(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1).to(device)

    # VGG 11
    # model = models.vgg11().to(device)
    # print(model)
    # num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_features, 1).to(device)

    # VGG 11 with batch normalization
    # model = models.vgg11_bn().to(device)
    # print(model)
    # num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_features, 1).to(device)

    # Densenet-121
    # model = models.densenet121().to(device)
    # print(model)
    # num_features = model.classifier.in_features
    # model.classifier = nn.Linear(num_features, 1).to(device)

    # Shufflenet V2
    # model = models.shufflenet_v2_x1_0().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Mobilenet v2
    # model = models.mobilenet_v2().to(device)
    # print(model)
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_features, 1).to(device)

    # ResNeXt-100-32x8d
    # model = models.resnext101_32x8d().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # Wide Resnet-50-2
    # model = models.wide_resnet50_2().to(device)
    # print(model)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 1).to(device)

    # MNASNet 0.5
    # model = models.mnasnet0_5().to(device)
    # print(model)
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_features, 1).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    mean_error_list = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        mean_error = test(model, device, test_loader)
        mean_error_list.append(mean_error)
        t = np.arange(1, epoch + 1)
        plt.xlim(1, epochs)
        plt.ylim(0, 5000)
        plt.plot(t, mean_error_list)
        plt.pause(0.1)
        plt.clf()
        scheduler.step()
    print(f"Best MAE: {float(min(mean_error_list))}")

    # torch.save(model.state_dict(), "cnn_reg.pt")


if __name__ == '__main__':
    main()
# %%
