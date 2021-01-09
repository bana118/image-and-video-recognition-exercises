# %%
from __future__ import print_function

import os
import csv
from PIL import Image

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time


# %%

class ShodouDataset(torch.utils.data.Dataset):
    def __init__(self, img_file_list, time_dict, transform=None):
        self.img_file_list = img_file_list
        self.transform = transform
        self.time_dict = time_dict

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.img_file_list)

    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.img_file_list[index]
        img = Image.open(img_path).convert("RGB")

        # 画像の前処理を実施
        img_transformed = self.transform(img)

        # 画像ラベルをファイル名から抜き出す
        # label = self.file_list[index].split('/')[2][10:]
        img_id = self.img_file_list[index].split("/")[-1].split(".")[0]

        # ラベル名を数値に変換
        write_time = self.time_dict[img_id]

        return img_transformed, write_time
# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7938, 128)
        self.fc2 = nn.Linear(128, 10)

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
        output = F.log_softmax(x, dim=1)
        return output

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
    sho_num_data = len(sho_file_list)
    sho_num_split = int(sho_num_data * 0.8)
    dou_num_data = len(dou_file_list)
    dou_num_split = int(dou_num_data * 0.8)

    sho_train_file_list += [os.path.join(renamed_data_dir_path, sho_unicode,
                                         file_name).replace("\\", "/") for file_name in sho_file_list[:sho_num_split]]
    sho_valid_file_list += [os.path.join(renamed_data_dir_path, sho_unicode,
                                         file_name).replace("\\", "/") for file_name in sho_file_list[sho_num_split:]]
    dou_train_file_list += [os.path.join(renamed_data_dir_path, dou_unicode,
                                         file_name).replace("\\", "/") for file_name in dou_file_list[:dou_num_split]]
    dou_valid_file_list += [os.path.join(renamed_data_dir_path, dou_unicode,
                                         file_name).replace("\\", "/") for file_name in dou_file_list[dou_num_split:]]

    # TODO 場合分け
    return sho_train_file_list, sho_valid_file_list


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
        loss = F.mse_loss(output, target)
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
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))


# %%
def main():
    train_target = "sho"
    test_target = "sho"

    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_filelist, test_filelist = create_file_list(
        train_target=train_target, test_target=test_target)

    time_dict = create_time_dict()

    train_dataset = ShodouDataset(
        img_file_list=train_filelist, time_dict=time_dict, transform=transform)

    test_dataset = ShodouDataset(
        img_file_list=test_filelist, time_dict=time_dict, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 15):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "cnn_reg.pt")


if __name__ == '__main__':
    main()
# %%
