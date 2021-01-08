# Reference to https://venoda.hatenablog.com/entry/2020/10/11/221117
# %%
# ライブラリの読み込み
import os
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt
# %%
sho_unicode = "U66F8"
dou_unicode = "U9053"
renamed_data_dir_path = os.path.join(os.path.dirname(__file__), "renamed_data")


def make_filepath_list():
    sho_train_file_list = []
    sho_valid_file_list = []
    dou_train_file_list = []
    dou_valid_file_list = []

    sho_file_list = os.listdir(os.path.join(renamed_data_dir_path, sho_unicode))
    dou_file_list = os.listdir(os.path.join(renamed_data_dir_path, dou_unicode))

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
    return sho_train_file_list, sho_valid_file_list, dou_train_file_list, dou_valid_file_list


# 画像データへのファイルパスを格納したリストを取得する
sho_train_file_list, sho_valid_file_list, dou_train_file_list, dou_valid_file_list = make_filepath_list()

print('「書」学習データ数 : ', len(sho_train_file_list))
# 先頭3件だけ表示
print(sho_train_file_list[:3])

print('「書」検証データ数 : ', len(sho_valid_file_list))
# 先頭3件だけ表示
print(sho_valid_file_list[:3])

print('「道」学習データ数 : ', len(dou_train_file_list))
# 先頭3件だけ表示
print(dou_train_file_list[:3])

print('「道」検証データ数 : ', len(dou_valid_file_list))
# 先頭3件だけ表示
print(dou_valid_file_list[:3])

# %%


class ImageTransform(object):
    """
    入力画像の前処理クラス
    画像のサイズをリサイズする

    Attributes
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャンネルの平均値
    std: (R, G, B)
        各色チャンネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_trasnform = {
            'train': transforms.Compose([
                # データオーグメンテーション
                # transforms.RandomHorizontalFlip(),
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)


# 動作確認
img = Image.open(os.path.join(renamed_data_dir_path, sho_unicode,
                              f"{sho_unicode}_00001.png")).convert("RGB")

# リサイズ先の画像サイズ
resize = 256

# 今回は簡易的に(0.5, 0.5, 0.5)で標準化
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

transform = ImageTransform(resize, mean, std)
img_transformed = transform(img, 'train')

plt.imshow(img)
plt.show()

plt.imshow(img_transformed.numpy().transpose((1, 2, 0)))
plt.show()

# %%


class ShodouDataset(data.Dataset):
    def __init__(self, file_list, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")

        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)

        # 画像ラベルをファイル名から抜き出す
        # label = self.file_list[index].split('/')[2][10:]
        label = self.file_list[index].split("/")[-1].split("_")[0]

        # ラベル名を数値に変換
        label = self.classes.index(label)

        return img_transformed, label


# 動作確認
# クラス名
chr_classes = [
    "U66F8", "U9053"
]

# リサイズ先の画像サイズ
resize = 256

# 今回は簡易的に(0.5, 0.5, 0.5)で標準化
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

# Datasetの作成
sho_train_file_list.extend(dou_train_file_list)
shodou_train_dataset = ShodouDataset(
    file_list=sho_train_file_list, classes=chr_classes,
    transform=ImageTransform(resize, mean, std),
    phase='train'
)
sho_valid_file_list.extend(dou_valid_file_list)
shodou_valid_dataset = ShodouDataset(
    file_list=sho_valid_file_list, classes=chr_classes,
    transform=ImageTransform(resize, mean, std),
    phase='valid'
)

# sho_train_dataset = ShodouDataset(
#     file_list=sho_train_file_list, classes=chr_classes,
#     transform=ImageTransform(resize, mean, std),
#     phase='train', label="sho"
# )

# sho_valid_dataset = ShodouDataset(
#     file_list=sho_valid_file_list, classes=chr_classes,
#     transform=ImageTransform(resize, mean, std),
#     phase='valid', label="sho"
# )

# dou_train_dataset = ShodouDataset(
#     file_list=dou_train_file_list, classes=chr_classes,
#     transform=ImageTransform(resize, mean, std),
#     phase='train', label="dou"
# )

# dou_valid_dataset = ShodouDataset(
#     file_list=dou_valid_file_list, classes=chr_classes,
#     transform=ImageTransform(resize, mean, std),
#     phase='valid', label="dou"
# )


index = 0
print(shodou_train_dataset.__getitem__(index)[0].size())
print(shodou_train_dataset.__getitem__(index)[1])
# print(sho_train_dataset.__getitem__(index)[0].size())
# print(sho_train_dataset.__getitem__(index)[1])

# %%
# バッチサイズの指定
batch_size = 64

# DataLoaderを作成
train_dataloader = data.DataLoader(
    shodou_train_dataset, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
    shodou_valid_dataset, batch_size=32, shuffle=False)

# 辞書にまとめる
dataloaders_dict = {
    'train': train_dataloader,
    'valid': valid_dataloader
}

# 動作確認
# イテレータに変換
batch_iterator = iter(dataloaders_dict['train'])

# 1番目の要素を取り出す
inputs, labels = next(batch_iterator)

print(inputs.size())
print(labels)
# %%


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=128 * 64 * 64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=5)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = x.view(-1, 128 * 64 * 64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


net = Net()
print(net)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# %%
# エポック数
num_epochs = 30

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-------------')

    for phase in ['train', 'valid']:

        if phase == 'train':
            # モデルを訓練モードに設定
            net.train()
        else:
            # モデルを推論モードに設定
            net.eval()
        # 損失和
        epoch_loss = 0.0
        # 正解数
        epoch_corrects = 0

        # DataLoaderからデータをバッチごとに取り出す
        for inputs, labels in dataloaders_dict[phase]:

            # optimizerの初期化
            optimizer.zero_grad()

            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)

                # 損失を計算
                loss = criterion(outputs, labels)

                # ラベルを予測
                _, preds = torch.max(outputs, 1)

                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    # パラメータの更新
                    optimizer.step()

                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)

                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

# %%
print("finish")

# %%
