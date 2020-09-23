import torch
import torch.nn as nn
from torchvision import datasets, transforms
from skimage.color import rgb2lab, lab2rgb
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import visdom


# модель нейронной сети
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        a = self.layer1(x)
        a = self.layer2(a)
        a = self.layer3(a)
        a = self.layer4(a)
        a = self.layer5(a)
        out = self.layer6(a)
        return out


# предобработка изображений
class GrayColor(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = torch.transpose(self.transform(img), 0, 1)
            img_original = np.asarray(torch.transpose(img_original, 2, 1))
            img_lab = rgb2lab(img_original)
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = img_lab[:, :, 0]
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target  # чб изображение, lab - изображение, target = 0


# соединение слоев и сохранение
def visual(gray, ab, save_path, save_name):
    color_image = torch.cat((gray, ab), 0).cpu().numpy()  # соединить 2 канала
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0] = color_image[:, :, 0] * (max_l.item() - min_l.item()) + mean_l.item()
    color_image[:, :, 1:3] = color_image[:, :, 1:3] - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = gray.squeeze().cpu().numpy()
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


# поиск средних, минимальных и максимальных значений по каналам
def find_values(trainfolder):
    # средние значения по каналам
    mean_l = 0.0
    mean_ab = 0.0
    # максимальные и минмальне значения по каналам
    max_l, max_a, max_b = -500.0, -500.0, -500.0
    min_a, min_b, min_l = 500.0, 500.0, 500.0

    for gray, ab, _ in trainfolder:
        mean_l += gray.mean([1, 2])
        l_max, l_min = torch.max(gray[:, :, :]), torch.min(gray[:, :, :])
        if l_max > max_l:
            max_l = l_max
        if l_min < min_l:
            min_l = l_min
        mean_ab += ab.mean([1, 2])
        a_max, a_min = torch.max(ab[0, :, :]), torch.min(ab[0, :, :])
        if a_max > max_a:
            max_a = a_max
        if a_min < min_a:
            min_a = a_min
        b_max, b_min = torch.max(ab[1, :, :]), torch.min(ab[1, :, :])
        if b_max > max_b:
            max_b = b_max
        if b_min < min_b:
            min_b = b_min
    mean_ab /= len(trainfolder)
    mean_l /= len(trainfolder)

    return max_l, min_l, max_a, min_a, max_b, min_b, mean_l, mean_ab


use_gpu = torch.cuda.is_available()  # использование gpu
device = torch.device('cuda:0' if use_gpu else 'cpu')

num_epochs = 500  # число эпох
batch_size = 3  # размер батча
learning_rate = 0.01  # скорость обучения

# загрузка данных
base_dir_1 = 'C:/Users/MM/Desktop/трен/train/..'
base_dir_2 = 'C:/Users/MM/Desktop/валид/valid/..'
base_dir_3 = 'C:/Users/MM/Desktop/тест/test/..'

# предобработка, формирование батчей
#  тренировочная выборка
trainfolder = GrayColor(base_dir_1, transforms.ToTensor())
train = torch.utils.data.DataLoader(dataset=trainfolder,
                                    batch_size=batch_size,
                                    shuffle=True)  # подача данных
#  валидационая выборка
validfolder = GrayColor(base_dir_2, transforms.ToTensor())
valid = torch.utils.data.DataLoader(dataset=validfolder,
                                   batch_size=batch_size,
                                   shuffle=True)
#  тестовая выборка
testfolder = GrayColor(base_dir_3, transforms.ToTensor())
test = torch.utils.data.DataLoader(dataset=testfolder,
                                   batch_size=batch_size,
                                   shuffle=True)

vis = visdom.Visdom()  # визуализация

max_l, min_l, max_a, min_a, max_b, min_b, mean_l, mean_ab = find_values(trainfolder)  # поиск мин, макс, средних значений для нормализации

net = CNN()
if use_gpu:  # использоване gpu
    net.to(device)

criterion = nn.CrossEntropyLoss().to(device) if use_gpu else nn.CrossEntropyLoss()  # целевая функция
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # weight_decay=0.0)

losses = []  # ошибки
A_list = []
B_list = []
ep = []  # шаги
step_train = 0  # шаг
picture = 0  # номера изображений

# Train
for epoch in range(num_epochs):
    for i, (input_gray, input_ab, target) in enumerate(train):
        # нормализация
        input_gray[:, 0, :, :] = (input_gray[:, 0, :, :] - mean_l) / (max_l - min_l)
        input_ab[:, :, :, :] = input_ab[:, :, :, :] + 128.0

        input_gray_variable = Variable(input_gray).to(device) if use_gpu else Variable(input_gray)

        output_ab = net(input_gray_variable)
        # деление на 2 блока
        output_ab_1 = output_ab[:, :256, :, :]
        output_ab_2 = output_ab[:, 256:, :, :]

        # определение номеров классов
        label = torch.as_tensor(input_ab, dtype=torch.long).to(device)

        # ошибки по a, b
        Loss_1 = criterion(output_ab_1, label[:, 0, :, :])
        Loss_2 = criterion(output_ab_2, label[:, 1, :, :])
        # средняя ошибка
        loss = (Loss_1 + Loss_2) / 2
        # вычисление векторов уверенностей
        softmax = nn.Softmax(dim=1).to(device)
        output_ab_1 = softmax(output_ab_1)
        output_ab_2 = softmax(output_ab_2)
        # вычисление номера класса
        A = Variable(torch.argmax(output_ab_1, dim=1, keepdim=True)).to(device)
        B = Variable(torch.argmax(output_ab_2, dim=1, keepdim=True)).to(device)
        # соединение каналов a, b
        ab = torch.cat((A, B), dim=1)
        ab = Variable(torch.as_tensor(ab, dtype=torch.float)).to(device)
        # оптимизация
        optimizer.zero_grad()
        loss.backward()  # накопление градиентa
        optimizer.step()  # оптимизация

        # сохранение изображений
        if (epoch + 1) % 30 == 0:
            for j in range(len(output_ab)):
                save_path = {'grayscale': 'C:/Users/MM/Desktop/outputs/gray_train_1/',
                             'colorized': 'C:/Users/MM/Desktop/outputs/experiment_train_1/'}
                save_name = 'img-{}.jpg'.format(picture)
                picture += 1
                visual(gray=input_gray_variable[j], ab=ab[j].data, save_path=save_path, save_name=save_name)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}], Loss_TR: {:.5f}'.format(epoch + 1, num_epochs, i + 1, loss.item()))
            losses.append(loss.item())
            ep.append(step_train)
            step_train += 1
            vis.line(X=np.array(ep), Y=np.array(losses), win='train_loss', opts=dict(title='Loss_train'))  # визуализация
            torch.save(net.state_dict(), 'C:/Users/MM/Desktop/outputs/model_train.pth')  # сохранение модели

max_l, min_l, max_a, min_a, max_b, min_b, mean_l, mean_ab = find_values(validfolder)
# точные прогнозы для каналов
correct_a = 0
correct_b = 0
total = 0
losses = []  # ошибки
A_list = []
B_list = []
ep = []  # шаги
step_val = 0  # шаг
picture = 0  # номера изображений
# валидация
for epoch in range(num_epochs):
    for i, (input_gray, input_ab, target) in enumerate(valid):
        # нормализация
        input_gray[:, 0, :, :] = (input_gray[:, 0, :, :] - mean_l) / (max_l - min_l)
        input_ab[:, :, :, :] = input_ab[:, :, :, :] + 128.0

        input_gray_variable = Variable(input_gray).to(device) if use_gpu else Variable(input_gray)

        output_ab = net(input_gray_variable)
        # деление на 2 блока
        output_ab_1 = output_ab[:, :256, :, :]
        output_ab_2 = output_ab[:, 256:, :, :]

        # определение номеров классов
        label = torch.as_tensor(input_ab, dtype=torch.long).to(device)

        # ошибки по a, b
        Loss_1 = criterion(output_ab_1, label[:, 0, :, :])
        Loss_2 = criterion(output_ab_2, label[:, 1, :, :])
        # средняя ошибка
        loss = (Loss_1 + Loss_2) / 2
        # вычисление векторов уверенностей
        softmax = nn.Softmax(dim=1).to(device)
        output_ab_1 = softmax(output_ab_1)
        output_ab_2 = softmax(output_ab_2)
        # вычисление номера класса
        A = Variable(torch.argmax(output_ab_1, dim=1, keepdim=True)).to(device)
        B = Variable(torch.argmax(output_ab_2, dim=1, keepdim=True)).to(device)
        # соединение каналов a, b
        ab = torch.cat((A, B), dim=1)
        ab = Variable(torch.as_tensor(ab, dtype=torch.float)).to(device)
        # оптимизация
        optimizer.zero_grad()
        loss.backward()  # накопление градиентa
        optimizer.step()  # оптимизация

        # сохранение изображений
        if (epoch + 1) % 30 == 0:
            for j in range(len(output_ab)):
                save_path = {'grayscale': 'C:/Users/MM/Desktop/outputs/gray_val/',
                             'colorized': 'C:/Users/MM/Desktop/outputs/color_val/'}
                save_name = 'img-{}.jpg'.format(picture)
                picture += 1
                visual(gray=input_gray_variable[j], ab=ab[j].data, save_path=save_path, save_name=save_name)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}], Loss_TR: {:.5f}'.format(epoch + 1, num_epochs, i + 1, loss.item()))
            losses.append(loss.item())
            ep.append(step_val)
            step_val += 1
            A_list.append(Loss_1.item())
            B_list.append(Loss_2.item())
            vis.line(X=np.array(ep), Y=np.array(losses), win='valid_loss',
                     opts=dict(title='Loss_valid'))  # визуализация
            vis.line(X=np.array(ep), Y=np.array(A_list), win='loss_a', opts=dict(title='valid_Loss_A'))
            vis.line(X=np.array(ep), Y=np.array(B_list), win='loss_b', opts=dict(title='valid_Loss_B'))
            torch.save(net.state_dict(), 'C:/Users/MM/Desktop/outputs/model_valid.pth')  # сохранение модели

        # accuracy для a, b
        for z in range(len(output_ab)):
            a, b = 0, 0
            for r in range(256):
                for j in range(256):
                    a += (output_ab[z, 0, r, j] == input_ab[z, 0, r, j]).sum()
                    b += (output_ab[z, 1, r, j] == input_ab[z, 1, r, j]).sum()
            correct_a += a / (256 * 256)
            correct_b += b / (256 * 256)

        total += target.size(0)
