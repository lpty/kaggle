# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('../..'))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from quora import config
from quora.feature.feature import MatchPyramidFeature


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.params = config.model_params['MatchPyramid']
        self.conv1_1 = nn.Conv2d(*self.params['conv1_1'])
        self.bn1_1 = nn.BatchNorm2d(self.params['conv1_1'][1])
        self.conv1_2 = nn.Conv2d(*self.params['conv1_2'])
        self.bn1_2 = nn.BatchNorm2d(self.params['conv1_2'][1])
        self.pool1 = nn.AdaptiveMaxPool2d(*self.params['pool1'])

        self.conv2 = nn.Conv2d(*self.params['conv2'])
        self.bn2 = nn.BatchNorm2d(self.params['conv2'][1])
        self.pool2 = nn.MaxPool2d(*self.params['pool2'])

        self.mlp3 = nn.Linear(*self.params['mlp3'])
        self.bn3 = nn.BatchNorm1d(self.params['mlp3'][1])

        self.mlp4 = nn.Linear(*self.params['mlp4'])

        if self.params['init_weight']:
            nn.init.xavier_normal_(self.conv1_1.weight.data)
            nn.init.xavier_normal_(self.conv1_2.weight.data)
            nn.init.xavier_normal_(self.conv2.weight.data)

    def forward(self, x):
        # layer 1
        conv1_1 = self.conv1_1(x)
        bn1_1 = self.bn1_1(conv1_1)
        act1_1 = F.relu(bn1_1)

        conv1_2 = self.conv1_2(act1_1)
        bn1_2 = self.bn1_2(conv1_2)
        act1_2 = F.relu(bn1_2)
        pool1 = self.pool1(act1_2)

        # layer 2
        conv2 = self.conv2(pool1)
        bn2 = self.bn2(conv2)
        act2 = F.relu(bn2)
        pool2 = self.pool2(act2)

        # layer 3
        reshape = pool2.view(-1, self.params['mlp3'][0])
        mlp3 = self.mlp3(reshape)
        bn3 = self.bn3(mlp3)
        act3 = F.relu(bn3)

        # layer 4
        mlp4 = self.mlp4(act3)
        act4 = F.softmax(mlp4, dim=1)
        return act4


class MatchPyramid(object):

    @classmethod
    def _init(cls):
        cls.net = Model()
        if torch.cuda.is_available():
            cls.net = cls.net.cuda()

        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.net.parameters(), lr=cls.net.params['lr'][0], momentum=cls.net.params['momentum'])
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=cls.net.params['auto_lr_epoch'],
                                                  gamma=cls.net.params['lr_gamma'])
        cls.running_loss, cls.correct, cls.total, cls.count = 0, 0, 0, 0
        cls.tmp = []

    @classmethod
    def _loss(cls, inputs, labels):
        cls.net.train()
        cls.optimizer.zero_grad()
        outputs = cls.net(inputs)
        loss = cls.criterion(outputs, labels)
        loss.backward()
        cls.optimizer.step()
        cls.running_loss += loss.item()

    @classmethod
    def _accuracy(cls, inputs, labels, mode='train'):
        with torch.no_grad():
            cls.net.eval()
            outputs = cls.net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            proba = outputs.cpu().data.numpy()[:, 1].tolist()
            if mode == 'train':
                cls.total += labels.size(0)
                cls.correct += (predicted == labels).sum().item()
            return proba

    @classmethod
    def _summary(cls, epoch):
        loss = round((cls.running_loss / cls.net.params['summary_count']), 4)
        accuracy = round((cls.correct / cls.total), 4)
        print('Epoch: {}, Count: {}, Loss/PerCount: {}, Total: {}, Accuracy {}'.format(
            epoch, cls.count, loss, cls.total, accuracy))
        cls.tmp.append((loss, accuracy))
        cls.running_loss = 0.0

    @classmethod
    def train(cls):
        cls._init()
        loader = MatchPyramidFeature()
        for epoch in range(1, cls.net.params['epoch']+1):
            cls.scheduler.step()
            for inputs, labels in loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                if cls.count != 0:
                    cls._accuracy(inputs, labels)
                cls._loss(inputs, labels)
                if cls.count % cls.net.params['summary_count'] == 0 and cls.count != 0:
                    cls._summary(epoch)
                cls.count += 1
            cls.running_loss, cls.correct, cls.total = 0, 0, 0
            if epoch % cls.net.params['save_epoch'] == 0:
                cls._save(str(epoch), str(cls.tmp[-1][1]))
        cls._save(str(cls.net.params['epoch']), str(cls.tmp[-1][1]))

    @classmethod
    def predict(cls):
        cls._init()
        cls._load()
        loader = MatchPyramidFeature(mode='test')
        predicted = []
        for inputs, labels in loader:
            predicted = predicted + cls._accuracy(inputs, labels, mode='predict')
        submission = pd.read_csv(config.origin_submission_file)
        submission['is_duplicate'] = predicted
        submission = submission[['test_id', 'is_duplicate']].set_index('test_id')
        submission.to_csv(config.root_path.format('match_pyramid.csv'))

    @classmethod
    def _save(cls, epoch, acc):
        torch.save(cls.net.state_dict(), config.model_path.format('match_pyramid_{}_{}.model'.format(epoch, acc)))
        torch.save(cls.tmp, config.model_path.format('mp_tmp_{}_{}.model'.format(epoch, acc)))

    @classmethod
    def _load(cls, name='match_pyramid.model'):
        cls.net.load_state_dict(torch.load(config.model_path.format(name)))

    @classmethod
    def show(cls, name='tmp.model'):
        import matplotlib.pyplot as plt
        data = torch.load(config.model_path.format(name))
        count = 0
        for d in data:
            count += 50
            print('Count: {}, Loss: {}, Acc: {}'.format(count, d[0], d[1]))
        loss = [i[0] for i in data][::5]
        acc = [i[1] for i in data][::5]
        x = [i for i in range(1, len(data)+1)][::5]
        plt.figure()
        plt.plot(x, loss, color='red')
        plt.plot(x, acc, color='blue')
        plt.show()


if __name__ == '__main__':
    MatchPyramid.train()
