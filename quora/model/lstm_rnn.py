# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('../..'))
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from quora import config
from quora.feature.feature import LSTMFeature


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.params = config.model_params['LSTM_VEC']
        self.lstm = nn.LSTM(self.params['input_size'], self.params['hidden_size'],
                            dropout=self.params['dropout'], num_layers=self.params['num_layers'],
                            batch_first=True)
        if self.params['init_weight']:
            nn.init.xavier_normal_(self.lstm.all_weights[0][0])
            nn.init.xavier_normal_(self.lstm.all_weights[0][1])

    def forward(self, x):
        out, _ = self.lstm(x)
        data, index = pad_packed_sequence(out, batch_first=True)
        x_vector = [data[i, j-1] for i, j in enumerate(index.numpy().tolist())]
        return x_vector


class LSTM(object):

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
    def _pack(cls, x, index):
        x = sorted(zip(index, x), key=lambda x: x[0])
        x = [ix[1] for ix in x]
        res = torch.stack(x, dim=0)
        return res

    @classmethod
    def _forward(cls, x1_index, x1, x2_index, x2):
        x1_vector = cls.net(x1)
        x1_tensor = cls._pack(x1_vector, x1_index)
        x2_vector = cls.net(x2)
        x2_tensor = cls._pack(x2_vector, x2_index)
        sims = F.cosine_similarity(x1_tensor, x2_tensor)
        outputs = torch.stack((1 - sims, sims), dim=1)
        return outputs

    @classmethod
    def _loss(cls, x1_index, x1, x2_index, x2, labels):
        cls.net.train()
        cls.optimizer.zero_grad()
        outputs = cls._forward(x1_index, x1, x2_index, x2)
        loss = cls.criterion(outputs, labels)
        loss.backward()
        cls.optimizer.step()
        cls.running_loss += loss.item()

    @classmethod
    def _accuracy(cls, x1_index, x1, x2_index, x2, labels, mode='train'):
        with torch.no_grad():
            cls.net.eval()
            outputs = cls._forward(x1_index, x1, x2_index, x2)
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
        loader = LSTMFeature()
        for epoch in range(1, cls.net.params['epoch']+1):
            cls.scheduler.step()
            for x1_index, x1, x2_index, x2, labels in loader:
                if torch.cuda.is_available():
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    labels = labels.cuda()

                if cls.count != 0:
                    cls._accuracy(x1_index, x1, x2_index, x2, labels)
                cls._loss(x1_index, x1, x2_index, x2, labels)
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
        loader = LSTMFeature(mode='test')
        predicted = []
        for x1_index, x1, x2_index, x2, labels in loader:
            if torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
                labels = labels.cuda()
            predicted = predicted + cls._accuracy(x1_index, x1, x2_index, x2, labels, mode='predict')
        submission = pd.read_csv(config.origin_submission_file)
        submission['is_duplicate'] = predicted
        submission = submission[['test_id', 'is_duplicate']].set_index('test_id')
        submission.to_csv(config.root_path.format('lstm_vec.csv'))

    @classmethod
    def _save(cls, epoch, acc):
        torch.save(cls.net.state_dict(), config.model_path.format('lstm_vec_{}_{}.model'.format(epoch, acc)))
        torch.save(cls.tmp, config.model_path.format('lstm_vec_tmp_{}_{}.model'.format(epoch, acc)))

    @classmethod
    def _load(cls, name='lstm_vec.model'):
        cls.net.load_state_dict(torch.load(config.model_path.format(name)))

    @classmethod
    def show(cls, name='tmp_vec.model'):
        import matplotlib.pyplot as plt
        data = torch.load(config.model_path.format(name))
        loss = [i[0] for i in data][::5]
        acc = [i[1] for i in data][::5]
        x = [i for i in range(1, len(data)+1)][::5]
        plt.figure()
        plt.plot(x, loss, color='red')
        plt.plot(x, acc, color='blue')
        plt.show()


if __name__ == '__main__':
    LSTM.train()
