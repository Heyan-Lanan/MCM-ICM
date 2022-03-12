from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
import torch.optim as optim

class DataReader(object):
    def __init__(self, fpath, replace_missing=True):
        begin_date = datetime(2016, 9, 11)
        end_date = datetime(2021, 9, 10)
        length = (end_date - begin_date).days + 1
        validations = np.zeros(length, dtype=np.bool)
        prices = np.zeros(length)
        self.min = float('inf')
        self.max = float('-inf')
        with open(fpath, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx >= 1:
                    line = line.rstrip().split(',')
                    if len(line) >= 2 and len(line[1]) > 0:
                        price = float(line[1])
                        date = [int(i) for i in line[0].split('/')]
                        date = datetime(date[2]+2000, date[0], date[1])
                        idx = (date - begin_date).days
                        validations[idx] = True
                        prices[idx] = price
                        if price > self.max:
                            self.max = price
                        if price < self.min:
                            self.min = price
        
        if replace_missing:
            last_price = None
            for i in range(length):
                if validations[i]:
                    last_price = prices[i]
                elif last_price is not None:
                    prices[i] = last_price
        
        self.length = length
        self.validations = validations
        self.prices = prices
    
    def show_prices(self):
        plt.plot(self.prices[np.where(self.validations)[0][0]:])
        plt.show()

class LSTM(nn.Module):
    def __init__(self, hidden_size=100, output_size=1, use_gpu = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_gpu = use_gpu

        self.lstm = nn.LSTM(1, self.hidden_size)
        self.linear = nn.Linear(
            self.hidden_size,
            self.output_size
            )
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))
    
    def forward(self, x):
        self.clear_hidden_cells()
        res, self.hidden_cell = self.lstm(x.view(len(x), 1, -1), self.hidden_cell)  # FIXME: not sure
        res = F.sigmoid(res)
        pred = self.linear(res.view(len(x), -1))
        pred = F.sigmoid(pred)
        return pred[-1]

    def clear_hidden_cells(self):
        if self.use_gpu:
            self.hidden_cell = (torch.zeros(1,1,self.hidden_size).cuda(),
                                torch.zeros(1,1,self.hidden_size).cuda())
        else:
            self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                                torch.zeros(1,1,self.hidden_size))

    def reset(self):
        self = LSTM(self.hidden_size, self.output_size)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

class Scaler(object):
    def __init__(self, low=-1, high=1):
        self.low = low
        self.high = high
    def fit(self, x):
        assert isinstance(x, list)
        self.min = min(x)
        self.max = max(x)
        if self.min == self.max:    # FIXME
            print(x)
        self.rate = (self.high - self.low) / (self.max - self.min)

    def transform(self, x):
        ans = [(i - self.min) * self.rate + self.low for i in x]    # FIXME: might be wrong
        return ans

    def fit_transform(self, x):
        self.fit(x)
        ans = self.transform(x)
        return ans
    def retransform(self, x):
        return (x - self.low) / self.rate + self.min

class Trainer(object):
    def __init__(self, 
                 model, 
                 data_reader,
                 max_ref_length = 365,
                 train_ref_length = 12,
                 num_epoch = 10,
                 lr = 0.001,
                 use_gpu = True):
        self.model = model
        self.data_reader = data_reader
        self.max_ref_length = max_ref_length
        self.train_ref_length = train_ref_length
        self.num_epoch = num_epoch
        self.lr = lr
        self.use_gpu = use_gpu
        self.loss_func = nn.MSELoss()

        self.seq = []
        self.cur = 0

        self.optimizer = optim.Adam(
            self.model.parameters(),
            self.lr
        )
        self.scaler = None
        self.y_pred = None

        if self.use_gpu:
            self.model.cuda()

    def to_device(self, x):
        if self.use_gpu:
            return x.cuda()
        else:
            return x

    def to_tensor(self, x):
        if self.use_gpu:
            return torch.tensor(x, dtype=torch.float32).cuda()
        else:
            return torch.tensor(x, dtype=torch.float32)
    
    def is_available(self):
        return self.scaler is not None  # equivalent to self.y_pred is not None

    def train_one_step(self, is_training = True):
        while True:
            self.cur += 1
            if self.cur >= self.data_reader.length:
                return -1   # break
            if self.data_reader.validations[self.cur] == True:
                break

        self.seq.append(self.data_reader.prices[self.cur])
        if len(self.seq) > self.max_ref_length:
            self.seq.pop(0)


        if is_training and self.train_ref_length + self.model.output_size <= len(self.seq):
            l1, l2 = self.train_ref_length, self.model.output_size

            if self.scaler is not None:
                y = self.to_tensor(self.seq[-l2:])
                self.lst_y.append(y)
                self.lst_y_pred.append(self.y_pred)

            self.model.reset()
            # scaling
            self.scaler = Scaler(-1,1)
            self.scaled_seq = self.scaler.fit_transform(self.seq)

            # sampling training data
            data = []
            idx = 0
            while True:
                if idx + l1 + l2 > len(self.seq):
                    break
                data.append((self.to_tensor(self.scaled_seq[idx: idx+l1]), self.to_tensor(self.scaled_seq[idx+l1: idx+l1+l2])))
                idx += 1

            # train for several epoches
            for epoch in range(self.num_epoch):
                for x, y in data:
                    self.optimizer.zero_grad()

                    y_pred = self.model(x)
                    loss = self.loss_func(y_pred, y)
                    loss.backward()
                    self.optimizer.step()
                    return loss.item()
            
            # prediction
            x = self.to_tensor(self.scaled_seq[-l1:])
            self.y_pred = self.scaler.retransform(self.model(x))    # prediction of next valid price

        else:
            return -2   # continue

    def train(self):
        loss_curve = []
        self.lst_y = []
        self.lst_y_pred = []
        while True:
            print('(%d/%d)'%(self.cur, self.data_reader.length), end='\r')
            res = self.train_one_step()
            if res == -1:
                break
            elif res == -2:
                continue
            else:
                loss_curve.append(res)
        # print(loss_curve[:10])
        plt.plot(self.lst_y, 'g')
        plt.plot(self.lst_y_pred, 'b')
        plt.show()
    
    def test(self):
        while self.cur <= self.data_reader.length / 2:
            self.train_one_step(is_training=False)
        self.train_one_step()
        self.ref = self.seq[-self.train_ref_length:]
        while True:
            pass

class Agent(object):
    def __init__(self, tr_gold, tr_btc, dr_gold, dr_btc, initial_state = (1000,0,0), rate_gold = 0.01, rate_btc = 0.02):
        self.tr_gold = tr_gold
        self.tr_btc = tr_btc
        self.dr_gold = dr_gold
        self.dr_btc = dr_btc
        self.rate_gold = rate_gold
        self.rate_btc = rate_btc
        self.state = initial_state  # (Dollar, Gold, BTC)
        self.cur = -1

    def asset(self):
        gold_price = self.dr_gold.prices[self.cur]
        btc_price = self.dr_btc.prices[self.cur]
        return (self.state[0], self.state[1] * gold_price, self.state[2] * btc_price)
    def total_asset(self):
        return sum(self.asset())
    def asset_distribution(self):
        return tuple([i / self.total_asset() for i in self.asset()])

    def step(self):
        self.cur += 1
        if self.dr_gold.validations[self.cur]:
            self.tr_gold.train_one_step()
        if self.dr_btc.validations[self.cur]:
            self.tr_btc.train_one_step()
        # TODO: somehow use the prediction to change the state

        return self.cur < self.dr_btc.length - 1    # become False from the last day

    def rearrage(self, lamb=1, lr=0.01, eps=1e-6):
        
        b1 = torch.tensor(self.asset_distribution())  # current distribution

        gold_price = self.dr_gold.prices[self.cur]
        btc_price = self.dr_btc.prices[self.cur]
        gold_pred = self.tr_gold.y_pred
        btc_pred = self.tr_btc.y_pred
        x = torch.tensor([1, gold_pred/gold_price, btc_pred/btc_price])
        weight = torch.tensor([self.rate_gold, self.rate_btc])

        b = b1.clone()[1:].requires_grad_(True)
        while True:
            b2 = torch.cat(((1-b.sum()).reshape(-1), b))
            Fw = torch.inner(x, b2)
            Ft = torch.inner(weight, b*torch.log2(b / b1[1:]))
            gain = lamb * Fw - Ft
            gain.backward()
            # new_b?
            with torch.no_grad():
                new_b = b - lr * b.grad
                if new_b[0] <= 0 and new_b[1] <= 0:
                    new_b = torch.tensor([0.0, 0.0])
                elif new_b[1] - new_b[0] >= 1 and new_b[1] >= 1:
                    new_b = torch.tensor([0.0, 1.0])
                elif new_b[0] - new_b[1] >= 1 and new_b[0] >= 1:
                    new_b = torch.tensor([1.0, 0.0])
                elif new_b[0] <= 0:
                    new_b[0] = 0.0
                elif new_b[1] <= 0:
                    new_b[1] = 0.0
                else:
                    new_b[0], new_b[1] = (1+new_b[0]-new_b[1])/2, (1-new_b[0]+new_b[1])/2
            if ((new_b - b) ** 2).sum() <= eps:
                break
            b = new_b.requires_grad_(True)
            for i in [b, b2, Fw, Ft, gain]:
                i.grad.zero_()
        return b

if __name__ == '__main__':
    dataset_root = '2022_Problem_C_DATA'
    dr_gold = DataReader(os.path.join(dataset_root, 'LBMA-GOLD.csv'))
    dr_btc = DataReader(os.path.join(dataset_root, 'BCHAIN-MKPRU.csv'))

    lstm_gold = LSTM()
    trainer_gold = Trainer(lstm_gold, dr_gold)
    lstm_btc = LSTM()
    trainer_btc = Trainer(lstm_btc, dr_btc)

    agent = Agent(trainer_gold, trainer_btc, dr_gold, dr_btc)
    
    asset_curve = []
    while True:
        is_last_day = agent.step()
        asset_curve.append(agent.total_asset())
        if is_last_day:
            break
    print('Final state:', agent.state())
    print('Assets:', agent.total_asset())
    plt.plot(asset_curve)
    plt.show()