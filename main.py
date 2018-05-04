# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os
import pickle
from functools import wraps

import numpy as np

import nsml
import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform
from torch.utils.data import DataLoader

from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, GPU_NUM, HAS_DATASET, IS_ON_NSML


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, dataset, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),
                os.path.join(dir_name, 'model'))
        with open(os.path.join(dir_name, 'dict'), 'wb') as f:
            pickle.dump({'dict': dataset.dict.word2id,
                        'max_len': dataset.max_len}, f)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        print('Model loading')
        checkpoint = torch.load(os.path.join(dir_name, 'model'))
        model.load_state_dict(checkpoint)

        with open(os.path.join(dir_name, 'dict'), 'rb') as f:
            params = pickle.load(f)

        dataset.dict.word2id = params['dict']
        dataset.max_len = params['max_len']
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        data = preprocess(raw_data, dataset.dict, dataset.max_len)
        max_len = max(map(lambda x: len(x), data))
        review = np.zeros((len(data), max_len), dtype=np.int64)
        for i, row in enumerate(data):
            length = len(row)
            review[i, :length] = row
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(Variable(torch.from_numpy(review)).cuda())
        point = ((output_prediction.data.squeeze(1).cpu().numpy() * 9 + 11) / 2).tolist()
        #point = np.clip(output_prediction.data.squeeze(1).cpu().numpy(), 1, 10).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """

    max_len = max(map(lambda x: len(x[0]), data))
    batch_size = len(data)

    review = np.zeros((batch_size, max_len), dtype=np.int64)
    labels = []

    for i, row in enumerate(data):
        text, label = row
        length = len(text)
        review[i, :length] = text
        labels.append(label)

    return torch.from_numpy(review), torch.Tensor(labels)

# From https://github.com/salesforce/awd-lstm-lm

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class Movie(nn.Module):
    def __init__(self, n_char, n_embed, n_hidden, dropout,
                dropout_in=0.1 attention=True):
        super().__init__()

        self.lockdrop = LockedDropout()

        self.h0 = nn.Parameter(torch.zeros(2, 1, n_hidden))
        self.c0 = nn.Parameter(torch.zeros(2, 1, n_hidden))

        self.embed = nn.Embedding(n_char, n_embed)

        self.h0_2 = nn.Parameter(torch.zeros(2, 1, n_hidden))
        self.c0_2 = nn.Parameter(torch.zeros(2, 1, n_hidden))

        encoder = nn.LSTM(n_embed, n_hidden,
                            batch_first=True, bidirectional=True)
        self.encoder = WeightDrop(encoder,
                    ['weight_hh_l0', 'weight_hh_l0_reverse'], dropout=dropout)
        
        encoder2 = nn.LSTM(n_hidden * 2, n_hidden,
                            batch_first=True, bidirectional=True)
        self.encoder2 = WeightDrop(encoder2,
                    ['weight_hh_l0', 'weight_hh_l0_reverse'], dropout=dropout)

        self.attn = nn.Linear(n_hidden, n_hidden)
        # self.attn_w = nn.Linear(n_hidden, 1)
        self.attn_w = nn.Linear(n_hidden, n_hidden)

        self.linear1 = nn.Linear(n_hidden * 2, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

        self.dropout = dropout
        self.attention = attention
        self.drop_in = dropout_in

        # self.init_weights()

    def encode_sent(self, sent, h0, c0, h0_2, c0_2, all_step=True):
        b_size = sent.size(0)
        sent = self.embed(sent)
        sent = self.lockdrop(sent, self.drop_in)

        output, (_, _) = self.encoder(sent, (h0, c0))
        output = self.lockdrop(output, self.drop_in)
        output, (encode, _) = self.encoder2(output, (h0_2, c0_2))

        if all_step:
            output = self.lockdrop(output, self.dropout)
            encode = self.linear1(output)

        else:
            encode = encode.permute(1, 0, 2).contiguous().view(b_size, -1)
            encode = F.dropout(encode, self.dropout, training=self.training)
            encode = self.linear1(encode)

        return encode


    def forward(self, sent1):
        b_size = sent1.size(0)

        h0 = self.h0.repeat(1, b_size, 1)
        c0 = self.c0.repeat(1, b_size, 1)
        h0_2 = self.h0_2.repeat(1, b_size, 1)
        c0_2 = self.c0_2.repeat(1, b_size, 1)

        lin1 = self.encode_sent(sent1, h0, c0, h0_2, c0_2,
                                all_step=self.attention)

        if self.attention:
            attn = self.attn_w(F.tanh(self.attn(lin1)))
            attn = F.softmax(attn, 1)
            attn = lin1 * attn
            lin1 = attn.sum(1)

        out = self.linear2(F.relu(lin1))

        return out.clamp(-1, 1)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=8)
    args.add_argument('--batch', type=int, default=128)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=8)
    config = args.parse_args()

    learning_rate = 1e-3
    grad_clip = True
    dropout = 0.2
    n_char = 4500
    n_embed = 256
    n_hidden = 256

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    if config.mode == 'train':
        dataset = MovieReviewDataset(DATASET_PATH, flip=True)

    else:
        dataset = MovieReviewDataset('', build=False, flip=False)

    model = Movie(n_char, n_embed, n_hidden, dropout)
    model_run = Movie(n_char, n_embed, n_hidden, dropout)
    if GPU_NUM:
        model = model.cuda()
        model_run.cuda()
    accumulate(model_run, model, 0)

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model_run, dataset, config)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                data = Variable(data)
                if GPU_NUM:
                    data = data.cuda()

                predictions = model(data)

                label_vars = Variable((labels * 2 - 11) / 9)
                if GPU_NUM:
                    label_vars = label_vars.cuda()

                loss = criterion(predictions, label_vars)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    grad_norm = nn.utils.clip_grad_norm(model.parameters(), 1)
                    grad_norm = grad_norm
                else:
                    grad_norm = 0
                optimizer.step()
                avg_loss += loss.data[0]
                accumulate(model_run, model)
                print('batch', i, 'loss', loss.data[0], 'norm', grad_norm)
                nsml.report(summary=True, scope=locals(), epoch=epoch,
                            epoch_total=config.epochs, train__loss=float(loss.data[0]),
                            step=epoch * total_batch + i)
                #print('batch:', i, 'loss:', loss.data[0])
            print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))

            if epoch in [4, 6]:
                optimizer.param_groups[0]['lr'] /= 10
                print('reduce learning rate')

            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            #
            #nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
            #            train__loss=float(avg_loss/total_batch), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
