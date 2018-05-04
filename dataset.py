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

import os

import numpy as np
from torch.utils.data import Dataset
import random

from kor_char_parser import decompose_str_as_one_hot


import re


class Dictionary:
    def __init__(self):
        self.word2id = {}
        self.id = 3 # 0 for null, 1 for unk 2 for eos

    def add(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.id
            self.id += 1

        return self.word2id[word]


class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, build=True, flip=False):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """

        self.dict = Dictionary()

        self.max_len = 0

        self.flip = flip

        if build:
            # 데이터, 레이블 각각의 경로
            data_review = os.path.join(dataset_path, 'train', 'train_data')
            data_label = os.path.join(dataset_path, 'train', 'train_label')

            self.construct_dict(data_review)

            # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
            with open(data_review, 'rt', encoding='utf-8') as f:
                self.reviews = preprocess(f.readlines(), self.dict, self.max_len)
            # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
            with open(data_label) as f:
                self.labels = [float(x) for x in f.readlines()]

    def construct_dict(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf8') as f:
            for line in f:
                q1 = line.strip()

                self.max_len = max(self.max_len, len(q1))

                for ch in q1:
                    self.dict.add(ch)

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """

        if self.flip:
            review = []

            for i in self.reviews[idx]:
                if i != 2 and random.randint(0, 99) < 1:
                    review.append(1)

                else:
                    review.append(i)

        else:
            review = self.reviews[idx]

        return review, self.labels[idx]


def preprocess(data: list, dictionary, max_len):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    result = []
    for idx, seq in enumerate(data):
        q1 = seq.strip()
        line = []

        for i, ch in enumerate(q1):
            try:
                line.append(dictionary.word2id[ch])

            except IndexError:
                continue

            except KeyError:
                line.append(1)

        line.append(2)

        result.append(line)

    return result