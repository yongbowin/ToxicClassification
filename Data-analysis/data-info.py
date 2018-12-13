#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: ToxicClassification - data-info.py
@time: 8/31/18 11:06 PM
@description: 
"""
import pandas as pd
import matplotlib.pyplot as plt


class DataStatistics:
    def __init__(self):
        self.train = pd.read_csv('../input/train.csv')
        self.test = pd.read_csv('../input/test.csv')

    def freq_statistics(self):
        lens = self.train.comment_text.str.len()
        print(lens.mean(), lens.std(), lens.max(), lens.min())

        plt.hist(lens, bins=40, facecolor='blue', edgecolor='black', alpha=0.7)
        plt.xlabel('The length of comments')
        plt.ylabel('Frequency')
        # plt.title('Histogram of Word frequency distribution')
        plt.text(2000, 30000, r'$\mu=' + str(round(lens.mean(), 4)) + r',\ \sigma=' + str(round(lens.std(), 4)) + r'$')
        plt.grid(False)
        plt.savefig('freq_distribution.jpg')

        # plt.show()

    def ratio_statistics(self):
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.train['none'] = 1-self.train[label_cols].max(axis=1)
        # print(self.train.describe())
        count = 0
        for i in list(self.train['none'].values):
            if i == 1:
                count += 1
        print('count:', count)

        labels = 'Have labels', 'No labels'
        sizes = [len(self.train)-count, count]
        explode = (0.2, 0)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        # plt.title('The ratio of whether or not there is a label')
        plt.savefig('retio_labels.jpg')
        # plt.show()

        print(len(self.train), len(self.test))


if __name__ == '__main__':
    ds = DataStatistics()
    ds.freq_statistics()
    ds.ratio_statistics()