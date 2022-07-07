import pandas as pd
import csv
import re
from random import shuffle
from tqdm import tqdm
from zhconv import convert
from create_corpus_pretrain_dataset import *
from gen_ecom import *


def preprocess_one(query):
    q = str(query)
    q = convert(q, 'zh-cn')
    q = q.lower()
    q = remove_punctuation(q)
    return q


def concat_big():
    df_b = pd.read_csv('./data/big.brand.query.corpus.csv')
    df_q = pd.read_csv('./data/big.query.corpus.csv')
    df_p = pd.read_csv('./data/corpus_pretrain_dataset.csv')
    df_e = pd.read_csv('./data/ecom_corpus_pretrain_dataset.csv')
    df_et = pd.read_csv('./data/ecom/train.data.csv', sep='\t')
    df_1 = pd.read_csv('./data/big.1.brand.commodity.corpus.csv')
    df_2 = pd.read_csv('./data/big.2.brand.commodity.corpus.csv')
    df = df_b.append(df_q)
    df = df.append(df_p)
    df = df.append(df_e)
    df = df.append(df_et)
    df = df.append(df_1)
    df = df.append(df_2)
    df['query'] = df['query'].map(lambda x: preprocess_one(x))
    df['doc'] = df['doc'].map(lambda x: preprocess_one(x))
    print(df.shape)
    # df = df.drop_duplicates(subset=['query','doc'], keep='first', inplace=True)
    # print("after drop duplicates",df.shape)
    df.to_csv('./data/pretrain.all.csv', index=None, header=None)


def make_qrels(qrels_file='./data/pretrain.all.csv',
               writer_file='./data/pretrain.train.csv',
               test_file='./data/pretrain.test.csv',
               test_num=3000,
               ):
    reader = csv.reader(open(qrels_file, encoding='utf-8'), delimiter=',')
    writer = csv.writer(open(writer_file, 'w', encoding='utf-8'))
    test_writer = csv.writer(open(test_file, 'w', encoding='utf-8'))
    reader = [line for line in reader]
    shuffle(reader)
    train_lines = reader[:-test_num]
    test_lines = reader[-test_num:]
    print(len(train_lines), len(test_lines))
    max_len = 0

    writer.writerow(['query', 'doc'])
    test_writer.writerow(['query', 'doc'])

    for line in tqdm(train_lines):
        writer.writerow([line[0], line[1]])
        max_len = max(len(line[0]), max_len)
        max_len = max(len(line[1]), max_len)

    for line in tqdm(test_lines):
        test_writer.writerow([line[0], line[1]])
        max_len = max(len(line[0]), max_len)
        max_len = max(len(line[1]), max_len)
    print(max_len)


if __name__ == "__main__":
    # base corpus
    all_en = []
    all_wd = []
    all_corpus = []
    base_corpus = load_corpus('data/corpus.tsv')
    create_word(base_corpus, all_en, all_wd, all_corpus)
    create_dataset('./data/corpus_pretrain_dataset.csv', all_en, all_wd, all_corpus)

    # ecom corpus
    all_en = []
    all_wd = []
    all_corpus = []
    corpus_ecom = load_corpus('./data/ecom/corpus.tsv')
    create_word(corpus_ecom, all_en, all_wd, all_corpus)
    create_dataset('./data/ecom_corpus_pretrain_dataset.csv', all_en, all_wd, all_corpus)

    gen_ecom()
    gen_big_pretrain()

    concat_big()
    make_qrels()
