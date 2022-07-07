import pandas as pd
import csv
from random import shuffle
from tqdm import tqdm
import re
from zhconv import convert


TRAIN_FILE = './data/train.query.txt'
DEV_FILE = './data/dev.query.txt'
CORPUS_FILE = './data/corpus.tsv'
QUERY_FILE = './data/qrels.train.tsv'
QUERY_DOC_FILE = './data/train.data.tsv'

def gen_new_query_doc():
    corpus_data = pd.read_csv(CORPUS_FILE, sep="\t", encoding='utf-8', header=None, names=["doc_id", "doc"])
    train_data = pd.read_csv(TRAIN_FILE, sep="\t", encoding='utf-8', header=None, names=["query_id", "query"])
    dev_data = pd.read_csv(DEV_FILE, sep="\t", encoding='utf-8', header=None, names=["query_id", "query"])
    
    qrels = pd.read_csv(QUERY_FILE, sep="\t", encoding='utf-8', header=None, names=["query_id", "doc_id"])
    print(qrels.shape)
    qrels = pd.merge(qrels, train_data[["query_id", "query"]], on='query_id', how='inner')
    print(qrels.shape)
    qrels = pd.merge(qrels, corpus_data[["doc_id", "doc"]], on='doc_id', how='inner')

    print(qrels.shape)
    qrels.to_csv(QUERY_DOC_FILE,sep='\t',encoding='utf-8',header=None, index=False)

def make_qrels(qrels_file='./data/train.data.tsv',
               writer_file='./data/query.doc.csv',
               test_file='./data/query.doc.test.csv',
               test_num=1000,
               ):
    reader = csv.reader(open(qrels_file,encoding='utf-8'),delimiter='\t')
    writer = csv.writer(open(writer_file, 'w',encoding='utf-8'))
    test_writer = csv.writer(open(test_file, 'w',encoding='utf-8'))
    reader = [line for line in reader]
    shuffle(reader)
    train_lines = reader[:-test_num]
    test_lines = reader[-test_num:]
    print(len(train_lines),len(test_lines))
    max_len = 0

    writer.writerow(['query', 'doc'])
    test_writer.writerow(['query', 'doc'])

    for line in tqdm(train_lines):
        writer.writerow([line[2], line[3]])
        max_len = max(len(line[2]), max_len)
        max_len = max(len(line[3]), max_len)


    for line in tqdm(test_lines):
        test_writer.writerow([line[2], line[3]])
        max_len = max(len(line[2]), max_len)
        max_len = max(len(line[3]), max_len)
    print(max_len)


if __name__ == '__main__':
    gen_new_query_doc()
    make_qrels(test_num=1000)
