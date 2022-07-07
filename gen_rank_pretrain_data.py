import os
import pandas as pd
from  gen_recall_pretrain_data import *

if __name__ == "__main__":
    input_file = './data/pretrain.all.csv'
    if not os.path.exists(input_file):
        print("file not exists") 
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
    print("file exists") 
    writer_file = './data/rank_pretrain_data/rank_pretrain.txt'
    writer = csv.writer(open(writer_file, 'w',encoding='utf-8'))
    df = pd.read_csv(input_file, header=None, names=['query', 'doc', '_'], low_memory=False)
    print(df)
    df = df.drop_duplicates(['doc'], keep='first', inplace=False)
    rank_file = './data/rank_pretrain_data/rank_pretrain_all.csv'
    df.to_csv(rank_file, index= None, header=None)
    print(df)
    
    reader = csv.reader(open(rank_file,encoding='utf-8'),delimiter=',')
    lines = [line for line in reader]
    shuffle(lines)

    for line in tqdm(lines):
        writer.writerow([line[0]])
        writer.writerow([line[1]])
        writer.writerow([])


