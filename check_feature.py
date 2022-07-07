import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import jieba
import re
from zhconv import convert
import pdb
import datetime
import faiss
import sys
import os 
import csv
import torch
import copy
import random
from random import shuffle
from tqdm import tqdm
from tqdm import tqdm_notebook
from simcse.models import BertForCL
from transformers import AutoTokenizer, BertTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda:0"
batch_size = 100
use_pinyin = False

def encode_fun(tokenizer, texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return inputs['input_ids'], embeddings

def get_query_doc_embedding(FEAT, query, doc):
    #tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenizer = BertTokenizer.from_pretrained("./models/result_recall")
    model = BertForCL.from_pretrained("./models/result_recall")
    batch_text = [query, doc]
    input_ids, temp_embedding = encode_fun(tokenizer, batch_text, model)
    query_str = [format(s, '.8f') for s in temp_embedding[0].tolist()]
    query_str = ','.join(query_str)
    query_ids_str = input_ids[0].tolist()
    if len(query_ids_str) <= 128:
        ids_str_app = []
        for k in range(len(ids_str),128):
            ids_str_app.append(0)
        query_ids_str.extend(ids_str_app)
    else:
        print(query_ids_str)
    query_ids_str = [str(s) for s in query_ids_str[:128]]
    #query_ids_str = ','.join(query_ids_str)
    
    doc_str = [format(s, '.8f') for s in temp_embedding[1].tolist()]
    doc_str = ','.join(doc_str)
    doc_ids_str = input_ids[0].tolist()
    if len(doc_ids_str) <= 129:
        ids_str_app = []
        for k in range(len(ids_str)-1,128):
            ids_str_app.append(0)
        doc_ids_str.extend(ids_str_app)
    else:
        print(doc_ids_str)
    doc_ids_str = [str(s) for s in doc_ids_str[1:129]]
    #doc_ids_str = ','.join(doc_ids_str)
    return query_ids_str, doc_ids_str

def get_embedding(FEAT, type_='base'):
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    if type_ == 'base':
        model = BertForCL.from_pretrained("./models/result_recall_base")
    else:
        model = BertForCL.from_pretrained("./models/result_recall")
    model.to(device)
    corpus = [line[1] for line in csv.reader(open("./data/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./data/dev.query.txt"), delimiter='\t')]
    train = [line[1] for line in csv.reader(open("./data/train.query.txt"), delimiter='\t')]
    query_embedding_file = csv.writer(open(FEAT + 'query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        input_ids, temp_embedding = encode_fun(tokenizer, batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            ids_str = input_ids[j].tolist()
            if len(ids_str) <= 128:
                ids_str_app = []
                for k in range(len(ids_str),128):
                    ids_str_app.append(0)
                ids_str.extend(ids_str_app)
            else:
                print(ids_str)
            ids_str = [str(s) for s in ids_str[:128]]
            ids_str = ','.join(ids_str)
            query_embedding_file.writerow([i + j + 200001, writer_str, ids_str])

    doc_embedding_file = csv.writer(open(FEAT + 'doc_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        input_ids, temp_embedding = encode_fun(tokenizer, batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            ids_str = input_ids[j].tolist()
            if len(ids_str) <= 129:
                ids_str_app = []
                for k in range(len(ids_str) - 1,128):
                    ids_str_app.append(0)
                ids_str.extend(ids_str_app)
            else:
                print(ids_str)
            ids_str = [str(s) for s in ids_str[1:129]]
            ids_str = ','.join(ids_str)
            doc_embedding_file.writerow([i + j + 1, writer_str, ids_str])
    train_embedding_file = csv.writer(open(FEAT + 'train_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(train), batch_size)):
        batch_text = train[i:i + batch_size]
        input_ids, temp_embedding = encode_fun(tokenizer, batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            ids_str = input_ids[j].tolist()
            if len(ids_str) <= 128:
                ids_str_app = []
                for k in range(len(ids_str),128):
                    ids_str_app.append(0)
                ids_str.extend(ids_str_app)
            else:
                print(ids_str)
            ids_str = [str(s) for s in ids_str[:128]]
            ids_str = ','.join(ids_str)
            train_embedding_file.writerow([i + j + 1, writer_str, ids_str])


def split_a(feature):
    return [float(i) for i in feature.split(",")]

def query_faiss_score(query_feature, index, a, b):
    # I 为 score D 为 ID
    index.nprobe = 10
    D, I = index.search(query_feature[a: b], 50)
    I = I.tolist()
    I = [[j + 1 for j in i] for i in I]
    return D.tolist(), I


def corpus_faiss_score(corpus_feature, index, a, b, k):
    # I 为 score D 为 ID
    D, I = index.search(corpus_feature[a: b], k)
    return D.tolist(), I.tolist()

def load_embedding(FEAT, type_):
    corpus_feature_data = pd.read_csv( FEAT + "doc_embedding", sep="\t", names=["query", "feature", "token"])
    dev_feature_data = pd.read_csv(FEAT + "query_embedding", sep="\t", names=["query", "feature", "token"])
    query_feature_data = pd.read_csv(FEAT + "train_embedding", sep="\t", names=["query", "feature", "token"])
    
    
    corpus_feature_data = corpus_feature_data.set_index("query")
    dev_feature_data = dev_feature_data.set_index("query")
    query_feature_data = query_feature_data.set_index("query")
    
    corpus_feature = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in corpus_feature_data["feature"])
    dev_feature = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in dev_feature_data["feature"])
    query_feature = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in query_feature_data["feature"])
    
    corpus_feature = np.array(corpus_feature, dtype=np.float32)
    dev_feature = np.array(dev_feature, dtype=np.float32)
    query_feature = np.array(query_feature, dtype=np.float32)
    
    
    query_list = {}
    
    for line in open("./data/qrels.train.tsv", "r+"):
        lines = line.strip().split("\t")
        q_idx = int(lines[0])
        c_idx = int(lines[1])
        query_list[q_idx] = c_idx
    

    PROCESSNUM = 50
    query_num = 100000
    index_list = []
    # INDEX_KEY = "IDMap,IVF4096"
    quantizer = faiss.IndexFlatL2(128)  # the other index
    index = faiss.IndexIVFFlat(quantizer, 128, 4096, faiss.METRIC_L2)
    # index = faiss.index_factory(128, INDEX_KEY, faiss.METRIC_L2)
    assert not index.is_trained
    index.train(corpus_feature)
    assert index.is_trained
    index.add(corpus_feature)
    
    
    import csv
    corpus = [[line[0],line[1]] for line in csv.reader(open("./data/corpus.tsv"), delimiter='\t')]
    train_query = [[line[0],line[1],line[2]] for line in csv.reader(open("./data/train.data.tsv"), delimiter='\t')]
    dev_query = [[line[0],line[1]] for line in csv.reader(open("./data/dev.query.txt"), delimiter='\t')]
    
    print("load query success !!!!!")
    a = 0
    b = 1000
    h = 16
    
    from multiprocessing import Process, Manager
    manager = Manager()
   
    if type_=='doc':
        match_doc_train = []
        for query, doc in query_list.items():#query_inter_list:
            in_train = False
            num_list = []
            top_num = 30
            s_out, d = corpus_faiss_score(corpus_feature, index, doc - 1, doc, top_num)
            n_d_list = []
            for m in d[0]:
                n_d_list.append(corpus[m][0])
            s_list = '#'.join(map(str, s_out[0]))
            d_list = '#'.join(map(str, n_d_list))
            match_doc_train.append([query,doc,s_list,d_list])
        df_new = pd.DataFrame(match_doc_train,columns=['query_id','doc_id','score_list','doc_list'])
        df_new.to_csv(FEAT + 'doc.top30.train.query.txt', sep='\t', encoding='utf-8',index=False)
    elif type_ == "query":
        match_query_train=[]
        a = 0
        b = 100000
        h = 8
        for i in range(a, b, h):
            s, d = query_faiss_score(query_feature, index, i, min(b, i + h))
            for k, j in enumerate(range(i, min(b, i + h))):
                q_list = []
                q_score_list = []
                for m in range(30):
                    doc_id = corpus[d[k][m] - 1][0]
                    q_list.append(str(doc_id))
                    q_score_list.append(str(s[k][m]))
                match_query_train.append([train_query[j][0],train_query[j][1],"#".join(q_score_list),"#".join(q_list)])
            if i % 10000 == 0: print("success idx:{}".format(i))
        df_new = pd.DataFrame(match_query_train,columns=['query_id','doc_id','score_list','query_list'])
        df_new.to_csv(FEAT + 'query.top30.train.query.txt', sep='\t', encoding='utf-8',index=False)

def map_full_match(x):
    query = x['query']
    doc = x['top10_doc']
    if str(query).lower().replace(' ','') in str(doc).lower().replace(' ', ''):
        return 1
    else:
        return 0

def extract_train_topn(FEAT, from_, to_):
    df = pd.read_csv(FEAT + 'doc.top30.train.query.txt',sep='\t')#,header=None, names=['query_id','doc_id','query_list','doc_list'])
    df_train = pd.read_csv('./data/train.query.txt', sep='\t', encoding='utf-8',names=['query_id','query'])
    df = pd.merge(df, df_train, on ='query_id', how='left')

    df_corpus = pd.read_csv('./data/corpus.tsv', sep='\t', encoding='utf-8',names=['doc_id','doc'])
    query_list = []
    doc_list = []
    error_count = 0
    for i in tqdm(range(df.shape[0])):
        try:
            query_id = df.iloc[i]['query_id']
            query = df.iloc[i]['query']
            doc_id = df.iloc[i]['doc_id']
            
            s_list = str(df.iloc[i]['score_list']).split('#')
            q_list = str(df.iloc[i]['doc_list']).split('#')
            select_list = []
            for i in range(from_,to_,1):
                if int(float(q_list[i])) != doc_id:
                    select_list.append([query_id, doc_id, int(float(q_list[i])), query,round(float(s_list[i]),6)])
            select_list = random.sample(select_list,10)
            query_list.extend(select_list)
        except Exception as e:
            error_count += 1
            pdb.set_trace()
            print(query_id)
    df_out = pd.DataFrame(query_list,columns=['query_id','doc_id','top10_doc_id','query','score'])
    df_out = pd.merge(df_out, df_corpus, on ='doc_id',how='left')
    df_corpus['top10_doc_id'] = df_corpus['doc_id']
    df_corpus['top10_doc'] = df_corpus['doc']
    df_out = pd.merge(df_out, df_corpus[['top10_doc_id','top10_doc']], on ='top10_doc_id',how='left')

    df_out = df_out.drop(columns=['score'])
    print(df_out['top10_doc_id'].nunique())
    df_out['is_full_match'] = df_out.apply(lambda x: map_full_match(x), axis=1)
    print(df_out['is_full_match'].value_counts())
    df_out = df_out[df_out.is_full_match==0]
    df_out = df_out.drop(columns=['is_full_match'])
    
    df_out.to_csv(FEAT + 'hard.train.data.tsv', sep='\t', encoding='utf-8',index=False, header=None)

def extract_train_set(FEAT, from_, to_, path_pre, out_path):

    df = pd.read_csv(FEAT + 'query.top30.train.query.txt',sep='\t')#,header=None, names=['query_id','doc_id','query_list','doc_list'])
    query_list = []
    doc_list = []
    error_count = 0
    for i in tqdm(range(df.shape[0])):
        try:
            query_id = df.iloc[i]['query_id']
            doc_id = df.iloc[i]['doc_id']
            
            q_list = [int(item) for item in str(df.iloc[i]['query_list']).split('#')[from_:to_]]
            #pdb.set_trace()
            if doc_id in q_list:
                q_list.remove(doc_id)
            q_list_str = '#'.join(map(str, q_list))
            lenq = len(q_list)
            if lenq != 10:
                error_count +=1
            query_list.append([query_id, doc_id, q_list_str])
            for top10_doc_id in q_list:
                doc_list.append([query_id, doc_id, top10_doc_id])
        except Exception as e:
            error_count += 1
            pdb.set_trace()
            print(query_id)
    print(error_count)
    df_out = pd.DataFrame(query_list,columns=['query_id','doc_id','top10_doc_id'])
    df_out.to_csv(out_path + path_pre + '.train.set.tsv', sep='\t', encoding='utf-8',index=False, header=None)
    
def make_recall_qrels(FEAT):
    qrels_file=FEAT + 'hard.train.data.tsv'
    writer_file='./data/hard.query.doc.csv'
    test_file='./data/hard.query.doc.test.csv'
    test_num=1000
    reader = csv.reader(open(qrels_file,encoding='utf-8'),delimiter='\t')
    writer = csv.writer(open(writer_file, 'w',encoding='utf-8'))
    test_writer = csv.writer(open(test_file, 'w',encoding='utf-8'))
    reader = [line for line in reader]
    shuffle(reader)
    train_lines = reader[:-test_num]
    test_lines = reader[-test_num:]
    print(len(train_lines),len(test_lines))
    max_len = 0

    writer.writerow(['query', 'doc', 'hard'])
    test_writer.writerow(['query', 'doc', 'hard'])

    for line in tqdm(train_lines):
        writer.writerow([line[3], line[4], line[5]])
        max_len = max(len(line[3]), max_len)
        max_len = max(len(line[4]), max_len)
        max_len = max(len(line[5]), max_len)


    for line in tqdm(test_lines):
        test_writer.writerow([line[3], line[4], line[5]])
        max_len = max(len(line[3]), max_len)
        max_len = max(len(line[4]), max_len)
        max_len = max(len(line[5]), max_len)
    print(max_len)


if __name__ == '__main__':
    make_qrels(test_num=1000)
    FEAT = 'data/feature_base_recall/'
    if not os.path.exists(FEAT): os.mkdir(FEAT)
    get_embedding(FEAT, 'base')
    load_embedding(FEAT, "doc")
    FEAT = './data/feature_recall/'
    if not os.path.exists(FEAT): os.mkdir(FEAT)
    get_embedding(FEAT, 'recall')
    load_embedding(FEAT, "query")
