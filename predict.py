import tensorflow as tf
import os
import argparse
import numpy as np
import time
import pandas as pd
from bert import modeling
from bert import optimization
from rank_model import Ranker
from tqdm import tqdm
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
import pdb
import pandas as pd
from wrapper import *
from simcse.models import BertForCL
from transformers import AutoTokenizer, BertTokenizer
import torch
BASE = './data/'
FEAT = BASE + 'feature_recall/'
DATA = BASE

def encode_fun(tokenizer, texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return inputs['input_ids'], embeddings

def get_query_doc_embedding(FEAT, query, doc):
    tokenizer = BertTokenizer.from_pretrained("./models/result_recall")
    model = BertForCL.from_pretrained("./models/result_recall")
    batch_text = [query, doc]
    input_ids, temp_embedding = encode_fun(tokenizer, batch_text, model)
    query_str = [format(s, '.8f') for s in temp_embedding[0].tolist()]
    query_str = ','.join(query_str)
    query_ids_str = input_ids[0].tolist()
    if len(query_ids_str) <= 128:
        ids_str_app = []
        for k in range(len(query_ids_str),128):
            ids_str_app.append(0)
        query_ids_str.extend(ids_str_app)
    else:
        print(query_ids_str)
    query_ids_str = [str(s) for s in query_ids_str[:128]]
    
    doc_str = [format(s, '.8f') for s in temp_embedding[1].tolist()]
    doc_str = ','.join(doc_str)
    doc_ids_str = input_ids[0].tolist()
    if len(doc_ids_str) <= 129:
        ids_str_app = []
        for k in range(len(doc_ids_str)-1,128):
            ids_str_app.append(0)
        doc_ids_str.extend(ids_str_app)
    else:
        print(doc_ids_str)
    doc_ids_str = [str(s) for s in doc_ids_str[1:129]]
    return query_ids_str, doc_ids_str

def split_a(feature):
    return [int(i) for i in feature.split(",")]

def get_dev_inputs(saved_model_path, topn=20):
    corpus_feature_data = pd.read_csv( FEAT + "doc_embedding", sep="\t", names=["top10_doc_id", "feature", "token"])
    dev_feature_data = pd.read_csv(FEAT + "query_embedding", sep="\t", names=["query_id", "feature", "token"])

    #corpus_feature_data = corpus_feature_data.set_index("doc_id")
    #dev_feature_data = dev_feature_data.set_index("query_id")
    #print(corpus_feature_data['token'])
    

    corpus_feature_data['doc_input_ids'] = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in corpus_feature_data["token"])
    dev_feature_data['query_input_ids'] = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in dev_feature_data["token"])

    #doc_input_ids = np.array(doc_input_ids, dtype=np.float32)
    #dev_input_ids = np.array(dev_input_ids, dtype=np.float32)
    #query_input_ids = np.array(query_input_ids, dtype=np.float32)
    
    df = pd.read_csv(FEAT + 'query.top50.dev.query.txt', sep='\t')
    query_list = []
    doc_list = []
    error_count = 0
    for i in tqdm(range(df.shape[0])):
        try:
            query_id = df.iloc[i]['query_id']
            
            q_list = [int(item) for item in str(df.iloc[i]['query_list']).split('#')[:topn]]
            for top10_doc_id in q_list:
                doc_list.append([query_id, top10_doc_id])
        except Exception as e:
            error_count += 1
            pdb.set_trace()
            print(query_id)
    #print(error_count)
    
    df_out = pd.DataFrame(doc_list,columns=['query_id','top10_doc_id'])
    df_train = pd.read_csv(DATA + './data/dev.query.txt', sep='\t', encoding='utf-8',names=['query_id','query'])
    df_out = pd.merge(df_out, df_train, on ='query_id', how='inner')
    
    df_corpus = pd.read_csv(DATA + './data/corpus.tsv', sep='\t', encoding='utf-8',names=['top10_doc_id','top10_doc'])
    df_out = pd.merge(df_out, df_corpus, on ='top10_doc_id',how='inner')
    #print(df_out)
    #print(query_feature_data)
    df_out = pd.merge(df_out, dev_feature_data[['query_id','query_input_ids']], on ='query_id', how='inner')
    #print(df_out)
    df_out = pd.merge(df_out, corpus_feature_data[['top10_doc_id','doc_input_ids']], on ='top10_doc_id', how='inner')
    grouped = df_out.groupby('query_id')

    df_result = None
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        loaded = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        graph = tf.get_default_graph()
        query_i = sess.graph.get_tensor_by_name('query_ids:0')
        doc_i = sess.graph.get_tensor_by_name('doc_ids:0')
        score = sess.graph.get_tensor_by_name('score:0')
        starttime = time.time()
    '''
	
    is_first = True
    if True:
        for query_id, group in grouped:
            result_score = []
            query = group['query'].unique()[0]
            top10_doc_id = group['top10_doc_id'].tolist()
            top10_doc = group['top10_doc'].tolist()
            #import pdb
            #pdb.set_trace()
            query_input_ids = np.array([group['query_input_ids'].tolist()[0]]).reshape(1, 128)
            doc_input_ids = np.array(group['doc_input_ids'].tolist()).reshape(-1, 128)
            #score_result = sess.run(score, feed_dict={query_i:query_input_ids, doc_i:doc_input_ids})
            score = test_saved_model(saved_model_path, query_input_ids, doc_input_ids)
            print(score.size)
            print(score)
            print(len(score))
            for num in range(len(score)):
                result_score.append([query_id, top10_doc_id[num], query, score[num], top10_doc[num]])
            df_r = pd.DataFrame(result_score,columns=['query_id', 'top10_doc_id', 'query', 'score', 'top10_doc'])
            df_r.sort_values(by='score',axis=0,ascending=False, inplace=True)
            if is_first:
                df_result = df_r
                is_first = False
            else:
                df_result = df_result.append(df_r)
    df_result.to_csv(FEAT + "dev.score.tsv",sep='\t',index=False)       
    print("*** The saved_model is available.")


def get_train_inputs(saved_model_path, topn=20):
    corpus_feature_data = pd.read_csv( FEAT + "doc_embedding", sep="\t", names=["doc_id", "feature", "token"])
    #dev_feature_data = pd.read_csv(FEAT + "query_embedding", sep="\t", names=["query_id", "feature", "token"])
    query_feature_data = pd.read_csv(FEAT + "train_embedding", sep="\t", names=["query_id", "feature", "token"])

    #corpus_feature_data = corpus_feature_data.set_index("doc_id")
    #dev_feature_data = dev_feature_data.set_index("query_id")
    #query_feature_data = query_feature_data.set_index("query_id")
    #print(corpus_feature_data['token'])
    

    corpus_feature_data['doc_input_ids'] = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in corpus_feature_data["token"])
    #dev_feature_data['dev_input_ids'] = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in dev_feature_data["token"])
    query_feature_data['query_input_ids'] = Parallel(n_jobs=4, backend="multiprocessing")(delayed(split_a)(feature) for feature in query_feature_data["token"])

    #doc_input_ids = np.array(doc_input_ids, dtype=np.float32)
    #dev_input_ids = np.array(dev_input_ids, dtype=np.float32)
    #query_input_ids = np.array(query_input_ids, dtype=np.float32)
    
    df = pd.read_csv(FEAT + 'query.top30.train.query.txt', sep='\t')
    query_list = []
    doc_list = []
    error_count = 0
    #for i in tqdm(range(df.shape[0])):
    for i in tqdm(range(100)):
        try:
            query_id = df.iloc[i]['query_id']
            doc_id = df.iloc[i]['doc_id']
            
            q_list = [int(item) for item in str(df.iloc[i]['query_list']).split('#')[:topn]]
            for top10_doc_id in q_list:
                doc_list.append([query_id, doc_id, top10_doc_id])
        except Exception as e:
            error_count += 1
            pdb.set_trace()
            print(query_id)
    #print(error_count)
    
    df_out = pd.DataFrame(doc_list,columns=['query_id','doc_id','top10_doc_id'])
    df_train = pd.read_csv(DATA + './data/train.query.txt', sep='\t', encoding='utf-8',names=['query_id','query'])
    df_out = pd.merge(df_out, df_train, on ='query_id', how='inner')
    
    df_corpus = pd.read_csv(DATA + './data/corpus.tsv', sep='\t', encoding='utf-8',names=['doc_id','doc'])
    df_out = pd.merge(df_out, df_corpus, on ='doc_id',how='inner')
    df_corpus['top10_doc_id'] = df_corpus['doc_id']
    df_corpus['top10_doc'] = df_corpus['doc']
    df_out = pd.merge(df_out, df_corpus[['top10_doc_id','top10_doc']], on ='top10_doc_id',how='inner')
   
    #print(df_out)
    #print(query_feature_data)
    df_out = pd.merge(df_out, query_feature_data[['query_id','query_input_ids']], on ='query_id', how='inner')
    corpus_feature_data['top10_doc_id'] = corpus_feature_data['doc_id']
    
    #print(df_out)
    corpus_feature_data['top10_doc_id'] = corpus_feature_data['doc_id']
    df_out = pd.merge(df_out, corpus_feature_data[['top10_doc_id','doc_input_ids']], on ='top10_doc_id', how='inner')
    grouped = df_out.groupby('query_id')

    df_result = None
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        loaded = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        graph = tf.get_default_graph()
        query_i = sess.graph.get_tensor_by_name('query_ids:0')
        doc_i = sess.graph.get_tensor_by_name('doc_ids:0')
        score = sess.graph.get_tensor_by_name('score:0')
        starttime = time.time()
    '''
	
    is_first = True
    dist_score = []
    if True:
        for query_id, group in grouped:
            result_score = []
            query = group['query'].unique()[0]
            doc_id = group['doc_id'].unique()[0]
            doc = group['doc'].unique()[0]
            top10_doc_id = group['top10_doc_id'].tolist()
            top10_doc = group['top10_doc'].tolist()
            #import pdb
            #pdb.set_trace()
            query_input_ids = np.array([group['query_input_ids'].tolist()[0]]).reshape(1, 128)
            doc_input_ids = np.array(group['doc_input_ids'].tolist()).reshape(-1, 128)
            #score_result = sess.run(score, feed_dict={query_i:query_input_ids, doc_i:doc_input_ids})
            score = test_saved_model(saved_model_path, query_input_ids, doc_input_ids)
            for num in range(len(score)):
                result_score.append([query_id, doc_id, top10_doc_id[num], query, doc, score[num], top10_doc[num]])
            df_r = pd.DataFrame(result_score,columns=['query_id','doc_id','top10_doc_id', 'query', 'doc','score', 'top10_doc'])
            df_r.sort_values(by='score',axis=0,ascending=False, inplace=True)
            df_r = df_r[:10]
            if is_first:
                df_result = df_r
                is_first = False
            else:
                df_result = df_result.append(df_r)
            r = 0
            for num in range(10):
                #import pdb
                #pdb.set_trace()
                if doc_id == df_r.iloc[num]['top10_doc_id']:
                    r = 1 / (num + 1)
            dist_score.append(r)
    print(dist_score)
    sum = np.sum(dist_score) / len(dist_score)
    print(sum)
    df_result.to_csv(FEAT + "train.score.tsv",sep='\t',index=False)       
    print("*** The saved_model is available.")

def test_saved_model(saved_model_path, query_input_ids, doc_input_ids):
    """
    The input_ids of query and doc are fake.
    Note:
        If you use bert-like model, 
        *** don't forget add the index of '[CLS]' and '[SEP]' in the input_ids of query. 
        *** don't forget add the index of '[SEP]' in the input_ids of doc.
    """
    #query_input_ids = np.array([[101, 4508, 7942, 7000, 7350, 2586, 3296, 2225, 4275, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(1, 128)
    #doc_input_ids = np.array([[2408, 691, 7556, 705, 4385, 6573, 6862, 1355, 524, 11361, 120, 2608, 4448, 5687, 1788, 4508, 4841, 7000, 7350, 2364, 3296, 2225, 4275, 121, 119, 8132, 8181, 115, 8108, 4275, 120, 4665, 166, 8197, 5517, 7608, 5052, 5593, 4617, 3633, 1501, 3241, 3309, 5310, 1394, 6956, 5593, 4617, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[727, 1310, 4377, 4508, 4841, 7000, 796, 827, 3296, 2225, 5540, 1718, 9695, 8181, 115, 8114, 5108, 120, 4665, 5498, 5301, 5528, 4617, 1146, 1265, 1798, 4508, 4307, 5593, 4617, 2229, 6956, 3241, 3309, 3186, 5664, 2421, 2135, 3175, 3633, 1501, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[928, 6449, 3710, 4508, 4384, 7000, 4275, 8114, 3710, 7185, 2128, 4508, 4384, 7000, 5790, 4508, 4841, 7000, 1079, 3302, 1366, 3302, 928, 2139, 7212, 1853, 3632, 6117, 5542, 4508, 7000, 1980, 6612, 3714, 4508, 3705, 3186, 5664, 2421, 3130, 1744, 772, 3709, 5790, 4275, 6820, 677, 3862, 4508, 3710, 3710, 7000, 5540, 1718, 5162, 7942, 7000, 3714, 2135, 3175, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1,128)

    with tf.Session(graph=tf.Graph()) as sess:
        loaded = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        graph = tf.get_default_graph()
        query = sess.graph.get_tensor_by_name('query_ids:0')
        doc = sess.graph.get_tensor_by_name('doc_ids:0')
        score = sess.graph.get_tensor_by_name('score:0')
        starttime = time.time()
        score = sess.run(score, feed_dict={query:query_input_ids, doc:doc_input_ids})
        print("score: ", score)
        print("*** Please check if this score is correct.")
        return score
    print("*** The saved_model is available.")


if __name__ == "__main__":
    #model_path = '../models/chinese_roberta_wwm_ext_tensorflow/'
    model_path = './models/result_rank_pretrain/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_path", type=str, default=model_path + 'bert_config.json', help="the path of bert_config file") 
    parser.add_argument("--ckpt_to_convert", type=str, default="./models/result_rank",help="the checkpoint of trained model to convert") 
    parser.add_argument("--output_dir", type=str, default='./data/feature_recall/model', help="the path of saved_model") 
    parser.add_argument("--query", type=str, default='美赞臣亲舒一段', help="the path of saved_model") 
    parser.add_argument("--doc", type=str, default='领券满减】美赞臣安婴儿A+亲舒 婴儿奶粉1段850克 0-12个月宝宝', help="the path of saved_model") 
    parser.add_argument("--max_seq_length", type=int, default=113)
    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    #run_save_model(args)
    #test_saved_model(args.output_dir)
    #get_train_inputs(args.output_dir, topn=20)
    #get_dev_inputs(args.output_dir, topn=20)
    query_ids, doc_ids = get_query_doc_embedding('./data/feature_recall', args.query, args.doc)
    query_input_ids = np.array(query_ids).reshape(1, 128)
    doc_input_ids = np.array(doc_ids).reshape(-1, 128)
            #score_result = sess.run(score, feed_dict={query_i:query_input_ids, doc_i:doc_input_ids})
    score = test_saved_model(args.output_dir, query_input_ids, doc_input_ids)
    print("query:",args.query, "doc:", args.doc, "score:",score[0][0])
