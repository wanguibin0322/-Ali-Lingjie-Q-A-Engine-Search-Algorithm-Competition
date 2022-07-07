import pandas as pd

# original
CORPUS_FILE = './data/ecom/corpus.tsv'
TRAIN_FILE = './data/ecom/train.query.txt'
DEV_FILE = './data/ecom/dev.query.txt'

TRAIN_QUERY_FILE = './data/ecom/qrels.train.tsv'
DEV_QUERY_FILE = './data/ecom/qrels.dev.tsv'
QUERY_DOC_FILE = './data/ecom/train.data.csv'


def gen_ecom():
    corpus_data = pd.read_csv(CORPUS_FILE, sep="\t", encoding='utf-8', header=None, names=["doc_id", "doc"])
    train_data = pd.read_csv(TRAIN_FILE, sep="\t", encoding='utf-8', header=None, names=["query_id", "query"])
    dev_data = pd.read_csv(DEV_FILE, sep="\t", encoding='utf-8', header=None, names=["query_id", "query"])
    train_qrels = pd.read_csv(TRAIN_QUERY_FILE, sep="\t", encoding='utf-8', header=None,
                              names=["query_id", "query_", "doc_id", 'doc_'])
    dev_qrels = pd.read_csv(DEV_QUERY_FILE, sep="\t", encoding='utf-8', header=None,
                            names=["query_id", "query_", "doc_id", 'doc_'])
    # print("dev:",dev_data)

    print(train_qrels.shape)
    train_qrels = pd.merge(train_qrels[["query_id", "doc_id"]], train_data, on='query_id', how='inner')
    dev_qrels = pd.merge(dev_qrels[["query_id", "doc_id"]], dev_data, on='query_id', how='inner')
    print(train_qrels.shape)
    print(dev_qrels.shape)
    train_qrels = pd.merge(train_qrels, corpus_data, on='doc_id', how='inner')
    # print(train_qrels)
    dev_qrels = pd.merge(dev_qrels, corpus_data, on='doc_id', how='inner')
    # print(dev_qrels)
    qrels = train_qrels.append(dev_qrels)
    # corpus_data = corpus_data.set_index("doc_id")
    # train_data = train_data.set_index("query_id")
    # dev_data = dev_data.set_index("query_id")
    # qrels = qrels.set_index("query_id")
    # print(qrels)
    qrels[['query', 'doc']].to_csv(QUERY_DOC_FILE, encoding='utf-8', index=False)


def new_brand(x):
    if x == 'MJX R/C Technic/美嘉欣':
        return "MJX RC Technic/美嘉欣"
    elif x == 'petkit/佩奇/小佩':
        return "petkit佩奇/小佩"
    elif x == 'SHINO//丝诺':
        return "SHINO/丝诺"
    elif x == 'D/IMAGE/第一形象':
        return "D IMAGE/第一形象"
    elif x == 'THE EIGHTY TWENTY/80/20':
        return "THE EIGHTY TWENTY/80 20"
    elif x == 'Enjoy Shopping/－加/印/上/品－':
        return "Enjoy Shopping/加印上品"
    else:
        return x

"""
如果品牌 没有 '/',
    a.brand 在query 里
        return None
    b.
        query+brand
如果品牌有 '/'
    a.brand中文 在query
        query.replace(中文,英文)
    b.brand英文 在query
        query.replace(英文,中文) 
"""

#big.brand.query.corpus.csv
def map_brand(x):
    query = x['query']
    brand = x['brand']
    brand_list = brand.split('/') if pd.notnull(brand) else []
    if len(brand_list) == 2:
        if brand_list[1] in query and (not brand_list[0] in query):
            return query.replace(brand_list[1], brand_list[0])
        elif brand_list[0] in query and (not brand_list[1] in query):
            return query.replace(brand_list[0], brand_list[1])
    elif len(brand_list) == 1:
        if brand_list[0] in query:
            return None
        else:
            return query + brand_list[0]
    return None

#如果brand有'/'
    #英文+类别
#brand 没有'/'
    #brand+类别

#big.1.brand.commodity.corpus.csv
def map_commodity_1(x):
    commodity = x['commodity']
    brand = x['brand']
    brand_list = brand.split('/') if pd.notnull(brand) else []
    if len(brand_list) == 2:
        return brand_list[0] + commodity
    else:
        return brand + commodity

#如果brand有'/'
    #中文+类别
#brand 没有'/'
    #brand+类别

#big.2.brand.commodity.corpus.csv
def map_commodity_2(x):
    commodity = x['commodity']
    brand = x['brand']
    brand_list = brand.split('/') if pd.notnull(brand) else []
    if len(brand_list) == 2:
        return brand_list[1] + commodity
    else:
        return None


def gen_big_pretrain():
    df = pd.read_csv('./data/big_corpus/big_corpus.txt', sep='\t', header=None,
                     names=['doc', 'doc_', 'query', 'query_', 'brand', 'brand_', 'commodity'])
    print("big corpus:", df.shape)
    df['brand'] = df['brand'].map(lambda x: new_brand(x))

    # brand replace query doc
    # df['brand_len'] = df['brand'].map(lambda x:len(x.split('/')))
    # print(df.brand_len.value_counts())
    df['new_query'] = df.apply(lambda x: map_brand(x), axis=1)
    df_brand_query = df[pd.notnull(df.new_query)][['new_query', 'doc']]
    df_brand_query['query'] = df_brand_query['new_query']
    df_brand_query[['query', 'doc']].to_csv('./data/big.brand.query.corpus.csv', index=None)
    print(df_brand_query.shape)
    # query doc
    df[['query', 'doc']].to_csv('./data/big.query.corpus.csv', index=None)

    df['new_1'] = df.apply(lambda x: map_commodity_1(x), axis=1)
    df_1 = df[pd.notnull(df.new_1)][['new_1', 'doc']]
    df_1['query'] = df_1['new_1']
    df_1[['query', 'doc']].to_csv('./data/big.1.brand.commodity.corpus.csv', index=None)
    print(df_1.shape)
    df['new_2'] = df.apply(lambda x: map_commodity_2(x), axis=1)
    df_2 = df[pd.notnull(df.new_2)][['new_2', 'doc']]
    df_2['query'] = df_2['new_2']
    df_2[['query', 'doc']].to_csv('./data/big.2.brand.commodity.corpus.csv', index=None)
    print(df_2.shape)
