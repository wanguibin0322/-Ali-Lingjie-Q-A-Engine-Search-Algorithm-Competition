from check_feature import *

if __name__ == '__main__':
    FEAT = './data/feature_base_recall/'
    if not os.path.exists(FEAT): os.mkdir(FEAT)
    get_embedding(FEAT, 'base')
    load_embedding(FEAT, "doc")

    extract_train_topn(FEAT, 10, 30)
    make_recall_qrels(FEAT)
