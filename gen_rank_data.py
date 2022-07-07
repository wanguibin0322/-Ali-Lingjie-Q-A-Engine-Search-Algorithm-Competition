from check_feature import *

if __name__ == '__main__':
    FEAT = './data/feature_recall/'
    if not os.path.exists(FEAT): os.mkdir(FEAT)
    get_embedding(FEAT, 'recall')
    load_embedding(FEAT, "query")

    extract_train_set(FEAT, 10, 20, "top1020", './data/tokenize/trainset/')
    extract_train_set(FEAT, 20, 30, "top2030", './data/tokenize/trainset/')
