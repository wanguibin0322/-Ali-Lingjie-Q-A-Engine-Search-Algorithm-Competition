import csv
from random import shuffle
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from query_preprocess import *

dc_p = re.compile("[a-zA-Z]{1,}")
# {query, en, wdq, wds}


def has_dig_n(a):
    # pattern = re.compile('[0-9a-zA-Z]+')
    # match = pattern.findall(a)
    # if len(match) == 0: return True
    # else: return False
    for i in a:
        if '\u4e00' <= i <= '\u9fa5': return True
    return False


def load_corpus(in_file):
    corpus_ = [preprocess(line[1], default_query_prep_config)
          for line in csv.reader(open(in_file), delimiter='\t')]
    return corpus_


def create_word(corpus, all_en, all_wd, all_corpus):
    for i, cp in enumerate(corpus):
        query = copy.deepcopy(cp)
        try:
            # 大于12 做裁剪
            ens = []
            for e in dc_p.findall(query):
                if len(e) > 12: ens.append(e[:6]); ens.append(e[6:])
                else: ens.append(e)
            all_en.extend(ens)
            ens.sort(key=len, reverse=True)

            for e in ens:
                query = query.replace(e, "")

            wds_q, wds_s = [], []
            wds = [w for w in jieba.lcut(query) if has_dig_n(w) and len(w) > 1]
            all_wd.extend(wds)

            all_corpus.append([ens, wds, cp])
            # 无共现的字

            # 有共现的字

            # 连续数字字母的大于等于3个

            # 连续字母的大于4个

            # for cont in range(3):
            #     q = random.sample(ens, 1) if len(ens) != 0 else []
            #     if len(q) == 0: p = random.sample(wds, 4) if len(wds) <= 4 else random.sample(wds, 2)
            #     else: p = random.sample(wds, 3)
            #     q = "".join(q + p)
            #     writer.writerow([q, cp])
        except:
            print(traceback.format_exc())
            print(i)
            print(cp)
            continue


def create_dataset(path, all_en, all_wd, all_corpus):
    writer = csv.writer(open(path, "w+"))
    writer.writerow(['query', 'doc'])
    enc = Counter(all_en)
    wdc = Counter(all_wd)
    for line in all_corpus:
        en = line[0]
        wd = line[1]
        enl = [e for e in en if enc[e] < 15]
        wdl = [e for e in wd if wdc[e] < 15]
        wdlm = [e for e in wd if wdc[e] >= 15]

        q = []
        if len(enl) > 3: q = random.sample(enl, 3)
        elif len(enl) == 3: q = enl
        elif len(enl) == 2: q = enl + random.sample(enl, 1)
        elif len(enl) == 1: q = enl + enl + [""]
        elif len(enl) == 0: q = ["", "", ""]

        try:
            for i in range(3):
                p = []
                if len(wdl) >= 3:
                    if len(wdlm) >= 2: p = random.sample(wdl, 2) + random.sample(wdlm, 2)
                    elif len(wdlm) >= 1: p = random.sample(wdl, 2) + random.sample(wdlm, 1)
                    else: p = random.sample(wdl, 2)
                elif len(wdl) == 2 or len(wdl) == 1:
                    if len(wdlm) >= 3: p = random.sample(wdl, 1) + random.sample(wdlm, 3)
                    elif len(wdlm) >= 2: p = random.sample(wdl, 1) + random.sample(wdlm, 2)
                    elif len(wdlm) >= 1: p = random.sample(wdl, 1) + random.sample(wdlm, 1)
                    else: p = random.sample(wdl, 1)
                elif len(wdl) == 0:
                    if len(wdlm) >= 5: p = random.sample(wdlm, 5)
                    else: p = wdlm
                qq = "".join([q[i]] + p)
                if len(qq) == 0: print(line[2]); continue
                writer.writerow([qq, line[2]])
        except:
            print(traceback.format_exc())
            print(line[2])
            continue


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
    corpus_ecom = load_corpus('data/ecom/corpus.tsv')
    create_word(corpus_ecom, all_en, all_wd, all_corpus)
    create_dataset('./data/ecom_corpus_pretrain_dataset.csv', all_en, all_wd, all_corpus)








