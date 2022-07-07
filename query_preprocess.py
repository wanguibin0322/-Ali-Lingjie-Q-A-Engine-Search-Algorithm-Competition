import numpy as np
import pandas as pd
import os
import copy
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
from collections import Counter
import jieba
import re
import random
import traceback
from zhconv import convert
import pdb

default_query_prep_config = {
    "use_list": False,
    "use_jieba": False,
    "ch_to_en": True,
    "to_lower": True,
    "del_all_punc": False,
    "del_some_punc1": True,  # error
    "have_kg": False,
    "have_kg_del_ch": False,
    "use_pinyin": False,
    "delete_mul_char": True,
    "copy_query": False,
    "tongyici": True,
    "delete_dig_char": False,
}

tongyici = {
    "bd云盘": "百度云盘",
    "候王丹": "猴王丹",
}

from pypinyin import pinyin
import pypinyin

yunmuDict = {'a': 'cyma', 'o': 'cymo', 'e': 'cyme', 'i': 'cymi',
            'u': 'cymu', 'v': 'cymv', 'ai': 'cymai', 'ei': 'cymei',
            'ui': 'cymui', 'ao': 'cymao', 'ou': 'cymou', 'iou': 'cymviou',  # 有：you->yiou->iou->iu
            'ie': 'cymie', 've': 'cymve', 'er': 'cymer', 'an': 'cyman',
            'en': 'cymen', 'in': 'cymin', 'un': 'cymun', 'vn': 'cymvn',  # 晕：yun->yvn->vn->ven
            'ang': 'cymang', 'eng': 'cymeng', 'ing': 'cyming', 'ong': 'cymong',
            'van': 'cymvan', 'uai': 'cymuai', 'uan': 'cymuan', 'ia': 'cymia',
            'uen': 'cymuen', 'uo':  'cymuo', 'uei':  'cymuei', 'ua': 'cymua',
            'iao': 'cymiao', 'ian': 'cymian', 'uang': 'cymuang', 'iang': 'cymiang',
            'iong': 'cymiong', 'ueng': 'cymueng',
             }

shengmuDict = {'b': 'csmb', 'p': 'csmp', 'm': 'csmm', 'f': 'csmf',
                    'd': 'csmd', 't': 'csmt', 'n': 'csmn', 'l': 'csml',
                    'g': 'csmg', 'k': 'csmk', 'h': 'csmh', 'j': 'csmj',
                    'q': 'csmq', 'x': 'csmx', 'zh': 'csmzh', 'ch': 'csmch',
                    'sh': 'csmsh', 'r': 'csmr', 'z': 'csmz', 'c': 'csmc',
                    's': 'csms', 'y': 'csmy', 'w': 'csmw', '0': 'csm0',

               }

pydict = {}


def has_dig_n(a):
    # pattern = re.compile('[0-9a-zA-Z]+')
    # match = pattern.findall(a)
    # if len(match) == 0: return True
    # else: return False
    for i in a:
        if '\u4e00' <= i <= '\u9fa5': return True
    return False


def delete_dig_char(q):
    if not has_dig_n(q):
        q = " "
    return q


def tongyici_pro(q):
    for k, v in tongyici.items():
        if k in q:
            q = q.replace(k, v)
    return q


def remove_punctuation(query):
    # punctuation = """！？｡＂*-＃＄％＆＇（）/()＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    # re_punctuation = "[{}]+".format(punctuation)
    # query = re.sub(re_punctuation, "", query)
    q = re.sub("[——()?【】“”！，:. ：+\[\]\-《》；.;/。？、~@#￥%……&*（）]+", "", query)
    # q = query.replace(" ", "")
    q = q.replace(",", "")
    q = q.replace(u'\u200b', '')
    p = re.compile(r"\d{1,4}\*\d{1,4}(?:cm|mm|m|)")
    q = re.sub(p, "", q)
    return q.strip()


def remove_all_punctuation(q):
    q = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", q)
    q = "".join(q)
    q = q.replace(" ", "")
    return q


def remove_multi_char(q):
    q = list(q)
    e = Counter(q).most_common(10)
    qq = list(reversed(copy.deepcopy(q)))
    for p, c in e:
        if ('\u4e00' <= p <= '\u9fa5' or p.isdigit()) and c >= 3:
            idx = qq.index(p); qq.pop(idx)
            idx = qq.index(p); qq.pop(idx)
    q = "".join(reversed(copy.deepcopy(qq)))

    # p = re.compile("[a-zA-Z]{2,}")
    # res = p.findall(q)
    # import pdb; pdb.set_trace()
    return q


def preprocess(query, config, i=-1, type=None):
    if config["use_list"]:
        return preprocess_list_q(query, config, type=type)
    else:
        return preprocess_one_q(query, config, type=type)


def preprocess_list_q(query, config, i=-1, type=None):
    q = query
    qs = []
    if config["ch_to_en"]: q = convert(q, 'zh-cn')
    if config["to_lower"]: q = q.lower()
    if config["del_all_punc"]: q = remove_all_punctuation(q)
    if config["del_some_punc1"]: q = remove_punctuation(q)

    q_1 = copy.deepcopy(q)
    if config["have_kg"]:
        q_1 = " ".join(q_1)
    if config["have_kg_del_ch"]:
        c = [i for i in " ".join(q_1).split() if not '\u4e00' <= i <= '\u9fa5']
        q_1 = q_1 + " " + " ".join(list(set(c)))
    qs.append(q_1)

    if config["use_jieba"]:
        q_2 = copy.deepcopy(q)
        j = jieba.lcut(q_2)

        def has_no_dig_n(a):
            pattern = re.compile('[0-9a-zA-Z]+')
            match = pattern.findall(a)
            if len(match) == 0:
                return True
            else:
                return False
        r = []
        for i in j:
            if has_no_dig_n(i):
                if len(i) >= 2:
                    r.append("WORDBEGIN")
                    r.append(i)
                    r.append("WORDEND")
                else:
                    r.append(i)
        q_2 = "".join(r)
        qs.append(q_2)

    if config["use_pinyin"]:
        q_3 = copy.deepcopy(q)
        q_can = []
        for i in q_3:
            if '\u4e00' <= i <= '\u9fa5' and i not in pydict.keys():
                try:
                    ym = pinyin(i, style=pypinyin.FINALS, heteronym=False, strict=True)[0][0]  # 韵母
                    sm = pinyin(i, style=pypinyin.INITIALS, heteronym=False, strict=False)[0][0]  # 声母
                    sm = shengmuDict[sm] if sm != "" else ""
                    ym = yunmuDict[ym] if ym != "" else ""
                    rs = "{}{}".format(sm, ym)
                    pydict[i] = [sm, ym, rs]
                except Exception as e:
                    print(e)
                    pdb.set_trace()
            if i in pydict.keys():
                q_can.append(pydict[i][2])
        qs.append("".join(q_can))
    return ",".join(qs)


def preprocess_one_q(query, config, i=-1, type=None):
    q = query
    if config["ch_to_en"]: q = convert(q, 'zh-cn')
    if config["to_lower"]: q = q.lower()
    if config["del_all_punc"]: q = remove_all_punctuation(q)
    if config["del_some_punc1"]: q = remove_punctuation(q)
    if config["tongyici"]: q = tongyici_pro(q)
    if type == "query" and config["copy_query"]: q = q + " " + q + " " + q
    if type is None and config["delete_mul_char"]: q = remove_multi_char(q)
    if type is None and config["delete_dig_char"]: q = delete_dig_char(q)

    if config["use_jieba"]:
        j = jieba.lcut(q)
        r = []
        for i in j:
            if len(i) >= 2:
                r.append("WORDBEGIN")
                r.append(i)
                r.append("WORDEND")
            else:
                r.append(i)
        q = "".join(r)
    if config["have_kg"]:
        q = " ".join(q)
        if config["use_jieba"]:
            q = q.replace("w o r d b e g i n", "WORDBEGIN")
            q = q.replace("w o r d e n d", "WORDEND")
    if config["have_kg_del_ch"]:
        c = [i for i in " ".join(q).split() if not '\u4e00' <= i <= '\u9fa5']
        q = q + " " + " ".join(list(set(c))) if len(set(c)) != 0 else q
    if config["use_pinyin"]:
        qq = q
        for i in q:
            if '\u4e00' <= i <= '\u9fa5' and i not in pydict.keys():
                try:
                    ym = pinyin(i, style=pypinyin.FINALS, heteronym=False, strict=True)[0][0]  # 韵母
                    sm = pinyin(i, style=pypinyin.INITIALS, heteronym=False, strict=False)[0][0]  # 声母
                    sm = shengmuDict[sm] if sm != "" else ""
                    ym = yunmuDict[ym] if ym != "" else ""
                    rs = "{}{}{}".format(i, sm, ym)
                    pydict[i] = [sm, ym, rs]
                except Exception as e:
                    print(e)
                    pdb.set_trace()
            if i in pydict.keys():
                qq = qq.replace(i, pydict[i][2])
        q = qq
    return q


if __name__ == "__main__":
    # print(remove_punctuation("神川722602/722609/722609S/7.22615/722302/722618电锤原装转子"))
    # print(preprocess("2 broke boys", default_query_prep_config))
    # print(preprocess("2BrokeBoys复古学院风拼色植绒oversize外套男女宽松情侣加厚ins", default_query_prep_config))
    # print(preprocess("正宗冻梨broke br br 5b 6b东北特产大黑梨", default_query_prep_config))
    print(remove_multi_char("纯铜分茶勺茶刀茶夹茶叶勺茶匙茶侧功夫茶道铲子六君子套装具"))

