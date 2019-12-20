# -*- coding: utf-8 -*-
"""
   Description :   data augment
   Author :        xxm
"""

import http.client
import hashlib
import urllib
import random
import json
import numpy as np
import re


def baidu_translate(Sentences: str, fromLang: str, toLang: str) -> json:
    """
    回译
    :param Sentences:
    :param fromLang:
    :param toLang:
    :return:
    """
    appid = '20191216000366743'  # 填写你的appid
    secretKey = 'meHxl8e9jCRoXebnB1Bx'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = fromLang  # 原文语种
    toLang = toLang  # 译文语种
    salt = random.randint(32768, 65536)
    q = Sentences
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        return result

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def shuffle(d):
    return np.random.permutation(d)


def shuffle2(d):
    len_ = len(d)
    times = 2
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]], d[index[1]] = d[index[1]], d[index[0]]
    return d


def dropout(d, p=0.4):
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d


def clean(xx):
    xx2 = re.sub(r'\?', "", xx)
    xx1 = xx2.split(' ')
    return xx1


def shuffle_with_drop(X: list, y: list):
    """
    shuffle and drop randomly
    :param X:  input datas
    :param y:  input labels
    :return:
    """
    l = len(X)
    for i in range(l):
        item = clean(X[i])
        d1 = shuffle2(item)
        d11 = ' '.join(d1)
        d2 = dropout(item)
        d22 = ' '.join(d2)
        X.extend([d11, d22])
        y.extend([y[i], y[i]])
    return X, y


if __name__ == '__main__':
    sentence = 'The interim leader will be in a mission to organize democratic elections, said the agreement received by Xinhua correspondents, signed between the junta\'s captain Amadou Sanogo and mediators.'
    lang = ['zh', 'jp', 'kor', 'vie']
    res = []
    for l in lang:
        res.append(
            baidu_translate(sentence, 'en', l)['trans_result'][0]['dst']
        )

    for se in res:
        print(
            baidu_translate(se, 'auto', 'en')['trans_result'][0]['dst']
        )

    sentences = [
        'The interim leader will be in a mission to organize democratic elections, said the agreement received by Xinhua correspondents, signed between the junta\'s captain Amadou Sanogo and mediators.']
    Xp, yp = shuffle_with_drop(sentences, [1])
