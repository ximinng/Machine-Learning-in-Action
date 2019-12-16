# !/usr/bin/env python
# coding: utf-8

"""
   Description : NLP 文本数据增强
   Author :        xxm
"""

import http.client
import hashlib
import urllib
import random
import json


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

        print(result)
        return result

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


if __name__ == '__main__':
    sentence = 'The interim leader will be in a mission to organize democratic elections, said the agreement received by Xinhua correspondents, signed between the junta\'s captain Amadou Sanogo and mediators.'
    lang = ['zh', 'jp', 'kor', 'vie']
    res = []
    for l in lang:
        res.append(
            baidu_translate(sentence, 'en', l)['trans_result'][0]['dst']
        )

    for se in res:
        baidu_translate(se, 'auto', 'en')
