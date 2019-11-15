# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""

from sklearn import datasets
from mlic import path

news = datasets.fetch_20newsgroups(data_home=path.get('BASE_PATH').__add__('/data'), subset='train')

print(news.DESCR)

