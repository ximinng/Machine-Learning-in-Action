# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import jsonlines
from w3lib.html import remove_tags

jsonlines_path = '/Users/ximingxing/PycharmProjects/Machine-Learning-in-Action/examples/news_data_generator/timesall.jl'
news = []
with open(jsonlines_path, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        news.append(item)
        print(item)

# 这里解析完毕之后，每个item已经是dict类型
print("文件大小: {}".format(len(news)))

for i in news[:3]:
    print(i)
    str_ = i
    print('url: {}\ntitle: {}\ninfo: {}\nlabel: {}\ncontent: {}\n'.format(str_['url'],
                                                                          str_['title'],
                                                                          str_['info'],
                                                                          str_['module'],
                                                                          remove_tags(str(str_['content']))))