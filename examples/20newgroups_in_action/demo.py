#!/usr/bin/env python
# coding: utf-8

# In[158]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import jsonlines
from w3lib.html import remove_tags
import pickle

# ##  1. Loading Dataset (20 newsgroups dataset)

# In[175]:


from sklearn.datasets import fetch_20newsgroups

train_set = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

train_set.target

# ## 1. Loading News Dataset

# In[159]:


base_path = '/Github/Machine-Learning-in-Action/examples/news_data_generator/news_data_generator/spiders/'
file_name = 'timesall.jl'
data_path = base_path + file_name

news = []
with open(data_path, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        news.append(item)
len(news)

# In[210]:


## 小数据集用作测试
test = news[:2000]
print(len(test))

# In[ ]:


from functools import reduce


def list_dict_duplicate_removal(data_list):
    run_function = lambda x, y: x if y in x else x + [y]
    return reduce(run_function, [[], ] + data_list)


news_set = list_dict_duplicate_removal(news)
len(news_set)

# In[173]:


news[0]

# In[172]:


remove_tags(''.join(news[0]['content']))

# In[ ]:


X = []
y = []
for i in news_set:
    X.append(i['content'])
    y.append(i['module'])

# ## Train Test dataset split

# In[ ]:


from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# ## 2. Data preprocessing

# ### Data Cleaning:  regular expersion

# In[103]:


import re

# 过滤不了\\ \ 中文（）还有
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
# 者中规则也过滤不完全
r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
# \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
# 去掉括号和括号内的所有内容
r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+" "'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

sentence = "hello! wo?rd!. \n"
cleanr = re.compile('<.*?>')  # 匹配HTML标签规则
sentence = re.sub(cleanr, ' ', sentence)  # 去除HTML标签
sentence = re.sub(r4, '', sentence)
print(sentence)

for i in tqdm(range(len(train_set.data))):
    train_set.data[i] = re.sub(cleanr, ' ', train_set.data[i])
    train_set.data[i] = re.sub(r4, '', train_set.data[i])
    train_set.data[i] = re.sub('\n\r', '', train_set.data[i])  # TODO: 这里并没有去掉\n
    train_set.data[i] = train_set.data[i].lower()
print(train_set.data[:1])

# ### Data Cleaning: stop words

# In[121]:


import nltk
from nltk.tokenize import word_tokenize

# nltk.download()
# nltk.download('stopwords')
"""引入停用词表"""
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
print('English Stop Words List：\n', stop)

# sentence = "this is a apple"
# filter_sentence = [
#     w for w in sentence.split(' ') if w not in stopwords.words('english')
# ]
# print(filter_sentence)

"""匹配停用词"""
for i in tqdm(range(len(train_set.data))):
    train_set.data[i] = " ".join([
        w for w in train_set.data[i].split(' ')
        if w not in stopwords.words('english')
    ])
print(train_set.data[:1])

# ### Normalization: lemmatization

# In[122]:


"""stemming -- 词干提取(no use)"""
from nltk.stem import SnowballStemmer

# stemmer = SnowballStemmer("english") # 选择语言
# stemmer.stem("leaves") # 词干化单词

"""lemmatization -- 词型还原(use)"""
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
wnl = WordNetLemmatizer()
# print(wnl.lemmatize('leaves'))

for i in tqdm(range(len(train_set.data))):
    train_set[i] = wnl.lemmatize(train_set.data[i])

print(train_set.data[:1])

# ### Extracting Features

# In[130]:


from sklearn.feature_extraction.text import CountVectorizer

"""build data dict"""
count_vect = CountVectorizer()  # 特征向量计数函数
X_train_counts = count_vect.fit_transform(train_set.data)  # 对文本进行特征向量处理

print(X_train_counts[:0])
"""TF-IDF: Term Frequency-Inverse Document Frequency"""
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf[:0])
print(tfidf_transformer)

# ## 3. Bayes Classifier

# ### 3.1 Train Bayes

# In[134]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

multinomialNB_pipeline = Pipeline([('Vectorizer',
                                    CountVectorizer(stop_words='english',
                                                    max_df=0.5)),
                                   ('TF_IDF', TfidfTransformer()),
                                   ('MultinomialNB', MultinomialNB())])
multinomialNB_pipeline.fit(train_set.data, train_set.target)
print(" Show gaussianNB_pipeline:\n", multinomialNB_pipeline)

# In[6]:


# 自定义文档测试分类器
docs_new = ['God is love', 'OpenGL on the GPU is fast']  # 文档

predicted = multinomialNB_pipeline.predict(docs_new)

print(predicted)  # 预测类别 [3 1]，一个属于3类，一个属于1类
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train_set.target_names[category]))

# ### 3.2 Evaluation Bayes

# In[145]:


from sklearn.metrics import classification_report, confusion_matrix

test_set = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = test_set.data
predicted = multinomialNB_pipeline.predict(docs_test)

print(
    classification_report(test_set.target,
                          predicted,
                          target_names=test_set.target_names))

# calculate confusion_matrix and plot it
confusion_mat = confusion_matrix(test_set.target, predicted)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_mat,
            annot=True,
            fmt='d',
            xticklabels=test_set.target_names,
            yticklabels=test_set.target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ## 4. SVM Classifier

# ### 4.1 Train svm

# In[136]:


from sklearn.linear_model import SGDClassifier

SGDClassifier_pipline = Pipeline([('Vectorizer',
                                   CountVectorizer(stop_words='english',
                                                   max_df=0.5)),
                                  ('TF_IDF', TfidfTransformer()),
                                  ('SGDClassifier',
                                   SGDClassifier(loss='hinge',
                                                 penalty='l2',
                                                 alpha=1e-3,
                                                 random_state=42))])
print(" Show SGDClassifier_pipline:\n", SGDClassifier_pipline)

# ### 4.2 Evaluation SVM

# In[137]:


SGDClassifier_pipline.fit(train_set.data, train_set.target)
predicted = SGDClassifier_pipline.predict(docs_test)
print(classification_report(test_set.target, predicted, target_names=test_set.target_names))
print(confusion_matrix(test_set.target, predicted))

# ### 4.3 SVM

# In[138]:


from sklearn.svm import LinearSVC

SVC_pipline = Pipeline([('Vectorizer', CountVectorizer()),
                        ('TF_IDF', TfidfTransformer()),
                        ('SVCClassifier', LinearSVC(random_state=42))])
print(" Show SVC_pipline:\n", SVC_pipline)

# In[139]:


from sklearn.metrics import classification_report, confusion_matrix

test_set = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = test_set.data
SVC_pipline.fit(train_set.data, train_set.target)
predicted = SVC_pipline.predict(docs_test)
print(classification_report(test_set.target, predicted, target_names=test_set.target_names))
print(confusion_matrix(test_set.target, predicted))

# ## GridSearch

# In[205]:


from sklearn.model_selection import GridSearchCV

bayes_params = {
    'Vectorizer__ngram_range': [(1, 1), (1, 2)],
    'TF_IDF__use_idf': (True, False),
    'MultinomialNB__alpha': (1e-2, 1e-3),
}

grid = GridSearchCV(multinomialNB_pipeline, bayes_params, cv=5, iid=False, n_jobs=-1)
grid.fit(train_set.data, train_set.target)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# In[207]:


svm_params = {
    'Vectorizer__ngram_range': [(1, 1), (1, 2)],
    'TF_IDF__use_idf': (True, False),
    'SGDClassifier__alpha': (1e-2, 1e-3),
}

grid = GridSearchCV(SGDClassifier_pipline, svm_params, cv=5, iid=False, n_jobs=-1)
grid.fit(train_set.data, train_set.target)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# In[ ]:
