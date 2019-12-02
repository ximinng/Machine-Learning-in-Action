#!/usr/bin/env python
# coding: utf-8

# In[107]:


# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import numpy as np
import pandas as pd
from tqdm import tqdm


# ##  1. Loading Dataset (20 newsgroups dataset)

# In[102]:


from sklearn.datasets import fetch_20newsgroups

train_set = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

print(train_set.data[:1])


# ## 2. Data preprocessing

# ### Data Cleaning:  regular expersion

# In[103]:


import re

# 过滤不了\\ \ 中文（）还有
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  #用户也可以在此进行自定义过滤字符
# 者中规则也过滤不完全
r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
# \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
# 去掉括号和括号内的所有内容
r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+" "'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

sentence = "hello! wo?rd!. \n"
cleanr = re.compile('<.*?>') # 匹配HTML标签规则
sentence = re.sub(cleanr, ' ', sentence)  # 去除HTML标签
sentence = re.sub(r4, '', sentence)
print(sentence)

for i in tqdm(range(len(train_set.data))):
    train_set.data[i] = re.sub(cleanr, ' ', train_set.data[i])
    train_set.data[i] = re.sub(r4, '', train_set.data[i])
    train_set.data[i] = re.sub('\n\r', '', train_set.data[i]) # TODO: 这里并没有去掉\n
    train_set.data[i] = train_set.data[i].lower()
print(train_set.data[:1])


# ### Data Cleaning: stop words

# In[104]:


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

# In[105]:


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

# In[111]:


from sklearn.feature_extraction.text import CountVectorizer
"""build data dict"""
count_vect = CountVectorizer()  # 特征向量计数函数
X_train_counts = count_vect.fit_transform(train_set.data)  # 对文本进行特征向量处理

print(X_train_counts[:1])
"""TF-IDF: Term Frequency-Inverse Document Frequency"""
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf[:0])
print(tfidf_transformer)


# ## 3. Bayes Classifier

# ### 3.1 Train Bayes

# In[112]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

multinomialNB_pipeline = Pipeline([('Vectorizer', CountVectorizer()),
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

# In[113]:


from sklearn.metrics import classification_report, confusion_matrix

test_set = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = test_set.data
predicted = multinomialNB_pipeline.predict(docs_test)
print(np.mean(predicted == test_set.target))

print(
    classification_report(test_set.target,
                          predicted,
                          target_names=test_set.target_names))
print(confusion_matrix(test_set.target, predicted))


# ## 4. SVM Classifier

# ### 4.1 Train svm

# In[84]:


from sklearn.linear_model import SGDClassifier

SGDClassifier_pipline = Pipeline([('Vectorizer', CountVectorizer()),
                                  ('TF_IDF', TfidfTransformer()),
                                  ('SGDClassifier',
                                   SGDClassifier(loss='hinge',
                                                 penalty='l2',
                                                 alpha=1e-3,
                                                 random_state=42))])
print(" Show SGDClassifier_pipline:\n", SGDClassifier_pipline)


# ### 4.2 Evaluation SVM

# In[85]:


SGDClassifier_pipline.fit(train_set.data, train_set.target)
predicted = SGDClassifier_pipline.predict(docs_test)
print(classification_report(test_set.target, predicted, target_names=test_set.target_names))
print(confusion_matrix(test_set.target, predicted))


# In[ ]:




