#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import jsonlines
from w3lib.html import remove_tags
import pickle


# ## 1. Loading News Dataset

# In[4]:


get_ipython().run_line_magic('time', '')
nowLen = 100
base_path = '/Github/Machine-Learning-in-Action/examples/news_data_generator/news_data_generator/spiders/'
file_name = 'timesall.jl'
data_path = base_path + file_name

train_set = {}
train_set['data'] = []
train_set['target'] = []
raw_counts = 0  # 统计原始数据数量
with open(data_path, "r+", encoding="utf8") as f:
    for i in jsonlines.Reader(f):
        # item: dict 表示一条原始数据
        """组装数据集"""
        title = ''.join(i['title']).lower()  # from list to lower str
        url = ''.join(i['url']).lower()
        if i['type'] != []:
            if i['content'] != []:
                train_set['target'].append(i['type'][0])
                train_set['data'].append(i['content'])
        elif i['module'] != []:
            if i['content'] != []:
                train_set['target'].append(i['module'][0])
                train_set['data'].append(i['content'])
        raw_counts += 1

data_len = len(train_set['data'])
target_len = len(train_set['target'])
print('原始数据集数量: {}'.format(raw_counts))
print('无类别文章个数: {}'.format(raw_counts - data_len))
print('可用数据集数据: {} - {}'.format(data_len, target_len))


# In[5]:


target = train_set['target']
data = train_set['data']

c_max = 0
record_n = []
record_t = []
for i in set(target): # 统计分类下文章个数
    if target.count(i) > c_max:
        c_max = target.count(i)
        record_n.append(str(i))
        record_t.append(i)
    print('{}的个数: {}'.format(i, target.count(i)))


# In[6]:


get_ipython().run_line_magic('time', '')
"""合并小分类 -> 大分类"""
Sport = [
    'Soccer', 'Basketball', 'Baseball', 'Rugby', 'Hockey', 'Tennis', 'Golf',
    'Winter Sport', 'Olympics', 'Motorsport', 'Boxing', 'Eye on the Ball',
    'Track and Field', 'Water Sport', 'Miscellany', 'SPORT', 'SPORTS'
]
Americas = ['Americas', 'AMERICAS']
Europe = ['Europe', 'EUROPE']
Mid_East = ['Mid-East', 'MID_EAST','MID_east']
Economy = ['Eye on The Economy', 'Economy']
Politics = ['Politics', 'POLITICS', 'Politics']
Military = ['Military', 'Diplomacy', 'MILITARY']
Life = ['LIFE', 'PHOTOS', 'life']
Companies = ['COMPANIES', 'Companies']
SouthAsia = ['Central & South Asia', 'CENTRAL & SOUTH ASIA', 'Miscellany']

categorys = [
    Sport, Americas, Europe, Mid_East, Economy, Politics, Military, Life,
    Companies, SouthAsia
]  # 十个类别

print('初始数据集大小: {} - {}'.format(len(train_set['data']),
                                len(train_set['target'])))
SPORT, AMERICAS, EUROPE, MID_EAST, ECONOMY, POLITICS, MILITARY, LIFE,     COMPANY, SOUTHASIA = [],[],[],[],[],[],[],[],[],[]
stores = [
    SPORT, AMERICAS, EUROPE, MID_EAST, ECONOMY, POLITICS, MILITARY, LIFE,
    COMPANY, SOUTHASIA
]
# 分类

for category, store in zip(categorys, stores):
    for index, i in (enumerate(zip(train_set['data'], train_set['target']))):
        for item in category:
            if i[1].lower() in item.lower():
                # 填充分类数据
                store.append(i[0])
                if len(store) > 155000:
                    break;
                #数据集中删除数据
#                 del train_set['data'][index]
#                 del train_set['target'][index]
#     print('个数: {}'.format(len(store)))
#     print('数据集大小: {} - {}'.format(len(train_set['data']),
#                                   len(train_set['target'])))

# 分类持久化数据
category_name = [
    'Sport', 'Americas', 'Europe', 'Mid_East', 'Economy', 'Politics',
    'Military', 'Life', 'Companies', 'SouthAsia'
]
for file_name, file in zip(category_name, stores):
    f = open('{}.pkl'.format(file_name), 'wb')
    print('{}的数量: {}'.format(file_name, len(file)))
    pickle.dump(file, f)
    f.close()


# In[7]:


get_ipython().run_line_magic('time', '')
# 制作数据集
X = []  # list[str]
y = []  # list[int] int: category_name中的下标
for index, fileName in enumerate(category_name):
    fr = open('{}.pkl'.format(fileName), 'rb')
    categoty_data = []
    categoty_data = pickle.load(fr)
    for data in categoty_data:
        X.append(data)
        y.append(index)
    fr.close()
    
f = open('X', 'wb')
pickle.dump(X, f)
f.close()

f = open('y', 'wb')
pickle.dump(y, f)
f.close()

for i in set(y): # 统计分类下文章个数
    print('%s的个数：%s' %(i,y.count(i)))


# ## Train Test dataset split

# In[8]:


from sklearn.model_selection import train_test_split

X = []
y = []
fr_X = open('X_now.pkl', 'rb')
X = pickle.load(fr_X)
fr_X.close()
fr_y = open('y_now.pkl', 'rb')
y = pickle.load(fr_y)
fr_y.close()

# X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print('训练集大小:{}'.format(len(X_train)))
print('测试集大小:{}'.format(len(X_test)))


# ## 2. Data preprocessing

# ### Data Cleaning:  regular expersion

# In[9]:


get_ipython().run_line_magic('time', '')
import re

r1 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+" "'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
r2 = '[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+'

"""
sentence = "hello! wo?rd!. \n\n\n\n      \n ,,,,,\t\t\t\t ////t/t/t/t /n/n//n"

cleanr = re.compile('<.*?>') # 匹配HTML标签规则
sentence = re.sub(cleanr, '', sentence)  # 去除HTML标签
sentence = re.sub(r1, '', sentence) 
sentence = re.sub(r2, '', sentence)
print(sentence)
"""
def find_unchinese(file):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    unchinese = re.sub(pattern,"",file)
    return unchinese

cleanr = re.compile('<.*?>') # 匹配HTML标签规则
for index, sentence in enumerate(X_train):
    sentence = ''.join(sentence) # list -> str
    sentence = re.sub(cleanr, '', sentence)  # 去除HTML标签
    sentence = re.sub(r1, '', sentence) 
    sentence = re.sub(r2, '', sentence)
    sentence = sentence.lower() # to lower case
    find_unchinese(sentence)
    X_train[index] = sentence
    
for index, sentence in enumerate(X_test):
    sentence = ''.join(sentence) # list -> str
    sentence = re.sub(cleanr, '', sentence)  # 去除HTML标签
    sentence = re.sub(r1, '', sentence) 
    sentence = re.sub(r2, '', sentence)
    sentence = sentence.lower()
    find_unchinese(sentence)
    X_test[index] = sentence


# In[10]:


print(X_train[211])
print(X_test[211])


# ### Data Cleaning: stop words

# In[11]:


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

X_train_sw = X_train[:200]
"""匹配停用词"""
for index, item in tqdm(enumerate(X_train_sw)):
    X_train[index] = " ".join([
        w for w in X_train[index].split(' ')
        if w not in stopwords.words('english')
    ])
print(X_train[20])

# for index, item in tqdm(enumerate(X_test)):
#     X_test[index] = " ".join([
#         w for w in X_test[index].split(' ')
#         if w not in stopwords.words('english')
#     ])
# print(X_test[90001])


# ### Normalization: lemmatization

# In[12]:


"""stemming -- 词干提取(no use)"""
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english") # 选择语言
# stemmer.stem("leaves") # 词干化单词
X_stem = X_train[:nowLen]
for index, item in tqdm(enumerate(X_stem)):
    X_stem[index] = stemmer.stem(item)

"""lemmatization -- 词型还原(use)"""
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
wnl = WordNetLemmatizer()
# print(wnl.lemmatize('leaves'))

# for i in tqdm(range(len(train_set.data))):
#     train_set[i] = wnl.lemmatize(train_set.data[i])    
# print(train_set.data[:1])

for index, item in tqdm(enumerate(X_train)):
    X_train[index] = wnl.lemmatize(item)
for index, item in tqdm(enumerate(X_test)):
    X_test[index] = wnl.lemmatize(item)


# In[13]:


print("词干提取:")
print(X_stem[nowLen -1])

print("词型还原:")
print("{}".format(X_train[211]))
print("{}".format(X_test[211]))


# ### Extracting Features

# In[14]:


get_ipython().run_line_magic('time', '')
"""build data dict"""
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()  # 特征向量计数函数
X_train_counts = count_vect.fit_transform(X_train)  # 对文本进行特征向量处理
print(X_train_counts.shape)
print(X_train_counts[211])

"""TF-IDF: Term Frequency-Inverse Document Frequency"""
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print('X_train_tfidf shape: {}'.format(X_train_tfidf.shape))
print(X_train_tfidf[211])


# ## 3. Bayes Classifier

# ### 3.1 Train Bayes

# In[15]:


get_ipython().run_line_magic('time', '')
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

multinomialNB_pipeline = Pipeline([('Vectorizer',
                                    CountVectorizer(stop_words='english',
                                                    ngram_range=(1,1),
                                                    max_df=0.5)),
                                   ('TF_IDF', TfidfTransformer(use_idf=True)),
                                   ('MultinomialNB', MultinomialNB(alpha=1e-4))]) # 1e-5
multinomialNB_pipeline.fit(X_train, y_train)
print(" Show gaussianNB_pipeline:\n", multinomialNB_pipeline)


# ### 3.2 Evaluation Bayes

# In[16]:


get_ipython().run_line_magic('time', '')
from sklearn.metrics import classification_report, confusion_matrix

# test_set = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
# docs_test = test_set.data
predicted = multinomialNB_pipeline.predict(X_test)

print(
    classification_report(y_test,
                          predicted,
                          digits=4,
                          target_names=category_name))


# In[17]:


# calculate confusion_matrix and plot it
confusion_mat = confusion_matrix(y_test, predicted)
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(confusion_mat,
            annot=True,
            fmt='d',
            cmap="RdPu",
            linewidths=.3,
            xticklabels=category_name,
            yticklabels=category_name)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# ## 4. SVM Classifier

# ### 4.1 Train svm

# In[18]:


get_ipython().run_line_magic('time', '')
from sklearn.linear_model import SGDClassifier

SGDClassifier_pipline = Pipeline([('Vectorizer',
                                   CountVectorizer(stop_words='english',
                                                   ngram_range=(1,3),
                                                   max_df=0.5)),
                                  ('TF_IDF', TfidfTransformer(use_idf=True)),
                                  ('SGDClassifier',
                                   SGDClassifier(loss='hinge', # linear SVM
                                                 penalty='l2', 
                                                 alpha=1e-6, # 1e-6
                                                 n_jobs = -1,
                                                 random_state=42))])
SGDClassifier_pipline.fit(X_train, y_train)
print(" Show SGDClassifier_pipline:\n", SGDClassifier_pipline)


# ### 4.2 Evaluation SVM

# In[19]:


get_ipython().run_line_magic('time', '')
predicted_2 = SGDClassifier_pipline.predict(X_test)
print(classification_report(y_test, predicted_2, digits=4, target_names=category_name))


# In[20]:


# calculate confusion_matrix and plot it
confusion_mat_2 = confusion_matrix(y_test, predicted_2)
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(confusion_mat_2,
            annot=True,
            fmt='d',
            cmap="RdPu",
            linewidths=.3,
            xticklabels=category_name,
            yticklabels=category_name)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# ## Grid Search

# In[1]:


get_ipython().run_line_magic('time', '')
from sklearn.model_selection import GridSearchCV
from  sklearn.model_selection import RandomizedSearchCV

bayes_params = {
#     'Vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
#     'TF_IDF__use_idf': (True, False),
    'MultinomialNB__alpha': (1e-3, 1e-4, 1e-5),
    'MultinomialNB__fit_prior' : (True, False),
}

grid = GridSearchCV(multinomialNB_pipeline, bayes_params, cv=5, iid=False, n_jobs=25)
# grid = RandomizedSearchCV(multinomialNB_pipeline, bayes_params, cv=5, iid=False, refit=True, n_jobs=25)
grid.fit(X_train, y_train)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# In[2]:


get_ipython().run_line_magic('time', '')
svm_params = {
#     'Vectorizer__ngram_range': [(1, 2), (1, 3), (1, 4)],
#     'TF_IDF__use_idf': (True, False),
#     'SGDClassifier__loss': ('hinge', 'squared_hinge'),
#     'SGDClassifier__penalty': ('l2', 'elasticnet'),
    'SGDClassifier__alpha': (1e-4, 1e-5, 1e-6, 1e-7),
#     'SGDClassifier__power_t': (0.4, 0.5, 0.6)
}

# grid = GridSearchCV(SGDClassifier_pipline, svm_params, cv=5, iid=False, refit=True ,n_jobs=25)
grid = RandomizedSearchCV(SGDClassifier_pipline, svm_params, cv=5, iid=False, refit=True, n_jobs=25)
grid.fit(X_train, y_train)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# ## 模型持久化

# In[27]:


from sklearn.externals import joblib

joblib.dump(multinomialNB_pipeline, 'multinomialNB_pipeline.pkl') 
multinomialNB_pipeline = joblib.load('multinomialNB_pipeline.pkl') 


# In[28]:


from sklearn.externals import joblib

joblib.dump(SGDClassifier_pipline, 'SGDClassifier_pipline.pkl') 
SGDClassifier_pipline = joblib.load('SGDClassifier_pipline.pkl') 


# In[ ]:




