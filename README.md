<h1 id="mlic" align="center">Machine Learning in Action</h1>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue" alt="Pyhton 3">
    </a>
    <a href="http://www.apache.org/licenses/">
        <img src="https://img.shields.io/badge/license-Apache-blue" alt="GitHub">
    </a>
    <a href="#">
        <img src="https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=flat-square" alt="welcome">
    </a>
</p>

<p align="center">
    <a href="#clipboard-getting-started">快速开始 - Getting Started</a> •
    <a href="#table-of-contents">内容 - Table of Contents</a> •
    <a href="#about">关于 - About</a> •
    <a href="#acknowledgment">鸣谢 - Acknowledgment</a> •
    <a href="#speech_balloon-faq">FAQ</a> •
</p>

<h6 align="center">Made by ximing Xing • :milky_way: 
<a href="https://ximingxing.github.io/">https://ximingxing.github.io/</a>
</h6>

Machine-Learning-in-Action是基于Peter Harrington的<<机器学习实战>>这本书, 将书中的机器学习算法和案例以scikit-learn的代码组织形式呈现.

除了算法机器学习算法之外，更重要的是算法的使用场景，这个代码仓库中也提供机器学习[实战案例](#center)。包括：百万英文新闻文本分类实战等.

Machine-Learning-in-Action is based on Peter Harrington`s <<Macine Learning in Action>> , 
The machine learning algorithms and cases in the book are presented in the form of scikit-learn code organization.

<h2 align="center">:clipboard: 快速开始 -  Getting Started</h2>

1. Using [Pycharm with conda plugin](https://www.jetbrains.com/pycharm/promo/anaconda/) IDE makes getting started easier.

    - check out from version control.

    - chose Git.
    
2. python setup.py --develop

<h2 align="center">内容 - Table of Contents</h2>
<p align="right"><a href="#mlic"><sup>▴ Back to top</sup></a></p>

```
├── LICENSE
├── README.md
├── data
│   └── 20news-bydate_py3.pkz
├── examples
│   └── 20newsgroup_in_action
├── mlic
│   ├── cluster
│   ├── linear_model
│   ├── metrics
│   ├── naive_bayes
│   ├── neighbors
│   ├── neural_network
│   ├── svm
│   ├── tree
│   └── utils
├── requestments.txt
├── setup.py
└── tests
    ├── Bayes
    ├── KNN
    └── Linear
```

<h2 align="center">关于 - About</h2>

完整的数据挖掘过程 :
1. 网络爬虫 Network Crawler
    - 按分类爬取环球网英文本新闻(处理静态网站与需要js渲染的内容)
    - Scrapy-Splash based Crawler crawls information from [globaltimes.cn](http://www.globaltimes.cn/)
    - CNN Crawler
    - BBC Crawler

2. 文本分类实战

    - DataLoader : 20newsgroup
    - Data preprocessing
        - Data Cleaning: regular expression
        - Data Cleaning: stop words
        - Normalization: lemmatization
    - Extracting Features : Word Dict and TF-IDF
    - Model: bayes and svm
    - Evaluation
    
    <div style="margin-left: 30px">
    <img src="https://github.com/ximingxing/Images/raw/master/mlic/svm_pipline.jpg" width="700px" height="500px">
    <img src="https://github.com/ximingxing/Images/raw/master/mlic/report.jpg" width="600px" height="500px">    
    <br/>
    <img src="https://github.com/ximingxing/Images/raw/master/mlic/confusion.jpg" width="600px" height="600px">
    </div>    

你无需担心`example/`与`tests/`中案例所使用的数据集，因为数据集都是自动下载的.

如果有问题也希望你指出.

<h2 align="center">鸣谢 - Acknowledgment</h2>
<p align="right"><a href="#mlic"><sup>▴ Back to top</sup></a></p>

* 代码组织参考[scikit-learn](https://github.com/scikit-learn/scikit-learn)中的组织结构

<h2 align="center">:speech_balloon: FAQ</h2>
<p align="right"><a href="#mlic"><sup>▴ Back to top</sup></a></p>
