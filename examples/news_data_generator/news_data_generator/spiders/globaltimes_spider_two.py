# -*- coding: utf-8 -*-
import scrapy
from ..items import GlobalTimesArticleItems
from scrapy_splash import SplashRequest
from w3lib.html import remove_tags
import time

"""
Use Scrapy-Splash get dynamic website(which built by js).
1. Start Scrapy-Splash: 
                       docker run -p 8050:8050 scrapinghub/splash
                    
2. debug crawl:
                scrapy shell 'http://localhost:8050/render.html?url=http://globaltimes.cn/&timeout=10&wait=0.5'
                
3. run crawl:
            scrapy crawl times -o items.jl        
            
            {'china': 'http://globaltimes.cn/china', 
            'source': 'http://globaltimes.cn/source', 
            'world': 'http://globaltimes.cn/world', 
            'opinion': 'http://globaltimes.cn/opinion', 
            'life': 'http://globaltimes.cn/life', 
            'arts': 'http://globaltimes.cn/arts', 
            'sci-tech': 'http://globaltimes.cn/sci-tech', 
            'odd': 'http://globaltimes.cn/odd', 
            'sport': 'http://globaltimes.cn/sport', 
            'metro': 'http://globaltimes.cn/metro'}    
"""


class GlobaltimesSpiderSpider(scrapy.Spider):
    name = 'timetwo'

    # allowed_domains = ['globaltimes.cn']

    def __init__(self, *args, **kwargs):
        super(GlobaltimesSpiderSpider, self).__init__(*args, **kwargs)
        self.args = {
            # optional; parameters passed to Splash HTTP API
            'wait': 20,
            # 'lua_source': lua_script,  # when endpoint="execute"
            # 'url' is prefilled from request url
            # 'http_method' is set to 'POST' for POST requests
            # 'body' is set to request body for POST requests
        }
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        }

    def start_requests(self):
        urls = {'china': 'http://globaltimes.cn/china',
                'source': 'http://globaltimes.cn/source',
                'world': 'http://globaltimes.cn/world',
                'opinion': 'http://globaltimes.cn/opinion',
                'life': 'http://globaltimes.cn/life',
                'arts': 'http://globaltimes.cn/arts',
                'sci-tech': 'http://globaltimes.cn/sci-tech',
                'odd': 'http://globaltimes.cn/odd',
                'sport': 'http://globaltimes.cn/sport',
                'metro': 'http://globaltimes.cn/metro'}
        for site, url in urls.items():
            yield scrapy.Request(url, callback=self.parse,
                                 dont_filter=True,
                                 meta={'label': site},
                                 headers=self.headers)

    def parse(self, response):
        """
        处理分类,调用 parse_item 爬取信息
        :param response: start_url的相应
        :return:
        """
        label = response.meta['label']
        urls = response.xpath('//@href').getall()
        article_urls = self._select_article_url(urls)
        for article in article_urls:
            self.log("正在处理链接: {}".format(article))
            yield scrapy.Request(article, callback=self.parse_article,
                                 meta={'label': label},
                                 headers=self.headers)

    def parse_article(self, response):
        """
        处理具体的文章
        :param response:
        :return: the instance of GlobalTimesArticleItems.
        """
        item = GlobalTimesArticleItems()
        item['label'] = response.meta['label']
        item['url'] = response.url
        self.log("正在处理文章 {}".format(response.url))

        title = response.xpath('//div[@class="row-fluid article-title"]/h3/text()').getall()[0]
        if title:
            item['title'] = str(title)
        content = response.xpath('//div[@class="span12 row-content"]/text()').extract()
        if isinstance(content, list):
            item['content'] = ''.join(content)

        self.log("title: {}".format(title))
        yield item

    def _select_article_url(self, urls):
        res = [url for url in urls if url and 'content' in url]
        return res
