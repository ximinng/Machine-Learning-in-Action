# -*- coding: utf-8 -*-
import scrapy
from ..items import GlobaltimesItem

"""
Use Scrapy-Splash get dynamic website(which built by js).
1. Start Scrapy-Splash: 
                       docker run -p 8050:8050 scrapinghub/splash
                    
2. debug crawl:
                scrapy shell 'http://localhost:8050/render.html?url=http://globaltimes.cn/&timeout=10&wait=0.5'
                
3. run crawl:
            scrapy crawl times -o items.jl        
            
            {'china': 'http://globaltimes.cn/china', 'source': 'http://globaltimes.cn/source', 'world': 'http://globaltimes.cn/world', 'opinion': 'http://globaltimes.cn/opinion', 'life': 'http://globaltimes.cn/life', 'arts': 'http://globaltimes.cn/arts', 'sci-tech': 'http://globaltimes.cn/sci-tech', 'odd': 'http://globaltimes.cn/odd', 'sport': 'http://globaltimes.cn/sport', 'metro': 'http://globaltimes.cn/metro'}    
"""


class GlobaltimesSpiderSpider(scrapy.Spider):
    name = 'timesall'
    allowed_domains = ['globaltimes.cn']
    start_urls = ['http://globaltimes.cn/']

    def parse(self, response):
        base_url = "http://www.globaltimes.cn/content/"

        # 循环新的url请求加入待爬队列，并调用回调函数 parse_page 1172897
        for page in range(1, 1136094):
            self.log(base_url + str(page) + '.shtml')
            yield scrapy.Request(base_url + str(page) + '.shtml', dont_filter=True, callback=self.parse_page)

    def parse_page(self, response):
        item = GlobaltimesItem()
        item['url'] = response.url
        item['title'] = response.xpath('//*[@id="left"]/div[2]/h3/text()').extract()
        item['info'] = response.xpath('//*[@id="left"]/div[3]/div[1]/text()').extract()
        item['module'] = response.xpath('//*[@id="left"]/div[1]/a/text()').extract()
        item['type'] = response.xpath('//*[@id="left"]/div[5]/div[1]/a/text()').extract()
        item['content'] = response.xpath('//*[@id="left"]/div[4]').extract()
        # content = response.xpath('//div[@class="span12 row-content"]/text()').extract()
        yield item
