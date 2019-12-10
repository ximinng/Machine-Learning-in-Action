# -*- coding: utf-8 -*-
import scrapy
from ..items import GlobalTimesArticleItems
from scrapy_splash import SplashRequest
from w3lib.html import remove_tags

"""
Use Scrapy-Splash get dynamic website(which built by js).
1. Start Scrapy-Splash: 
                       docker run -p 8050:8050 scrapinghub/splash
                    
2. debug crawl:
                scrapy shell 'http://localhost:8050/render.html?url=http://globaltimes.cn/&timeout=10&wait=0.5'
                
3. run crawl:
            scrapy crawl times -o items.jl            
"""


class GlobaltimesSpiderSpider(scrapy.Spider):
    name = 'times'

    # allowed_domains = ['globaltimes.cn']

    def __init__(self, category=None, *args, **kwargs):
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
        lua_script = '''
        function main(splash, args)
            local ok, reason = splash:go(args.url)
        end
        '''

        urls = [
            'http://globaltimes.cn/',
        ]
        for url in urls:
            # yield scrapy.Request(url=url, callback=self.parse)
            yield SplashRequest(url, self.parse, args=self.args,
                                endpoint='render.html',  # endpoint="execute" 执行lua
                                splash_headers=self.headers)

    def parse(self, response):
        """
        处理分类,调用 parse_item 爬取信息
        :param response: start_url的相应
        :return:
        """
        category: list[str] = response.xpath(
            "//div[@class='nav-collapse collapse nav-channels']/ul/li[@class='dropdown']/a/text()"
        ).getall()
        # sites: list[str] = ['http://globaltimes.cn/' + i.lower() for i in category[:-2]]
        sites = {i.lower(): 'http://globaltimes.cn/' + i.lower() for i in category[:-2]}

        self.log(sites)

        for site, url in sites:
            yield scrapy.Request(url, callback=self.parse_site(response, label=site), headers=self.headers)

        # if category:
        #     complete_url = response.urljoin(next_url)  # 构造了翻页的绝对url地址
        #     yield SplashRequest(complete_url, args={'timeout': 8, 'images': 0})
        # self.log('Saved file %s' % category)
        # yield category

    def parse_site(self, response, label: str):
        """
        处理分类后的网站
        :param label: 网站分类
        :param response: 分类网站的相应
        :return:
        """
        urls = response.xpath('//@href').getall()
        article_urls = self._select_article_url(urls)
        for article in article_urls:
            yield scrapy.Request(article, callback=self.parse_article(response, label), headers=self.headers)

    def parse_article(self, response, label: str):
        """
        处理具体的文章
        :param response:
        :return: the instance of GlobalTimesArticleItems.
        """
        title: str = response.xpath('//div[@class="row-fluid article-title"]/h3/text()').getall()[0].strip()
        content: str = str(response.xpath('//div[@class="span12 row-content"]/text()').getall()).strip()
        print(title)

        item = GlobalTimesArticleItems()
        item['label'] = label
        item['title'] = title
        item['content'] = content
        return item

    def _select_article_url(self, urls: list[str]) -> list[str]:
        res = [url for url in urls if url and 'content' in url]
        return res
