# -*- coding: utf-8 -*-
import scrapy
from ..items import GlobalTimesItems
from scrapy_splash import SplashRequest
from w3lib.html import remove_tags

"""
Use Scrapy-Splash get dynamic website(which built by js).
1. Start Scrapy-Splash: 
                       docker run -p 8050:8050 scrapinghub/splash
                    
2. debug website:
                scrapy shell 'http://localhost:8050/render.json?url=http://globaltimes.cn/&timeout=10&wait=0.5'
"""


class GlobaltimesSpiderSpider(scrapy.Spider):
    """
    How to debug xpath: scrapy shell "http://globaltimes.cn/"
    """
    name = 'times'
    allowed_domains = ['globaltimes.cn']
    script = """
        function main(splash)
            splash:go(splash.args.url)
            splash:wait(10)
            splash:runjs('document.getElementById("J-global-toolbar").scrollIntoView()')
            splash:wait(10) 
            return splash:html()
        end
    """

    def start_requests(self):
        urls = [
            'http://globaltimes.cn/',
        ]
        for url in urls:
            # yield scrapy.Request(url=url, callback=self.parse)
            yield SplashRequest(url, self.parse,
                                args={
                                    # optional; parameters passed to Splash HTTP API
                                    'wait': 0.5,
                                    'lua_source': scrapy,
                                    # 'url' is prefilled from request url
                                    # 'http_method' is set to 'POST' for POST requests
                                    # 'body' is set to request body for POST requests
                                },
                                endpoint='render.html',  # optional; default is render.html
                                splash_headers={
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                    'Accept-Language': 'en',
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
                                }
                                )

    def parse(self, response):
        category: list[str] = response.xpath(
            "//div[@class='nav-collapse collapse nav-channels']/ul/li/a/text()"
        ).getall()

        # filename = 'globalTimes.jl'
        # with open(filename, 'wb') as f:
        #     f.write(categroys)
        self.log('Saved file %s' % category)

        yield category
