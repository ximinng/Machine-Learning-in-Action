# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class CnnSpiderSpider(CrawlSpider):
    name = 'cnn'
    allowed_domains = ['https://edition.cnn.com/']
    start_urls = ['https://edition.cnn.com/world']
    # start_urls = ['https://edition.cnn.com/travel',
    #               'https://edition.cnn.com/style',
    #               'https://edition.cnn.com/politics',
    #               'https://edition.cnn.com/europe',
    #               'https://edition.cnn.com/sport',
    #               'https://edition.cnn.com/china',
    #               'https://edition.cnn.com/entertainment',
    #               'https://edition.cnn.com/health',
    #               'https://edition.cnn.com/world',
    #               'https://edition.cnn.com/business']

    rules = (
        Rule(LinkExtractor(allow=r'Items/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        self.logger.info('Start parse %: ', response.url)
        item = {}
        # item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        # item['name'] = response.xpath('//div[@id="name"]').get()
        # item['description'] = response.xpath('//div[@id="description"]').get()
        return item
