# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader.processors import Join, MapCompose, TakeFirst
from w3lib.html import remove_tags


class NewsDataGeneratorItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class GlobalTimesArticleItems(scrapy.Item):
    url = scrapy.Field()  # 网址
    label = scrapy.Field()  # 分类
    title = scrapy.Field()  # 标题
    content = scrapy.Field()  # 内容
    pass


class GlobaltimesItem(scrapy.Item):
    """
    for crawl name: timeall
    """
    url = scrapy.Field()
    title = scrapy.Field()
    module = scrapy.Field()
    type = scrapy.Field()
    info = scrapy.Field()
    content = scrapy.Field()
    pass


class BaseItem(scrapy.Item):
    source = scrapy.Field(output_processor=TakeFirst())
    crawled_at = scrapy.Field(
        output_processor=TakeFirst()
    )
    image_urls = scrapy.Field()
    images = scrapy.Field()
    url = scrapy.Field(
        output_processor=TakeFirst()
    )


class NewsBaseItem(BaseItem):
    '''
     Base Cls for newsItem
    '''
    title = scrapy.Field(
        input_processor=MapCompose(remove_tags),
        output_processor=TakeFirst(),
    )
    summary = scrapy.Field(
        input_processor=MapCompose(remove_tags),
        output_processor=TakeFirst()
    )
    timestamp = scrapy.Field(
        input_processor=MapCompose(remove_tags),
        output_processor=TakeFirst(),
    )
    text = scrapy.Field(input_processor=MapCompose(remove_tags))


class CNNItem(NewsBaseItem):
    tag = scrapy.Field(output_processor=TakeFirst())


class BBCItem(NewsBaseItem):
    tag = scrapy.Field(output_processor=TakeFirst())
