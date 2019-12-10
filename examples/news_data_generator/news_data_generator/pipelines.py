# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.exceptions import DropItem
import json


class NewsDataGeneratorPipeline(object):
    """
    1. save item to json line
    2. duplicate item found
    """

    def __init__(self):
        self.file = open('items.jl', 'wb')
        self.titleSet = set()

    def process_item(self, item, spider):
        if item['title'] in self.titleSet:
            raise DropItem("Duplicate item found: %s" % item)
        else:
            self.titleSet.add(item['title'])
            line = json.dumps(dict(item)) + "\n"
            self.file.write(line)
            return item
