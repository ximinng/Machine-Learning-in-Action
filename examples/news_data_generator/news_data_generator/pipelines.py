# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.exceptions import DropItem
import json
import jsonlines


class NewsDataGeneratorPipeline(object):
    """
    duplicate item found
    """

    def __init__(self):
        self.urlSet = set()

    def process_item(self, item, spider):
        if item['url'] in self.urlSet:
            raise DropItem("Duplicate item found: %s" % item)
        else:
            self.urlSet.add(item['title'])
            return item


class JsonWriterPipeline(object):
    """
    save item to json line
    """

    def __init__(self):
        self.file = open('items.jl', 'wb')

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)

        # jl = json.loads(item)
        # with jsonlines.open('output.jl', mode='a') as writer:
        #     writer.write(jl)
        return item
