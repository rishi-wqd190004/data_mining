# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class WebScraperItem(scrapy.Item):
    # define the fields for your item here like:
    all_laptops = scrapy.Field()
    price_laptop_in_USD = scrapy.Field()
    rated_from = scrapy.Field()
    
