# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import sqlite3

class WebScraperPipeline(object):

	def __init__(self):
		self.create_connection()
		self.create_table()

	def create_connection(self):
		self.conn = sqlite3.connect("amazon_gaming_laptop.db")
		self.curr = self.conn.cursor()

	def create_table(self):
		self.curr.execute("""DROP TABE IF EXISTS amazon_laptop""")
		self.curr.execute("""create table amazon_laptop(
				 		all_laptop text,
				 		price int,
				 		rated_from int
						)""")

    def process_item(self, item, spider):
    	self.store_db(item)
        return item

    def store_db(self,item):
    	self.curr.execute("""insert into amazon_laptop values (?,?,?)""",(
    				item['all_laptop'][0],
    				item['price'][0],
    				item['rated_from'][0]
    				))
    	self.conn.commit()