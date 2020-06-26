import scrapy
from ..items import WebScraperItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
#from selenium import webdriver

class lazadaSpider(scrapy.Spider):
	name = 'amazon'
	start_urls = [
		#'https://www.amazon.com/s?k=gaming+laptop&i=computers&rh=n%3A172282%2Cn%3A541966%2Cn%3A13896617011%2Cn%3A565108%2Cp_72%3A2661621011%2Cp_n_graphics_type_browse-bin%3A14292273011&dc&qid=1593160381&rnid=493964&ref=sr_nr_n_1'
		'https://www.amazon.com/s?k=gaming+laptop&ref=nb_sb_noss_1'
	]

	rules = (Rule(LinkExtractor(allow=(), restrict_xpaths=("//li[@class='a-last']",)), callback="parse", follow= True),)

	def parse(self, response):
		#title = response.css('title::text').extract() #just want the text
		#yield {'titleText': title} #show the response as a dictionary

		items = WebScraperItem()
		#for looop to loop for  all the pages(later)
		for each in all_laptops_names:
			all_laptops_names = response.xpath("//a[@class='a-link-normal a-text-normal']//span/text()").extract()
			price_laptop = response.xpath("//span[@class='a-price-whole']/text()").extract()
			rated_from_people = response.xpath("//span[@class='a-size-base']/text()").extract()


		#driver.find_element_by_xpath("//li[@class='a-last']").click()
			items['all_laptops'] = all_laptops_names
			items['price_laptop_in_USD'] = price_laptop
			items['rated_from'] = rated_from_people
		
			yield items
