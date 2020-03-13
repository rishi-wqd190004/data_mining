from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import pandas as pd
url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/"
html = urlopen(url).read()
soap_html = soup(html, "html.parser")
div = soap_html.findAll("div", {"class": "cmc-table__table-wrapper-outer"})

#table = div[2].findAll("table")
#print(table)
#thead = div[2].findAll("thead")
#th = thead[0].findAll("th")
#th_list =[]
#for element in th:
#	th_list.append(element.text)
#print(th_list)
tbody = div[2].findAll("tr")
header = tbody[0].findAll("th")
header_list = []
for h in header:
	header_list.append(h.text)

#print(header_list)
tbody2 = tbody[1:]
#print(tbody2)
content = []
for element in tbody2:
	t = []
	td = element.findAll("td")
	for ele in td:
		t.append(ele.text)
	content.append(t)

#print(content)
df = pd.DataFrame(content,columns=header_list)
print(df)

df.to_csv("~/masters_semester_2/data_mining_WQD7005/tests_script/bitcoin_price.csv")
#print(tbody)
