from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import pandas as pd
import matplotlib.pyplot as plt

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
df[["Date"]] = df[["Date"]].apply(pd.to_datetime,errors='ignore')
df[["Open*", "High", "Low", "Close**"]] = df[["Open*", "High", "Low", "Close**"]].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df[["Volume", "Market Cap"]] = df[["Volume", "Market Cap"]].apply(lambda a: a.str.replace(',', '').astype(int), axis=1)
#df[["Open*"]] = df[["Open*"]].apply(pd.to_numeric, downcast='int')
#print(df.info())
#df.drop(df.columns[[1]], axis=1)
print(df)

df.to_csv("~/masters_semester_2/data_mining_WQD7005/tests_script/bitcoin_price.csv")
#print(tbody)
plt.figure(figsize=(16,8))
plt.plot(df.Date, df['Close**'], color='r')
plt.xlabel('Dates')
plt.ylabel('Closing rates on that day')
plt.show()
