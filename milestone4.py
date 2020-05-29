from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:20,.2f}'.format
#crawling the first data
url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/"
html = urlopen(url).read()
soap_html = soup(html, "html.parser")
div = soap_html.findAll("div", {"class": "cmc-table__table-wrapper-outer"})

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

#cleaning the data
df[["Date"]] = df[["Date"]].apply(pd.to_datetime,errors='ignore')
df[["Open*", "High", "Low", "Close**"]] = df[["Open*", "High", "Low", "Close**"]].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df[["Volume", "Market Cap"]] = df[["Volume", "Market Cap"]].apply(lambda a: a.str.replace(',', '').astype(int), axis=1)

#print(df.head())

#decision tree to continue mining or not
columns_names = list(df.columns)
#print(columns_names)
#selecting the features and splitting the df
feature_cols = ['Open*', 'High', 'Low']
x = df[feature_cols]
y = df['Market Cap']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#building the model
dtr = DecisionTreeClassifier()
#training model
dtr = dtr.fit(x_train, y_train)
#predicting the model
y_pred = dtr.predict(x_test)
#accuracy
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))
