from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import pandas as pd
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn import metrics
from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:20,.2f}'.format
#crawling the first data
url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130429"
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
df.rename(columns={ "Date": "date_value","Open*": "Openning", "Close**": "Closing", "Market Cap": "Market_cap"}, inplace=True)
#print(df.head())
df.to_csv('/home/richi/masters_semester_2/data_mining_WQD7005/assignments/saved_df.csv')
columns_names = list(df.columns)
#print(columns_names)

#checking the data types for each column
#print(df.dtypes)

#set date as index value
df.set_index('date_value', inplace=True)
print(df.head(5))

#as known bitcoin transaction became famous from january 2017
bc = df[['Closing']].loc[:'2017-01-01'].sort_values(by='date_value',ascending=[True])
print(bc.head(5))
#move this set of data for later usage
with open("curr_bitcoin.pickle", 'wb') as fp:
    pickle.dump(bc, fp)

#plot the historical values of bitcoin
bc.plot(figsize=(16,5))
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.title('Bitcoin Price')
plt.savefig('btcprice.png')
plt.show()

#detrending
#method 1
# Differencing the price
bc_diff = bc.diff(1).dropna()
# Plotting the differences daily
bc_diff.plot(figsize=(12,5))
plt.title('Plot of the Daily Changes in Price for BTC')
plt.ylabel('Change in USD')
plt.show()
#testing the stationarity
results = adfuller(bc_diff.Closing)
print(f"P-value: ",{results[1]})
#method 2 (taking log)
bc_log = pd.DataFrame(np.log(df.Closing))
# Plotting the log of the data
plt.figure(figsize=(16,8))
plt.plot(bc_log)
plt.title('Log of BTC')
plt.xlabel('Dates')
plt.savefig('btc_log.png')
plt.show()
# Differencing the log values
log_diff = bc_log.diff().dropna()
# Plotting the daily log difference
plt.figure(figsize=(16,8))
plt.plot(log_diff)
plt.title('Differencing Log')
plt.savefig('logdiff.png')
#plt.show()
#testing the stationarity
results = adfuller(log_diff.Closing)
print(f"P-value with taking log: {results[1]}")

#AS THE P-VALUES ARE LESS THAN 0.05: REJECT NULL HYPOTHESIS

#PACF and ACF for differentiating
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8))
plot_acf(bc_diff, ax=ax1, lags=40)
plot_pacf(bc_diff, ax=ax2, lags=40)
plt.show()
#apperas to be some correlation in day 5 and day 10
#ACF and PACF for the Log Difference
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8))
plot_acf(log_diff, ax=ax1, lags=40)
plot_pacf(log_diff, ax=ax2, lags=40)
plt.savefig('acfpacf.png')
plt.show()

#modelling
#SARIMA Model for Differencing
#Finding the Best Parameters for ARIMA

# model = ARIMA(log_diff, order=(8, 1, 0))  
# results_AR = model.fit(disp=-1)  
# plt.plot(bc_diff)
# plt.plot(results_AR.fittedvalues, color='red', label = 'order 8')
# RSS = results_AR.fittedvalues-ts_diff_logtrans
# RSS.dropna(inplace=True)
# plt.title('RSS: %.4f'% sum(RSS**2))
# plt.legend(loc = 'best')



def best_param(model, data, pdq, pdqs):
    """
    Loops through each possible combo for pdq and pdqs
    Runs the model for each combo
    Retrieves the model with lowest AIC score
    """
    ans = []
    for comb in tqdm(pdq):
        for combs in tqdm(pdqs):
            try:
                mod = model(data,
                            order=comb,
                            seasonal_order=combs,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            freq='D')

                output = mod.fit()
                ans.append([comb, combs, output.aic])
                #print(ans.head(2))
            except:
                continue

    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    ans_df['pdq'] = ans_df['pdq'].astype(float)
    ans_df['pdqs'] = ans_df['pdqs'].astype(float)
    ans_df['aic'] = ans_df['aic'].astype(float)
    return #ans_df.loc[ans_df.aic.idxmin()]

# Assigning variables for p, d, q.
p = d = q = range(0,6)
d = range(2)

# Creating a list of all possible combinations of p, d, and q.
pdq = list(itertools.product(p, d, q))
pdq_array = np.asarray(pdq)
#print(pdq_array)
#print(float(i) for i in pdq_array)
# Keeping seasonality at zeroes
pdqs = [(0,0,0,0)]
# Finding the best parameters
#best_param(SARIMAX, bc_log, pdq, pdqs)

# Splitting 80/20
index = np.random.rand(len(df)) < 0.8

train = bc_log[index:]
test = bc_log[:~index]

# Fitting the model to the training set
model = SARIMAX(train, 
                order=(1, 0, 0), 
                seasonal_order=(0,0,0,0), 
                freq='D', 
                enforce_stationarity=False, 
                enforce_invertibility=False)
output = model.fit()
print(output.summary())
output.plot_diagnostics(figsize=(15,8))
plt.show()













#selecting the features and splitting the df
# feature_cols = ['Open*', 'High', 'Low']
# x = df[feature_cols]
# y = df['Market Cap']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# #building the model
# dtr = DecisionTreeClassifier()
# #training model
# dtr = dtr.fit(x_train, y_train)
# #predicting the model
# y_pred = dtr.predict(x_test)
# #accuracy
# print("accuracy: ", metrics.accuracy_score(y_test, y_pred))