from pyhive import hive


conn = hive.connect(host='127.0.0.1', port=10000, database='bitcoin_db')
curr = conn.cursor()
    #curr.execute('create table bitcoin_rate (id int, rate float)')
curr.execute('select * from  bitcoin_db.btc_hv_csv_table limit 5')
for result in curr.fetchall():
	print(result)

