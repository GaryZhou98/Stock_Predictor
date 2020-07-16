from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime
import csv 

start = datetime(2020, 1, 20)
end = datetime(2020, 7, 15)

fb = Stock('FB',  token="pk_69c9cac10e344939be9ee5694af27d49")
hist = get_historical_data('FB', start, end, close_only=True, token="pk_69c9cac10e344939be9ee5694af27d49")
# pe = fb.get_earnings(period='year', token="pk_69c9cac10e344939be9ee5694af27d49")[0]['actualEPS']

eps = 7.30 #TTM

data = []
data.append(['date', 'close_price', 'simple_avg', 'pe'])
counter = 1

data_file = open('primary.csv', 'w')
csv_writer = csv.writer(data_file)

for day in hist:
  price = hist[day]['close']
  temp = [day, price]
  if counter == 1:
    temp.append(price)
  elif counter == 2:
    temp.append((price + data[counter - 1][1])/2)
  elif counter == 3:
    temp.append((price + data[counter - 1][1] + data[counter - 2][1])/3)
  elif counter == 4:
    temp.append((price + data[counter - 1][1] + data[counter - 2][1] + data[counter - 3][1])/4)
  else:
    temp.append((price + data[counter - 1][1] + data[counter - 2][1] + data[counter - 3][1] + data[counter - 4][1])/5)
  
  counter += 1
  temp.append(price/eps)
  data.append(temp)

csv_writer.writerows(data)




# print(fb.get_quote())
# print(fb.get_price())