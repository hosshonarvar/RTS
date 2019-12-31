import sys
import os
import pandas as pd
from datetime import datetime

# Use six to import urllib so it is working for Python2/3
from six.moves import urllib
# If you don't want to use six, please comment out the line above
# and use the line below instead (for Python3 only).
#import urllib.request, urllib.parse, urllib.error

import time

'''
@author: c0redumb
https://github.com/c0redumb/yahoo_quote_download

Starting on May 2017, Yahoo financial has terminated its service on
the well used EOD data download without warning. This is confirmed
by Yahoo employee in forum posts.

Yahoo financial EOD data, however, still works on Yahoo financial pages.
These download links uses a "crumb" for authentication with a cookie "B".
This code is provided to obtain such matching cookie and crumb.
'''

# Build the cookie handler
cookier = urllib.request.HTTPCookieProcessor()
opener = urllib.request.build_opener(cookier)
urllib.request.install_opener(opener)

# Cookie and corresponding crumb
_cookie = None
_crumb = None

# Headers to fake a user agent
_headers={
	'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'
}

def _get_cookie_crumb():
	'''
	This function perform a query and extract the matching cookie and crumb.
	'''

	# Perform a Yahoo financial lookup on SP500
	req = urllib.request.Request('https://finance.yahoo.com/quote/^GSPC', headers=_headers)
	f = urllib.request.urlopen(req)
	alines = f.read().decode('utf-8')

	# Extract the crumb from the response
	global _crumb
	cs = alines.find('CrumbStore')
	cr = alines.find('crumb', cs + 10)
	cl = alines.find(':', cr + 5)
	q1 = alines.find('"', cl + 1)
	q2 = alines.find('"', q1 + 1)
	crumb = alines[q1 + 1:q2]
	_crumb = crumb

	# Extract the cookie from cookiejar
	global cookier, _cookie
	for c in cookier.cookiejar:
		if c.domain != '.yahoo.com':
			continue
		if c.name != 'B':
			continue
		_cookie = c.value

	# Print the cookie and crumb
	#print('Cookie:', _cookie)
	#print('Crumb:', _crumb)

def load_yahoo_quote(ticker, begindate, enddate, info = 'quote'):
	'''
	This function load the corresponding history/divident/split from Yahoo.
	'''
	# Check to make sure that the cookie and crumb has been loaded
	global _cookie, _crumb
	if _cookie == None or _crumb == None:
		_get_cookie_crumb()

	# Prepare the parameters and the URL
	tb = time.mktime((int(begindate[0:4]), int(begindate[4:6]), int(begindate[6:8]), 4, 0, 0, 0, 0, 0))
	te = time.mktime((int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 18, 0, 0, 0, 0, 0))

	param = dict()
	param['period1'] = int(tb)
	param['period2'] = int(te)
	param['interval'] = '1d'
	if info == 'quote':
		param['events'] = 'history'
	elif info == 'dividend':
		param['events'] = 'div'
	elif info == 'split':
		param['events'] = 'split'
	param['crumb'] = _crumb
	params = urllib.parse.urlencode(param)
	url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?{}'.format(ticker, params)
	#print(url)
	req = urllib.request.Request(url, headers=_headers)

	# Perform the query
	# There is no need to enter the cookie here, as it is automatically handled by opener
	f = urllib.request.urlopen(req)
	alines = f.read().decode('utf-8')
	#print(alines)
	return alines.split('\n')

def companies():
    dataset = pd.read_csv(os.path.join("data","dow30.csv"))
    return dataset

def symbol_list():
    dataset = pd.read_csv(os.path.join("data","dow30.csv"))
    return dataset['Symbol'].values.tolist()

class BaseData(object):
    def __init__(self,symbol:str):
        self.__symbol = symbol
    
    @property
    def symbol(self):
        return self.__symbol

    def save(self,file_dir:str,file_name:str,data:pd.DataFrame):
        try:
            if data is None:
                return
            full_path = os.path.join(file_dir,file_name)
            include_index = False if data.index.name == None else True
            if os.path.isdir(file_dir):
                data.to_csv(full_path,index=include_index)
            else:
                os.makedirs(file_dir)
                data.to_csv(full_path,index=include_index)
        except OSError as err:
            print("OS error for symbol {} : {}".format(self.symbol,err))
        except:
            print("Unexpected error for symbol {} : {}".format(self.symbol, sys.exc_info()[0]))


class Downloader(BaseData):
    def __init__(self,symbol:str,start_date:str, end_date:str):
        try:
            BaseData.__init__(self,symbol)
            self.__start_date = datetime.strptime(start_date,'%Y%m%d')
            self.__end_date = datetime.strptime(end_date,'%Y%m%d')
            self.__data = None

            #Download data from Yahoo.
            yah = load_yahoo_quote(symbol,start_date,end_date)
            header = yah[0].split(',')
            table = []
            for i in yah[1:]:
                quote = i.split(',')
                if len(quote)>1:
                    d = dict()
                    d[header[0]] = quote[0]
                    d[header[1]] = quote[1]
                    d[header[2]] = quote[2]
                    d[header[3]] = quote[3]
                    d[header[4]] = quote[4]
                    d[header[5]] = quote[5]
                    d[header[6]] = quote[6]
                    table.append(d)
            self.__data = pd.DataFrame(table)
            self.__size = len(self.__data)
        except OSError as err:
            print("OS error for symbol {} : {}".format(symbol,err))
    
    def save(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"quotes.csv",self.__data)

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date

    @property
    def data(self):
        return self.__data
    
    @property
    def size(self):
        return self.__size