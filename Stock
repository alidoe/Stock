import csv # allows to read data from aapl.csv file containing stock prices
import numpy as np # perform calculations on our data
from sklearn.svm import SVR # build a predictive model
import matplotlib.pyplot as plt # plot data points with models on our graph to analyze
import matplotlib.pyplot as p

plt.switch_backend('newbackend')  

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileRead = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict_prices(dates,prices,x):
	dates = np.reshape(dates,(len(dates), 1))

	svr_lin = SVR(kernal = 'linear', C=1e3)
	svr_poly = SVR(kernel = 'poly', C=1e2, degree = 2)
	svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma=0.1)
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates,prices)
	svr_rbf.fit(dates,prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red',label='RBF model')
	plot.plot(dates, svr_lin.predict(dates), color='green',label='Linear model')
	plt.plot(dates,svr_poly.predict(dates), color='blue', label='Polynomial model')
	plot.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

	get_data('aapl.csv')

	def predict_prices(dates,prices,x):

		predicted_price = predict_price(dates, prices, 29)
		print(predicted_price)
