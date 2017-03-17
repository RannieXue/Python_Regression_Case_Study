import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

"""
Clean up data and merge into one dataframe
"""
def clean_data(directory):
	# print os.listdir(directory)
	reservations = pd.read_csv(directory+'reservations_(5).csv')
	print reservations.shape[0]
	# print reservations.head()

	vehicles = pd.read_csv(directory+'vehicles_(6).csv')
	print vehicles.shape[0]
	# print vehicles.head()

	# reservations.iloc['Hourly_Count'] = reservations['vehicle_id'].count()
	# print reservations.head()
	reservation_total_count = reservations[['vehicle_id', 'reservation_type']].groupby(['vehicle_id']).count()
	reservation_total_count['vehicle_id'] = reservation_total_count.index
	reservation_total_count.reset_index(drop=True)
	reservation_total_count.columns = ['total_count', 'vehicle_id']
	# reservation_total_count.to_csv(directory+'total_count.csv', index=False)
	print reservation_total_count.shape[0]

	reservation_hourly_count = reservations[:][reservations['reservation_type'] == 1].groupby(['vehicle_id']).count()
	reservation_hourly_count['vehicle_id'] = reservation_hourly_count.index
	reservation_hourly_count.reset_index(drop=True)
	reservation_hourly_count.columns = ['hourly_count', 'vehicle_id']
	# reservation_hourly_count.to_csv(directory+'hourly_count.csv', index=False)
	print reservation_hourly_count.shape[0]

	reservation_daily_count = reservations[:][reservations['reservation_type'] == 2].groupby(['vehicle_id']).count()
	reservation_daily_count['vehicle_id'] = reservation_daily_count.index
	reservation_daily_count.reset_index(drop=True)
	reservation_daily_count.columns = ['daily_count', 'vehicle_id']
	# reservation_daily_count.to_csv(directory+'daily_count.csv', index=False)
	print reservation_daily_count.shape[0]

	reservation_weekly_count = reservations[:][reservations['reservation_type'] == 3].groupby(['vehicle_id']).count()
	reservation_weekly_count['vehicle_id'] = reservation_weekly_count.index
	reservation_weekly_count.reset_index(drop=True)
	reservation_weekly_count.columns = ['weekly_count', 'vehicle_id']
	# reservation_weekly_count.to_csv(directory+'weekly_count.csv', index=False)
	print reservation_weekly_count.shape[0]

	merged1 =  pd.merge(reservation_total_count, vehicles, on = 'vehicle_id', how='outer')
	merged1 =  pd.merge(reservation_hourly_count, merged1, on = 'vehicle_id', how='outer')
	merged1 =  pd.merge(reservation_daily_count, merged1, on = 'vehicle_id', how='outer')
	final_clean_data =  pd.merge(reservation_weekly_count, merged1, on = 'vehicle_id', how='outer')
	final_clean_data = final_clean_data.fillna(0)

	return final_clean_data



if __name__ == '__main__':
	directory = r'/Users/Jingran/TuroCaseStudy/'

	final_clean_data = clean_data(directory)

	# Linear Regression
	print '*'*20, 'Linear Regression', '*'*20
	X = final_clean_data[['technology', 'actual_price', 'recommended_price', 'num_images', 'street_parked', 'description']]
	y = final_clean_data['total_count']
	X = sm.add_constant(X)
	est = sm.OLS(y, X).fit()
	summ = est.summary()
	print est.params
	print summ

	# Logistic Regression
	print '*'*20, 'Logistic Regression', '*'*20
	total_count_mean = final_clean_data['total_count'].mean()
	final_clean_data['total_categorical'] = ''
	for index, row in final_clean_data.iterrows():
		if final_clean_data.loc[index,'total_count']>=total_count_mean:
			final_clean_data.loc[index, 'total_categorical'] = 1
		else:
			final_clean_data.loc[index, 'total_categorical'] = 0
	# print final_clean_data['total_categorical'].mean()
	# x = final_clean_data.copy()
	X = final_clean_data[['technology', 'street_parked']]
	y = final_clean_data['total_categorical']
	y = np.ravel(y)
	# instantiate a logistic regression model, and fit with X and y
	model = LogisticRegression()
	model = model.fit(X, y)

	# check the accuracy on the training set
	# model.score(X, y)
	# print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))





