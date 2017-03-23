import numpy as np
import pandas as pd
import copy

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn import linear_model, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class Preprocessor(object):
	"""docstring for Preprocessor"""
	def __init__(self):
		self.users_csv = pd.read_csv('data/users.csv')
		self.cycles_csv = pd.read_csv('data/cycles.csv')
		self.tracking_csv = pd.read_csv('data/tracking.csv')
		self.active_csv = None
		self.cycles0_csv = pd.read_csv('data/cycles0.csv')

		self.predict_categories = ['energy','emotion','pain','skin']
		self.predict_symptoms = self.tracking_csv.loc[self.tracking_csv['category'].isin(self.predict_categories)]['symptom'].unique()

		self.y_cols = None
		self.feature_list = None

		self.output_ix_j_user = None
		self.output_ix_j_day = None
		self.output_ix_i_symptom = None

		self.x_users = self.get_x_users()
		self.x_cycles =  None# self.get_x_cycles()
		self.x_tracking = None#self.get_x_tracking() # do last
		self.x_active = None # self.get_x_active()

		self.x, self.y = self.get_feature_vector()
		self.x_test = self.get_test_feature_vector()

		self.scaler_users = None

	def df_crossjoin(self, df1, df2, **kwargs):

		df1['_tmpkey'] = 1
		df2['_tmpkey'] = 1

		res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
		res.index = pd.MultiIndex.from_product((df1.index, df2.index))

		df1.drop('_tmpkey', axis=1, inplace=True)
		df2.drop('_tmpkey', axis=1, inplace=True)

		return res 

	def get_x_users(self):

		x_users = self.users_csv

		x_users['age'] = 2017.0-x_users['birthyear']
		del x_users['birthyear']

		x_users = x_users.fillna(x_users.median())

		# DUMMIES

		# platform
		platform_dummies = pd.get_dummies(x_users['platform'])
		x_users = pd.concat([x_users,platform_dummies],axis=1)
		del x_users['platform']
		
		# country analysis
		del x_users['country']

		# CUSTOM

		# age groups
		x_users['age_group_teenager'] = np.zeros(x_users['user_id'].shape)
		x_users['age_group_middle'] = np.zeros(x_users['user_id'].shape)
		x_users['age_group_old'] = np.zeros(x_users['user_id'].shape)

		x_users['age_group_teenager'].loc[x_users['age']<18] = 1
		x_users['age_group_middle'].loc[(x_users['age']>=18) & (x_users['age']<35)] = 1
		x_users['age_group_old'].loc[x_users['age']>=35] = 1

		# SCALE
		'''self.scaler_users = MinMaxScaler(feature_range=[0,1]) #RobustScaler(with_centering=False, with_scaling=True, quantile_range=(10.0, 90.0))
		x_users[['weight','height','age']] = self.scaler_users.fit_transform(x_users[['weight','height','age']])'''

		return x_users

	def get_x_cycles(self):
		
		x_cycles = self.cycles_csv

		# Unusual if outside of this range - no ovulation most likely
		x_cycles['cycle_length_high'] = np.zeros(x_cycles['expected_cycle_length'].shape)
		x_cycles['cycle_length_really_high'] = np.zeros(x_cycles['expected_cycle_length'].shape)
		x_cycles['cycle_length_low'] = np.zeros(x_cycles['expected_cycle_length'].shape)
		x_cycles['cycle_length_high'].loc[(x_cycles['expected_cycle_length'] > 35)] = 1
		x_cycles['cycle_length_really_high'].loc[(x_cycles['expected_cycle_length'] > 45)] = 1
		x_cycles['cycle_length_low'].loc[(x_cycles['expected_cycle_length'] < 21)] = 1

		# Season
		x_cycles['cycle_start'] = pd.to_datetime(x_cycles['cycle_start'], format='%Y-%m-%d')
		x_cycles['season'] = x_cycles['cycle_start'].apply(lambda x: x.month)

		def season(val):

			if (val==12) or (val==1) or (val==2):
				return 'Winter'
			if (val==3) or (val==4) or (val==5):
				return 'Spring'
			if (val==6) or (val==7) or (val==8):
				return 'Summer'
			if (val==9) or (val==10) or (val==11):
				return 'Autumn'

		x_cycles['season'] = x_cycles['season'].apply(lambda x: season(x))
		x_seasons = pd.get_dummies(x_cycles['season'])
		x_cycles = pd.concat([x_cycles,x_seasons],axis=1)

		try:
			x_cycles = x_cycles.drop(['season','cycle_length','cycle_start','period_length'],axis=1)
		except:
			x_cycles = x_cycles.drop(['season','cycle_start','period_length'],axis=1)
		#del x_cycles['season']
		#del x_cycles['cycle_length']
		#del x_cycles['cycle_start']

		#print x_cycles.head()

		return x_cycles

	def get_x_cycles0(self):
		
		x_cycles = self.cycles0_csv

		# Unusual if outside of this range - no ovulation most likely
		x_cycles['cycle_length_high'] = np.zeros(x_cycles['expected_cycle_length'].shape)
		x_cycles['cycle_length_really_high'] = np.zeros(x_cycles['expected_cycle_length'].shape)
		x_cycles['cycle_length_low'] = np.zeros(x_cycles['expected_cycle_length'].shape)
		x_cycles['cycle_length_high'].loc[(x_cycles['expected_cycle_length'] > 35)] = 1
		x_cycles['cycle_length_really_high'].loc[(x_cycles['expected_cycle_length'] > 45)] = 1
		x_cycles['cycle_length_low'].loc[(x_cycles['expected_cycle_length'] < 21)] = 1

		# Season
		x_cycles['cycle_start'] = pd.to_datetime(x_cycles['cycle_start'], format='%Y-%m-%d')
		x_cycles['season'] = x_cycles['cycle_start'].apply(lambda x: x.month)

		def season(val):

			if (val==12) or (val==1) or (val==2):
				return 'Winter'
			if (val==3) or (val==4) or (val==5):
				return 'Spring'
			if (val==6) or (val==7) or (val==8):
				return 'Summer'
			if (val==9) or (val==10) or (val==11):
				return 'Autumn'

		#x_cycles = x_cycles[['user_id','cycle_length_high','cycle_length_low','cycle_length_really_high','season']]
		x_cycles['season'] = x_cycles['season'].apply(lambda x: season(x))
		x_seasons = pd.get_dummies(x_cycles['season'])
		x_cycles = pd.concat([x_cycles,x_seasons],axis=1)

		del x_cycles['season']
		#del x_cycles['cycle_length']
		#del x_cycles['cycle_start']

		#print x_cycles.head()

		return x_cycles


	def get_x_tracking(self):

		x_tracking = self.tracking_csv

		'''ls_include = self.predict_symptoms
		all_symptoms = x_tracking['symptom'].unique()
		relevant_symptoms = ls_include
		irrelevant_symptoms = [symptom for symptom in all_symptoms if symptom not in relevant_symptoms]

		x_tracking['symptom'] = x_tracking['symptom'].replace(irrelevant_symptoms, np.nan)
		dummies = pd.get_dummies(x_tracking['symptom'], prefix='s_')
		x_tracking = pd.concat([x_tracking,dummies],axis=1)
		df_users_sympts = d

		x_tracking = pd.merge(x_tracking,df_users_sympts)
		print x_tracking.head(10)'''
		
		'''x = x_tracking.loc[x_tracking['symptom'].isin(ls_include)]
		x = x.drop(['cycle_id','date','category'],1)
		x = x.groupby(['user_id','symptom']).mean()'''


		x_tracking = x_tracking.drop(['category','date'],axis=1)
		
		return x_tracking

	def get_x_tracking2(self):

		x = self.tracking_csv

		ls_include = self.predict_symptoms
		x = x.loc[x['symptom'].isin(ls_include)]
		x = x.drop(['cycle_id','date','category'],1)

		x2 = pd.pivot_table(x, values='day_in_cycle', index='user_id', columns = ['symptom'], aggfunc=np.mean)
		x2 = x2.fillna(x2.median())

		# get cycle data to normalize on
		c = self.cycles0_csv
		c = c.set_index('user_id')
		x2['expected_cycle_length'] = c['expected_cycle_length']
		x2 = x2[[col for col in x2.columns if col is not 'expected_cycle_length']].div(x2.expected_cycle_length, axis=0)

		# make index column
		x2.reset_index(level=0,inplace=True)
		#print x2.head()

		return x2

	def get_x_active(self):
		pass

	def time_features(self, x):

		# Time location
		# Day in cycle generated by you for predict_proba
		x['tr'] = x['day_in_cycle']/x['expected_cycle_length']
		
		t_ovulation = (x['expected_cycle_length']-15)
		x['tr_ovulation'] = x['day_in_cycle']/t_ovulation

		# In fertile window?
		x['fertile_window'] = np.zeros(x['tr'].shape)
		x['fertile_window'].loc[(x['day_in_cycle']>=(t_ovulation-6))] = 1
		x['fertile_window'].loc[(x['day_in_cycle']>(t_ovulation))] = 0

		# In pms window?
		x['pms_window'] = np.zeros(x['tr'].shape)
		x['pms_window'].loc[(x['day_in_cycle']>=(x['expected_cycle_length']-3))] = 1

		# In Luteal phase?
		x['tr_luteal'] = (x['day_in_cycle']-t_ovulation)/(x['expected_cycle_length']-t_ovulation)
		x['tr_luteal'].loc[x['tr_luteal']<0]=0

		return x

	def get_feature_vector(self):

		x_tracking = self.get_x_tracking()
		x_cycles = self.get_x_cycles()
		x_users = self.x_users
		x_tracking2 = self.get_x_tracking2()
		x = pd.merge(x_tracking, x_cycles, on=['user_id','cycle_id'])
		x = pd.merge(x,x_users,on='user_id')
		x = pd.merge(x,x_tracking2,on='user_id')
		x = self.time_features(x)

		#######################################

		# Summary dataframe

		y = self.binary_classification(x['symptom'])
		self.output_ix_i_symptom = y.columns
		x = x.drop('symptom',1)
		xy = pd.concat([x,y],axis=1) # Summary
		xy.to_csv('xy.csv')

		# X
		#print self.y_cols
		x = xy.drop(['user_id','cycle_id','day_in_cycle','expected_cycle_length'],axis=1) # Just used for calculating other features
		y = x[self.y_cols]
		x = x.drop(self.y_cols,axis=1)
		x.to_csv('x.csv',index=False)
		y.to_csv('y.csv',index=False)

		self.feature_list = list(x.columns)
		#print self.feature_list

		return x,y

	def get_test_feature_vector(self):

		x_test = pd.DataFrame()

		# Generate day in cycle
		df_days = pd.DataFrame()
		max_length = int(self.cycles0_csv['expected_cycle_length'].max())
		df_days['day_in_cycle'] = list(range(1,max_length))

		df_cycles = self.cycles0_csv[['user_id','expected_cycle_length']]

		x_test = self.df_crossjoin(df_days, df_cycles, suffixes=('_days', '_cycles'))
		x_test = x_test[x_test['day_in_cycle'] <= x_test['expected_cycle_length']]  

		#self.output_ix_j_day = x_test['day_in_cycle'].values

		x_test = self.time_features(x_test)

		# Apply other features

		x_cycles = self.get_x_cycles0()
		x_users = self.x_users
		x_tracking2 = self.get_x_tracking2()
		x_test = pd.merge(x_test,x_cycles,on='user_id')
		x_test = pd.merge(x_test,x_users,on='user_id')
		x_test = pd.merge(x_test,x_tracking2,on='user_id')

		x_test.index = x_test['user_id']
		self.output_ix_j_user, self.output_ix_j_day = x_test['day_in_cycle'].index, x_test['day_in_cycle'].values
		x_test = x_test[self.feature_list]
		x_test.to_csv('x_test.csv')

		#print len(x_test.index)

		return x_test

	def binary_classification(self, symptom):

		df_symptom = copy.deepcopy(symptom)
		all_symptoms = df_symptom.unique()
		relevant_symptoms = self.predict_symptoms
		#print relevant_symptoms
		irrelevant_symptoms = [symptom for symptom in all_symptoms if symptom not in relevant_symptoms]

		df_symptom = df_symptom.replace(irrelevant_symptoms, np.nan)
		df_symptom = pd.get_dummies(df_symptom, prefix='y_')
		self.y_cols = list(df_symptom.columns)

		return df_symptom

	def output_file(self):

		pass

def output(j_user, j_day, i_symptom, prob):

	# remove prefix
	symptom = []
	#print i_symptom
	for i in range(0,len(i_symptom)):
		entry = i_symptom[i].replace('y__','')
		symptom.append(entry)
	#print symptom

	#print len(j_day)
	#print len(j_user)

	output = []
	str_header = "user_id"+','+"day_in_cycle"+','+"symptom"+','+"probability"
	output.append(str_header)
	#print output

	'''ls_symptoms = []
	str_header = "symptom"+','+"probability"
	ls_symptoms.append(str_header)
	print len(prob)
	for j in range(0,len(symptom)):
		str_symptom = str(symptom[j])
		px = prob[j][:,1]
		for i in range(0,len(px)):
			str_px = str(px[i])
			str_current = str_symptom+','+str_px
		ls_symptoms.append(str_current)
		print ls_symptoms[j]'''

	for i in range(0,len(j_day)):
		x_user = str(j_user[i])
		x_day = str(j_day[i])
		for j in range(0,len(symptom)):
			x_symptom = str(symptom[j])
			x_px = str(prob[j][:,1][i])
			str_current = x_user+','+x_day+','+x_symptom+','+x_px
			output.append(str_current)
		#print output[i]

	with open("result.txt", "w") as f:
		f.writelines([x+'\n' for x in output])

	return output

def main():

	a = Preprocessor()

	# training set


	# train model
	clf = RandomForestClassifier(n_estimators = 1000)
	X = a.x.as_matrix()
	y = a.y.as_matrix()
	X_test = a.x_test.as_matrix()
	#print X.shape
	#print y.shape
	#print X_test.shape
	clf.fit(X,y)
	px = clf.predict_proba(X_test)
	#print len(px)
	#print px[1][:,1][0:10]

	# results
	output(a.output_ix_j_user, a.output_ix_j_day, a.output_ix_i_symptom, px)

if __name__ == "__main__":

	try:
		main()
	except KeyboardInterrupt:
		pass
