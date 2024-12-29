# -*- coding: utf-8 -*-

__author__ = "wangzhefeng"


"""
	- 应用LightGBM模型进行分类任务；
	- 数据和问题来源: 
		- https://www.kaggle.com/c/home-credit-default-risk
	- 项目参考: 
		- https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/code
		- https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
	- 只做了适当的特征工程, 进行详细的特征工程构建；
	- 希望参考这个项目实现一个应用以下常用算法的更多项目:
		- XGBoost
		- GBDT
		- RandomForest
		- Logistic Regression
		- Logistic Regression with LASSO, Ridge, Elastic Net
		- CART
		- SVM
	-----------------
	Build 2018-10-15:
	-----------------
	V 1.0
"""


import sys
sys.path.append(r'E:\tools')
import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


# -------------------------- CONFIGURATIONS -------------------------------
# General Configurations
NUM_THREADS = 4
DATA_DIRECTORY = "./data"
SUBMISSION_SUFIX = "_model1_01"

# Installments Trend Peridos
INSTALLMENTS_LAST_K_TREND_PERIODS = [12, 24, 60, 120]

# LightGBM Configuration and Hyper-Parameters
GENERATE_SUBMISSION_FILE = True
STRATIFIED_KFOLDS = False
RANDOM_SEED = 737851
NUM_FOLDS = 10
EARLY_STOPPING = 100

LIGHTGBM_PARAMS = {
	'boosting_type': 'goss',
	'n_estimators': 10000,
	'learning_rate': 0.005134,
	'max_depth': 10,
	'num_leaves': 54,
	'subsample_for_bin': 240000,
	'reg_alpha': 0.436193,
	'reg_lambda': 0.479169,
	'colsample_bytree': 0.508716,
	'min_split_gain': 0.024766,
	'subsample': 1,
	'is_unbalance': False,
	'silent': -1,
	'verbose': -1
}

# Aggregations

BUREAU_AGG = {
    'SK_ID_BUREAU': ['nunique'],
    'DAYS_CREDIT': ['min', 'max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
    'AMT_ANNUITY': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean', 'sum'],
    'MONTHS_BALANCE_MEAN': ['mean', 'var'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    # Categorical
    'STATUS_0': ['mean'],
    'STATUS_1': ['mean'],
    'STATUS_12345': ['mean'],
    'STATUS_C': ['mean'],
    'STATUS_X': ['mean'],
    'CREDIT_ACTIVE_Active': ['mean'],
    'CREDIT_ACTIVE_Closed': ['mean'],
    'CREDIT_ACTIVE_Sold': ['mean'],
    'CREDIT_TYPE_Consumer credit': ['mean'],
    'CREDIT_TYPE_Credit card': ['mean'],
    'CREDIT_TYPE_Car loan': ['mean'],
    'CREDIT_TYPE_Mortgage': ['mean'],
    'CREDIT_TYPE_Microloan': ['mean'],
    # Group by loan duration features (months)
    'LL_AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'LL_DEBT_CREDIT_DIFF': ['mean'],
    'LL_STATUS_12345': ['mean'],
}

BUREAU_ACTIVE_AGG = {
    'DAYS_CREDIT': ['max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['min', 'mean'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'CREDIT_TO_ANNUITY_RATIO': ['mean'],
    'MONTHS_BALANCE_MEAN': ['mean', 'var'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
}

BUREAU_CLOSED_AGG = {
    'DAYS_CREDIT': ['max', 'var'],
    'DAYS_CREDIT_ENDDATE': ['max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
    'DAYS_CREDIT_UPDATE': ['max'],
    'ENDDATE_DIF': ['mean'],
    'STATUS_12345': ['mean'],
}

BUREAU_LOAN_TYPE_AGG = {
    'DAYS_CREDIT': ['mean', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
    'AMT_CREDIT_SUM': ['mean', 'max'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'max'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'DAYS_CREDIT_ENDDATE': ['max'],
}

BUREAU_TIME_AGG = {
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'STATUS_0': ['mean'],
    'STATUS_12345': ['mean'],
}

PREVIOUS_AGG = {
    'SK_ID_PREV': ['nunique'],
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    'DAYS_TERMINATION': ['max'],
    # Engineered features
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
    'DOWN_PAYMENT_TO_CREDIT': ['mean'],
}

PREVIOUS_ACTIVE_AGG = {
    'SK_ID_PREV': ['nunique'],
    'SIMPLE_INTERESTS': ['mean'],
    'AMT_ANNUITY': ['max', 'sum'],
    'AMT_APPLICATION': ['max', 'mean'],
    'AMT_CREDIT': ['sum'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    # Engineered features
    'AMT_PAYMENT': ['sum'],
    'INSTALMENT_PAYMENT_DIFF': ['mean', 'max'],
    'REMAINING_DEBT': ['max', 'mean', 'sum'],
    'REPAYMENT_RATIO': ['mean'],
}

PREVIOUS_APPROVED_AGG = {
    'SK_ID_PREV': ['nunique'],
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['max'],
    'AMT_GOODS_PRICE': ['max'],
    'HOUR_APPR_PROCESS_START': ['min', 'max'],
    'DAYS_DECISION': ['min', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    'DAYS_TERMINATION': ['mean'],
    # Engineered features
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['max'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    # The following features are only for approved applications
    'DAYS_FIRST_DRAWING': ['max', 'mean'],
    'DAYS_FIRST_DUE': ['min', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE': ['max', 'mean'],
    'DAYS_LAST_DUE_DIFF': ['min', 'max', 'mean'],
    'SIMPLE_INTERESTS': ['min', 'max', 'mean'],
}

PREVIOUS_REFUSED_AGG = {
    'AMT_APPLICATION': ['max', 'mean'],
    'AMT_CREDIT': ['min', 'max'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    # Engineered features
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
    'APPLICATION_CREDIT_RATIO': ['min', 'mean'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}

PREVIOUS_LATE_PAYMENTS_AGG = {
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    # Engineered features
    'APPLICATION_CREDIT_DIFF': ['min'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}

PREVIOUS_LOAN_TYPE_AGG = {
    'AMT_CREDIT': ['sum'],
    'AMT_ANNUITY': ['mean', 'max'],
    'SIMPLE_INTERESTS': ['min', 'mean', 'max', 'var'],
    'APPLICATION_CREDIT_DIFF': ['min', 'var'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['max'],
    'DAYS_LAST_DUE_1ST_VERSION': ['max', 'mean'],
    'CNT_PAYMENT': ['mean'],
}

PREVIOUS_TIME_AGG = {
    'AMT_CREDIT': ['sum'],
    'AMT_ANNUITY': ['mean', 'max'],
    'SIMPLE_INTERESTS': ['mean', 'max'],
    'DAYS_DECISION': ['min', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    # Engineered features
    'APPLICATION_CREDIT_DIFF': ['min'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}

POS_CASH_AGG = {
    'SK_ID_PREV': ['nunique'],
    'MONTHS_BALANCE': ['min', 'max', 'size'],
    'SK_DPD': ['max', 'mean', 'sum', 'var'],
    'SK_DPD_DEF': ['max', 'mean', 'sum'],
    'LATE_PAYMENT': ['mean']
}

INSTALLMENTS_AGG = {
    'SK_ID_PREV': ['size', 'nunique'],
    'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DPD': ['max', 'mean', 'var'],
    'DBD': ['max', 'mean', 'var'],
    'PAYMENT_DIFFERENCE': ['mean'],
    'PAYMENT_RATIO': ['mean'],
    'LATE_PAYMENT': ['mean', 'sum'],
    'SIGNIFICANT_LATE_PAYMENT': ['mean', 'sum'],
    'LATE_PAYMENT_RATIO': ['mean'],
    'DPD_7': ['mean'],
    'DPD_15': ['mean'],
    'PAID_OVER': ['mean']
}

INSTALLMENTS_TIME_AGG = {
    'SK_ID_PREV': ['size'],
    'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DPD': ['max', 'mean', 'var'],
    'DBD': ['max', 'mean', 'var'],
    'PAYMENT_DIFFERENCE': ['mean'],
    'PAYMENT_RATIO': ['mean'],
    'LATE_PAYMENT': ['mean'],
    'SIGNIFICANT_LATE_PAYMENT': ['mean'],
    'LATE_PAYMENT_RATIO': ['mean'],
    'DPD_7': ['mean'],
    'DPD_15': ['mean'],
}

CREDIT_CARD_AGG = {
    'MONTHS_BALANCE': ['min'],
    'AMT_BALANCE': ['max'],
    'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
    'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
    'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
    'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
    'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
    'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum', 'var'],
    'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
    'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
    'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
    'CNT_DRAWINGS_POS_CURRENT': ['mean'],
    'SK_DPD': ['mean', 'max', 'sum'],
    'SK_DPD_DEF': ['max', 'sum'],
    'LIMIT_USE': ['max', 'mean'],
    'PAYMENT_DIV_MIN': ['min', 'mean'],
    'LATE_PAYMENT': ['max', 'sum'],
}

CREDIT_CARD_TIME_AGG = {
    'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
    'SK_DPD': ['max', 'sum'],
    'AMT_BALANCE': ['mean', 'max'],
    'LIMIT_USE': ['max', 'mean']
}


# -------------------------- UTILITY FUNCTIONS -------------------------------

@contextmanager
def timer(name):
	start = time.time()
	yield
	print("{} - done in {:.0f}s".format(name, time.time() - start))





def oneHotEncoding(df, nan_as_category = True):
	"""
	One-hot encoding for categorical columns with pd.get_dummies()
	:param df: dataframe without one-hot encoded
	:param nan_as_category: nan options
	:return:
		df: dataframe one-hot encoded
		new_columns: new columns in df in the process of encoding
	"""
	original_columns = list(df.columns)
	categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
	# categorical_columns = [col for col in df.columns if len(pd.DataFrame(df.loc[:, col]).drop_duplicates()) <= 10 or \
	# 													  df[col].dtype == "object"]
	df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category)
	new_columns = [c for c in df.columns if c not in original_columns]

	return df, new_columns

def bin_factorize(df):
	"""
	Categorical features with Binary encode (0 or 1: two categories)
	:param df:
	:return:
	"""
	bin_features = [col for col in df.columns if df[col].dtype == "object" and len(pd.DataFrame(df[col]).drop_duplicates()) == 2]
	for bin_column in bin_features:
		df[bin_features], uniques = pd.factorize(df[bin_features])

	return df

def read_train_test(data_path):
	"""
	Read training and testing data
	:param data_path:
	:return:
	"""
	train = pd.read_csv(data_path + "train.csv")
	test = pd.read_csv(data_path + "test.csv")
	print("Train samples: {},\n Test sample: {}".format(len(train), len(test)))
	df = train.append(test).reset_index()

	return train, test, df


# def readProjData(path = "./data", suffix = ".csv"):
# 	all_files = os.listdir(path)
# 	csv_files = [filename for filename in all_files if filename.endswith(suffix)]
# 	for fname in csv_files:
# 		filename = os.path.join(path, fname)
# 		file = pd.read_csv(filename)
# 	return file



def Nan_fill():
	pass
























# -------------------------- MAIN FUNCTION -------------------------------

def main(debug = False):
	num_rows = 30000 if debug else None
	df = application_train_test(num_rows)

	with timer("Process bureau and bureau_balance"):
		bureau = bureau_and_balance(num_rows)
		print("Bureau df shape:", bureau.shape)
		df = df.join(bureau, how = 'left', on = 'SK_ID_CURR')
		del bureau
		gc.collect()

	with timer("Process previous_applications"):
		prev = previous_applications(num_rows)
		print("Previous application of shape:", prev.shape)
		df = df.join(prev, how = 'left', on = 'SK_ID_CURR')
		del prev
		gc.collect()

	with timer("Process POS-CASH balance"):
		pos = pos_cash(num_rows)
		print("Pos-cash balance df shape:", pos.shape)
		df = df.join(pos, how = 'left', on = 'SK_ID_CURR')
		del pos
		gc.collect()

	with timer("Porcess installments payments"):
		ins = installments_payments(num_rows)
		print("Installments payments df shape:", ins.shape)
		df = df.join(ins, how = 'left', on = 'SK_ID_CURR')
		del ins
		gc.collect()

	with timer("Process credit card balance"):
		cc = credit_card_balance(num_rows)
		print("Credit card balance df shape:", cc.shape)
		df = df.join(cc, how = 'left', on = 'SK_ID_CURR')
		del cc
		gc.collect()

	with timer("Run LightGBM with kfold"):
		feat_importances = kfold_lightgbm(df, num_folds = 10, stratified = False, debug = debug)




# ------------------------- APPLICATION PIPELINE -------------------------

def application_train_test(num_rows = None, nan_as_category = False):
	"""
	Preprocess application_train.csv and application_test.csv
	:param num_rows:
	:param nan_as_category:
	:return:
	"""
	# read data and merge
	train_df = pd.read_csv("./data/application_train.csv", nrows = num_rows)
	test_df = pd.read_csv("./data/application_test.csv", nrows = num_rows)
	print("Train samples: {},\n Test sample: {}".format(len(train_df), len(test_df)))
	df = train_df.append(test_df).reset_index()

	# Optional: remove 4 apllications with XNA CODE_GENDER (train data set)
	df = df[df["CODE_GENDER"] != "XNA"]

	# Categorical features with Binary encode (0 or 1: two categories)
	for bin_features in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
		df[bin_features], uniques = pd.factorize(df[bin_features])

	# Categorical features with One-Hot encode
	df, cat_cols = oneHotEncoding(df, nan_as_category)

	# NaN values for DAYS_EMPLOYED: 365,243 -> nan
	df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace = True)

	# Some simple new features (percentages)
	df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
	df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
	df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
	df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
	df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

	del test_df
	gc.collect()

	return df


# ------------------------- BUREAU PIPELINE -------------------------

def bureau_and_balance(num_rows = None, nan_as_category = True):
	"""
	Preprocess bureau.csv and bureau_balance.csv
	:param num_rows:
	:param nan_as_category:
	:return:
	"""
	bureau = pd.read_csv("./data/bureau.csv", nrows = num_rows)
	bb = pd.read_csv("./data/bureau_balance.csv", nrows = num_rows)
	bureau, bureau_cat = oneHotEncoding(bureau, nan_as_category)
	bb, bb_cat = oneHotEncoding(bb, nan_as_category)

	# Bureau balance: Perform aggregations and merge with bureau.csv
	bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
	for col in bb_cat:
		bb_aggregations[col] = ["mean"]
	bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
	bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
	bureau = bureau.join(bb_agg, how = "left", on = "SK_ID_BUREAU")
	bureau.drop(["SK_ID_BUREAU"], axis = 1, inplace = True)
	del bb, bb_agg
	gc.collect()

	# Bureau and bureau_balance numeric features
	num_aggregations = {
		'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
	}

	# Bureau and bureau_balance categorical features
	cat_aggregations = {}
	for cat in bureau_cat:
		cat_aggregations[cat] = ['mean']
	for cat in bb_cat:
		cat_aggregations[cat + '_MEAN'] = ['mean']

	bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
	bureau_agg.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() for e in bureau_agg.columns.tolist()])

	# Bureau: Active credits - using only numerical aggregations
	active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
	active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
	active_agg.columns = pd.Index(['ACTIVE_' + e[0] + '_' + e[1].upper() for e in active_agg.columns.tolist()])
	bureau_agg = bureau_agg.join(active_agg, how = 'left', on = 'SK_ID_CURR')
	del active, active_agg

	# Bureau: Closed credits - using only numerical aggregations
	closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
	closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
	closed_agg.columns = pd.Index(['CLOSED_' + e[0] + '_' + e[1].upper() for e in closed_agg.columns.tolist()])
	bureau_agg = bureau_agg.join(closed_agg, how = 'left', on = 'SK_ID_CURR')
	del closed, closed_agg, bureau

	return bureau_agg



# ------------------------- PREVIOUS PIPELINE -------------------------

def previous_applications(num_rows = None, nan_as_category = True):
	prev = pd.read_csv("./data/previous_application.csv", nrows = num_rows)
	prev, prev_cat = oneHotEncoding(prev, nan_as_category)

	# Days 365.234 values -> nan
	prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
	prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
	prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
	prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
	prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

	# Add feature: value ask / value received percentage
	prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev['AMT_CREDIT']

	# Previous applications numeric features
	num_aggregations = {
		'AMT_ANNUITY': ['min', 'max', 'mean'],
		'AMT_APPLICATION': ['min', 'max', 'mean'],
		'AMT_CREDIT': ['min', 'max', 'mean'],
		'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
		'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
		'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
		'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
		'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
		'DAYS_DECISION': ['min', 'max', 'mean'],
		'CNT_PAYMENT': ['mean', 'sum'],
	}

	# Previous applications categorical features
	cat_aggregations = {}
	for cat in prev_cat:
		cat_aggregations[cat] = ['mean']

# ------------------------- POS-CASH PIPELINE -------------------------

def pos_cash(num_rows = None, nan_as_category = True):
	"""
	Preprocess POS_CASH_balance.csv
	:param num_rows:
	:param nan_as_category:
	:return:
	"""
	pos = pd.read_csv('./data/POS_CASH_balance.csv', nrows = num_rows)
	pos, pos_cat = oneHotEncoding(pos, nan_as_category)
	# Features
	aggregations = {
		'MONTHS_BALANCE': ['max', 'mean', 'size'],
		'SK_DPD': ['max', 'mean'],
		'SK_DPD_DEF': ['max', 'mean']
	}
	for cat in pos_cat:
		aggregations[cat] = ['mean']
	pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
	pos_agg.columns = pd.Index(['POS_' + e[0] + '_' + e[1].upper() for e in pos_agg.columns.tolist()])
	# Count pos cash accounts
	pos_agg['POS_COUNT'] = pos_agg.groupby('SK_ID_CURR').size()
	del pos
	gc.collect()
	return pos_agg

# ------------------------- INSTALLMENTS PIPELINE -------------------------

def installments_payments(num_rows = None, nan_as_category = True):
	"""
	Preprocess installments_payments.csv
	:param num_rows:
	:param nan_as_category:
	:return:
	"""
	ins = pd.read_csv('./data/installments_payments.csv', nrows = num_rows)
	ins, ins_cat = oneHotEncoding(ins, nan_as_category)

	# Percnetage and difference paid in each installment (amount paid and installment value)
	ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
	ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

	# Days past due and days before due (no negative values)
	ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
	ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
	ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
	ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

	# Feature: Perform aggregations
	aggregations = {
		'NUM_INSTALMENT_VERSION': ['nunique'],
		'DPD': ['max', 'mean', 'sum'],
		'DBD': ['max', 'mean', 'sum'],
		'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
		'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
		'AMT_INSTALMENT': ['max', 'mean', 'sum'],
		'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
		'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
	}
	for cat in ins_cat:
		aggregations[cat] = ['mean']
	ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
	ins_agg.colunms = pd.Index(['INSTAL_' + e[0] + '_' + e[1].upper() for e in ins_agg.columns.tolist()])

	# Count installments accounts
	ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
	del ins
	gc.collect()

	return ins_agg


# ------------------------- CREDIT CARD PIPELINE -------------------------

def credit_card_balance(num_rows = None, nan_as_category = True):
	"""
	Preprocess credit_card_balance.csv
	:param num_rows:
	:param nan_as_category:
	:return:
	"""
	cc = pd.read_csv('./data/credit_card_balance.csv', nrows = num_rows)
	cc, cc_cat = oneHotEncoding(cc, nan_as_category)

	# General aggregations
	cc.drop(['SK_ID_PREV'], axis = 1, inplace = True)
	cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
	cc_agg.columns = pd.Index(['CC_' + e[0] + '_' + e[1].upper() for e in cc_agg.columns.tolist()])

	# Count credit card lines
	cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
	del cc
	gc.collect()

	return cc_agg




# ------------------------- LIGHTGBM MODEL -------------------------

def kfold_lightgbm(df, num_folds, stratified = False, debug = False):
	"""
	LightGBM with KFold or Stratified KFold
	Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
	:param df:
	:param num_folds:
	:param stratified:
	:param debug:
	:return:
	"""
	# Divide in training / validation and testing data
	train_df = df[df['TARGET'].notnull()]
	test_df = df[df['TARGET'].isnull()]
	print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
	del df
	gc.collect()

	# Cross validation model
	if stratified:
		folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
	else:
		folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)

	# Create array and dataframes to store results
	oof_preds = np.zeros(train_df.shape[0])
	sub_preds = np.zeros(test_df.shape[0])
	feature_importance_df = pd.DataFrame()
	feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV', 'index']]
	for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
		train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
		valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

		# LightGBM parameters found by Bayesian optimization
		clf = LGBMClassifier(
			nthread = 4,
			n_estimators = 10000,
			learning_rate = 0.02,
			num_leaves = 34,
			colsample_bytree = 0.9497036,
			subsample = 0.8715623,
			max_depth = 8,
			reg_alpha = 0.041545473,
			reg_lambda = 0.0735294,
			min_split_gain = 0.0222415,
			min_child_weight = 39.3259775,
			silent = -1,
			verbose = -1,
		)
		clf.fit(train_x, train_y,
				eval_set = [(train_x, train_y), (valid_x, valid_y)],
				eval_metric = 'auc',
				verbose = 200,
				early_stopping_rounds = 200)
		oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, -1]
		sub_preds += clf.predict_proba(test_df[feats], num_iteration = clf.best_iteration_)[:, -1] / folds.n_splits

		fold_importance_df = pd.DataFrame()
		fold_importance_df['feature'] = feats
		fold_importance_df['importance'] = clf.feature_importances_
		fold_importance_df['fold'] = n_fold + 1
		feature_importance_df = pd.concat([feature_importance_df, fold_importance_df])
		print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
		del clf, train_x, train_y, valid_x, valid_y
		gc.collect()

	print("Full AUC score %.6f" % roc_auc_score(train_df['TARGET'], oof_preds))

	# Write submission file and plot feature importance
	if not debug:
		test_df['TARGET'] = sub_preds
		test_df[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join('./submission/', submission_file_name), index = False)

	# Display feature importance
	display_importances(feature_importance_df)

	return feature_importance_df


def display_importances(feature_importance_df_):
	"""
	Display / plot feature importance
	:param feature_importance_df_:
	:return:
	"""
	cols = feature_importance_df_[['feature', 'importance']] \
			   .groupby('feature') \
			   .mean() \
			   .sort_values(by = 'importance', ascending = False)[:40].index
	best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
	plt.figure(figsize = (8, 10))
	sns.barplot(x = 'importance',
				y = 'feature',
				data = best_features.sort_values(by = 'importance', ascending = False))
	plt.title("LightGBM Features (avg over folds)")
	plt.tight_layout()
	plt.savefig("lgbm_importances01.png")




if __name__ == "__main__":
	submission_file_name = "submission_kernel01.csv"
	with timer("Full model run"):
		main()
