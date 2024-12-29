# -*- coding: utf-8 -*-

__author__ = "wangzhefeng"


import pymssql
from sqlalchemy import create_engine
import pandas as pd


def get_engine():
	db_info = {
		'host': 'WANGZF-PC',
		'port': '1433',
		'user': 'tinker.wang',
		'password': 'alvin123',
		'database': 'tinker'
	}
	conn_info = 'mssql+pymssql://%(user)s:%(password)s@%(host)s:%(port)s/%(database)s' % db_info
	engine = create_engine(conn_info, encoding = 'utf-8')
	return engine


def write_db(df, table_name, engine):
	pd.io.sql.to_sql(df,
					 name = table_name,
					 con = engine,
					 if_exists = "append",
					 index = False)


def main():
	engine = get_engine()
	# train_df = pd.read_csv("./data/application_train.csv")
	test_df = pd.read_csv("./data/application_test.csv")
	# bureau = pd.read_csv("./data/bureau.csv")
	# bb = pd.read_csv("./data/bureau_balance.csv")
	# prev = pd.read_csv("./data/previous_application.csv")
	# pos = pd.read_csv('./data/POS_CASH_balance.csv')
	# ins = pd.read_csv('./data/installments_payments.csv')
	# cc = pd.read_csv('./data/credit_card_balance.csv')
	print(test_df.head())
	train_name = "application_train"
	test_name = "application_test"
	bureau_name = "bureau"
	bb_name = "bureau_balance"
	prev_name = "previous_application"
	pos_name = "POS_CASH_balance"
	ins_name = "installments_payments"
	cc_name = "credit_card_balance"
	write_db(test_df, test_name, engine)




if __name__ == '__main__':
	main()