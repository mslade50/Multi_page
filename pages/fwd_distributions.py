import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
from datetime import timedelta
import smtplib, ssl
from pandas import ExcelWriter
from email.message import EmailMessage
import yfinance as yf
from yahoo_fin import options as op
import pandas as pd
import pandas_datareader.data as web
import scipy.stats
from scipy import stats
import ta
from pandas_datareader import data as pdr
from matplotlib import cm
import plotly.io as pio
from mpl_toolkits import mplot3d
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.stats import norm
from yahoo_fin import stock_info as si
import streamlit as st

# ['^NDX','^GSPC','^RUT','^SOX','^DJT','^DJI','XLE','XLF','ETH-USD','BTC-USD','CL=F','HG=F','GC=F','SI=F','RB=F','NG=F','AAPL','TSLA','MSFT','OXY','JPM','GS','WMT','TLT','HYG',
# 'FFTY','MTUM','XBI','EEM','DX-Y.NYB','EURUSD=X','AUDUSD=X','CHFUSD=X','CADUSD=X','^VIX','ZC=F']

###cycle years are 1.pre-election 2.election 3.post election 4.midterm

ticker='^GSPC'
cycle_start=1951
cycle_label='Third Year of Cycle'
cycle_var='pre_election'

def seasonal_return_ranks(ticker,end):
	spx1=yf.Ticker(ticker)
	spx = spx1.history(period="max")

	spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))*100

	spx["day_of_month"] = spx.index.day
	spx['day_of_year'] = spx.index.day_of_year
	spx['month'] = spx.index.month
	spx['Fwd_5dR']=spx.log_return.shift(-5).rolling(window=5).sum().round(2)
	spx['Fwd_10dR']=spx.log_return.shift(-10).rolling(window=10).sum().round(2)
	spx['Fwd_21dR']=spx.log_return.shift(-21).rolling(window=21).sum().round(2)
	spx["year"] = spx.index.year

	#second dataframe explicity to count the number of trading days so far this year
	now = dt.datetime.now()+timedelta(days=1)
	days = yf.download(ticker, start="2023-01-01", end=now)
	days["log_return"] = np.log(days["Close"] / days["Close"].shift(1))*100
	days['day_of_year'] = days.index.day_of_year
	days['this_yr']=days.log_return.cumsum()


	#create your list of all years
	start=spx['year'].min()
	stop=end
	r=range(0,(stop+3-start+1),1)
	print(start)
	years=[]
	for i in r:
		j=start+i
		years.append(j)
	print(years)

	def yearly(time):
			rslt_df2 = spx.loc[spx['year']==time] 
			grouped_by_day = rslt_df2.groupby("day_of_year").log_return.mean()
			day_by_day=[]
			for day in grouped_by_day:
				cum_return = day
				day_by_day.append(cum_return)
			return day_by_day

	def yearly_5d(time):
		rslt_df2 = spx.loc[spx['year']==time]
		fwd_5_by_day=rslt_df2.groupby("day_of_year").Fwd_5dR.mean()
		day_by_day_5=[]
		for day in fwd_5_by_day:
			cum_return = day
			day_by_day_5.append(cum_return)
		return day_by_day_5

	def yearly_10d(time):
		rslt_df2 = spx.loc[spx['year']==time]
		fwd_5_by_day=rslt_df2.groupby("day_of_year").Fwd_10dR.mean()
		day_by_day_5=[]
		for day in fwd_5_by_day:
			cum_return = day
			day_by_day_5.append(cum_return)
		return day_by_day_5

	def yearly_21d(time):
		rslt_df2 = spx.loc[spx['year']==time]
		fwd_5_by_day=rslt_df2.groupby("day_of_year").Fwd_21dR.mean()
		day_by_day_5=[]
		for day in fwd_5_by_day:
			cum_return = day
			day_by_day_5.append(cum_return)
		return day_by_day_5


	yr_master=[]
	for year in years:
		yearly(year)
		yr_master.append(yearly(year))

	#create list of midterm years
	l=range(0,19,1)
	years_mid=[]
	for i in l:
		j=end-i*4
		years_mid.append(j)
	print(years_mid)
	years_mid2=[]
	for i in l:
		j=end-1-(i*4)
		years_mid2.append(j)

	years_mid3=[]
	for i in l:
		j=end-2-(i*4)
		years_mid3.append(j)

	years_mid4=[]
	for i in l:
		j=end+1-(i*4)
		years_mid4.append(j)

	###cycle years are 1.pre-election 2.election 3.post election 4.midterm

	yr_mid_master=[]
	yr_mid_master2=[]
	yr_mid_master3=[]
	yr_mid_master4=[]
	for year in years_mid:
		yearly(year)
		yr_mid_master.append(yearly(year))
	for year in years_mid2:
		yearly(year)
		yr_mid_master2.append(yearly(year))
	for year in years_mid3:
		yearly(year)
		yr_mid_master3.append(yearly(year))
	for year in years_mid4:
		yearly(year)
		yr_mid_master4.append(yearly(year))

	yr_master2=[]
	for year in years:
		yearly_5d(year)
		yr_master2.append(yearly_5d(year))

	yr_master_mid2=[]
	yr_master_mid22=[]
	yr_master_mid23=[]
	yr_master_mid24=[]
	for year in years_mid:
		yearly_5d(year)
		yr_master_mid2.append(yearly_5d(year))
	for year in years_mid2:
		yearly_5d(year)
		yr_master_mid22.append(yearly_5d(year))
	for year in years_mid3:
		yearly_5d(year)
		yr_master_mid23.append(yearly_5d(year))
	for year in years_mid4:
		yearly_5d(year)
		yr_master_mid24.append(yearly_5d(year))

	yr_master3=[]
	for year in years:
		yearly_10d(year)
		yr_master3.append(yearly_10d(year))

	yr_master_mid3=[]
	yr_master_mid32=[]
	yr_master_mid33=[]
	yr_master_mid34=[]
	for year in years_mid:
		yearly_10d(year)
		yr_master_mid3.append(yearly_10d(year))
	for year in years_mid2:
		yearly_10d(year)
		yr_master_mid32.append(yearly_10d(year))
	for year in years_mid3:
		yearly_10d(year)
		yr_master_mid33.append(yearly_10d(year))
	for year in years_mid4:
		yearly_10d(year)
		yr_master_mid34.append(yearly_10d(year))

	yr_master4=[]
	for year in years:
		yearly_21d(year)
		yr_master4.append(yearly_21d(year))

	yr_master_mid4=[]
	yr_master_mid42=[]
	yr_master_mid43=[]
	yr_master_mid44=[]
	for year in years_mid:
		yearly_21d(year)
		yr_master_mid4.append(yearly_21d(year))
	for year in years_mid2:
		yearly_21d(year)
		yr_master_mid42.append(yearly_21d(year))
	for year in years_mid3:
		yearly_21d(year)
		yr_master_mid43.append(yearly_21d(year))
	for year in years_mid4:
		yearly_21d(year)
		yr_master_mid44.append(yearly_21d(year))



		###you are now converting your lists of returns into dataframes, and then manipulating the resulting data to get averages across all years from the same day.
	###this process is repeated for each cycle year, and for 5d 10d and 21d forward returns. 
	df_all_5d=pd.DataFrame(yr_master2).round(3)
	df_all_5d_mean=df_all_5d.mean().round(2)
	rank=df_all_5d.rank(pct=True).round(3)*100

	df_mt_5d=pd.DataFrame(yr_master_mid2).round(3)
	df_mt_5d_mean=df_mt_5d.mean().round(2)
	rank2=df_mt_5d.rank(pct=True).round(3)*100

	df_mt2_5d=pd.DataFrame(yr_master_mid22).round(3)
	df_mt2_5d_mean=df_mt2_5d.mean().round(2)
	rank22=df_mt2_5d.rank(pct=True).round(3)*100

	df_mt3_5d=pd.DataFrame(yr_master_mid23).round(3)
	df_mt3_5d_mean=df_mt3_5d.mean().round(2)
	rank23=df_mt3_5d.rank(pct=True).round(3)*100

	df_mt4_5d=pd.DataFrame(yr_master_mid24).round(3)
	df_mt4_5d_mean=df_mt4_5d.mean().round(2)
	rank24=df_mt4_5d.rank(pct=True).round(3)*100

	df_all_10d=pd.DataFrame(yr_master3).round(3)
	df_all_10d_mean=df_all_10d.mean().round(2)
	rank3=df_all_10d.rank(pct=True).round(3)*100

	df_mt_10d=pd.DataFrame(yr_master_mid3).round(3)
	df_mt_10d_mean=df_mt_10d.mean().round(2)
	rank4=df_mt_10d.rank(pct=True).round(3)*100

	df_mt2_10d=pd.DataFrame(yr_master_mid32).round(3)
	df_mt2_10d_mean=df_mt2_10d.mean().round(2)
	rank42=df_mt2_10d.rank(pct=True).round(3)*100

	df_mt3_10d=pd.DataFrame(yr_master_mid33).round(3)
	df_mt3_10d_mean=df_mt3_10d.mean().round(2)
	rank43=df_mt3_10d.rank(pct=True).round(3)*100

	df_mt4_10d=pd.DataFrame(yr_master_mid34).round(3)
	df_mt4_10d_mean=df_mt4_10d.mean().round(2)
	rank44=df_mt4_10d.rank(pct=True).round(3)*100

	df_all_21d=pd.DataFrame(yr_master4).round(3)
	df_all_21d_mean=df_all_21d.mean().round(2)
	rank5=df_all_21d_mean.rank(pct=True).round(3)*100

	df_mt_21d=pd.DataFrame(yr_master_mid4).round(3)
	df_mt_21d_mean=df_mt_21d.mean().round(2)
	rank6=df_mt_21d_mean.rank(pct=True).round(3)*100

	df_mt2_21d=pd.DataFrame(yr_master_mid42).round(3)
	df_mt2_21d_mean=df_mt2_21d.mean().round(2)
	rank62=df_mt2_21d_mean.rank(pct=True).round(3)*100

	df_mt3_21d=pd.DataFrame(yr_master_mid43).round(3)
	df_mt3_21d_mean=df_mt3_21d.mean().round(2)
	rank63=df_mt3_21d_mean.rank(pct=True).round(3)*100

	df_mt4_21d=pd.DataFrame(yr_master_mid44).round(3)
	df_mt4_21d_mean=df_mt4_21d.mean().round(2)
	rank64=df_mt4_21d_mean.rank(pct=True).round(3)*100

	pre_election=[]
	election=[]
	post_election=[]
	midterms=[]

	pre_election_list=[df_mt_5d_mean,df_mt_10d_mean,df_mt_21d_mean]
	election_list=[df_mt2_5d_mean,df_mt2_10d_mean,df_mt2_21d_mean]
	post_election_list=[df_mt3_5d_mean,df_mt3_10d_mean,df_mt3_21d_mean]
	midterms_list=[df_mt4_5d_mean,df_mt4_10d_mean,df_mt4_21d_mean]

	for g in pre_election_list:
	    pre_election.append(g)
	pre_election_df=pd.DataFrame(pre_election).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	pre_election_df['avg']=pre_election_df.mean(axis=1).round(2)

	for g in election_list:
	    election.append(g)
	election_df=pd.DataFrame(election).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	election_df['avg']=election_df.mean(axis=1).round(2)

	for g in post_election_list:
	    post_election.append(g)
	post_election_df=pd.DataFrame(post_election).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	post_election_df['avg']=post_election_df.mean(axis=1).round(2)

	for g in midterms_list:
	    midterms.append(g)
	midterms_df=pd.DataFrame(midterms).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	midterms_df['avg']=midterms_df.mean(axis=1).round(2)

	cycles_df=pd.concat([pre_election_df['avg'],election_df['avg'],post_election_df['avg'],midterms_df['avg']],axis=1,keys=['pre_election','election','post_election','midterms'])
	cycles_df=cycles_df.stack().reset_index()
	cycles_df.columns.values[2]="avg"
	cycles_df.columns.values[1]=ticker
	cycles_df['rnk']=cycles_df.avg.rank(pct=True).round(3)*100


	length=len(days)

	print_df=cycles_df[cycles_df[ticker] == cycle_var].reset_index(drop=True)
	# print_df=print_df[print_df['level_0'] == length]
	# print_df=print_df.reset_index(drop=True)
	# true_cycle_rnk=print_df['rnk'].iat[-1].round(1)


	returns=[]
	tuples=[df_all_5d_mean,df_mt_5d_mean,df_all_10d_mean,df_mt_10d_mean,df_all_21d_mean,df_mt_21d_mean]
	for data in tuples:
	    returns.append(data)
	new_df=pd.DataFrame(returns).transpose().rename(columns={
	                                                        0:'Fwd_R5',1:'Fwd_R5_MT',
	                                                        2:'Fwd_R10',3:'Fwd_R10_MT',
	                                                        4:'Fwd_R21',5:'Fwd_R21_MT'      
	})

	#5d stuff
	new_df['Returns_5_rnk']=new_df.Fwd_R5.rank(pct=True).round(3)*100
	new_df['Returns_5_rnk_mt']=new_df.Fwd_R5_MT.rank(pct=True).round(3)*100

	r_5=new_df['Fwd_R5'][[length]].round(2)
	r_5_mt=new_df['Fwd_R5_MT'][[length]].round(2)
	r_5_ptile=new_df['Returns_5_rnk'][[length]].round(2)
	r_5_ptile_mt=new_df['Returns_5_rnk_mt'][[length]].round(2)

	#10d stuff
	length=len(days)
	new_df['Returns_10_rnk']=new_df.Fwd_R10.rank(pct=True).round(3)*100
	new_df['Returns_10_rnk_mt']=new_df.Fwd_R10_MT.rank(pct=True).round(3)*100

	r_10=new_df['Fwd_R10'][[length]].round(2)
	r_10_mt=new_df['Fwd_R10_MT'][[length]].round(2)
	r_10_ptile=new_df['Returns_10_rnk'][[length]].round(2)
	r_10_ptile_mt=new_df['Returns_10_rnk_mt'][[length]].round(2)

	#21d stuff
	new_df['Returns_21_rnk']=new_df.Fwd_R21.rank(pct=True).round(3)*100
	new_df['Returns_21_rnk_mt']=new_df.Fwd_R21_MT.rank(pct=True).round(3)*100

	##Calculate average ranks across the row
	new_df['Returns_all_avg']=new_df[['Returns_21_rnk','Returns_10_rnk','Returns_5_rnk']].mean(axis=1)
	new_df['Returns_all_avg_mt']=new_df[['Returns_21_rnk_mt','Returns_10_rnk_mt','Returns_5_rnk_mt']].mean(axis=1)
	new_df['Returns_all_avg_10dt']=new_df['Returns_all_avg'].rolling(window=10).mean().shift(1)
	new_df['Returns_all_avg_mt_10dt']=new_df['Returns_all_avg_mt'].rolling(window=10).mean().shift(1)
	new_df['Seasonal_delta']=new_df.Returns_all_avg - new_df.Returns_all_avg_10dt
	new_df['Seasonal_delta_cycle']=new_df.Returns_all_avg_mt - new_df.Returns_all_avg_mt_10dt

	r_21=new_df['Fwd_R21'][[length]].round(2)
	r_21_mt=new_df['Fwd_R21_MT'][[length]].round(2)
	r_21_ptile=new_df['Returns_21_rnk'][[length]].round(2)
	r_21_ptile_mt=new_df['Returns_21_rnk_mt'][[length]].round(2)
	new_df['Returns_rnk_avg']=new_df[['Returns_21_rnk','Returns_10_rnk','Returns_5_rnk']].mean(axis=1)

	##Output
	# all_5d=r_5_ptile.values[0]
	# mt_5d=r_5_ptile_mt.values[0]
	# all_10d=r_10_ptile.values[0]
	# mt_10d=r_10_ptile_mt.values[0]
	# all_21d=r_21_ptile.values[0]
	# mt_21d=r_21_ptile_mt.values[0]
	# all_avg=((all_5d+all_10d+all_21d)/3).round(2)
	# cycle_avg=true_cycle_rnk
	# total_avg=((all_avg+true_cycle_rnk)/2).round(2)
	df2=pd.DataFrame().assign(Cycle_returns_rank=new_df['Returns_all_avg_mt'],Returns_rank=new_df['Returns_rnk_avg'])

	df2['Rank'] = (3 * df2['Cycle_returns_rank'] + df2['Returns_rank']) / 4
	df3=pd.DataFrame().assign(Expectancy=df2['Rank'])
	# df3=pd.DataFrame().assign(Expectancy=df2['Cycle_returns_rank'])
	df4=df3.values.tolist()

	return df4 
yf.pdr_override() 
# ticker = '^VIX'
# tgt_date_range=12
# PF_size=70000
# MDM_ptile=2
# cycle_rnk=5
# direction="Short"
# end_date=(dt.date.today()+dt.timedelta(days=1)).strftime('%Y-%m-%d') 
	# end_date = '2020-02-18'  
def fig_creation(ticker,tgt_date_range,end_date,sigma,days,atr):
	end_date = end_date
	tgt_date_range=tgt_date_range
	data = pdr.get_data_yahoo(ticker,end=end_date) 
	ticker=ticker
	# Find the year of data inception
	start_year = data.index.min().year


	# Start the loop 8 years after data inception
	start_year += 4

	# Define the current year
	end_date_dt = end_date
	current_year = end_date_dt.year
	last_close=data['Close'][-1]
	def mdm(ticker, years=list(range(start_year, current_year-3))):	 
		dfs = []
		for year in years:
			try:
				g = seasonal_return_ranks(ticker, year)
				df_year = pd.DataFrame(g, columns=[ticker])
				df_year['Year'] = year + 4  # Adjust here
				df_year['IndexWithinYear'] = df_year.groupby('Year').cumcount()
				df_year['Average'] = df_year[ticker]
				df_year['Average_rnk'] = (df_year['Average']/10).round(0)
				dfs.append(df_year)
			except Exception:
				print(f"No data on {ticker} for the year {year}")

		df_final = pd.concat(dfs)
	    
		# Load your CSV file
		csv_url = "https://raw.githubusercontent.com/mslade50/Multi_page/main/trading_days.csv"
		trading_dates_df = pd.read_csv(csv_url)

		# Convert 'Date' column to datetime and extract year
		trading_dates_df['Date'] = pd.to_datetime(trading_dates_df['Date'])
		trading_dates_df['Year'] = trading_dates_df['Date'].dt.year
		trading_dates_df['IndexWithinYear'] = trading_dates_df.groupby('Year').cumcount()

		# Merge df_final and trading_dates_df based on 'Year' and 'IndexWithinYear'
		df_final = pd.merge(df_final.reset_index(drop=True), trading_dates_df, on=['Year', 'IndexWithinYear']).set_index('Date')
		
		return df_final

	df = mdm(ticker)
	data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

	# Calculate performance with respect to ATR
	# List of lookbacks
	lookbacks = [5, 21, 63, 126, 252]
	most_recent_atr = data['ATR'].iloc[-1]

	# Calculate Trailing and Forward returns for each lookback
	for lookback in lookbacks:
		data[f'Trailing_{lookback}d'] = (data['Close'] - data['Close'].shift(lookback)) / data['ATR'].shift(lookback)
		if atr == "atr":
			data[f'Forward_{lookback}d'] = (data['Close'].shift(-lookback) - data['Close']) / data['ATR'].shift(-lookback)
		else:
			data[f'Forward_{lookback}d'] = (data['Close'].shift(-lookback) - data['Close']) / data['Close']
	for lookback in lookbacks:
		if atr == "atr":
			data[f'Trailing_{lookback}d_pct_rank'] = (data[f'Trailing_{lookback}d'].rank(pct=True) * 10).round(0)
			data[f'Forward_{lookback}d_pct_rank'] = (data[f'Forward_{lookback}d'] * most_recent_atr)/data['Close'][-1]*100
		else:
			data[f'Trailing_{lookback}d_pct_rank'] = (data[f'Trailing_{lookback}d'].rank(pct=True) * 10).round(0)
			data[f'Forward_{lookback}d_pct_rank'] = data[f'Forward_{lookback}d'] * 100
	for lookback in lookbacks:
		data[f'Forward_{lookback}d_pct_rank'].fillna(0, inplace=True)
		data['pct_change'] = data['Close'].pct_change()
		
	data['pct_change'] = data['Close'].pct_change()
	# Square the percentage changes
	data['squared_pct_change'] = data['pct_change'] ** 2

	# Calculate the alternative volatility measure for various windows
	windows = [5, 21, 63]  # the desired time windows for realized volatility
	for window in windows:
	    data[f'alt_vol_{window}d'] = data['squared_pct_change'].rolling(window).sum()/window

	# Calculate forward realized volatility change as a percentage increase 
	# vs a blended trailing average realized vol over the last 21d and 63d
	for window in windows:
	    blended_vol = (data['alt_vol_21d'] + data['alt_vol_63d']) / 2
	    data[f'forward_vol_change_{window}d'] = (data[f'alt_vol_{window}d'].shift(-window) - blended_vol) / blended_vol

	# Drop the NA values from the 'Trailing' columns
	columns_to_dropna = [f'Trailing_{lookback}d_pct_rank' for lookback in lookbacks]
	data.dropna(subset=columns_to_dropna, inplace=True)

	# Create a subset of the dataframe
	columns_subset = [f'Trailing_{lookback}d_pct_rank' for lookback in lookbacks] + [f'Forward_{lookback}d_pct_rank' for lookback in lookbacks]
	alt_vol_columns = [f'alt_vol_{window}d' for window in windows]
	forward_vol_change_columns = [f'forward_vol_change_{window}d' for window in windows]

	columns_subset.extend(alt_vol_columns)
	columns_subset.extend(forward_vol_change_columns)

	data_subset = data[columns_subset]
	# Perform the merge
	merged_df = pd.merge(df, data_subset, left_index=True, right_index=True)
	# filepath=r"C:\Users\McKinley\Dropbox\MS Docs\Work\Sublime_Misc\52whigh.py"


	def matplotlib_to_plotly(cmap, pl_entries):
	    h = 1.0/(pl_entries-1)
	    pl_colorscale = []

	    for k in range(pl_entries):
	        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
	        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

	    return pl_colorscale

	seismic_cmap= plt.cm.seismic
	seismic = matplotlib_to_plotly(seismic_cmap, 255)

	# create subplot structure
	# Calculate 85th percentile
	max_21d = np.percentile(merged_df['Forward_21d_pct_rank'], 95)
	max_5d = np.percentile(merged_df['Forward_5d_pct_rank'], 95)
	# Get the values from the last row of the DataFrame
	today_21d_rank = merged_df['Trailing_21d_pct_rank'].iloc[-1]
	today_5d_rank = merged_df['Trailing_5d_pct_rank'].iloc[-1]
	today_63d_rank = merged_df['Trailing_63d_pct_rank'].iloc[-1]
	today_126d_rank = merged_df['Trailing_126d_pct_rank'].iloc[-1]
	today_252d_rank = merged_df['Trailing_252d_pct_rank'].iloc[-1]
	today_avg_rank = merged_df['Average_rnk'].iloc[-1]
	# For 21d
	rows_for_21d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_21d_pct_rank'] == today_21d_rank)]
	mean_forward_21d_for_today = round(rows_for_21d['Forward_21d_pct_rank'].mean(), 2)
	median_forward_21d_for_today = round(rows_for_21d['Forward_21d_pct_rank'].median(), 2)
	sample_size_21d = len(rows_for_21d)


	# For 5d
	rows_for_5d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_5d_pct_rank'] == today_5d_rank)]
	mean_forward_5d_for_today = round(rows_for_5d['Forward_5d_pct_rank'].mean(), 2)
	median_forward_5d_for_today = round(rows_for_5d['Forward_5d_pct_rank'].median(), 2)
	sample_size_5d = len(rows_for_5d)

	# For 63d
	rows_for_63d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_63d_pct_rank'] == today_63d_rank)]
	mean_forward_63d_for_today = round(rows_for_63d['Forward_63d_pct_rank'].mean(), 2)
	median_forward_63d_for_today = round(rows_for_63d['Forward_63d_pct_rank'].median(), 2)
	sample_size_63d = len(rows_for_63d)

	# For 126d
	rows_for_126d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_126d_pct_rank'] == today_126d_rank)]
	mean_forward_126d_for_today = round(rows_for_126d['Forward_126d_pct_rank'].mean(), 2)
	median_forward_126d_for_today = round(rows_for_126d['Forward_126d_pct_rank'].median(), 2)
	sample_size_126d = len(rows_for_126d)

	# For 252d
	rows_for_252d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_252d_pct_rank'] == today_252d_rank)]
	mean_forward_252d_for_today = round(rows_for_252d['Forward_252d_pct_rank'].mean(), 2)
	median_forward_252d_for_today = round(rows_for_252d['Forward_252d_pct_rank'].median(), 2)
	sample_size_252d = len(rows_for_252d)



	# Adding shapes
	shapes = []

	# We need to add two lines per subplot: one vertical and one horizontal

	fig = make_subplots(
	    rows=2, cols=2,
	    vertical_spacing=0.05,
	    subplot_titles=("Forward_21d_pct_rank", "Forward_5d_pct_rank", "Forward_21d_pct_rank", "Forward_5d_pct_rank"),
	    specs=[[{"b":0.1}, {"b":0.1}],
	           [{"b":0.1}, {"b":0.1}]])

	fig.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_21d_pct_rank'], 
	    z=merged_df['Forward_21d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    showscale=False,  # Disable the colorbar
	    zsmooth='best',
	    zmin=-max_21d, 
	    zmax=max_21d),
	    row=1, col=1
	)

	fig.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_21d_pct_rank'], 
	    z=merged_df['Forward_5d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    colorbar=dict(len=0.36, y=0.82, tickfont=dict(color='black')),
	    zsmooth='best',
	    zmin=-max_5d, 
	    zmax=max_5d),
	    row=1, col=2)

	fig.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_5d_pct_rank'], 
	    z=merged_df['Forward_21d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    showscale=False,  # Disable the colorbar
	    zsmooth='best',
	    zmin=-max_21d, 
	    zmax=max_21d),
	    row=2, col=1
	)

	fig.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_5d_pct_rank'], 
	    z=merged_df['Forward_5d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    colorbar=dict(len=0.36, y=0.28, tickfont=dict(color='black')),
	    zsmooth='best',
	    zmin=-max_5d, 
	    zmax=max_5d),
	    row=2, col=2)

	# Update x-axis titles
	fig.update_xaxes(title_text="Seasonal Rank", title_font_color="black", row=2, col=1)
	fig.update_xaxes(title_text="Seasonal Rank", title_font_color="black", row=2, col=2)


	# Update y-axis labels
	fig.update_yaxes(title_text="Trailing 21d Rank",title_font_color="black", row=1, col=1)
	fig.update_yaxes(title_text="Trailing 5d Rank",title_font_color="black", row=2, col=1)
	fig.update_yaxes(row=1, col=1, tickfont=dict(color='black'))
	
	# For the second heatmap
	fig.update_yaxes(row=1, col=2, tickfont=dict(color='black'))
	
	# For the third heatmap
	fig.update_xaxes(title_text="Seasonal Rank", row=2, col=1, tickfont=dict(color='black'))
	fig.update_yaxes(row=2, col=1, tickfont=dict(color='black'))
	
	# For the fourth heatmap
	fig.update_xaxes(title_text="Seasonal Rank", row=2, col=2, tickfont=dict(color='black'))
	fig.update_yaxes(row=2, col=2, tickfont=dict(color='black'))
	
	# Adding lines to each subplot
	fig.add_hline(y=today_21d_rank, line_width=3, line_color="black", row=1, col=1)
	fig.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=1, col=1)
	fig.add_hline(y=today_21d_rank, line_width=3, line_color="black", row=1, col=2)
	fig.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=1, col=2)
	fig.add_hline(y=today_5d_rank, line_width=3, line_color="black", row=2, col=1)
	fig.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=2, col=1)
	fig.add_hline(y=today_5d_rank, line_width=3, line_color="black", row=2, col=2)
	fig.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=2, col=2)
	subplot_titles = ("Forward_21d_pct_rank", "Forward_5d_pct_rank", "Forward_21d_pct_rank", "Forward_5d_pct_rank")
	for i, title in enumerate(subplot_titles, 1):
	    fig.layout.annotations[i-1].update(font=dict(color="black"))
	# Set font colors for all subplots' x-axes and y-axes
	for axis in ['xaxis', 'xaxis2', 'yaxis', 'yaxis2']:
	    fig.update_layout({
	        axis: dict(
	            titlefont=dict(
	                color="black"
	            ),
	            tickfont=dict(
	                color="black"
	            )
	        )
	    })


	fig.update_layout(
	    title_text=f"Heatmaps for {ticker}",
	    title_font_color="black",
	    plot_bgcolor='white',
	    paper_bgcolor='white',
	    width=800,  # This is a typical width, but you may adjust as needed
	    height=600,  # Adjust based on your preferred height
	    legend=dict(
	        font=dict(
	            color="black"
	        )
	    )
	)
	
	
	def compute_distance(row, target_values, avg_rank_weight=2.5):
	    return np.sqrt(
	        avg_rank_weight * (row['Average_rnk'] - target_values[0])**2 +  # Increased importance
	        (row['Trailing_5d_pct_rank'] - target_values[1])**2 +
	        (row['Trailing_21d_pct_rank'] - target_values[2])**2 +
	        (row['Trailing_63d_pct_rank'] - target_values[3])**2 + 
	        (row['Trailing_126d_pct_rank'] - target_values[4])**2 + 
	        (row['Trailing_252d_pct_rank'] - target_values[5])**2
	    )

	# Assuming you've added today_63d_rank, today_126d_rank, and today_252d_rank for current situation values
	# Assuming you've added today_63d_rank, today_126d_rank, and today_252d_rank for current situation values
	current_values = [today_avg_rank, today_5d_rank, today_21d_rank, today_63d_rank, today_126d_rank, today_252d_rank]

	# Exclude the most recent 63 rows
	filtered_df = merged_df.iloc[:-63].copy()
	filtered_df.index = pd.to_datetime(filtered_df.index)  # Convert the index to a datetime object
	filtered_df = filtered_df[filtered_df.index.year > 2000] 

	# Calculate the distance for each row and sort the dataframe by these distances
	filtered_df['distance'] = filtered_df.apply(lambda row: compute_distance(row, current_values), axis=1)
	closest_rows = filtered_df.nsmallest(50, 'distance').round(2)


	# Compute the mean and median of Forward_21d_pct_rank from the 50 closest rows

	mean_forward_5d_closest = round(closest_rows['Forward_5d_pct_rank'].mean(), 2)
	median_forward_5d_closest = round(closest_rows['Forward_5d_pct_rank'].median(), 2)
	mean_forward_21d_closest = round(closest_rows['Forward_21d_pct_rank'].mean(), 2)
	median_forward_21d_closest = round(closest_rows['Forward_21d_pct_rank'].median(), 2)
	mean_forward_63d_closest = round(closest_rows['Forward_63d_pct_rank'].mean(), 2)
	median_forward_63d_closest = round(closest_rows['Forward_63d_pct_rank'].median(), 2)

	mean_forward_vol_change_5d_closest = round(closest_rows['forward_vol_change_5d'].mean(), 2)
	median_forward_vol_change_5d_closest = round(closest_rows['forward_vol_change_5d'].median(), 2)

	mean_forward_vol_change_21d_closest = round(closest_rows['forward_vol_change_21d'].mean(), 2)
	median_forward_vol_change_21d_closest = round(closest_rows['forward_vol_change_21d'].median(), 2)

	mean_forward_vol_change_63d_closest = round(closest_rows['forward_vol_change_63d'].mean(), 2)
	median_forward_vol_change_63d_closest = round(closest_rows['forward_vol_change_63d'].median(), 2)

	percentile_mean_forward_5d = round(stats.percentileofscore(filtered_df['Forward_5d_pct_rank'], mean_forward_5d_closest), 2)
	percentile_median_forward_5d = round(stats.percentileofscore(filtered_df['Forward_5d_pct_rank'], median_forward_5d_closest), 2)
	percentile_mean_forward_21d = round(stats.percentileofscore(filtered_df['Forward_21d_pct_rank'], mean_forward_21d_closest), 2)
	percentile_median_forward_21d = round(stats.percentileofscore(filtered_df['Forward_21d_pct_rank'], median_forward_21d_closest), 2)
	percentile_mean_forward_63d = round(stats.percentileofscore(filtered_df['Forward_63d_pct_rank'], mean_forward_63d_closest), 2)
	percentile_median_forward_63d = round(stats.percentileofscore(filtered_df['Forward_63d_pct_rank'], median_forward_63d_closest), 2)

	percentile_mean_forward_vol_5d = round(stats.percentileofscore(filtered_df['forward_vol_change_5d'], mean_forward_vol_change_5d_closest), 2)
	percentile_median_forward_vol_5d = round(stats.percentileofscore(filtered_df['forward_vol_change_5d'], median_forward_vol_change_5d_closest), 2)
	percentile_mean_forward_vol_21d = round(stats.percentileofscore(filtered_df['forward_vol_change_21d'], mean_forward_vol_change_21d_closest), 2)
	percentile_median_forward_vol_21d = round(stats.percentileofscore(filtered_df['forward_vol_change_21d'], median_forward_vol_change_21d_closest), 2)
	percentile_mean_forward_vol_63d = round(stats.percentileofscore(filtered_df['forward_vol_change_63d'], mean_forward_vol_change_63d_closest), 2)
	percentile_median_forward_vol_63d = round(stats.percentileofscore(filtered_df['forward_vol_change_63d'], median_forward_vol_change_63d_closest), 2)


	# Define the number of metrics based on tgt_date_range
	num_metrics = 2 if tgt_date_range <= 30 else 3

	# Calculate average of mean values for Forward Vol Change and their percentiles
	avg_mean_forward_vol_change = round((mean_forward_vol_change_5d_closest + mean_forward_vol_change_21d_closest + (0 if num_metrics == 2 else mean_forward_vol_change_63d_closest)) / num_metrics, 2)
	avg_percentile_mean_forward_vol = round((percentile_mean_forward_vol_5d + percentile_mean_forward_vol_21d + (0 if num_metrics == 2 else percentile_mean_forward_vol_63d)) / num_metrics, 2)

	# Calculate average of median values for Forward Vol Change and their percentiles
	avg_median_forward_vol_change = round((median_forward_vol_change_5d_closest + median_forward_vol_change_21d_closest + (0 if num_metrics == 2 else median_forward_vol_change_63d_closest)) / num_metrics, 2)
	avg_percentile_median_forward_vol = round((percentile_median_forward_vol_5d + percentile_median_forward_vol_21d + (0 if num_metrics == 2 else percentile_median_forward_vol_63d)) / num_metrics, 2)

	# Calculate average of mean values for Forward Returns and their percentiles
	avg_mean_forward_pct = round((mean_forward_5d_closest + mean_forward_21d_closest + (0 if num_metrics == 2 else mean_forward_63d_closest)) / num_metrics, 2)
	avg_percentile_mean_forward_pct = round((percentile_mean_forward_5d + percentile_mean_forward_21d + (0 if num_metrics == 2 else percentile_mean_forward_63d)) / num_metrics, 2)

	# Calculate average of median values for Forward Returns and their percentiles
	avg_median_forward_pct = round((median_forward_5d_closest + median_forward_21d_closest + (0 if num_metrics == 2 else median_forward_63d_closest)) / num_metrics, 2)
	avg_percentile_median_forward_pct = round((percentile_median_forward_5d + percentile_median_forward_21d + (0 if num_metrics == 2 else percentile_median_forward_63d)) / num_metrics, 2)

	# Calculate the final averages
	final_avg_forward_vol_change = round((avg_mean_forward_vol_change + avg_median_forward_vol_change) / 2, 2)
	final_avg_percentile_forward_vol = round((avg_percentile_mean_forward_vol + avg_percentile_median_forward_vol) / 2, 2)
	final_avg_forward_pct = round((avg_mean_forward_pct + avg_median_forward_pct) / 2, 2)
	final_avg_percentile_forward_pct = round((avg_percentile_mean_forward_pct + avg_percentile_median_forward_pct) / 2, 2)

	# Print the final averages
	
	S = last_close  # Current stock price
	sigma = sigma  # Annualized implied volatility
	T_days = days  # Time to expiration in days
	r = 0.03  # Risk-free rate

	# Convert days to years
	T = T_days / 252.0

	# Calculate the parameters for the distribution of the logarithm of the stock price
	mu = np.log(S) + (r - 0.5 * sigma**2) * T
	sigma_adjusted = sigma * np.sqrt(T)

	# Calculate the x-axis range based on ±4 standard deviations
	x_min = np.exp(mu - 4*sigma_adjusted)
	x_max = np.exp(mu + 4*sigma_adjusted)

	# Generate data points for the stock price and the associated PDF
	stock_prices = np.linspace(x_min, x_max, 400)
	pdf_values = norm.pdf(np.log(stock_prices), mu, sigma_adjusted) / stock_prices

	# Sample 50 random outcomes based on the log-normal distribution
	random_outcomes = np.random.lognormal(mean=mu, sigma=sigma_adjusted, size=50)

	# Plot
	fig6 = go.Figure()

	# Add the PDF line (y-axis on the left)
	fig6.add_trace(go.Scatter(x=stock_prices, y=pdf_values, mode='lines', name='Probability Density'))

	# Add histogram (y-axis on the right)
	fig6.add_trace(go.Histogram(x=random_outcomes, yaxis='y2', name='Sample Outcomes', opacity=0.7, nbinsx=40))

	# Update layout to include a secondary y-axis and adjust x-axis range
	fig6.update_layout(title='Naive Implied Probability Distribution & Sample Outcomes',
	                  xaxis_title='Stock Price',
	                  yaxis_title='Naive Distribution',
	                  yaxis2=dict(title='Sample Count', overlaying='y', side='right'),
	                  xaxis=dict(tickformat='$,.2f', range=[x_min, x_max]))
	
	mean_forward_prices = {}
	median_forward_prices = {}
	# Calculate the forward 21-day prices from the percentage returns
	for lookback in lookbacks:
		if atr== "atr":
			most_recent_atr = data['ATR'].iloc[-1]
			closest_rows[f'Forward_{lookback}d_price'] = last_close + (data[f'Forward_{lookback}d'] * most_recent_atr)
		else:
			closest_rows[f'Forward_{lookback}d_pct_proportion'] = closest_rows[f'Forward_{lookback}d_pct_rank'] / 100
			closest_rows[f'Forward_{lookback}d_price'] = last_close * (1 + closest_rows[f'Forward_{lookback}d_pct_proportion'])
		mean_forward_prices[f'{lookback}d'] = closest_rows[f'Forward_{lookback}d_price'].mean()
		median_forward_prices[f'{lookback}d'] = closest_rows[f'Forward_{lookback}d_price'].median()
	


	# 1. Limiting the x-axis to 4 standard deviations
	mean_price = closest_rows['Forward_21d_price'].mean()
	std_price = closest_rows['Forward_21d_price'].std()

	x_range = [mean_price - 4 * std_price, mean_price + 4 * std_price]

	# 2. Calculate KDE
	kde_x = np.linspace(x_range[0], x_range[1], 400)
	kde_y = stats.gaussian_kde(closest_rows['Forward_21d_price'].values)(kde_x)
	kde_trace = fig6.data[0]
	kde_trace.name = "Market Implied Distribution"
	kde_trace.line.color = 'gray'
	# Create subplots and specify secondary y-axis for KDE
	fig2 = make_subplots(specs=[[{"secondary_y": True}]])

	# Add histogram
	fig2.add_trace(
	    go.Histogram(
	        x=closest_rows['Forward_21d_price'],
	        nbinsx=40,
		showlegend=False
	    )
	)
	fig2.add_trace(kde_trace, secondary_y=True)

	# Add KDE line to secondary y-axis
	fig2.add_trace(
	    go.Scatter(x=kde_x, y=kde_y, mode='lines', line=dict(width=2, color='goldenrod'), name='Seasonal Implied Dist.'),
	    secondary_y=True
	)

	fig2.add_vline(x=mean_forward_prices['21d'], line_color="green")
	fig2.add_vline(x=median_forward_prices['21d'], line_color="blue")
	fig2.add_vline(x=last_close, line_color="white")
	# Add annotations at the left edge
	annotations_y = [0.95, 0.90, 0.85]  # positions to stack the annotations
	texts = [
	    f"Mean = {mean_forward_prices['21d']:.2f}",
	    f"Median = {median_forward_prices['21d']:.2f}",
	    f"Last Close = {last_close:.2f}"
	]
	colors = ["green", "blue", "white"]
	# Add annotations
	for i, (y, text, color) in enumerate(zip(annotations_y, texts, colors)):
	    fig2.add_annotation(
	        xref="paper",
	        yref="paper",
	        x=0,
	        y=y,
	        text=text,
	        showarrow=False,
	        font=dict(color=color, size=12)
	    )

	fig2.update_layout(annotations=dict(xanchor='left', xshift=10), title=f'Forward 21 Day Distribution for {ticker}')
	fig2.update_layout(yaxis2=dict(showticklabels=False))
	fig2.update_xaxes(range=x_range)  # limit x-axis to ±4 standard deviations


	# Creating a new 5x2 figure layout
	fig3 = make_subplots(
	    rows=2, cols=2,
	    vertical_spacing=0.05,
	    subplot_titles=("Forward_21d_pct_rank for Trailing_252d", "Forward_5d_pct_rank for Trailing_252d",
	                    "Forward_21d_pct_rank for Trailing_63d", "Forward_5d_pct_rank for Trailing_63d"),
	    specs=[[{"b":0.1}, {"b":0.1}],
	           [{"b":0.1}, {"b":0.1}]])

	fig3.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_252d_pct_rank'], 
	    z=merged_df['Forward_21d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    colorbar=dict(len=0.36, x=0.47,y=0.82),
	    zsmooth='best',
	    zmin=-max_21d, 
	    zmax=max_21d),
	    row=1, col=1)

	fig3.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_252d_pct_rank'], 
	    z=merged_df['Forward_5d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    colorbar=dict(len=0.36,y=0.82),
	    zsmooth='best',
	    zmin=-max_5d, 
	    zmax=max_5d),
	    row=1, col=2)

	fig3.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_63d_pct_rank'], 
	    z=merged_df['Forward_21d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    colorbar=dict(len=0.36, x=0.47,y=0.28),
	    zsmooth='best',
	    zmin=-max_21d, 
	    zmax=max_21d),
	    row=2, col=1)

	fig3.add_trace(go.Heatmap(
	    x=merged_df['Average_rnk'], 
	    y=merged_df['Trailing_63d_pct_rank'], 
	    z=merged_df['Forward_5d_pct_rank'], 
	    colorscale=seismic,
	    reversescale=True,
	    colorbar=dict(len=0.36,y=0.28),
	    zsmooth='best',
	    zmin=-max_5d, 
	    zmax=max_5d),
	    row=2, col=2)

	# Update x-axis titles
	fig3.update_xaxes(title_text="Seasonal Rank", row=1, col=1)
	fig3.update_xaxes(title_text="Seasonal Rank", row=1, col=2)
	fig3.update_xaxes(title_text="Seasonal Rank", row=2, col=1)
	fig3.update_xaxes(title_text="Seasonal Rank", row=2, col=2)

	# Update y-axis labels
	fig3.update_yaxes(title_text="Trailing 252d Rank", row=1, col=1)
	fig3.update_yaxes(title_text="Trailing 252d Rank", row=1, col=2)
	fig3.update_yaxes(title_text="Trailing 63d Rank", row=2, col=1)
	fig3.update_yaxes(title_text="Trailing 63d Rank", row=2, col=2)

	# Adding lines to each subplot
	fig3.add_hline(y=today_21d_rank, line_width=3, line_color="black", row=1, col=1)
	fig3.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=1, col=1)
	fig3.add_hline(y=today_21d_rank, line_width=3, line_color="black", row=1, col=2)
	fig3.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=1, col=2)
	fig3.add_hline(y=today_5d_rank, line_width=3, line_color="black", row=2, col=1)
	fig3.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=2, col=1)
	fig3.add_hline(y=today_5d_rank, line_width=3, line_color="black", row=2, col=2)
	fig3.add_vline(x=today_avg_rank, line_width=3, line_color="black", row=2, col=2)

	fig3.update_layout(title_text=f"Heatmaps for {ticker}", shapes=shapes)



	# Randomly sample 50 rows from the dataframe
	random_sample = merged_df.sample(50)

	# Compute the mean and median of Forward_21d_pct_rank from the random sample
	mean_forward_21d_random = round(random_sample['Forward_21d_pct_rank'].mean(), 2)
	median_forward_21d_random = round(random_sample['Forward_21d_pct_rank'].median(), 2)

	# Create the histogram for the random sample using px.histogram
	fig4 = px.histogram(
	    random_sample, 
	    x='Forward_21d_pct_rank',
	    nbins=40, 
	    title='Baseline Histogram of Forward 21d Returns for 50 Random Samples'
	)

	fig4.add_vline(x=mean_forward_21d_random, line_color="green")
	fig4.add_vline(x=median_forward_21d_random, line_color="blue")
	fig4.add_vline(x=0, line_color="black")

	# Add annotations at the left edge
	annotations_y_random = [0.95, 0.90]  # positions to stack the annotations
	texts_random = [
	    f"Mean = {mean_forward_21d_random}",
	    f"Median = {median_forward_21d_random}"
	]

	for i, (y, text, color) in enumerate(zip(annotations_y_random, texts_random, colors)):
	    fig4.add_annotation(
	        xref="paper",
	        yref="paper",
	        x=0, 
	        y=y, 
	        text=text,
	        showarrow=False,
	        font=dict(color=color, size=12)
	    )

	fig4.update_layout(annotations=dict(xanchor='left', xshift=10))

	T_days_5d = 5  # Time to expiration in days for the 5-day window

	# Convert days to years
	T_5d = T_days_5d / 252.0

	# Calculate the parameters for the distribution of the logarithm of the stock price
	mu_5d = np.log(S) + (r - 0.5 * sigma**2) * T_5d
	sigma_adjusted_5d = sigma * np.sqrt(T_5d)

	# Calculate the x-axis range based on ±4 standard deviations
	x_min_5d = np.exp(mu_5d - 4*sigma_adjusted_5d)
	x_max_5d = np.exp(mu_5d + 4*sigma_adjusted_5d)

	# Generate data points for the stock price and the associated PDF
	stock_prices_5d = np.linspace(x_min_5d, x_max_5d, 400)
	pdf_values_5d = norm.pdf(np.log(stock_prices_5d), mu_5d, sigma_adjusted_5d) / stock_prices_5d

	# Sample 50 random outcomes based on the log-normal distribution
	random_outcomes_5d = np.random.lognormal(mean=mu_5d, sigma=sigma_adjusted_5d, size=50)

	# Plot for 5 days
	fig9 = go.Figure()

	# Add the PDF line (y-axis on the left)
	fig9.add_trace(go.Scatter(x=stock_prices_5d, y=pdf_values_5d, mode='lines', name='Probability Density'))

	# Add histogram (y-axis on the right)
	fig9.add_trace(go.Histogram(x=random_outcomes_5d, yaxis='y2', name='Sample Outcomes', opacity=0.7, nbinsx=40))

	# Update layout to include a secondary y-axis and adjust x-axis range
	fig9.update_layout(title='Naive Implied Probability Distribution & Sample Outcomes (5d)',
	                   xaxis_title='Stock Price',
	                   yaxis_title='Naive Distribution',
	                   yaxis2=dict(title='Sample Count', overlaying='y', side='right'),
	                   xaxis=dict(tickformat='$,.2f', range=[x_min_5d, x_max_5d]))


	# Calculate mean and standard deviation for Forward_5d_price
	mean_price_5d = closest_rows['Forward_5d_price'].mean()
	std_price_5d = closest_rows['Forward_5d_price'].std()

	# Define the x range for KDE based on mean and standard deviation
	x_range_5d = [mean_price_5d - 4 * std_price_5d, mean_price_5d + 4 * std_price_5d]

	# Calculate KDE for Forward_5d_price
	kde_x_5d = np.linspace(x_range_5d[0], x_range_5d[1], 400)
	kde_y_5d = stats.gaussian_kde(closest_rows['Forward_5d_price'].values)(kde_x_5d)


	# Incorporate the probability density function from fig9
	pdf_trace_5d = go.Scatter(x=stock_prices_5d, y=pdf_values_5d, mode='lines', name='Probability Density')
	pdf_trace_5d.name="Market Implied Distribution"
	pdf_trace_5d.line.color = 'gray'



	fig5 = px.histogram(
	    closest_rows, 
	    x='Forward_5d_price',
	    nbins=40, 
	    title='Histogram of Forward 5d Prices with KDE & PDF for 50 Closest Samples'
	)

	# Add the KDE to fig5
	fig5.add_trace(go.Scatter(x=kde_x_5d, y=kde_y_5d, mode='lines', name='Seasonal Implied Dist.', yaxis='y2', line=dict(color='gold')))

	# Assuming you have already extracted the PDF trace from fig9
	fig5.add_trace(pdf_trace_5d.update(yaxis='y2'))  

	# Add vertical lines for mean, median, and last close
	fig5.add_vline(x=mean_forward_prices['5d'], line_color="green")
	fig5.add_vline(x=median_forward_prices['5d'], line_color="blue")
	fig5.add_vline(x=last_close, line_color="white")
	
	# Add annotations at the left edge
	annotations_y_5d = [0.95, 0.90, 0.85]  # positions to stack the annotations
	texts_5d = [
	    f"Mean = {mean_forward_prices['5d']:.2f}",
	    f"Median = {median_forward_prices['5d']:.2f}",
	    f"Last Close = {last_close:.2f}"
	]
	colors_5d = ["green", "blue", "white"]
	annotations_y_5d.extend([0.85])  # Add another position for the new annotation
	texts_5d.extend([f"Last Close = {last_close:.2f}"])
	colors.extend(["white"])
	for i, (y, text, color) in enumerate(zip(annotations_y_5d, texts_5d, colors)):
	    fig5.add_annotation(
	        xref="paper",
	        yref="paper",
	        x=0, 
	        y=y, 
	        text=text,
	        showarrow=False,
	        font=dict(color=color, size=12)
	    )
	fig5.update_layout(title_text=f'Forward 5 Day Distribution for {ticker}')
	fig5.update_layout(yaxis2=dict(showticklabels=False))
	fig5.update_layout(
	    yaxis2=dict(title=None, overlaying='y', side='right'),
	    annotations=dict(xanchor='left', xshift=10)
	)


	fig7 = make_subplots(
	    rows=2, cols=1,
	    shared_xaxes=True,
	    vertical_spacing=0.1,  # adjust this if needed for spacing between plots
	    subplot_titles=(f"My implied distribution for {ticker}", f"Naive implied distribution for {ticker}"),
	    row_heights=[0.5, 0.5], # Ensure both subplots have equal height
	    specs=[[{"secondary_y": True}],
	           [{"secondary_y": True}]]  # enable secondary y-axis for both rows
	)

	# Add the traces (plots) from fig2
	for trace in fig2.data:
	    secondary_y = "yaxis2" in trace.yaxis if hasattr(trace, "yaxis") and trace.yaxis else False
	    fig7.add_trace(trace, row=1, col=1, secondary_y=secondary_y)

	# Add the traces (plots) from fig6
	for trace in fig6.data:
	    secondary_y = "yaxis4" in trace.yaxis if hasattr(trace, "yaxis") and trace.yaxis else False
	    fig7.add_trace(trace, row=2, col=1, secondary_y=secondary_y)

	# Update any necessary layout details
	fig7.update_layout(
	    # combined layout properties go here
	)

	# fig7.show()
	S = last_close  # Current stock price
	sigma = sigma  # Annualized implied volatility
	T_days = 63  # Time to expiration in days, changed to 63 days
	r = 0.03  # Risk-free rate

	# Convert days to years
	T = T_days / 252.0

	# Calculate the parameters for the distribution of the logarithm of the stock price
	mu = np.log(S) + (r - 0.5 * sigma**2) * T
	sigma_adjusted = sigma * np.sqrt(T)

	# Calculate the x-axis range based on ±4 standard deviations
	x_min = np.exp(mu - 4*sigma_adjusted)
	x_max = np.exp(mu + 4*sigma_adjusted)

	# Generate data points for the stock price and the associated PDF
	stock_prices = np.linspace(x_min, x_max, 400)
	pdf_values = norm.pdf(np.log(stock_prices), mu, sigma_adjusted) / stock_prices

	# Sample 50 random outcomes based on the log-normal distribution
	random_outcomes = np.random.lognormal(mean=mu, sigma=sigma_adjusted, size=50)

	# Plot
	fig11 = go.Figure()  # Changed from fig6 to fig9

	# Add the PDF line (y-axis on the left)
	fig11.add_trace(go.Scatter(x=stock_prices, y=pdf_values, mode='lines', name='Probability Density'))

	# Add histogram (y-axis on the right)
	fig11.add_trace(go.Histogram(x=random_outcomes, yaxis='y2', name='Sample Outcomes', opacity=0.7, nbinsx=40))

	# Update layout to include a secondary y-axis and adjust x-axis range
	fig11.update_layout(title='Naive Implied Probability Distribution & Sample Outcomes',
	                  xaxis_title='Stock Price',
	                  yaxis_title='Naive Distribution',
	                  yaxis2=dict(title='Sample Count', overlaying='y', side='right'),
	                  xaxis=dict(tickformat='$,.2f', range=[x_min, x_max]))



	# 1. Limiting the x-axis to 4 standard deviations
	mean_price = closest_rows['Forward_63d_price'].mean()  # Changed 21d to 63d
	std_price = closest_rows['Forward_63d_price'].std()  # Changed 21d to 63d

	x_range = [mean_price - 4 * std_price, mean_price + 4 * std_price]

	# 2. Calculate KDE
	kde_x = np.linspace(x_range[0], x_range[1], 400)
	kde_y = stats.gaussian_kde(closest_rows['Forward_63d_price'].values)(kde_x)  # Changed 21d to 63d
	kde_trace = fig11.data[0]  # Refers to fig9 instead of fig6
	kde_trace.name = "Market Implied Distribution"
	kde_trace.line.color = 'gray'

	# Create subplots and specify secondary y-axis for KDE
	fig10 = make_subplots(specs=[[{"secondary_y": True}]])  # Changed from fig2 to fig10

	# Add histogram
	fig10.add_trace(
	    go.Histogram(
	        x=closest_rows['Forward_63d_price'],  # Changed 21d to 63d
	        nbinsx=40, 
		showlegend=False
	    )
	)
	fig10.add_trace(kde_trace, secondary_y=True)

	# Add KDE line to secondary y-axis
	fig10.add_trace(
	    go.Scatter(x=kde_x, y=kde_y, mode='lines', line=dict(width=2, color='goldenrod'), name='Seasonal Implied Dist.'),
	    secondary_y=True
	)

	fig10.add_vline(x=mean_forward_prices['63d'], line_color="green")  # Changed 21d to 63d
	fig10.add_vline(x=median_forward_prices['63d'], line_color="blue")  # Changed 21d to 63d
	fig10.add_vline(x=last_close, line_color="white")
	
	# Add annotations at the left edge
	annotations_y_63d = [0.95, 0.90, 0.85]  # positions to stack the annotations
	texts_63d = [
	    f"Mean = {mean_forward_prices['63d']:.2f}",  # Changed 21d to 63d
	    f"Median = {median_forward_prices['63d']:.2f}",  # Changed 21d to 63d
	    f"Last Close = {last_close:.2f}"
	]
	colors_63d = ["green", "blue", "white"]


	# Add annotations
	for i, (y, text, color) in enumerate(zip(annotations_y, texts_63d, colors)):
	    fig10.add_annotation(
	        xref="paper",
	        yref="paper",
	        x=0,
	        y=y,
	        text=text,
	        showarrow=False,
	        font=dict(color=color, size=12)
	    )

	fig10.update_layout(annotations=dict(xanchor='left', xshift=10), title=f'Forward 63 Day Distribution for {ticker}')  # Changed 21d to 63d
	fig10.update_layout(yaxis2=dict(showticklabels=False))
	fig10.update_xaxes(range=x_range)  # limit x-axis to ±4 standard deviations
	st.plotly_chart(fig) #traditional 5 and 21d heatmap
	st.plotly_chart(fig5) #fwd 5 histogram
	st.plotly_chart(fig2) #Fwd 21 histogram
	st.plotly_chart(fig10) #Fwd 63 histogram
	# # fig6.show() #naive implied distribution 21d
	# fig3.show() #trailing 252 and 63d heatmap
	# # fig4.show() #random sample of 21 returns histogram
	

# fig_creation('XLK', 12, (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d'), .173, 21) 
# fig_creation('HD', 12, "2010-04-23", .293, 21)

# Calculate KDE for Forward_5d_price using your provided method
ticker = st.text_input("Enter Ticker:")
tgt_date_range = st.number_input("Enter Target Holding Period in Days:")
end_date_default = dt.date.today() + dt.timedelta(days=1)
end_date = st.date_input("Enter End Date:", value=end_date_default)
sigma = st.number_input("Enter Implied Vol:", value=0.25)  # Example default value of 1.0, adjust as needed
days = st.number_input("Enter Days:", value=21, format="%i")  # Example default value of 21, adjust as needed
atr = st.text_input("ATR's or % Historical Returns?:")

# Call the function with the user input
if st.button("Generate Figures"):
    fig_creation(ticker, tgt_date_range, end_date, sigma, days,atr)	

	
	
	
