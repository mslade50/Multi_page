import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
from datetime import timedelta
import yfinance as yf
from yahoo_fin import options as op
import pandas as pd
import pandas_datareader.data as web
import scipy.stats
from scipy import stats
import ta
from pandas_datareader import data as pdr
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.integrate import quad
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

	days_list=days
	print(days_list)
	# Start the loop 8 years after data inception
	start_year += 4

	# Define the current year
	end_date_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
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
				df_year['Average_rnk'] = df_year[ticker]/10
				# df_year['Average_rnk'] = (df_year['Average'].rank(pct=True) * 10).round(0)
				dfs.append(df_year)
			except Exception:
				print(f"No data on {ticker} for the year {year}")

		df_final = pd.concat(dfs)
	    
		# Load your CSV file
		trading_dates_df = pd.read_csv('trading_days.csv')

		# Convert 'Date' column to datetime and extract year
		trading_dates_df['Date'] = pd.to_datetime(trading_dates_df['Date'])
		trading_dates_df['Year'] = trading_dates_df['Date'].dt.year
		trading_dates_df['IndexWithinYear'] = trading_dates_df.groupby('Year').cumcount()

		# Merge df_final and trading_dates_df based on 'Year' and 'IndexWithinYear'
		df_final = pd.merge(df_final.reset_index(drop=True), trading_dates_df, on=['Year', 'IndexWithinYear']).set_index('Date')
		
		return df_final

	df = mdm(ticker)
	df.to_csv('spot_check.csv',index=True)
	data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

	# Calculate performance with respect to ATR
	# List of lookbacks
	lookbacks = [5, 21, 63, 126, 252]
	most_recent_atr = data['ATR'].iloc[-1]
	# Calculate Trailing and Forward returns for each lookback
	# Calculate Trailing and Forward returns for each lookback
	# Calculate Trailing returns and percentage rank for each lookback
	for lookback in lookbacks:
	    data[f'Trailing_{lookback}d'] = (data['Close'] - data['Close'].shift(lookback)) / data['ATR'].shift(lookback)
	    data[f'Trailing_{lookback}d_pct_rank'] = (data[f'Trailing_{lookback}d'].rank(pct=True) * 10).round(0)

	for days in days_list:
	    if atr == "atr":
	        data[f'Forward_{days}d'] = (data['Close'].shift(-days) - data['Close']) / data['ATR']
	    else:
	        data[f'Forward_{days}d'] = (data['Close'].shift(-days) - data['Close']) / data['Close']

	    if atr == "atr":
	        data[f'Forward_{days}d_pct_rank'] = (data[f'Forward_{days}d'] * most_recent_atr) / data['Close'][-1] * 100
	    else:
	        data[f'Forward_{days}d_pct_rank'] = data[f'Forward_{days}d'] * 100

	    data[f'Forward_{days}d_pct_rank'].fillna(0, inplace=True)


        

	data['pct_change'] = data['Close'].pct_change()

	# Square the percentage changes
	data['squared_pct_change'] = data['pct_change'] ** 2


	# Drop the NA values from the 'Trailing' columns
	columns_to_dropna = [f'Trailing_{lookback}d_pct_rank' for lookback in lookbacks]
	data.dropna(subset=columns_to_dropna, inplace=True)

	# Create a subset of the dataframe
	columns_subset = [f'Trailing_{lookback}d_pct_rank' for lookback in lookbacks]
	for days in days_list:
	    columns_subset.append(f'Forward_{days}d_pct_rank')
	data_subset = data[columns_subset]
	# Perform the merge
	merged_df = pd.merge(df, data_subset, left_index=True, right_index=True)

	# Get the values from the last row of the DataFrame
	today_21d_rank = merged_df['Trailing_21d_pct_rank'].iloc[-1]
	today_5d_rank = merged_df['Trailing_5d_pct_rank'].iloc[-1]
	today_63d_rank = merged_df['Trailing_63d_pct_rank'].iloc[-1]
	today_126d_rank = merged_df['Trailing_126d_pct_rank'].iloc[-1]
	today_252d_rank = merged_df['Trailing_252d_pct_rank'].iloc[-1]
	today_avg_rank = merged_df['Average_rnk'].iloc[-1]
	print(today_avg_rank)
	# For 21d
	rows_for_21d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_21d_pct_rank'] == today_21d_rank)]

	# For 5d
	rows_for_5d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_5d_pct_rank'] == today_5d_rank)]
	# For 63d
	rows_for_63d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_63d_pct_rank'] == today_63d_rank)]
	# For 126d
	rows_for_126d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_126d_pct_rank'] == today_126d_rank)]
	# For 252d
	rows_for_252d = merged_df[(merged_df['Average_rnk'] == today_avg_rank) & (merged_df['Trailing_252d_pct_rank'] == today_252d_rank)]

	def compute_distance(row, target_values, avg_rank_weight=2.5):
	    return np.sqrt(
	        avg_rank_weight * (row['Average_rnk'] - target_values[0])**2 +  # Increased importance
	        (row['Trailing_5d_pct_rank'] - target_values[1])**2 +
	        (row['Trailing_21d_pct_rank'] - target_values[2])**2 +
	        (row['Trailing_63d_pct_rank'] - target_values[3])**2 + 
	        (row['Trailing_126d_pct_rank'] - target_values[4])**2 + 
	        (row['Trailing_252d_pct_rank'] - target_values[5])**2
	    )

	# def apply_decay(row, decay_rate=0.99999):
	#     date = row.name
	#     if isinstance(date, str):
	#         date = pd.Timestamp(date) # Only parse if it's a string
	#     days_diff = (pd.Timestamp.today() - date).days
	#     weight = 1 / (decay_rate ** days_diff)
	#     return row['distance'] * weight

	current_values = [today_avg_rank, today_5d_rank, today_21d_rank, today_63d_rank, today_126d_rank, today_252d_rank]

	# Exclude the most recent 63 rows
	filtered_df = merged_df.iloc[:-63].copy()
	filtered_df.index = pd.to_datetime(filtered_df.index)  # Convert the index to a datetime object
	filtered_df = filtered_df[filtered_df.index.year > 2000]  # Keep only rows where the year is greater than 2000

	# Compute the distance
	filtered_df['distance'] = filtered_df.apply(lambda row: compute_distance(row, current_values), axis=1)

	filtered_df.to_csv("filtered_df.csv", index=True)
	closest_rows = filtered_df.nsmallest(50, 'distance').round(2)
	closest_rows.to_csv("closest_rows.csv", index=True)
	results = {}
	days_1,days_2,days_3,days_4=days_list
	for days in days_list:
	    if atr == "atr":
	        most_recent_atr = data['ATR'].iloc[-1]
	        closest_rows[f'Forward_{days}d_price'] = last_close + (data[f'Forward_{days}d'] * most_recent_atr)
	    else:
	        closest_rows[f'Forward_{days}d_pct_proportion'] = closest_rows[f'Forward_{days}d_pct_rank'] / 100
	        closest_rows[f'Forward_{days}d_price'] = last_close * (1 + closest_rows[f'Forward_{days}d_pct_proportion'])
	    
	    # Save the result for this day into the results dictionary
	    results[days] = closest_rows[f'Forward_{days}d_price']

	    
	return closest_rows[[f'Forward_{days_1}d_price', f'Forward_{days_2}d_price', f'Forward_{days_3}d_price',f'Forward_{days_4}d_price']]

if __name__ == "__main__":
	pass
    # This block will only run if mdm_module.py is executed directly.
    # pass  # You can replace 'pass' with any standalone tests or commands you want to run.

	
