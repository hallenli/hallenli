import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
from lunardate import LunarDate
plt.switch_backend('agg')


path='/home/ling/prophet_dir/normal_data/204001_1h.csv'
#计算正确率所需要的，可接受范围内的误差
allowable_error=0.5
data=pd.read_csv(path,usecols=('ds','y'),engine='python')
data['ds']=pd.to_datetime(data['ds'],format='%Y-%m-%d %H:%M:%S')

train_len=int(len(data.index)*0.8)
test_len=int(len(data.index)-train_len)
train_data=data.iloc[:train_len,:]
test_data=data.iloc[train_len:,:]
#获取数据的开始日期以及结束日期，以便获取该时间段内的节假日
start=data['ds'][0]
# Timestamp('2017-06-22 09:30:00')
end=data['ds'][len(data.index)-1]
# Timestamp('2018-11-30 14:00:00')

#收集除清明节的所有节假日，共6个。#清明节，每年阳历的4.4或4.5，不一定，所以先不加，暂时为
def get_holidays(start,end):
    years = [i for i in range(start.year, end.year + 1)]
    names=["New Year's Day","Chinese New Year",
           "Labor Day","Dragon Boat Festival",
           "Mid-Autumn Festival","National Day"]

    holidays={key:value for key,value in zip(names,[[]for i in range(len(names))])}
    # 筛选出数据时间范围年内的假期
    for year in years:
        #元旦
        holidays["New Year's Day"].append(date(year, 1, 1))
        #春节
        for offset in range(-1, 2, 1):
            new_year = LunarDate(year + offset, 1, 1).toSolarDate()
            if new_year.year == year:
                holidays["Chinese New Year"].append(new_year)
        #劳动节
        holidays["Labor Day"].append(date(year, 5, 1))
        #端午节
        for offset in range(-1, 2, 1):
            dragon_Boat = LunarDate(year + offset, 5, 5).toSolarDate()
            if dragon_Boat.year == year:
                holidays["Dragon Boat Festival"].append(dragon_Boat)
        #中秋节
        for offset in range(-1, 2, 1):
            mid_autumn = LunarDate(year + offset, 8, 15).toSolarDate()
            if mid_autumn.year == year:
                holidays["Mid-Autumn Festival"].append(mid_autumn)
        #国庆节
        holidays["National Day"].append(date(year, 10, 1))
    #筛选出数据时间，日期范围内的假期
    s=start.date()
    e=end.date()
    for k, v in holidays.items():
        for i in v:
            if i<s or i>e:
                holidays[k].remove(i)
    return holidays

holidays=get_holidays(start,end)

#2017年的清明节为4.4，4.1~4.4为非交易日，不在数据时间范围内，所以不加入假期字典
#2018年的清明节为4.5,4.5~4.8为非交易日，在数据时间范围给内，加入假期字典
holidays["Tomb-Sweeping Day"]=[date(2018,4,5)]
# print(holidays)
# {"New Year's Day": [datetime.date(2018, 1, 1)],
#  'Chinese New Year': [datetime.date(2018, 2, 16)],
#  'Labor Day': [datetime.date(2018, 5, 1)],
#  'Dragon Boat Festival': [datetime.date(2018, 6, 18)],
#  'Mid-Autumn Festival': [datetime.date(2017, 10, 4),
#   datetime.date(2018, 9, 24)],
#  'National Day': [datetime.date(2017, 10, 1), datetime.date(2018, 10, 1)],
#  'Tomb-Sweeping Day': [datetime.date(2018, 4, 5)]}


#传入假期，以及包假期的前几天或后几天，分别懂ffill以及bfill表示
def normal_holidays(holidays,ffill,bfill):
    holidays_data=pd.DataFrame(columns=['holiday','ds',
                                        'lower_window','upper_window'])
    #元旦,
    holidays_data=holidays_data.append({'holiday':"New Year's Day",
                                        'ds':pd.to_datetime(holidays["New Year's Day"][0]),
                                        'lower_window':-2,'upper_window':0},
                                        ignore_index=True)
    #除夕+春节
    holidays_data = holidays_data.append({'holiday': 'Chinese New Year',
                                          'ds': pd.to_datetime(holidays['Chinese New Year'][0]),
                                          'lower_window': -1, 'upper_window': 5},
                                         ignore_index=True)
    #清明节
    holidays_data = holidays_data.append({'holiday': 'Tomb-Sweeping Day',
                                          'ds': pd.to_datetime(holidays['Tomb-Sweeping Day'][0]),
                                          'lower_window': 0, 'upper_window': 3},
                                         ignore_index=True)
    #劳动节
    holidays_data = holidays_data.append({'holiday': 'Labor Day',
                                          'ds': pd.to_datetime(holidays['Labor Day'][0]),
                                          'lower_window': -3, 'upper_window': 0},
                                         ignore_index=True)
    #端午节
    holidays_data = holidays_data.append({'holiday': 'Dragon Boat Festival',
                                          'ds': pd.to_datetime(holidays['Dragon Boat Festival'][0]),
                                          'lower_window': -2, 'upper_window': 0},
                                         ignore_index=True)
    #中秋节,[datetime.date(2017, 10, 4),datetime.date(2018, 9, 24)]
    # 2017年10.1~10.8包含中秋以及国庆
    # 2018年9.22~9.24中秋
    holidays_data = holidays_data.append({'holiday': 'Mid-Autumn Festival',
                                          'ds': pd.to_datetime(holidays['Mid-Autumn Festival'][0]),
                                          'lower_window': 0, 'upper_window': 4},
                                         ignore_index=True)
    holidays_data = holidays_data.append({'holiday': 'Mid-Autumn Festival',
                                          'ds': pd.to_datetime(holidays['Mid-Autumn Festival'][1]),
                                          'lower_window': -2, 'upper_window': 0},
                                         ignore_index=True)
    #国庆节，[datetime.date(2017, 10, 1), datetime.date(2018, 10, 1)]
    #2017年，9.30~10.8
    # 2018年9.29~10.7
    holidays_data = holidays_data.append({'holiday': 'National Day',
                                          'ds': pd.to_datetime(holidays['National Day'][0]),
                                          'lower_window': -1, 'upper_window': 2},
                                         ignore_index=True)
    holidays_data = holidays_data.append({'holiday': 'National Day',
                                          'ds': pd.to_datetime(holidays['National Day'][1]),
                                          'lower_window': -2, 'upper_window': 6},
                                         ignore_index=True)
    holidays_data['lower_window']-=ffill
    holidays_data['upper_window']+= bfill
    holidays_data.sort_values(by='ds',inplace=True)
    return holidays_data


holidays_data=normal_holidays(holidays,2,1)
print('-----------------------------')
print('holidays_data:',holidays_data)

def FriDay(ds):
    if ds.weekday()==4:
        return 1
    else:
        return 0
train_data['Friday']=train_data['ds'].apply(FriDay)
train_data['cap']=12
# Prophet(growth='linear', changepoints=None, n_changepoints=25,
#         changepoint_range=0.8, yearly_seasonality='auto',
#         weekly_seasonality='auto', daily_seasonality='auto',
#         holidays=None, seasonality_mode='additive',
#         seasonality_prior_scale=10.0, holidays_prior_scale=10.0,
#         changepoint_prior_scale=0.05,
#         mcmc_samples=0, interval_width=0.8, uncertainty_samples=1000))


m=Prophet(growth='logistic',holidays=holidays_data,
          weekly_seasonality=True,yearly_seasonality=True,
          holidays_prior_scale=50,mcmc_samples=10)

#add_seasonality此函数功能，参数为周期性名称，周期时间，以及傅里叶级？和惩罚力度。
# m.add_seasonality(name='monthly', period=30.5, fourier_order=5,mode = 'additive')
#m.add_regressor添加额外的可视化组件，比如说只想看受周五影响的的数据性变化
m.add_regressor('Friday')
#训练
m.fit(train_data)
#设置预测1h为单位的90条数据
future=m.make_future_dataframe(freq='1h',periods=test_len,include_history=False)
future['Friday']=future['ds'].apply(FriDay)
future['cap']=12
#预测
forecast=m.predict(future)
print('forecast的维度:',forecast.shape)
# >>> forecast.columns
# Index(['ds', 'trend', 'trend_lower', 'trend_upper', 'yhat_lower', 'yhat_upper',
#        'Friday', 'Friday_lower', 'Friday_upper', 'additive_terms',
#        'additive_terms_lower', 'additive_terms_upper', 'daily', 'daily_lower',
#        'daily_upper', 'extra_regressors_additive',
#        'extra_regressors_additive_lower', 'extra_regressors_additive_upper',
#        'multiplicative_terms', 'multiplicative_terms_lower',
#        'multiplicative_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper',
#        'yearly', 'yearly_lower', 'yearly_upper', 'yhat'],
#       dtype='object')

# print("%s:"%(growth),
#       forecast[['ds','yhat','yhat_lower','yhat_upper']])

# m.plot(forecast)
# #画出变化潜在点的位置，默认情况只会在前80％的时间序列中推断25个变换点
# a = add_changepoints_to_plot(fig.gca(), m, forecast)
# #可以查看趋势性、周期性以及假期的变化。
# m.plot_components(forecast)
# plt.show()
row_forecast=pd.merge(left=test_data,right=forecast[['ds','yhat','yhat_lower','yhat_upper']],
                       on='ds')
print('row_forecast:',row_forecast)
row_forecast['y-yhat']=test_data['y']-forecast['yhat']
row_forecast['y-yhat_lower']=test_data['y']-forecast['yhat_lower']
row_forecast['y-yhat_upper']=test_data['y']-forecast['yhat_upper']
score=row_forecast[row_forecast['y-yhat'].abs()<allowable_error].shape[0]/row_forecast.shape[0]*100
print('按照误差在%.4f方位内为准确的方法确定预测得分，为：'%(allowable_error),score)

storage_path = '/home/ling/prophet_dir/model_data/1h_dir/'
row_forecast.to_csv(storage_path+'row_forecast_1h_自定义节假日.csv')
# mat = row_forecast.loc[:, ['ds', 'y', 'yhat']]
# mat.set_index('ds', drop=True, inplace=True)
# mat.plot(kind='line', title='row_forecast_tick', legend=True, figsize=(7, 8))
plt.gcf().autofmt_xdate()
row_forecast.plot(x='ds',y='y',title='row_forecast_1h', legend=True, figsize=(7, 8))
plt.savefig(storage_path+'row_forecast_1h_自定义节假日.jpg')
