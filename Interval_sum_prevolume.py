'''
python 指定周期内(5min/13min/20min等)累积成交量计算.
功能：
    1、展示指定时间段内（时间段左闭右开，如5min，第一个时间段区间为9:30~9:39:59，只显示左标签，即9:30）。
    2、将以上聚合后的数据写入特定文件夹，文件名为原数据名+间隔时间。
    3、画图展示各个区间变化

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#原始数据文件的存储路径
filepath='D:/python工作文件/间隔时间求总和/20180515_000001.csv'
#整合后数据存放的文件夹路径
storage_dir='D:/python工作文件/间隔时间求总和/'
#函数参数为原始文件、指定的间隔时间。打印累计成交量的数据以及返回pandas.DataFrame形式的聚合后的数据
def interval_sum(filepath,interval_time):
    df=pd.read_csv(filepath,engine='python',
                   usecols=['MDTime','PreVolume'])
    df['MDTime']=pd.to_datetime(df['MDTime'],
                                format='%H%M%S000')
    df=df.set_index('MDTime')
    # am以及pm，切片出上午以及下午的有效时间
    am = df['1900-01-01 09:30:00':'1900-01-01 11:30:00']
    pm=df['1900-01-01 13:00:00':'1900-01-01 15:00:00']
    # 合并有效数据
    data=pd.concat([am,pm],axis=0)
    #处理间隔时间，因为要求间隔聚合为左闭右开，且数据文件最小单位为s，所以间隔时间减去1s即可。
    interval_time=pd.Timedelta(minutes=interval_time-1,seconds=59)
    #starttime和endtime用于控制区间内的左和右。即开始和结束
    starttime=pd.Timestamp('1900-01-01 09:30:00')
    endtime=starttime+interval_time
    #用来存储按照间隔时间聚合成交量之后的数据
    aggdf=pd.DataFrame(columns=['MDTime','PreVolume'])
    # 循环控制条件是：开始区间一定要在交易结束之前
    while starttime<pd.Timestamp('1900-01-01 15:00:00'):
        # 获取区间内的数据
        data_slice=data[starttime:endtime]
        #聚合区间内的成交量之和。
        prevolume_sum=data_slice['PreVolume'].sum()
        # 如果区内开始时间在中午休市时间内
        if pd.Timestamp('1900-01-01 11:30:00')<starttime<pd.Timestamp('1900-01-01 13:00:00'):
            # 如果结束时间不在休市时间内的话，显示最早的开市时间，即13:00。下一个区间正常按照间隔时间显示
            if endtime>pd.Timestamp('1900-01-01 13:00:00'):
                time = pd.Timestamp('1900-01-01 13:00:00')
                print(time.strftime('%H:%M:%S'), ':', prevolume_sum)
                aggdf=aggdf.append(
                    {'MDTime':starttime.strftime('%H:%M:%S'),'PreVolume':prevolume_sum},
                    ignore_index=True)
            # 如果结束时间同样也在休市期，那就不展示该区间。
            else:
               pass
        else:
            print(starttime.strftime('%H:%M:%S'),':',prevolume_sum)
            aggdf = aggdf.append(
                {'MDTime':starttime.strftime('%H:%M:%S'), 'PreVolume': prevolume_sum},
                ignore_index=True)
        starttime=endtime+pd.Timedelta(seconds=1)
        endtime=starttime+interval_time
    return aggdf

#函数功能：格式化存储文件
#       如原始文件为20180515_000001.csv，那么处理后文件为20180515_000001_Tmin.csv
#       T为间隔时间
#函数参数为原始数据文件路径，处理过的数据即将要存储的文件夹，以及间隔时间
def handle_storage_path(filepath,storage_dir,interval_time):
    filename=os.path.basename(filepath)
    filename_list=list(os.path.splitext(filename))
    storage_filename=filename_list[0]+'_%dmin'%interval_time+filename_list[1]
    return os.path.join(storage_dir,storage_filename)


#函数功能是用bar和barh显示区间的成交量。
# 参数为聚合后的数据，以及间隔时间。
def show_plot(df,interval_time):
    # fig,axes=plt.subplots(2,1)
    figure=plt.figure('%dmin_prevolume'%interval_time,figsize=(12,12))
    df=pd.Series(df['PreVolume'].values,index=df['MDTime'].values)
    plt.subplot(211)
    df.plot(kind='bar',alpha=0.7)
    plt.subplot(212)
    df.plot(kind='barh', alpha=0.7)
    # plt.title()
    plt.ylabel('PreVolume')
    plt.xlabel('MDTime')
    plt.xticks(rotation=45)
    # plt.yticks(rotation=45)
    plt.legend(loc='upper left')
    plt.show()



if __name__=='__main__':
    interval_time=int(input('请输入间隔时间:'))
    aggdf=interval_sum(filepath,interval_time)
    show_plot(aggdf,interval_time)
    storage_path=handle_storage_path(filepath, storage_dir, interval_time)
    aggdf.to_csv(storage_path)


