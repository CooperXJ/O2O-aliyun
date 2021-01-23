# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from chinese_calendar import is_holiday,is_workday
# In[0]
def PreProcess(off_train,off_test):
    off_train_test.replace('null',np.nan,inplace = True)
    off_train_test['Date_received'] = pd.to_datetime(off_train_test['Date_received'],format = '%Y%m%d')
    off_train_test['Date'] = pd.to_datetime(off_train_test['Date'],format = '%Y%m%d')

    off_train_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

    off_test_test.replace('null',np.nan,inplace = True)
    off_test_test['Date_received'] = pd.to_datetime(off_test_test['Date_received'],format = '%Y%m%d')

    off_test_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
    return off_train,off_test

# In[2]:    
def Get_Extra_Feature(DataSet):
    '''
    1.用户领取到的所有优惠券的数量
    2.用户领取到的同一优惠券的数量
    3.最大接受的日期
    4.最小的接收日期
    5.优惠券最近接受时间
    6.优惠券最远接受时间
    7.同一用户同一天领取优惠券的数量
    8.同一用户不同天接收到的不同优惠券的具体数量
    '''
    
    #用户领取到的所有优惠券的数量
    t1 = DataSet[['user_id','merchant_id']].copy()   
    temp = t1.groupby('user_id').count()
    temp.rename(columns={'merchant_id':'this_month_user_receive_all_coupon_count'}, inplace = True)
    t1 = pd.merge(t1,temp,on = 'user_id')
    t1.drop('merchant_id',axis = 1,inplace = True)
    t1.drop_duplicates(inplace = True);
    
    #用户领取到的同一优惠券的数量
    t2 = DataSet[['user_id','coupon_id','merchant_id']].copy()
    temp = t2.groupby(['user_id','coupon_id']).count()
    temp.rename(columns={'merchant_id':'this_month_user_receive_same_coupn_count'}, inplace = True)
    t2 = pd.merge(t2,temp,on = ['user_id','coupon_id'])
    t2.drop('merchant_id',axis = 1,inplace = True)
    t2.drop_duplicates(inplace = True);
    
    t3 = DataSet[['user_id', 'coupon_id', 'date_received']].copy()
    # 如果出现相同的用户接收相同的优惠券在接收时间上用‘：’连接上第n次接受优惠券的时间
    
    temp = t3.groupby(['user_id', 'coupon_id']).count().reset_index()
    temp.rename(columns={'date_received':'receive_number'}, inplace = True)
    t3 = pd.merge(t3,temp,on =['user_id','coupon_id'])
    t3 = t3[t3['receive_number']>1][['user_id','coupon_id','date_received']]
    
    # 最大接受的日期
    temp = t3.groupby(['user_id', 'coupon_id'])['date_received'].agg(max).reset_index()
    temp.rename(columns={'date_received':'max_date_received'}, inplace = True)
    t3 = pd.merge(t3,temp,on =['user_id','coupon_id'])
    
    # 最小的接收日期
    temp = t3.groupby(['user_id', 'coupon_id'])['date_received'].agg(min).reset_index()
    temp.rename(columns={'date_received':'min_date_received'}, inplace = True)
    t3 = pd.merge(t3,temp,on =['user_id','coupon_id'])
    t3.drop_duplicates(['user_id','coupon_id'],inplace = True);
    t3.drop('date_received',axis = 1,inplace = True);
   
    
    t4 = DataSet[['user_id', 'coupon_id', 'date_received']]
    t4 = pd.merge(t4, t3, on=['user_id', 'coupon_id'], how='left')
    # 这个优惠券最近接受时间
    t4['this_month_user_receive_same_coupon_lastone'] = (t4.max_date_received - t4.date_received).map(lambda x:x.days)
    # 这个优惠券最远接受时间
    t4['this_month_user_receive_same_coupon_firstone'] = (t4.date_received - t4.min_date_received).map(lambda x:x.days)
    
    t4.this_month_user_receive_same_coupon_lastone = t4.this_month_user_receive_same_coupon_lastone.map(lambda x:1 if x==0 else(0 if x>0 else -1))
    t4.this_month_user_receive_same_coupon_firstone = t4.this_month_user_receive_same_coupon_firstone.map(lambda x:1 if x==0 else(0 if x>0 else -1))
    t4.drop(['max_date_received','min_date_received'],axis = 1,inplace = True)
    
    #同一用户同一天领取优惠券的数量
    t5 = DataSet[['user_id','date_received','merchant_id']].copy()
    temp = t5.groupby(['user_id','date_received']).count()
    temp.rename(columns={'merchant_id':'this_day_receive_all_coupon_count'}, inplace = True)
    t5 = pd.merge(t5,temp,on = ['user_id','date_received'])
    t5.drop('merchant_id',axis = 1,inplace = True)
    t5.drop_duplicates(inplace = True);
    
    #同一用户不同天接收到的不同优惠券的具体数量
    t6 = DataSet[['user_id','coupon_id','date_received','merchant_id']].copy()
    temp = t6.groupby(['user_id','coupon_id','date_received']).count()
    temp.rename(columns={'merchant_id':'this_day_user_receive_same_coupon_count'}, inplace = True)
    t6 = pd.merge(t6,temp,on = ['user_id','coupon_id','date_received'])
    t6.drop('merchant_id',axis = 1,inplace = True)
    t6.drop_duplicates(inplace = True);
    
    
    
    other_feature_test = pd.merge(t2, t1, on='user_id',how  = 'left')
    other_feature_test = pd.merge(other_feature_test, t4, on=['user_id', 'coupon_id'],how  = 'left')
    other_feature_test = pd.merge(other_feature_test, t5, on=['user_id', 'date_received'],how  = 'left')
    other_feature_test = pd.merge(other_feature_test, t6, on=['user_id', 'coupon_id', 'date_received'],how  = 'left')
    return other_feature_test

# In[3]:
def Get_Coupon_Feature(DataSet, Feature):
    '''
    1.周几
    2.月几
    3.日期和截止日之间的天数
    4.满了多少钱后开始减
    5.满减的减少的钱
    6.优惠券是否是满减券
    7.打折力度
    8.优惠券的数量
    '''
    #找到最大值
    t = max(Feature[pd.notna(Feature['date'])]['date'])
    dataset_test = DataSet.copy()
    #周几
    dataset_test['day_of_week'] = dataset_test.date_received.map(lambda x: x.weekday()+1)
    
    #月几
    dataset_test['day_of_month'] = dataset_test.date_received.map(lambda x: x.month)
    
    #日期和截止日之间的天数
    dataset_test['days_distance'] = dataset_test.date_received.map(lambda x:(x-t).days)
    
    #满了多少钱后开始减
    dataset_test['discount_man'] = dataset_test.discount_rate.map(lambda x: np.nan if ':' not in x else (str(x).split(':')[0]))
    
    ##满减的减少的钱
    dataset_test['discount_jian'] = dataset_test.discount_rate.map(lambda x: np.nan if ':' not in x else (str(x).split(':')[1]))
    
    #优惠券是否是满减券
    dataset_test['is_man_jian'] = dataset_test.discount_rate.map(lambda x: 0 if ':' not in x else 1)
    
    #打折力度
    dataset_test['discount_rate'] = dataset_test.discount_rate.map(lambda x: float(x) if ':' not in x else(float(x.split(':')[0]) - float(x.split(':')[1])) / float(x.split(':')[0]))
    
    #优惠券的数量
    d = dataset_test[['coupon_id','user_id']]
    temp = d.groupby('coupon_id').count()
    temp.rename(columns={'user_id':'coupon_count'}, inplace = True)
    dataset_test = pd.merge(dataset_test, temp, on='coupon_id',how  = 'left')
    
    return dataset_test
    
# In[4]:
def Get_Merchant_Feature(Feature):
    '''
    1.每个商家卖出的商品数量
    2.使用了优惠券消费的商品
    3.商品的优惠券的总数量
    4.填补距离中的null
    5.用户离商品的距离最小值
    6.用户离商品的距离最大值
    7.距离的平均值
    8.距离的中位值
    9.优惠券的使用率
    10.卖出商品中使用优惠券的占比
    '''
    
    merchant_test = Feature[['user_id','merchant_id', 'coupon_id', 'distance', 'date_received', 'date']].copy()
    
    #每个商家卖出的商品数量
    #这里之所以选择user_id是为了防止有些空值无法被count（）进去
    t1_test = merchant_test[pd.notna(merchant_test['date'])][['user_id','merchant_id']].copy()
    temp = t1_test.groupby('merchant_id').count()
    temp.rename(columns={'user_id': 'total_sales'}, inplace=True)
    t1_test = pd.merge(t1_test,temp,on="merchant_id")
    t1_test.drop('user_id',axis = 1,inplace = True)
    t1_test.drop_duplicates(inplace = True);
    
    # 使用了优惠券消费的商品，正样本
    t2_test = merchant_test[pd.notna(merchant_test.date) & pd.notna(merchant_test.coupon_id)][['merchant_id','user_id']].copy()
    temp = t2_test.groupby('merchant_id').count()
    temp.rename(columns={'user_id': 'sales_use_coupon'}, inplace=True)
    t2_test = pd.merge(t2_test,temp,on="merchant_id")
    t2_test.drop('user_id',axis = 1,inplace = True)
    t2_test.drop_duplicates(inplace = True);
    
    #商品的优惠券的总数量
    t3_test = merchant_test[pd.notna(merchant_test.coupon_id)][['merchant_id','user_id']].copy()
    temp = t3_test.groupby('merchant_id').count()
    temp.rename(columns={'user_id': 'total_coupon'}, inplace=True)
    t3_test = pd.merge(t3_test,temp,on="merchant_id")
    t3_test.drop('user_id',axis = 1,inplace = True)
    t3_test.drop_duplicates(inplace = True);
    
    #填补距离中的null
    t4_test = merchant_test[pd.notna(merchant_test.date) & pd.notna(merchant_test.coupon_id)][['merchant_id', 'distance']].fillna(-1).copy()
    t4_test['distance'] = t4_test.distance.astype(int)
    t4_test = t4_test.replace(-1,np.nan)
    
    #返回用户离商品的距离最小值  这里agg只会比较distance中不为np.nan的字段 而使用lambda的话会加上np.nan的字段进行比较
    temp = t4_test.groupby('merchant_id')['distance'].agg(min)
    t5_test = t4_test;
    t5_test = pd.merge(t5_test,temp,on = 'merchant_id')
    t5_test.rename(columns={'distance_x': 'distance','distance_y': 'merchant_min_distance'}, inplace=True)
    t5_test.drop_duplicates('merchant_id',inplace = True);
    t5_test.drop('distance',axis = 1,inplace = True)
    
    #返回用户离商品的距离最大值
    temp = t4_test.groupby('merchant_id')['distance'].agg(max)
    t6_test = t4_test;
    t6_test = pd.merge(t6_test,temp,on = 'merchant_id')
    t6_test.rename(columns={'distance_x': 'distance','distance_y': 'merchant_max_distance'}, inplace=True)
    t6_test.drop_duplicates('merchant_id',inplace = True);
    t6_test.drop('distance',axis = 1,inplace = True)
    
    #返回距离的平均值
    temp = t4_test.groupby('merchant_id')['distance'].apply(lambda x:np.mean(x))
    t7_test = t4_test;
    t7_test = pd.merge(t7_test,temp,on = 'merchant_id')
    t7_test.rename(columns={'distance_x': 'distance','distance_y': 'merchant_mean_distance'}, inplace=True)
    t7_test.drop_duplicates('merchant_id',inplace = True);
    t7_test.drop('distance',axis = 1,inplace = True)
    
    #返回距离的中位值
    temp = t4_test.groupby('merchant_id')['distance'].agg('median')
    t8_test = t4_test;
    t8_test = pd.merge(t8_test,temp,on = 'merchant_id')
    t8_test.rename(columns={'distance_x': 'distance','distance_y': 'merchant_median_distance'}, inplace=True)
    t8_test.drop_duplicates('merchant_id',inplace = True);
    t8_test.drop('distance',axis = 1,inplace = True)
    
    t_test = merchant_test[['merchant_id']].drop_duplicates()
    merchant_feature_test = pd.merge(t_test, t1_test, on='merchant_id', how='left')
    l = [t2_test,t3_test,t4_test,t5_test,t6_test,t7_test,t8_test]
    for i in range(7):
        if i == 2:
            pass
        else:
            merchant_feature_test = pd.merge(merchant_feature_test,l[i], on='merchant_id',how  = 'left')
            
    #将sales_use_coupon中的null用0代替
    merchant_feature_test['sales_use_coupon'].fillna(0,inplace = True)
    
    #优惠券的使用率
    merchant_feature_test['merchant_coupon_transfer_rate'] = merchant_feature_test['sales_use_coupon'].astype('float') / merchant_feature_test['total_coupon']
    
    #即卖出商品中使用优惠券的占比
    merchant_feature_test['coupon_rate'] = merchant_feature_test['sales_use_coupon'].astype('float') / merchant_feature_test['total_sales']
    

    return merchant_feature_test
# In[5] :    
def Get_User_Feature(Feature):
    user_test = Feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']].copy()
    '''
     1.客户一共买的商家个数
     2.客户使用优惠券线下购买距离商店的最小距离
     3.客户使用优惠券线下购买距离商店的最大距离
     4.客户使用优惠券线下购买距离商店的平均距离
     5.客户使用优惠券线下购买距离商店的中间距离
     6.客户使用优惠券购买的次数
     7.客户购买的总次数
     8.客户收到优惠券的总数
     9.客户从收优惠券到消费的时间间隔
     10.客户从收优惠券到消费的平均时间间隔
     11.客户从收优惠券到消费的最小时间间隔
     12.客户从收优惠券到消费的最大时间间隔
    '''
    #客户一共买的商家个数
    t1_test = user_test[pd.notna(user_test.date)][['user_id','merchant_id']].drop_duplicates().copy()
    temp = t1_test.groupby('user_id').count()
    temp.rename(columns={'merchant_id': 'user_buy_merchant_count'}, inplace=True)
    t1_test = pd.merge(t1_test,temp,on="user_id")
    t1_test.drop('merchant_id',axis = 1,inplace = True)
    t1_test.drop_duplicates('user_id',inplace = True);
    
    #初始化用户距离
    t2_test = user_test[pd.notna(user_test.date) & pd.notna(user_test.coupon_id)][['user_id', 'distance']].fillna(-1)
    t2_test['distance'] = t2_test.distance.astype(int)
    t2_test.replace(-1,np.nan)
    
    #客户使用优惠券线下购买距离商店的最小距离
    temp = t2_test.groupby('user_id')['distance'].agg(min)
    t3_test = t2_test;
    t3_test = pd.merge(t3_test,temp,on = 'user_id')
    t3_test.rename(columns={'distance_x': 'distance','distance_y': 'user_min_distance'}, inplace=True)
    t3_test.drop_duplicates('user_id',inplace = True);
    t3_test.drop('distance',axis = 1,inplace = True)
    
    #客户使用优惠券线下购买距离商店的最大距离
    temp = t2_test.groupby('user_id')['distance'].agg(max)
    t4_test = t2_test;
    t4_test = pd.merge(t4_test,temp,on = 'user_id')
    t4_test.rename(columns={'distance_x': 'distance','distance_y': 'user_max_distance'}, inplace=True)
    t4_test.drop_duplicates('user_id',inplace = True);
    t4_test.drop('distance',axis = 1,inplace = True)
    
    #客户使用优惠券线下购买距离商店的平均距离
    temp = t2_test.groupby('user_id')['distance'].apply(lambda x:np.mean(x))
    t5_test = t2_test;
    t5_test = pd.merge(t5_test,temp,on = 'user_id')
    t5_test.rename(columns={'distance_x': 'distance','distance_y': 'user_mean_distance'}, inplace=True)
    t5_test.drop_duplicates('user_id',inplace = True);
    t5_test.drop('distance',axis = 1,inplace = True)
    
    #客户使用优惠券线下购买距离商店的中位数距离
    temp = t2_test.groupby('user_id')['distance'].agg('median')
    t6_test = t2_test;
    t6_test = pd.merge(t6_test,temp,on = 'user_id')
    t6_test.rename(columns={'distance_x': 'distance','distance_y': 'user_median_distance'}, inplace=True)
    t6_test.drop_duplicates('user_id',inplace = True);
    t6_test.drop('distance',axis = 1,inplace = True)
    
    #客户使用优惠券购买的次数
    t7_test = user_test[pd.notna(user_test.date) & pd.notna(user_test.coupon_id)][['user_id','date']]
    temp = t7_test.groupby('user_id').count()
    temp.rename(columns={'date':'buy_use_coupon_count'}, inplace = True)
    t7_test = pd.merge(t7_test,temp,on = 'user_id')
    t7_test.drop('date',axis = 1,inplace = True)
    t7_test.drop_duplicates(inplace = True);
    
    #客户购买的总次数
    t8_test = user_test[pd.notna(user_test.date)][['user_id','date']]
    temp = t8_test.groupby('user_id').count()
    temp.rename(columns={'date':'buy_all_count'}, inplace = True)
    t8_test = pd.merge(t8_test,temp,on = 'user_id')
    t8_test.drop('date',axis = 1,inplace = True)
    t8_test.drop_duplicates(inplace = True);
    
    #客户收到优惠券的总数
    t9_test = user_test[pd.notna(user_test.coupon_id)][['user_id','coupon_id']]
    temp = t9_test.groupby('user_id').count()
    temp.rename(columns={'coupon_id':'coupon_received_count'}, inplace = True)
    t9_test = pd.merge(t9_test,temp,on = 'user_id')
    t9_test.drop('coupon_id',axis = 1,inplace = True)
    t9_test.drop_duplicates(inplace = True);
    
    #客户从收优惠券到消费的时间间隔
    t10_test = user_test[pd.notna(user_test.date_received) & pd.notna(user_test.date)][['user_id', 'date_received', 'date']]
    t10_test = user_test[pd.notna(user_test.date_received) & pd.notna(user_test.date)][['user_id', 'date_received', 'date']]
    t10_test['user_date_datereceived_gap'] = (t10_test.date - t10_test.date_received).map(lambda x:x.days)
    t10_test = t10_test[['user_id', 'user_date_datereceived_gap']]
    
    #客户从收优惠券到消费的平均时间间隔
    temp = t10_test.groupby('user_id')['user_date_datereceived_gap'].apply(lambda x:np.mean(x))
    t11_test = pd.merge(t10_test,temp,on = 'user_id')
    t11_test.rename(columns={'user_date_datereceived_gap_x': 'user_date_datereceived_gap','user_date_datereceived_gap_y': 'mean_user_date_datereceived_gap'}, inplace=True)
    t11_test.drop_duplicates('user_id',inplace = True);
    t11_test.drop('user_date_datereceived_gap',axis = 1,inplace = True)
    
    #客户从收优惠券到消费的最小时间间隔
    temp = t10_test.groupby('user_id')['user_date_datereceived_gap'].apply(lambda x:min(x))
    t12_test = pd.merge(t10_test,temp,on = 'user_id')
    t12_test.rename(columns={'user_date_datereceived_gap_x': 'user_date_datereceived_gap','user_date_datereceived_gap_y': 'min_user_date_datereceived_gap'}, inplace=True)
    t12_test.drop_duplicates('user_id',inplace = True);
    t12_test.drop('user_date_datereceived_gap',axis = 1,inplace = True)
    
    #客户从收优惠券到消费的最大时间间隔
    temp = t10_test.groupby('user_id')['user_date_datereceived_gap'].apply(lambda x:max(x))
    t13_test = pd.merge(t10_test,temp,on = 'user_id')
    t13_test.rename(columns={'user_date_datereceived_gap_x': 'user_date_datereceived_gap','user_date_datereceived_gap_y': 'max_user_date_datereceived_gap'}, inplace=True)
    t13_test.drop_duplicates('user_id',inplace = True);
    t13_test.drop('user_date_datereceived_gap',axis = 1,inplace = True)
    
    t_test = user_test[['user_id']].copy()
    t_test.drop_duplicates(inplace=True)
    
    
    user_feature_test = pd.merge(t_test, t1_test, on='user_id')
    l = [t3_test,t4_test,t5_test,t6_test,t7_test,t8_test,t9_test,t10_test,t11_test,t12_test,t13_test]
    for i in range(10):
        user_feature_test = pd.merge(user_feature_test,l[i], on='user_id',how  = 'left')
    
    user_feature_test['user_buy_merchant_count'].fillna(0,inplace = True)
    user_feature_test['buy_use_coupon_count'].fillna(0,inplace = True)
    user_feature_test['buy_use_coupon_rate'] = user_feature_test['buy_use_coupon_count'].astype('float') / user_feature_test['buy_all_count'].astype('float')
    user_feature_test['user_coupon_transfer_rate'] = user_feature_test['buy_use_coupon_count'].astype('float') / user_feature_test['coupon_received_count'].astype('float')
    user_feature_test['buy_all_count'].fillna(0,inplace = True)
    user_feature_test['coupon_received_count'].fillna(0,inplace = True)
    
    return user_feature_test

# In[6]
def Get_User_Merchant_Feature(Feature):
    '''
    1.一个客户在一个商家一共收到的优惠券并且消费日期不为空的数目
    2.一个客户在一个商家一共收到的优惠券
    3。一个客户在一个商家使用优惠券购买的次数
    4.一个客户在一个商家浏览的次数
    5.一个客户在一个商家没有使用优惠券购买的次数
    '''
    
    all_user_merchant_test = Feature[['user_id', 'merchant_id']].drop_duplicates().copy()
    
    #一个客户在一个商家一共收到的优惠券
    t1_test = Feature[pd.notna(Feature.date)][['user_id', 'merchant_id', 'date']].copy()
    temp = t1_test.groupby(['user_id','merchant_id']).count()
    temp.rename(columns={'date':'user_merchant_buy_total'}, inplace = True)
    t1_test = pd.merge(t1_test,temp,on = ['user_id','merchant_id'])
    t1_test.drop('date',axis = 1,inplace = True)
    t1_test.drop_duplicates(inplace = True);
    
    #一个客户在一个商家一共收到的优惠券
    t2_test = Feature[pd.notna(Feature.coupon_id)][['user_id', 'merchant_id', 'coupon_id']]
    temp = t2_test.groupby(['user_id','merchant_id']).count()
    temp.rename(columns={'coupon_id':'user_merchant_received'}, inplace = True)
    t2_test = pd.merge(t2_test,temp,on = ['user_id','merchant_id'])
    t2_test.drop('coupon_id',axis = 1,inplace = True)
    t2_test.drop_duplicates(inplace = True);
    
    #一个客户在一个商家使用优惠券购买的次数
    t3_test = Feature[pd.notna(Feature.date) & pd.notna(Feature.date_received)][['user_id', 'merchant_id', 'date']]
    temp = t3_test.groupby(['user_id','merchant_id']).count()
    temp.rename(columns={'date':'user_merchant_buy_use_coupon'}, inplace = True)
    t3_test = pd.merge(t3_test,temp,on = ['user_id','merchant_id'])
    t3_test.drop('date',axis = 1,inplace = True)
    t3_test.drop_duplicates(inplace = True);
    
    #一个客户在一个商家浏览的次数
    t4_test = Feature[['user_id', 'merchant_id','coupon_id']]
    t4_test = t4_test.fillna(-1)
    temp = t4_test.groupby(['user_id','merchant_id']).count()
    temp.rename(columns={'coupon_id':'user_merchant_any'}, inplace = True)
    t4_test = pd.merge(t4_test,temp,on = ['user_id','merchant_id'])
    t4_test.drop('coupon_id',axis = 1,inplace = True)
    t4_test.drop_duplicates(inplace=True)
    
    #一个客户在一个商家没有使用优惠券购买的次数
    t5_test = Feature[pd.notna(Feature.date) & pd.isna(Feature.coupon_id)][['user_id', 'merchant_id', 'date']]
    temp = t5_test.groupby(['user_id','merchant_id']).count()
    temp.rename(columns={'date':'user_merchant_buy_common'}, inplace = True)
    t5_test = pd.merge(t5_test,temp,on = ['user_id','merchant_id'])
    t5_test.drop('date',axis = 1,inplace = True)
    t5_test.drop_duplicates(inplace=True)
    
    l = [t2_test,t3_test,t4_test,t5_test]
    user_merchant_test = pd.merge(all_user_merchant_test, t1_test, on=['user_id', 'merchant_id'], how='left')
    for i in range(4):
        user_merchant_test = pd.merge(user_merchant_test,l[i], on=['user_id', 'merchant_id'],how  = 'left')

    user_merchant_test['user_merchant_buy_use_coupon'].fillna(0,inplace = True)
    user_merchant_test['user_merchant_buy_common'].fillna(0,inplace = True)
    user_merchant_test['user_merchant_coupon_transfer_rate'] = user_merchant_test['user_merchant_buy_use_coupon'].astype('float') / user_merchant_test['user_merchant_received'].astype('float')
    user_merchant_test['user_merchant_coupon_buy_rate'] = user_merchant_test['user_merchant_buy_use_coupon'].astype('float') / user_merchant_test['user_merchant_buy_total'].astype('float')
    user_merchant_test['user_merchant_rate'] = user_merchant_test['user_merchant_buy_total'].astype('float') / user_merchant_test['user_merchant_any'].astype('float')
    user_merchant_test['user_merchant_common_buy_rate'] = user_merchant_test['user_merchant_buy_common'].astype('float') / user_merchant_test['user_merchant_buy_total'].astype('float')
    return user_merchant_test

# In[7]
def getweekday(DataSet):
    dataset_test = DataSet
    dataset_test['is_weekend'] = dataset_test['day_of_week'].map(lambda x: 1 if x==5 or x==6 else 0)
    weekday_dummies = pd.get_dummies(dataset_test['day_of_week'])
    weekday_dummies.columns = ['weekday' + str(i) for i in range(weekday_dummies.shape[1])]
    dataset_test = pd.concat([dataset_test, weekday_dummies], axis=1)
    
    #新加的特征
    dataset_test['is_holiday'] = dataset_test['date_received'].map(lambda x:1 if is_holiday(x) else 0)
    dataset_test['is_workday'] = dataset_test['date_received'].map(lambda x:1 if is_workday(x) else 0)
    
    return dataset_test
    
def Process(DataSet, Feature, flag):
    other_feature_test = Get_Extra_Feature(DataSet)
    merchant_test = Get_Merchant_Feature(Feature)
    user_test = Get_User_Feature(Feature)
    user_merchant_test = Get_User_Merchant_Feature(Feature)
    coupon_test = Get_Coupon_Feature(DataSet, Feature)

    dataset_test = pd.merge(coupon_test, merchant_test, on='merchant_id', how='left')
    dataset_test = pd.merge(dataset_test, user_test, on='user_id', how='left')
    dataset_test = pd.merge(dataset_test, user_merchant_test, on=['user_id', 'merchant_id'], how='left')
    dataset_test = pd.merge(dataset_test, other_feature_test, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset_test.drop_duplicates(inplace=True)

    dataset_test['user_merchant_buy_total'] = dataset_test['user_merchant_buy_total'].replace(np.nan, 0)
    dataset_test['user_merchant_any'] = dataset_test['user_merchant_any'].replace(np.nan, 0)
    dataset_test['user_merchant_received'] = dataset_test['user_merchant_received'].replace(np.nan, 0)
    dataset_test = getweekday(dataset_test)
     
    if flag:
        dataset_test['label'] =list(map(lambda x, y: 1 if (x - y).days<= 15 else 0, dataset_test['date'],dataset_test['date_received']))
        dataset_test = dataset_test.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_count'], axis=1)
    else:
        dataset_test = dataset_test.drop(['merchant_id', 'day_of_week', 'coupon_count'], axis=1)
    dataset_test = dataset_test.fillna(np.nan)
    return dataset_test

# In[9]:
def GetResult(dataset_1,dataset_2,dataset_3,feature_1,feature_2,feature_3):
    print('数据集处理开始----')
    Processdataset_1 = Process(dataset_1, feature_1, True)
#    Processdataset_1.to_csv('./Processdataset_1.csv', index=None)
#    Processdataset_1 = pd.read_csv('./Processdataset_1.csv')
    Processdataset_1.drop('coupon_id',axis = 1,inplace = True)
    Processdataset_1.drop_duplicates(inplace=True)
 
    Processdataset_2 = Process(dataset_2, feature_2, True)
#    Processdataset_2.to_csv('./Processdataset_2.csv', index=None) 
#    Processdataset_2 = pd.read_csv('./Processdataset_2.csv')
    Processdataset_2.drop('coupon_id',axis = 1,inplace = True)
    Processdataset_2.drop_duplicates(inplace=True)
    
    Processdataset_3 = Process(dataset_3, feature_3, False)
#    Processdataset_3.to_csv('./Processdataset_3.csv', index=None)   
#    Processdataset_3 = pd.read_csv('./Processdataset_3.csv')
    Processdataset_3.drop_duplicates(inplace=True)
    print('数据集处理完成----')
    
    print('数据集训练开始----')
    ProcessDataSet = pd.concat([Processdataset_1,Processdataset_2],axis=0)   
    ProcessDataSet_y = ProcessDataSet.label
    ProcessDataSet_x = ProcessDataSet.drop(['user_id','label'],axis=1)    
    Processdataset_3_preds = Processdataset_3[['user_id','coupon_id','date_received']]
    Processdataset_3_x = Processdataset_3.drop(['user_id','coupon_id','date_received'],axis=1)    
    ProcessDataSet = xgb.DMatrix(ProcessDataSet_x,label=ProcessDataSet_y)
    Processdataset_3 = xgb.DMatrix(Processdataset_3_x)
    
    
    params={'booster':'gbtree', #迭代模型：树模型
            'objective': 'binary:logistic',
    	    'eval_metric':'auc',#计算目标函数值的方法
    	    'gamma':0.1,  #调整因为树的增加而设立的损失系数
    	    'min_child_weight':1.1, #最小叶子节点样本权重 避免过拟合的产生
    	    'max_depth':5, #树的最大深度 避免过拟合
    	    'lambda':10, #L2 正则惩罚系数
    	    'subsample':0.7, #样本采样，适当调整可以避免过拟合和欠拟合
    	    'colsample_bytree':0.7, #列采样，选择树的特征，防止过拟合
    	    'colsample_bylevel':0.7,#每一级进行分列时对列进行采样，防止过拟合
    	    'eta': 0.01, #学习率参数  目的是为了降低每一个树对于结果的影响，如果设置过大会导致无法收敛
    	    'tree_method':'exact',#树的生成方法 这里我们选择的是贪心算法 因为我们这里的数据集并不是很大
    	    'seed':0 #随机数种子 作用是使得结果可以复现
    	    #'nthread':12 #最大并行线程数 如果不指定则默认取得CPU所有的核  显然和训练模型速度有关
    	    }
    '''
    
    params = {
        'learning_rate': 1e-2,
        'n_estimators': 1260,
        'max_depth': 8,
        'min_child_weight': 4,
        'gamma': .2,
        'subsample': .6,
        'colsample_bytree': .8,
        'scale_pos_weight': 1,
        'reg_alpha': 0,
        'seed': 0
    }
     
    '''
    
    watchlist = [(ProcessDataSet,'train')]
    model = xgb.train(params,ProcessDataSet,num_boost_round=2000,evals=watchlist)  #此处我调节到5000时发生过拟合  几次实验都证明2000比较好
    Processdataset_3_preds['label'] = model.predict(Processdataset_3)
    Processdataset_3_preds.label = MinMaxScaler().fit_transform(Processdataset_3_preds.label.values.reshape(-1, 1))
    Processdataset_3_preds.sort_values(by=['coupon_id','label'],inplace=True)
    Processdataset_3_preds['date_received'] = Processdataset_3_preds['date_received'].map(lambda x:x.replace('-',''))
    Processdataset_3_preds.to_csv("./xgb_preds.csv",index=None,header=None)
    print("数据集训练完成-----")   

# In[10]
if __name__=="__main__":
    off_train_test = pd.read_csv('C:/Users/Aaron/Desktop/Data/ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
    off_test_test = pd.read_csv('C:/Users/Aaron/Desktop/Data/ccf_offline_stage1_test_revised.csv', header=0, keep_default_na=False)
    
    off_train_test,off_test_test = PreProcess(off_train_test,off_test_test)
    
    #这里一定要注意 如果写成doff_train_test['date']!=np.nan是没有用的 因为日期的空值NaT！=np.nan 
    
    #训练集
    dataset_1 = off_train_test[off_train_test['date_received'].isin(pd.date_range('2016/4/2', periods=43))]
    feature_1 = off_train_test[off_train_test['date'].isin(pd.date_range('2016/1/1',periods=104))|(pd.isna(off_train_test['date'])&off_train_test['date_received'].isin(pd.date_range('2016/1/1',periods=104)))]
    dataset_2 = off_train_test[off_train_test['date_received'].isin(pd.date_range('2016/5/15', periods=32))]    
    feature_2 = off_train_test[off_train_test['date'].isin(pd.date_range('2016/2/1',periods=104))|(pd.isna(off_train_test['date'])&off_train_test['date_received'].isin(pd.date_range('2016/2/1',periods=104)))]
    feature_3 = off_train_test[off_train_test['date'].isin(pd.date_range('2016/3/15',periods=108))|(pd.isna(off_train_test['date'])&off_train_test['date_received'].isin(pd.date_range('2016/3/15',periods=108)))]

    #测试集
    dataset_3 = off_test_test
    
    GetResult(dataset_1,dataset_2,dataset_3,feature_1,feature_2,feature_3)

















    

    