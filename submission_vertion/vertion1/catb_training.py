"""
Created on Fri Sep 22 22:25:42 2017

The submission version 1

@author: xiong
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm
from sklearn import linear_model
from sklearn.cross_validation import train_test_split 

train_df2016 = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
properties = pd.read_csv('data/properties_2016.csv', low_memory=False)
train_df2017 = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
properties1 = pd.read_csv('data/properties_2017.csv', low_memory=False)

#################################### data clear and features engineering ###################################################

############################################# for properties->2016 ########################################################
#add some other features
#living area proportions 
properties['living_area_prop'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
#tax value ratio
properties['value_ratio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
#tax value proportions
properties['value_prop'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']
#Price per square feet
properties['per_price'] = properties['taxvaluedollarcnt'] / properties['calculatedfinishedsquarefeet']

#landperprice
properties['landperprice']=properties['landtaxvaluedollarcnt']/properties['lotsizesquarefeet']
#structtax_ratio
properties['structtax_ratio']=properties['structuretaxvaluedollarcnt']/properties['taxvaluedollarcnt']
#landtax_ratio
properties['landtax_ratio']=properties['landtaxvaluedollarcnt']/properties['taxvaluedollarcnt']


#fill with 0
properties.fillna({'decktypeid':0,'threequarterbathnbr':0},inplace=True)
properties.fillna({'finishedfloor1squarefeet':0,'finishedsquarefeet6':0,'finishedsquarefeet12':0},inplace=True)
properties.fillna({'finishedsquarefeet15':0,'fips':0,'fireplacecnt':0,'fullbathcnt':0},inplace=True)
properties.fillna({'garagecarcnt':0,'garagetotalsqft':0,'pooltypeid2':0,'unitcnt':0},inplace=True)
properties.fillna({'numberofstories':0,'poolcnt':0,'poolsizesum':0,'pooltypeid10':0},inplace=True)
properties.fillna({'yardbuildingsqft17':0,'structuretaxvaluedollarcnt':0,'landtaxvaluedollarcnt':0,'taxdelinquencyyear':0},inplace=True)


#fill with mean
properties.fillna({'buildingqualitytypeid':properties.buildingqualitytypeid.mode()},inplace=True)
properties.fillna({'calculatedfinishedsquarefeet':properties.calculatedfinishedsquarefeet.mean()},inplace=True)
properties.fillna({'lotsizesquarefeet':properties.lotsizesquarefeet.mean()},inplace=True)
properties.fillna({'roomcnt':properties.roomcnt.mean()},inplace=True)
properties.fillna({'taxvaluedollarcnt':properties.taxvaluedollarcnt.mean()},inplace=True)
properties.fillna({'living_area_prop':properties.living_area_prop.mean()},inplace=True)
properties.fillna({'value_ratio':properties.value_ratio.mean()},inplace=True)
properties.fillna({'value_prop':properties.value_prop.mean()},inplace=True)
properties.fillna({'per_price':properties.per_price.mean()},inplace=True)
properties.fillna({'landperprice':properties.value_ratio.mean()},inplace=True)
properties.fillna({'structtax_ratio':properties.value_prop.mean()},inplace=True)
properties.fillna({'landtax_ratio':properties.per_price.mean()},inplace=True)


#deal with other missing data
properties.fillna({'hashottuborspa':0},inplace=True)
properties['hashottuborspa']=properties['hashottuborspa'].astype('int')#replace 'True' to 1
properties.latitude.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties.longitude.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties.regionidcounty.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties.regionidcity.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties.regionidneighborhood.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties.yearbuilt[0]=1948
properties.yearbuilt.interpolate(inplace=True) #replace NaN with the interpolation

#linearregression for the taxamount
#linreg=linear_model.LinearRegression()
#tax_y=properties.taxamount.dropna()
#tax_x=properties.taxvaluedollarcnt[properties.taxamount.notnull()]
#tax_x=tax_x.reshape((2953967,1))
#linreg.fit(tax_x,tax_y)
#taxpre=linreg.predict(properties.taxvaluedollarcnt.reshape((2985217,1)))
#taxpre=pd.Series(taxpre)
#properties.taxamount.loc[properties.taxamount.isnull()]=taxpre.loc[properties.taxamount.isnull()]

#add some other features
#location
properties['location']=properties['latitude']+properties['longitude']
#life of property
properties['life'] = 2018 - properties['yearbuilt']
#Average structuretaxvaluedollarcnt by city
group =properties.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
properties['N-Avg-structuretaxvaluedollarcnt'] = properties['regionidcity'].map(group)
#Deviation away from average
properties['N-Dev-structuretaxvaluedollarcnt'] = abs((properties['structuretaxvaluedollarcnt']-properties['N-Avg-structuretaxvaluedollarcnt']))/properties['N-Avg-structuretaxvaluedollarcnt']
#allroom
properties['allroomcnt']=properties['bedroomcnt']+properties['bathroomcnt']




############################################# for properties1->2017 ##########################################################################
#add some other features
#living area proportions 
properties1['living_area_prop'] = properties1['calculatedfinishedsquarefeet'] / properties1['lotsizesquarefeet']
#tax value ratio
properties1['value_ratio'] = properties1['taxvaluedollarcnt'] / properties1['taxamount']
#tax value proportions
properties1['value_prop'] = properties1['structuretaxvaluedollarcnt'] / properties1['landtaxvaluedollarcnt']
#Price per square feet
properties1['per_price'] = properties1['taxvaluedollarcnt'] / properties1['calculatedfinishedsquarefeet']

#landperprice
properties1['landperprice']=properties1['landtaxvaluedollarcnt']/properties1['lotsizesquarefeet']
#structtax_ratio
properties1['structtax_ratio']=properties1['structuretaxvaluedollarcnt']/properties1['taxvaluedollarcnt']
#landtax_ratio
properties1['landtax_ratio']=properties1['landtaxvaluedollarcnt']/properties1['taxvaluedollarcnt']


#fill with 0
properties1.fillna({'decktypeid':0,'threequarterbathnbr':0},inplace=True)
properties1.fillna({'finishedfloor1squarefeet':0,'finishedsquarefeet6':0,'finishedsquarefeet12':0},inplace=True)
properties1.fillna({'finishedsquarefeet15':0,'fips':0,'fireplacecnt':0,'fullbathcnt':0},inplace=True)
properties1.fillna({'garagecarcnt':0,'garagetotalsqft':0,'pooltypeid2':0,'unitcnt':0},inplace=True)
properties1.fillna({'numberofstories':0,'poolcnt':0,'poolsizesum':0,'pooltypeid10':0},inplace=True)
properties1.fillna({'yardbuildingsqft17':0,'structuretaxvaluedollarcnt':0,'landtaxvaluedollarcnt':0,'taxdelinquencyyear':0},inplace=True)


#fill with mean
properties1.fillna({'buildingqualitytypeid':properties1.buildingqualitytypeid.mode()},inplace=True)
properties1.fillna({'calculatedfinishedsquarefeet':properties1.calculatedfinishedsquarefeet.mean()},inplace=True)
properties1.fillna({'lotsizesquarefeet':properties1.lotsizesquarefeet.mean()},inplace=True)
properties1.fillna({'roomcnt':properties1.roomcnt.mean()},inplace=True)
properties1.fillna({'taxvaluedollarcnt':properties1.taxvaluedollarcnt.mean()},inplace=True)
properties1.fillna({'living_area_prop':properties1.living_area_prop.mean()},inplace=True)
properties1.fillna({'value_ratio':properties1.value_ratio.mean()},inplace=True)
properties1.fillna({'value_prop':properties1.value_prop.mean()},inplace=True)
properties1.fillna({'per_price':properties1.per_price.mean()},inplace=True)
properties1.fillna({'landperprice':properties1.value_ratio.mean()},inplace=True)
properties1.fillna({'structtax_ratio':properties1.value_prop.mean()},inplace=True)
properties1.fillna({'landtax_ratio':properties1.per_price.mean()},inplace=True)


#deal with other missing data
properties1.fillna({'hashottuborspa':0},inplace=True)
properties1['hashottuborspa']=properties1['hashottuborspa'].astype('int')#replace 'True' to 1
properties1.latitude.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties1.longitude.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties1.regionidcounty.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties1.regionidcity.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties1.regionidneighborhood.fillna(method='pad',inplace=True)#replace NaN with the privious data
properties1.yearbuilt[0]=1948
properties1.yearbuilt.interpolate(inplace=True) #replace NaN with the interpolation

#linearregression for the taxamount
#linreg=linear_model.LinearRegression()
#tax_y=properties1.taxamount.dropna()
#tax_x=properties1.taxvaluedollarcnt[properties1.taxamount.notnull()]
#tax_x=tax_x.reshape((2953967,1))
#linreg.fit(tax_x,tax_y)

#add some other features
#location
properties1['location']=properties1['latitude']+properties1['longitude']
#life of property
properties1['life'] = 2018 - properties1['yearbuilt']
#Average structuretaxvaluedollarcnt by city
group =properties1.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
properties1['N-Avg-structuretaxvaluedollarcnt'] = properties1['regionidcity'].map(group)
#Deviation away from average
properties1['N-Dev-structuretaxvaluedollarcnt'] = abs((properties1['structuretaxvaluedollarcnt']-properties1['N-Avg-structuretaxvaluedollarcnt']))/properties1['N-Avg-structuretaxvaluedollarcnt']
#allroom
properties1['allroomcnt']=properties1['bedroomcnt']+properties1['bathroomcnt']


#######################################################################################################################################################################

# get the year,month,day and quarter from transcationdate
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df

#drop out the abnormal logerror
#train_df2016=train_df2016[train_df2016.logerror<2]
#train_df2016=train_df2016[train_df2016.logerror>-2]
#train_df2017=train_df2017[train_df2017.logerror<2]
#train_df2017=train_df2017[train_df2017.logerror>-2]

#merge the trainset
train_df2016 = train_df2016.merge(properties, how='left', on='parcelid')
train_df2017 = train_df2017.merge(properties1, how='left', on='parcelid')
train_df = pd.concat([train_df2016,train_df2017])
train_df = add_date_features(train_df)

test_df = pd.read_csv('data/sample_submission.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']

#get the different labels accroding to the different months
test_df201610 = test_df.merge(properties, how='left', on='parcelid')
test_df201611 = test_df.merge(properties, how='left', on='parcelid')
test_df201612 = test_df.merge(properties, how='left', on='parcelid')

test_df201710 = test_df.merge(properties1, how='left', on='parcelid')
test_df201711 = test_df.merge(properties1, how='left', on='parcelid')
test_df201712 = test_df.merge(properties1, how='left', on='parcelid')

print("Train: ", train_df.shape)
print("Test: ", test_df.shape)

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

exclude_other = ['parcelid', 'logerror']  # for indexing/training only

exclude_other.append('propertyzoningdesc')
exclude_other.append('transaction_day')
exclude_other.append('transaction_year')
exclude_other.append('calculatedbathnbr')
exclude_other.append('finishedsquarefeet50')
exclude_other.append('censustractandblock')
exclude_other.append('assessmentyear')
exclude_other.append('threequarterbathnbr')

#exclude the correlative features
#exclude_other.append('bedroomcnt')
#exclude_other.append('fips')
#exclude_other.append('landtax_ratio')
#exclude_other.append('regionidcounty')
#exclude_other.append('garagecarcnt')
#exclude_other.append('roomcnt')
#exclude_other.append('taxvaluedollarcnt')
#exclude_other.append('landtaxvaluedollarcnt')
#exclude_other.append('structuretaxvaluedollarcnt')
#exclude_other.append('finishedfloor1squarefeet')
#exclude_other.append('fireplacecnt')
#exclude_other.append('pooltypeid10')
#exclude_other.append('propertylandusetypeid')

#decide the train features
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

#decide the cat features 
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'living_area_prop' in c \
       and not 'value_prop' in c \
       and not 'ratio' in c \
       and not 'per_price' in c \
       and not 'life' in c \
       and not 'yearbuilt' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

# some out of range int is a good choice
train_df.fillna(-999, inplace=True)
test_df201610.fillna(-999, inplace=True)
test_df201611.fillna(-999, inplace=True)
test_df201612.fillna(-999, inplace=True)
test_df201710.fillna(-999, inplace=True)
test_df201711.fillna(-999, inplace=True)
test_df201712.fillna(-999, inplace=True)

X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)

#X_train,test_X, y_train, test_y = train_test_split(X_train,  
#                                                   y_train,  
#                                                   test_size = 0.2,  
#                                                   random_state = 0) 

test_df201610['transactiondate'] = pd.Timestamp('2016-10-01')
test_df201611['transactiondate'] = pd.Timestamp('2016-11-01')  
test_df201612['transactiondate'] = pd.Timestamp('2016-12-01')  
test_df201710['transactiondate'] = pd.Timestamp('2017-10-01')  
test_df201711['transactiondate'] = pd.Timestamp('2017-11-01')  
test_df201712['transactiondate'] = pd.Timestamp('2017-12-01')  

test_df201610 = add_date_features(test_df201610)
test_df201611 = add_date_features(test_df201611)
test_df201612 = add_date_features(test_df201612)
test_df201710 = add_date_features(test_df201710)
test_df201711 = add_date_features(test_df201711)
test_df201712 = add_date_features(test_df201712)

X_test201610 = test_df201610[train_features]
X_test201611 = test_df201611[train_features]
X_test201612 = test_df201612[train_features]
X_test201710 = test_df201710[train_features]
X_test201711 = test_df201711[train_features]
X_test201712 = test_df201712[train_features]

num_ensembles = 6
y_pred201610 = 0.0
y_pred201611 = 0.0
y_pred201612 = 0.0
y_pred201710 = 0.0
y_pred201711 = 0.0
y_pred201712 = 0.0
for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations=550, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train,
        #eval_set=[test_X,test_y],
        cat_features=cat_feature_inds,
        verbose=True,
        #use_best_model=True
        )
    y_pred201610 += model.predict(X_test201610)
    y_pred201611 += model.predict(X_test201611)
    y_pred201612 += model.predict(X_test201612)
    y_pred201710 += model.predict(X_test201710)
    y_pred201711 += model.predict(X_test201711)
    y_pred201712 += model.predict(X_test201712)
    
y_pred201610 /= num_ensembles
y_pred201611 /= num_ensembles
y_pred201612 /= num_ensembles
y_pred201710 /= num_ensembles
y_pred201711 /= num_ensembles
y_pred201712 /= num_ensembles

submission = pd.DataFrame({
    'ParcelId': test_df['parcelid'],
})

submission['201610'] = y_pred201610
submission['201611'] = y_pred201611
submission['201612'] = y_pred201612
submission['201710'] = y_pred201710
submission['201711'] = y_pred201711
submission['201712'] = y_pred201712      

submission_major = 1
submission.to_csv(
    'finalsubmission_%03d.csv' % (submission_major),
    float_format='%.4f',
    index=False)
print("Great submission #%d :)" % submission_major)

feature_importance = model.get_feature_importance(X_train, y_train,cat_features=cat_feature_inds,
                                                  thread_count=1,fstr_type='FeatureImportance')
