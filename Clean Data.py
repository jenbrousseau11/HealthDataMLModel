# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:49:05 2023

@author: jbrousseau
"""
#%%
#### import needed packages
import xml.etree.ElementTree as ET
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns



#%%
#### edit settings
#set time frame
start_date='2023-03-01'
end_date='2023-09-30'


plt.style.use("fivethirtyeight")

#%%
#### Load Data




# create element tree object
tree = ET.parse('\\apple_health_export\\export.xml') 


### Sleep Data
# for every health record, extract the attributes
root = tree.getroot()
record_list = [x.attrib for x in root.iter('Record')]
record_data = pd.DataFrame(record_list)

### Workout Data
workout_list = [x.attrib for x in root.iter('Workout')]
workout_data = pd.DataFrame(workout_list)

#%%
#### Clean Data

### Sleep Data
sleep_data = record_data.loc[(record_data['type'].str.contains("HKCategoryTypeIdentifierSleepAnalysis"))]
sleep_data['value'] = sleep_data['value'].str.replace('HKCategoryValueSleepAnalys', '')
sleep_data['endDate'] = pd.to_datetime(sleep_data['endDate'])
sleep_data['startDate'] = pd.to_datetime(sleep_data['startDate'])
sleep_data['sleep_date'] = sleep_data['startDate'].apply(lambda x: x.date() if x.hour >= 20 else (x - pd.DateOffset(days=1)).date())

sleep_data['Mins'] = (sleep_data.endDate-sleep_data.startDate).dt.seconds/60
sleep_data = sleep_data[(sleep_data['startDate'] > start_date) & (sleep_data['endDate'] <= end_date)]
sleep_data_agg_Day = sleep_data.groupby(['sleep_date','value'])['Mins'].sum().reset_index()


sleep_pivot  = sleep_data_agg_Day.pivot(index='sleep_date', columns='value', values='Mins').reset_index()
sleep_pivot = sleep_pivot.drop(['isInBed','isAsleepUnspecified'],axis=1)

#I want to fill in the data but I think I will do weekend and weeday seperately
sleep_pivot['sleep_date'] = pd.to_datetime(sleep_pivot['sleep_date'])
sleep_pivot['Weekend'] = sleep_pivot['sleep_date'].dt.dayofweek.isin([4, 5]).astype(int)

all_dates = pd.date_range(start = sleep_pivot['sleep_date'].min(), end =  sleep_pivot['sleep_date'].max())
all_dates_df = pd.DataFrame(all_dates,columns=['sleep_date'])
all_dates_sleep = all_dates_df.merge(sleep_pivot,how='left' )

all_dates_sleep[all_dates_sleep['Weekend']==1]
# theres only 3 values here werid lets just remove weeekends
weekday_sleep = all_dates_sleep[all_dates_sleep['Weekend']==0]

# how many nas do I have?
weekday_sleep.isna().sum()
weekday_sleep = weekday_sleep.fillna(method="ffill")

### Workout Data
workout_data['workoutActivityType'] = workout_data['workoutActivityType'].str.replace('HKWorkoutActivityType', '')
workout_data = workout_data.rename({"workoutActivityType": "WorkoutType"}, axis=1)
# proper type to dates
for col in ['creationDate', 'startDate', 'endDate']:
    workout_data[col] = pd.to_datetime(workout_data[col])

# convert string to numeric   
workout_data['duration'] = pd.to_numeric(workout_data['duration'])

#drop columns that arent needed
workout_data_sub = workout_data.drop(columns=['sourceName','sourceVersion','device'])

#subset to data after getting apple watch
workout_data_sub = workout_data_sub[(workout_data_sub['startDate'] > start_date) & (workout_data_sub['endDate'] <= end_date)]

# look at counts of workout type
workout_data_sub.head()
print(workout_data_sub['WorkoutType'].value_counts())
#print(workout_data_sub['durationUnit'].unique())


# removing the activites that I didnt do regularly
regular_activities = ['FunctionalStrengthTraining','Walking','Yoga','TraditionalStrengthTraining','Tennis','Downhill Skiing']
apple_data = workout_data_sub[workout_data_sub['WorkoutType'].isin(regular_activities)]

print(apple_data['WorkoutType'].value_counts())

## add chuze data
chuze = pd.read_csv('Chuze check ins.csv')
chuze['Date'] = pd.to_datetime(chuze['Date'])
chuze = chuze[(chuze['Date'] > start_date) & (chuze['Date'] <= end_date)]
chuze['Date'] = pd.to_datetime(chuze['Date']).dt.date
chuze['Location'] = 'Gym'

# merge gym and apple watch data
apple_data_merge = apple_data.copy()
apple_data_merge['startDate'] = pd.to_datetime(apple_data_merge['startDate']).dt.date
chuze_apple = apple_data_merge.merge(chuze,how='left', left_on='startDate',right_on='Date')

chuze_apple['Location'] = np.where(chuze_apple['Location']== 'Gym', 'Gym', 
                                   np.where(chuze_apple['WorkoutType']== 'FunctionalStrengthTraining', 'PSF', 
                                            np.where(chuze_apple['WorkoutType']== 'TraditionalStrengthTraining', 'Gym', 
                                               chuze_apple['WorkoutType']   )))


#%%
#### Explore
#sleep data
# compare not filled with filled
sleep_chart = sleep_pivot.plot(x='sleep_date')
sleep_chart.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

sleep_chart_clean = weekday_sleep.plot(x='sleep_date')
sleep_chart_clean.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

sleep_pivot.boxplot()
weekday_sleep.boxplot()

# what percent am I awake
weekday_sleep["TotalSleepTime"] = weekday_sleep["isAsleepCore"]+weekday_sleep["isAsleepDeep"] + weekday_sleep["isAsleepREM"] + weekday_sleep["isAwake"]
weekday_sleep['PctAwake'] = weekday_sleep["isAwake"]/weekday_sleep["TotalSleepTime"]
weekday_sleep.describe()
weekday_sleep["BelowAvg"]= np.where(weekday_sleep['PctAwake']<weekday_sleep['PctAwake'].mean(),1,0)
weekday_sleep.count()
weekday_sleep['BelowAvg'].sum()

weekday_sleep.boxplot(column='PctAwake')

# Agg to see weekly
sleep_data['sleep_date'] = pd.to_datetime(sleep_data['sleep_date'])
sleep_data['Week'] = sleep_data["sleep_date"].dt.to_period('W').dt.start_time
sleep_agg_week = sleep_data.groupby(['Week','value'])['Mins'].sum().reset_index()

sleep_pivot_week  = sleep_agg_week.pivot(index='Week', columns='value', values='Mins').reset_index()
sleep_pivot_week = sleep_pivot_week.drop('isInBed',axis=1)
sleep_pivot_week["TotalSleepTime"] = sleep_pivot_week["isAsleepCore"]+sleep_pivot_week["isAsleepDeep"] + sleep_pivot_week["isAsleepREM"] + sleep_pivot_week["isAwake"]

sleep_pivot_week.plot(subplots=True,x='Week');

# Agg to see monthly
sleep_data['sleep_date'] = pd.to_datetime(sleep_data['sleep_date'])
sleep_data['Month'] = pd.DatetimeIndex( sleep_data["sleep_date"]).month
sleep_agg_month = sleep_data.groupby(['Month','value'])['Mins'].sum().reset_index()

sleep_pivot_month  = sleep_agg_month.pivot(index='Month', columns='value', values='Mins').reset_index()
sleep_pivot_month = sleep_pivot_month.drop('isInBed',axis=1)
sleep_pivot_month["TotalSleepTime"] = sleep_pivot_month["isAsleepCore"]+sleep_pivot_month["isAsleepDeep"] + sleep_pivot_month["isAsleepREM"] + sleep_pivot_month["isAwake"]

sleep_pivot_month.plot(subplots=True,x='Month');

### Workout Data
#see data summary
chuze_apple.groupby('Location').describe(percentiles=[])

#remove workouts less than 5 mins 
chuze_apple = chuze_apple[chuze_apple['duration']>5]

workout_agg_day = chuze_apple.groupby(['Location','startDate'])[['duration']].sum().reset_index()
workout_pivot_day = workout_agg_day.pivot(index="startDate", columns="Location",values='duration').reset_index()
workout_pivot_day = workout_pivot_day.fillna(0)

#plot visits by location over time
axes = plt.gca()
workout_pivot_day.plot(kind='line', x='startDate', y='Gym', ax=axes);
workout_pivot_day.plot(kind='line', x='startDate', y='PSF', ax=axes);
workout_pivot_day.plot(kind='line', x='startDate', y='Walking', ax=axes);
workout_pivot_day.plot(kind='line', x='startDate', y='Yoga', ax=axes);

workout_pivot_day.plot(subplots=True);

workout_pivot_day.boxplot()
plt.hist(workout_pivot_day['PSF'])
plt.hist(workout_pivot_day['Gym'])
plt.hist(workout_pivot_day['Walking'])
plt.hist(workout_pivot_day['Yoga'])
workout_pivot_day.describe()

fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].hist(workout_pivot_day['PSF'])
axes[0,0].set_title("PSF")
axes[0,1].hist(workout_pivot_day['Gym'])
axes[0,1].set_title("Gym")
axes[1,1].hist(workout_pivot_day['Walking'])
axes[1,1].set_title("Walking")
axes[1,0].hist(workout_pivot_day['Yoga'])
axes[1,0].set_title("Yoga")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# Agg to see weekly
workout_week = chuze_apple.copy()
workout_week['startDate'] = pd.to_datetime(workout_week['startDate'])
workout_week['Week'] = workout_week["startDate"].dt.to_period('W').dt.start_time
workout_agg_week = workout_week.groupby(['Location','Week'])[['duration']].sum().reset_index()
workout_pivot_week = workout_agg_week.pivot(index="Week", columns="Location",values='duration').reset_index()
workout_pivot_week = workout_pivot_week.fillna(0)


axes = plt.gca()
workout_pivot_week.plot(kind='line', x='Week', y='Gym', ax=axes);
workout_pivot_week.plot(kind='line', x='Week', y='PSF', ax=axes);
workout_pivot_week.plot(kind='line', x='Week', y='Walking', ax=axes);
workout_pivot_week.plot(kind='line', x='Week', y='Yoga', ax=axes);

workout_pivot_week.plot(subplots=True,x='Week');

# Agg to see monthly
workout_month = chuze_apple.copy()
workout_month['startDate'] = pd.to_datetime(workout_month['startDate'])
workout_month['Month'] = pd.DatetimeIndex( workout_month["startDate"]).month
workout_agg_month = workout_month.groupby(['Location','Month'])[['duration']].sum().reset_index()
workout_pivot_month = workout_agg_month.pivot(index="Month", columns="Location",values='duration').reset_index()
workout_pivot_month = workout_pivot_month.fillna(0)


axes = plt.gca()
workout_pivot_month.plot(kind='line', x='Month', y='Gym', ax=axes);
workout_pivot_month.plot(kind='line', x='Month', y='PSF', ax=axes);
workout_pivot_month.plot(kind='line', x='Month', y='Walking', ax=axes);
workout_pivot_month.plot(kind='line', x='Month', y='Yoga', ax=axes);

workout_pivot_month.plot(subplots=True,x='Month');

####
#put sleep times and workout times on same chart to see if theres a correlation
total_workout_time = workout_pivot_month.copy()
total_workout_time['TotalTime']= total_workout_time['PSF'] + total_workout_time['Gym'] + total_workout_time['Walking'] + total_workout_time['Yoga']


fig, axes = plt.subplots(nrows=3,ncols=1)
axes[0].plot(total_workout_time['Month'], total_workout_time['TotalTime'])
axes[0].title.set_text('Total Workout Time')
axes[1].title.set_text('Total Time Awake')
axes[1].plot(sleep_pivot_month['Month'], sleep_pivot_month['isAwake'])
axes[2].title.set_text('Pct Awake')
axes[2].plot(sleep_pivot_month['Month'], sleep_pivot_month['isAwake']/sleep_pivot_month['TotalSleepTime'])
plt.tight_layout()


plt.hist(total_workout_time['TotalTime'])
total_workout_time.boxplot()

### Put data together
weekday_sleep['sleep_date'] = pd.to_datetime(weekday_sleep['sleep_date'])
workout_pivot_day['startDate'] = pd.to_datetime(workout_pivot_day['startDate'])
workout_sleep = weekday_sleep.merge(workout_pivot_day, left_on='sleep_date',right_on='startDate',how='left')
workout_sleep = workout_sleep.fillna(0)

workout_sleep.corr()
plt.matshow(workout_sleep.corr())
plt.show()

corr = workout_sleep.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

## final dataset
# according to https://www.whoop.com/us/en/thelocker/average-sleep-stages-time/ 9-11% time awake is av
workout_sleep["BelowAvg"]= np.where(workout_sleep['PctAwake']<=.09,1,0)
model_cols = ['Gym','PSF','Tennis','Walking','BelowAvg']
workout_sleep_final = workout_sleep[model_cols]

workout_sleep_final.count()
workout_sleep_final['BelowAvg'].sum()

workout_sleep_final.to_csv("workout_sleep_classifier_model_dataset.csv")
