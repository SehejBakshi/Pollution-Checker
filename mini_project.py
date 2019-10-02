import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('datasets/IndiaAirQualityData.csv', encoding='ISO-8859-1', low_memory=False)
#print(df.head(5))

##Find out the states with minimum/maximum pollution parameters
fig, axes=plt.subplots(figsize=(20,12), ncols=5)
state_wise_max_so2=df[['state', 'so2']].dropna().groupby('state').median().sort_values(by='so2')
state_wise_max_no2=df[['state', 'no2']].dropna().groupby('state').median().sort_values(by='no2')
state_wise_max_rspm=df[['state', 'rspm']].dropna().groupby('state').median().sort_values(by='rspm')
state_wise_max_spm=df[['state', 'spm']].dropna().groupby('state').median().sort_values(by='spm')
state_wise_max_pm2_5=df[['state', 'pm2_5']].dropna().groupby('state').median().sort_values(by='pm2_5')

sns.barplot(x='so2', y=state_wise_max_so2.index, data=state_wise_max_so2, ax=axes[0])
axes[0].set_title("Average so2 observed in a state")

sns.barplot(x='no2', y=state_wise_max_no2.index, data=state_wise_max_no2, ax=axes[1])
axes[1].set_title("Average no2 observed in a state")

sns.barplot(x='rspm', y=state_wise_max_rspm.index, data=state_wise_max_rspm, ax=axes[2])
axes[2].set_title("Average rspm observed in a state")

sns.barplot(x='spm', y=state_wise_max_spm.index, data=state_wise_max_spm, ax=axes[3])
axes[3].set_title("Average spm observed in a state")

sns.barplot(x='pm2_5', y=state_wise_max_pm2_5.index, data=state_wise_max_pm2_5, ax=axes[4])
axes[4].set_title("Average pm2_5 observed in a state")
plt.tight_layout()

##Top 10 cities with highest risk of respiratory disease
state=df[['state', 'location', 'rspm']].groupby(['state', 'location']).median().reset_index()
state_location_max_rspm=state.loc[state.groupby('state')['rspm'].idxmax()].sort_values(by='rspm', ascending=False).head(10)
fig, ax=plt.subplots(figsize=(12,6))
sns.barplot(x='rspm', y='location', data=state_location_max_rspm, palette='coolwarm', axes=ax)
sns.despine(left=True)
ax.set_title('Bars showing average rspm values of the cities')
plt.tight_layout()

#plt.show()

##Show RSPM variation in Ghaziabad over time
rspm_data=df[df['location']=='Ghaziabad'][['date', 'rspm']].dropna()
fig, ax=plt.subplots(figsize=(12,8))
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
sns.lineplot(x='date', y='rspm', data=rspm_data, axes=ax, label='rspm')
fig.autofmt_xdate()
plt.tight_layout()

##Show SPM variation in Ghaziabad over time
spm_data=df[df['location']=='Ghaziabad'][['date', 'spm']].dropna()
fig, ax=plt.subplots(figsize=(12,8))
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
sns.lineplot(x='date', y='spm', data=spm_data, axes=ax, label='spm')
fig.autofmt_xdate()
plt.tight_layout()


##Show so2 variation in Ghaziabad over time
so2_data=df[df['location']=='Ghaziabad'][['date', 'so2']].dropna()
fig, ax=plt.subplots(figsize=(12,8))
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
sns.lineplot(x='date', y='so2', data=so2_data, axes=ax, label='so2')
fig.autofmt_xdate()
plt.tight_layout()
#plt.show()


##Show no2 variations in Ghaziabad over time
no2_data=df[df['location']=='Ghaziabad'][['date', 'no2']].dropna()
fig, ax=plt.subplots(figsize=(12,8))
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
sns.lineplot(x='date', y='no2', data=no2_data, axes=ax, label='no2')
fig.autofmt_xdate()
plt.tight_layout()

##pm2_5 is NA

##Top 5 states with highest number of monitoring stations
mon_station=df.drop_duplicates(subset=['location_monitoring_station'])
grouped_mon_station = mon_station[['state', 'location_monitoring_station']].groupby('state').count().sort_values(by='location_monitoring_station', ascending=False).head(5)
fig, ax=plt.subplots(figsize=(12,6))
sns.set_style('dark')
sns.barplot(x=grouped_mon_station.index, y='location_monitoring_station', data=grouped_mon_station, axes=ax)
sns.despine()
plt.tight_layout()

#plt.show()


##Top 5 sampling stations to produce highest samples
location_mon=df[['state', 'location_monitoring_station', 'date']].groupby(['state', 'location_monitoring_station']).count().reset_index()
location_mon.loc[location_mon.groupby('state')['date'].idxmax()].sort_values(by='date', ascending=False).head(5)


##5 sampling stations to produce least samples  
location_mon=df[['state', 'location_monitoring_station', 'date']].groupby(['state', 'location_monitoring_station']).count().reset_index()
location_mon.loc[location_mon.groupby('state')['date'].idxmax()].sort_values(by='date').head(5)


##Relation between so2 and no2
relation_df_so2_no2=df[(df['state']=='Uttar Pradesh') & (df['location']=='Ghaziabad')][['so2', 'no2']].dropna()
sns.jointplot(x='so2', y='no2', data=relation_df_so2_no2, height=12)

##Relation between rspm and spm
relation_df_rspm_spm=df[(df['state']=='Uttar Pradesh') & (df['location']=='Ghaziabad')][['rspm', 'spm']].dropna()
sns.jointplot(x='rspm', y='spm', data=relation_df_rspm_spm, height=12)

plt.show()

