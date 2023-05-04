# example of Exploratory Data Analysis with Python
# taken from: https://www.kaggle.com/code/robikscube/introduction-to-exploratory-data-analysis/notebook
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# use default matplotlibrc 
plt.style.use('default') 
import seaborn as sns
from IPython import embed
# read data
df = pd.read_csv('coaster_db.csv')

# Step 1. Understanding our data set
# shape of dataframe
print("Dataframe shape = ", df.shape)
# show first 5 rows of dataframe
print(df.head(5))
# print columns
# print(df.columns)
# print type of data inside
# print(df.dtypes)
# show basic statistics of our dataset
print("\nBasic dataframe statistics (describe()):", df.describe())

# Step 2. Data preparation
# Drop irrelevant columns and rows
# Indentify duplicated columns
# Rename columns
# Create features

# example of droppin column by its name would be
# df.drop(['Opening date'], axis=1)

# let's use the columns information to drop the columns we don't want to use but to keep track of unused information
df = df[['coaster_name', 
        # 'Length', 'Speed', 
        'Location', 'Status', 
        # 'Opening date', 'Type', 
        'Manufacturer', 
        # 'Height restriction', 'Model', 'Height',
        # 'Inversions', 'Lift/launch system', 'Cost', 'Trains', 'Park section',
        # 'Duration', 'Capacity', 'G-force', 'Designer', 'Max vertical angle',
        # 'Drop', 'Soft opening date', 'Fast Lane available', 'Replaced',
        # 'Track layout', 'Fastrack available', 'Soft opening date.1',
        # 'Closing date', 'Opened', 'Replaced by', 'Website',
        # 'Flash Pass Available', 'Must transfer from wheelchair', 'Theme',
        # 'Single rider line available', 'Restraint Style',
        # 'Flash Pass available', 'Acceleration', 'Restraints', 'Name',
       'year_introduced', 'latitude', 'longitude', 'Type_Main',
       'opening_date_clean', 
        # 'speed1', 'speed2', 'speed1_value', 'speed1_unit',
       'speed_mph', 
        # 'height_value', 'height_unit', 
       'height_ft',
       'Inversions_clean', 'Gforce_clean']].copy()
# and we make a copy of the df

# rewrite the opening_date_clean column to ensure it is a datetime type
df['opening_date_clean'] = pd.to_datetime(df['opening_date_clean'])
# print(df.dtypes)

# Rename columns using lowercase only
def convert_to_lower(columns) -> dict:

    lowercasecolumns = {}
    for column in columns:
        lowercasecolumns[column] = column.lower()
    return lowercasecolumns

df = df.rename(columns=convert_to_lower(df.columns))

# another way would be using directly
# df.columns = df.columns.str.lower()

# now let's have a look if there is some missing data in our set
# print(df.isna().sum())
# and to see if there are duplicates in our data
# print(df.duplicated(subset='coaster_name'))

# checking an exaple of duplicates
print("nChecking example of duplicates quering an specific coaster name:", df.query('coaster_name == "Derby Racer"'))

# now we will drop duplicated rows by matching on desired conditions
df = df.loc[~df.duplicated(subset=['coaster_name', 'location', 'opening_date_clean'])].reset_index(drop=True).copy()
print("\nResulting dataframe shape after dropping duplicated rows (coaster_name, location, opening_date):", df.shape)
# Step 3. Feature Understanding

# Count values of a given parameter using 'value_counts' in a pd Series
fig = plt.figure()
ax1 = df.sort_values('year_introduced', ascending=False)['year_introduced'].value_counts().head(15).plot(kind='bar', title='Top 15 years of Coasters Introduced')
ax1.set_xlabel('Year Introduced')
ax1.set_ylabel('Count')

# Plot distribution of a parameter
fig = plt.figure()
ax2 = df['speed_mph'].plot(kind='hist', bins=20)
ax2.set_xlabel('Coaster Speed (mph)')
# another way to look at features is makinga density plot
fig = plt.figure()
ax3 = df['speed_mph'].plot(kind='kde')
ax3.set_xlabel('Coaster Speed (mph)')

 # Step 4. Feature relationships
 # scatterplot
 # heatmap correlation
 # pairplot
 # groupby comparisons

fig = plt.figure()
ax4 = sns.scatterplot(x='speed_mph',
                y='height_ft',
                hue='year_introduced',
                data=df)
ax4.set_title('Coaster Speed vs. Height')

ax5 = sns.pairplot(df,
              vars=['year_introduced', 'speed_mph', 'height_ft', 'inversions_clean', 'gforce_clean'],
              hue='type_main'
            )

# another feature is to look at correlation between parameters
corr = df[['year_introduced', 'speed_mph', 'height_ft', 'inversions_clean', 'gforce_clean']].dropna().corr()
print("\nCorrelation matrix:", corr)

fig = plt.figure()
ax6 = sns.heatmap(corr, annot=True)

# Step 5. Ask a Question about our data
"""
    What are the locations with the fastest rollercoaster (minimum of 10)?
"""
# first of all we remove the 'Other' location from our set using a query
query = df.query('location != "Other"').groupby('location')['speed_mph'] \
        .agg(['mean', 'count']) \
        .query('count >= 10') \
        .sort_values('mean')

print("\nFilter locations with the fastest rollercoasters (minimum 10):\n", query)
fig = plt.figure()
ax7 = query['mean'].plot(kind='barh', figsize=(12,5), title='Average Coast Speed by Location')
ax7.set_xlabel('Mean speed (mph)')

plt.show()

embed()