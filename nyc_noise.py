#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import pandas as pd
import regex as re
from pandas import read_csv
from matplotlib import pyplot
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression


# In[7]:


# load data
# data is derived from NYC OpenData (https://opendata.cityofnewyork.us/) using the search term
#       "Noise Complaints (in 2017)"
nyc_data = pd.read_csv('nyc_noise_complaints.csv')
nyc_data.head()


# In[26]:


nyc_data.tail()


# In[8]:


"""
QUESTION:
1.1 How many rows are in the data set?
"""
print("\n*********\n1.1 ANSWER:")
print("Number of rows:",len(nyc_data))
print("*********\n")


# In[10]:


"""
QUESTION:
1.2 What fraction of noise complaints deal with music? A complaint is considered to deal with 
music if it has the string "Music" present in the value of the "descriptor" column.
"""
count = 0
for row in nyc_data["descriptor"]:
  if 'Music' in row:
    count = count + 1 
print("\n*********\n1.2 ANSWER:")
print("Fraction of noise complaints dealing with music:", count/len(nyc_data))
print("*********\n")


# In[11]:


"""
QUESTION:
1.3 For noise complaints with creation date in 2022, what is the probability a complaint's 
status is "Closed" given that it happened in Manhattan? Complaint creation date is logged 
in column "created_date", status of a complaint is in column "status" and the borough is 
in column "borough".
"""
nyc_data1 = nyc_data[  nyc_data['complaint_type'].str.contains('Noise')  ]
print("\nNumber of Noise rows:",len(nyc_data1))
nyc_data2 = nyc_data1[nyc_data1['created_date'].str.contains('2022')  ]
print("\nNumber of Noise 2022 rows:",len(nyc_data2))
nyc_data3 = nyc_data2[nyc_data2['status'].str.contains('Closed')  ]
print("\nNumber of Noise 2022 Closed rows:",len(nyc_data3))
nyc_data4 = nyc_data3[nyc_data3['borough'].str.contains('MANHATTAN')  ]
print("\nNumber of Noise 2022 Closed MANHATTAN rows:",len(nyc_data4))
print("\n***\n1.3 ANSWER:")
print("Probability of a closed, noise complaint located in Mahahattan in 2022:", len(nyc_data4)/len(nyc_data2))
print("***\n")


# In[16]:


"""
QUESTION:
1.4 How does construction noise vary across New York City? For each ZIP code, calculate 
fraction of noise complaints that are due to construction. For simplification, a 
complaint dealing with construction noise is one with with the string "Construction" 
appearing anywhere in the "descriptor" column. Once you have the fractions for each ZIP code, 
report the standard deviation. Exclude ZIP codes that do not have at least 100 complaints 
dealing with construction noise.
"""
# isolate construction noise complaints
nyc_data1 = nyc_data[  nyc_data['complaint_type'].str.contains('Noise')  ]
print("Number of Noise rows:",len(nyc_data1))
nyc_data5 = nyc_data1[nyc_data1['descriptor'].str.contains('Construction')  ]
print("Number of Construction Noise rows:",len(nyc_data5))
listy = nyc_data5['incident_zip'].unique()
print("Number of unique zipcodes:",listy.shape)
print("Dimensions of component dataframe containing only complaints due to construction noise", nyc_data5.shape)

# form dictionary of keys (zipcodes) : values (complaints)
ZIPCODES = Counter(nyc_data5['incident_zip']).keys()
COMPLAINTS = Counter(nyc_data5['incident_zip']).values()
print("\nZipcodes as Keys in Key:Value pairs:\n",ZIPCODES)
print("\nComplaints per zipcode as Values in Key:Value pairs:\n", COMPLAINTS)
zip_total_data = {'ZIP_CODE':list(ZIPCODES), 'COMPLAINTS':list(COMPLAINTS)}

# limit to at least 100 complaints, convert dictionary to dataframe of integers
construc_zips = pd.DataFrame(zip_total_data)
construc_zips_final = construc_zips[construc_zips['COMPLAINTS'] >= 100]
construc_zips_final = construc_zips_final.astype({"ZIP_CODE":"int","COMPLAINTS":"int"})


"""
print("\nZipcodes with over 100 complaints:")
print(construc_zips_final.head())
print(construc_zips_final.tail())
"""
# sum total complaints to form fractions
print(construc_zips_final['COMPLAINTS'].sum())
fractions = (construc_zips_final['COMPLAINTS']/construc_zips_final['COMPLAINTS'].sum())
# add new column for fractions
construc_zips_final['FRACTIONS'] = fractions
print(construc_zips_final.head())
print(construc_zips_final.tail())
print("\n*********\n1.4 ANSWER:")
print("Standard deviation for construction complaints in NYC zipcodes:", construc_zips_final['FRACTIONS'].std())
print("*********\n")


# In[20]:


"""
QUESTION:
1.5 As the population of a ZIP code increases so do the number of complaints. 
We can visualize this trend by plotting the number of complaints as a function of the 
ZIP code population. What is the slope of a line of best fit? A CSV file with the population 
data for each ZIP code can be downloaded here.
"""
# load data
# data is derived from NYC OpenData (https://opendata.cityofnewyork.us/) using the search term
#       "Modified Zip Code Tabulation Areas (MODZCTA)"
nyc_pop_data = pd.read_csv('nyc_population.csv')




# In[23]:


# limit analysis to NYC zip codes based on https://bklyndesigns.com/new-york-city-zip-code/
from IPython.display import Image
Image(filename = "nyc_zip_codes.jpg", width=500, height=500)


# In[28]:


# form dataframe of true NYC zips with associated populations
nyc_pop_true_zips = nyc_pop_data[ (nyc_pop_data['ZIP_CODE'] >= 10000) & (nyc_pop_data['ZIP_CODE'] <= 11300) ]
# form dataframe of true NYC zips with associated construction complaints
construc_zips_final = construc_zips_final[construc_zips_final['ZIP_CODE'].between(10000, 11300) ]
# merge two dataframes using shared zip code column
merged_nyc_pop_zips_complaints = pd.merge(construc_zips_final, nyc_pop_true_zips, on ='ZIP_CODE' )
merged_nyc_pop_zips_complaints.head()


# In[29]:


merged_nyc_pop_zips_complaints.tail()


# In[42]:


# extract x and y data for regression
x = np.array(merged_nyc_pop_zips_complaints['ZIP_CODE']).reshape((-1, 1))
y = np.array(merged_nyc_pop_zips_complaints['COMPLAINTS'])
complaint_zip_model = LinearRegression().fit(x,y)
COD = complaint_zip_model.score(x,y)
intercept = complaint_zip_model.intercept_
slope = complaint_zip_model.coef_
print("\n*********\n1.5 ANSWER:")
print("Model Coefficient of Determination:",COD)
print("Model Y-intercept:",intercept)
print("Model Slope:",slope)
print("*********\n")


# In[50]:


# plot linear regression data and best fit line
pyplot.plot(x, y, 'o')
pyplot.plot(x, slope*x + intercept)
pyplot.title("NYC Noise Complaint - Population Zip Code Regression")
pyplot.ylabel("Noise Complaints")
pyplot.xlabel("Zip Code")
pyplot.savefig("nyc_regress.pdf")


# In[ ]:




