
# coding: utf-8

# # Advanced programming for data analysis: pandas.
# 
# pandas is a Python package providing fast, flexible, and expressive data structures designed to work with relational or labeled data both. It is a fundamental high-level building block for doing practical, real world data analysis in Python.
# 
# pandas is well suited for:
# 
# + Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet
# + Ordered and unordered (not necessarily fixed-frequency) time series data.
# + Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels.
# 
# Key features:
# 
# + Easy handling of missing data
# + Size mutability: columns can be inserted and deleted from DataFrame
# + Powerful, flexible ``group by`` functionality to perform split-apply-combine operations on data sets
# + Intelligent label-based slicing, fancy indexing, and subsetting of large data sets
# + Intuitive merging and joining data sets
# + Flexible reshaping and pivoting of data sets
# + Hierarchical labeling of axes
# + Robust IO tools for loading data from flat files, Excel files, databases, and HDF5
# + Time series functionality: date range generation and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging, etc.
# 

# Importing pandas:

# In[1]:

import pandas as pd                            # convention, alias 'pd'


# ## Pandas Data Structure
# 
# ###Series
# 
# A Series is a single vector of data with an index that labels every element in the vector. If we do not specify the index, a sequence of integers is assigned as the index.

# In[2]:

c = pd.Series([1956, 1967, 1989, 2000])
c


# Its values are stored in a NumPy array (``values``) and the index in a pandas ``Index`` object:

# In[3]:

c.values


# In[4]:

c.index


# We can assign labels to the index, while creating the serie

# In[5]:

c = pd.Series([1956, 1967, 1989, 2000], index = ['a','b','c','d'])
c


# Labels can be used to refer to the values in the serie

# In[6]:

c['d']


# We can still use positional index

# In[7]:

c[3]


# We can give both the array of values and the index meaningful labels names

# In[8]:

c.name = 'years'
c.index.name = 'obs'
c


# ###Time Series

# In working with time series data, we will frequently seek to:
# 
# - generate sequences of fixed-frequency dates and time spans
# - conform or convert time series to a particular frequency
# - compute “relative” dates based on various non-standard time increments (e.g. 5 business days before the last business day of the year), or “roll” dates forward or backward
# 
# pandas provides a relatively compact and self-contained set of tools for performing the above tasks.
# 
# Create a range of dates:

# In[9]:

# 72 hours starting with midnight Jan 1st, 2011
rng = pd.date_range('1/1/2011', periods=72, freq='H')


# Index pandas objects with dates:

# In[10]:

import numpy as np
ts = pd.Series(np.random.randint(0,500,len(rng)),index = rng)
ts.head()


# Change frequency and fill gaps:

# In[11]:

# to 45 minute frequency and forward fill
converted = ts.asfreq('45Min', method='pad')
converted.head()


# In[12]:

ts.resample('D', how='mean')


# Time zone representation

# Convert to another time zone

# Convert to TimeStamps

# ###DataFrames 
# DataFrames are designed to store heterogeneous multivarite data, where for every index there are multiple fields or columns of data, often of different data type.
# 
# A `DataFrame` os a tabular data structure, encapsulating multiple series like columns in a spreadsheet. Data are stored interally as a 2-dimensional object, but the `DataFrame` allows us to represent and manipulate higher-dimensional data.
# 

# ##Reading tabular data
# 
# The ‘pandas’ Python library provides several operators, <code>read_csv(), read_table(), 
# read_excel() ...</code> that allows you to access data ﬁles in tabular format on your computer as well as data stored in web repositories.
# 
# Reading in a data table is simply a matter of knowing the name (and location) of the data set.

# ###EUROSTAT data
# 
# Eurostat is the home of the European Commssion data. Eurostat’s main role is to process and publish comparable statistical information at European level. Data in eurostat is provided by each member state. Eurostat's re-use policy is free re-use of its data, both for non-commercial and commercial purposes (with some minor exceptions).

# In[13]:

from IPython.display import HTML
HTML('<iframe src=http://epp.eurostat.ec.europa.eu/portal/page/portal/eurostat/home  width=900 height=500></iframe>')


# In this case study we are going to retrieve Eurostat data. The amount of data in the database is huge, thus we are going to use a small subset for illustration purposes. In our first study we are going to focus on **indicators on education finance data** among the member states. The data is already downloaded and provided as is in the file `educ_figdp_1_Data.csv`. You can download it directly following this links `Database by terms>Population and social conditions>Education and training>Indicators on education finance > Expenditure on education as % of GDP or public expenditure (educ_figdp)`
# 
# Let us start reading the data:

# In[14]:

edu=pd.read_csv('./educ_figdp/educ_figdp_1_Data.csv',na_values=':')


# Check the shape and type of `DataFrame`

# In[15]:

edu.shape 


# In[16]:

type(edu)


# It is also possible to create a `DataFrame` from a multidimensional numpy array or by passing a `dict` of objects that can be converted to series-like.

# In[17]:

import numpy as np
dates = pd.date_range('19781212',periods=7)
df = pd.DataFrame(np.random.randn(7,3),index=dates, columns=['A','B','C'])
df


# In[18]:

dictionary = dict({'A' : 1, 
                   'B' : pd.date_range('19781212',periods=7), 
                   'C' : range(7),
                   'D' : np.arange(7)[::-1],
                   'E' : 'foo' })
pd.DataFrame(dictionary, index=range(1,8))


# We can also read data directly from the *clipboard*. Just `copy` some rows to the clipboard and use `read_clipboard()` function. By default it uses `S+` (space characters) as column separator.

# In[19]:

clipboard = pd.read_clipboard()
clipboard


# ## Viewing Data
# 
# Take a look at the **Eurostat** data:
# 
# The 5 first rows:

# In[20]:

edu.head()


# The last 5 rows:

# In[21]:

edu.tail()


# Data in CSV and databases are often organized in what is called *stacked* or *record* formats. In this case for each year (`TIME`) and country (`GEO`) of the EU as well as some reference countries such as Japan and United States, we have twelve indicators (`INDIC_ED`) on education finance with their values (`Value`): 

# In[22]:

edu.columns  # This is not a function; it is an attribute of the data frame.


# The values of the indexes can be retrieved using:

# In[23]:

edu.index


# The values of the `DataFrame` can be retrieved as a numpy array using:

# In[24]:

edu.values


# To get quick stadistical information about the numeric columns in a data frame is with the function `describe()`. The result is itself a data frame.

# In[25]:

edu.describe()


# ## Sorting

# We can sort the `DataFrame` using any column. If we want to see the data sorted by Time, it can be done like this:

# In[26]:

s = edu.sort(columns='Value', ascending= False)
s.head(9)


# We can sort by index again, using the `sort_index` function and specifying `axis=0`

# In[27]:

s.sort_index(axis=0,ascending=True).head()


# ##Selection
# 
# We can acces to each column by name:

# In[28]:

edu['TIME']


# We can acces to a *slice* of rows using []

# In[29]:

edu[10:14]


# If we want to select a subset of columns and rows we can use `ix` indexing

# In[30]:

edu.ix[15:20,['TIME','GEO','Value']]


# We can filter a `DataFrame` using boolean indexing.

# In[31]:

edu[edu.Value > 6.5].sort(columns='Value', ascending= False).head(10)


# We can **set** new values to rows or columns by using `=`after a selection.

# In[32]:

edu['Flag and Footnotes']=0
edu.head()


# Adding a new column to a data frame can be done similarly to accessing a column.

# In[33]:

edu['ValueNorm'] = edu.Value/np.max(edu.Value) # or data['time']/60.
edu.head()


# NOTE: If instead of using Pandas/numpy `max` function, we would use python built-in `max` function:

# In[34]:

max(edu.Value) ## don't use max min python built-in functions!!!!


# By default, columns get inserted at the end. The <code>insert</code> function is available to insert at a particular location in the columns.

# In[35]:

edu.insert(4, 'ValueSub',  edu.Value - np.min(edu.Value))
edu.head()


# If we want to add a new row on the bottom of the table, we can do it by assigning the new row to the last index:

# In[36]:

edu.ix[len(edu)] = [2000,'a','b',5.00,np.nan,0,np.nan]
edu.tail()


# ## Missing Data

# Pandas uses the value `np.nan` to represent missing data. 
# 
# The pandas.isnull function can be used to tell whether or not a value is missing.
# We can use the `numpy` values for filtering rows with NaN values

# In[37]:

edu[pd.isnull(edu).values]


# We can either filling missing values with `fillna(value=<value>)` function or drop all rows using `dropna()` function

# In[38]:

eduDrop = edu.dropna(how='any') #returns a copy of the data!!!
eduDrop.head()


# In[39]:

edu.head()


# It is equivalent to use `drop` function over the indexes of rows with `NaN` values

# In[40]:

edu = edu.drop(edu[pd.isnull(edu).values].index) ### Overwritting data!!!!
edu.head()


# ##Operations
# ###Statistical Operations:
# Operations in general exclude missing data.
#  

# In[41]:

edu.mean()


# In[42]:

print edu.Value.count()    # number of non-NaN values
print edu['Value'].mean()     # mean value 
print edu.Value.sum()   # sum of values


# In[43]:

print edu.Value.argmin()   # index location at which min is obtained
print edu.Value.min()      # min value
edu.ix[edu.Value.argmin(),['TIME','GEO']].values


# In[44]:

print edu.Value.argmax()   # index location at which min is obtained
print edu.Value.max()      # min value
edu.ix[edu.Value.argmax(),['TIME','GEO']].values


# In[45]:

minim=edu.Value.min()
edu.Value=edu.Value.sub(minim) #Substraction and Overwritting data!!!!
edu.head()


# 
# When you encounter a function that isn’t supported by data frames, you can use ‘numpy’ functions or the special <code>apply</code> function built-into data frames.
# 
# Using the ``apply()``method, which takes an anonymous function, we can apply any function to each value in a column.

# In[46]:

edu.Value=edu.Value.apply(lambda d: d**2)
edu.head()


# In[47]:

edu.Value=edu.Value.apply(np.sqrt)
edu.head()


# 

# In[47]:




# ## Reshaping and Pivoting
# 

# Let us reshape the table into a feature vector style data set. To the process of reshaping stacked data into a table is sometimes called **pivoting**.

# In[48]:

#Pivot table in order to get a nice feature vector representation with dual indexing by TIME and GEO 
edu=pd.read_csv('./educ_figdp/educ_figdp_1_Data.csv',na_values=':')
edu.INDIC_ED.value_counts()


# In[67]:

pivedu=pd.pivot_table(edu, values='Value', index=['TIME', 'GEO'],columns = ['INDIC_ED'])
pivedu.head()


# In[50]:

pivedu.ix[2010:2011,:]


# In[51]:

pivedu.ix[[(2010,'Spain'),(2010,'Romania'),(2010, 'Denmark'),(2011,'Spain'),(2011,'Romania'),(2011, 'Denmark')]]


# ##Ranking Countries

# We want to rank "Total public expenditure on education as % of GDP, for all levels of education combined" for all the coutries by year.
# 
# First we clean the data. Removing all non-countries and Countries without values (NaN):

# In[52]:

eduCtry=edu.drop(edu[edu.ix[:,'GEO'].isin(['Euro area (13 countries)','Euro area (15 countries)',
                      'Euro area (17 countries)','Euro area (18 countries)',
                      'European Union (25 countries)','European Union (27 countries)',
                      'European Union (28 countries)'])].index)
eduCtry = eduCtry.dropna(subset = ['Value'])


# In[53]:

piveduCtry=pd.pivot_table(eduCtry, values='Value', rows=['GEO'],columns = ['INDIC_ED','TIME'])
piveduCtry= piveduCtry.rename(index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
piveduCtry.rank(ascending=False,method='dense')


# ##Grouping

# `group by` means:
# + Splittng the data into groups based on some criteria
# + Applying a function to each group independently
# + Combining the results int a `dataframe`
# 
# We can group by countries and apply sum. This returns all the `TIME` and `Value` columns added for the same country:

# In[54]:

group = eduCtry.groupby('GEO').sum()
group= group.rename(index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
group.head()


# We drop time column and ranking by Value, over all the time:

# In[55]:

group.drop('TIME', axis=1).rank(ascending=False,method='dense').sort('Value')


# ##Plotting

# In[56]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

group.drop('TIME', axis=1).plot(kind='bar',style='b', alpha=0.4)
plt.title("bar of weight and mpg")
plt.figure()


# In[57]:

piveduCtry=pd.pivot_table(eduCtry, values='Value', rows=['GEO'],columns = ['TIME'])
piveduCtry= piveduCtry.rename(index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
piveduCtry = piveduCtry.fillna(0) #FILL NaN with 0
my_colors = ['b', 'r', 'g', 'y', 'k']*3 #By default ColorMap has only 5 colours
ax=piveduCtry.plot(kind='barh',stacked=True, color=my_colors ,alpha=0.4)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #By default legend goes inside plot area


# #Merge
# 

# 1M MovieLens database (http://www.grouplens.org/node/73) contains 1,000,209 ratings of 3,900 films made during yerar 2000 for 6040 anonymous users from MovieLens recommender Online (http://www.movielens.org)
# 

# The contents of the database are:
# 
# ================================================================================
# 
# All ratings are contained in the file "ratings.dat" and are in the following format:
# 
# UserID::MovieID::Rating::Timestamp
# 
# - UserIDs range between 1 and 6040 
# - MovieIDs range between 1 and 3952
# - Ratings are made on a 5-star scale (whole-star ratings only)
# - Timestamp is represented in seconds since the epoch as returned by time(2)
# - Each user has at least 20 ratings
# 
# USERS FILE DESCRIPTION
# 
# ================================================================================
# 
# User information is in the file "users.dat" and is in the following format:
# 
# UserID::Gender::Age::Occupation::Zip-code
# 
# All demographic information is provided voluntarily by the users and is not checked for accuracy.  Only users who have provided some demographic information are included in this data set.
# 
# - Gender is denoted by a "M" for male and "F" for female
# - Age is chosen from the following ranges:
# 
# 	*  1:  "Under 18"
# 	* 18:  "18-24"
# 	* 25:  "25-34"
# 	* 35:  "35-44"
# 	* 45:  "45-49"
# 	* 50:  "50-55"
# 	* 56:  "56+"
# 
# - Occupation is chosen from the following choices:
# 
# 	*  0:  "other" or not specified
# 	*  1:  "academic/educator"
# 	*  2:  "artist"
# 	*  3:  "clerical/admin"
# 	*  4:  "college/grad student"
# 	*  5:  "customer service"
# 	*  6:  "doctor/health care"
# 	*  7:  "executive/managerial"
# 	*  8:  "farmer"
# 	*  9:  "homemaker"
# 	* 10:  "K-12 student"
# 	* 11:  "lawyer"
# 	* 12:  "programmer"
# 	* 13:  "retired"
# 	* 14:  "sales/marketing"
# 	* 15:  "scientist"
# 	* 16:  "self-employed"
# 	* 17:  "technician/engineer"
# 	* 18:  "tradesman/craftsman"
# 	* 19:  "unemployed"
# 	* 20:  "writer"
# 
# MOVIES FILE DESCRIPTION
# 
# ================================================================================
# 
# Movie information is in the file "movies.dat" and is in the following format:
# 
# MovieID::Title::Genres
# 
# - Titles are identical to titles provided by the IMDB (including year of release)
# - Genres are pipe-separated and are selected from the following genres:
# 
# 	* Action
# 	* Adventure
# 	* Animation
# 	* Children's
# 	* Comedy
# 	* Crime
# 	* Documentary
# 	* Drama
# 	* Fantasy
# 	* Film-Noir
# 	* Horror
# 	* Musical
# 	* Mystery
# 	* Romance
# 	* Sci-Fi
# 	* Thriller
# 	* War
# 	* Western
# 
# - Some MovieIDs do not correspond to a movie due to accidental duplicate entries and/or test entries
# - Movies are mostly entered by hand, so errors and inconsistencies may exist

# Download the database and copy it to a local directory on your machine. (./ml-1m/)

# Load the three files in the database into three `DataFrames`.

# In[58]:

import pandas as pd
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=rnames,  engine='python')
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=mnames,  engine='python')


# To work with these data, the first step is to obtain a unique structure containing all the information. To do this we can use the function `merge` of pandas. This function automatically infers which columns should be used for the `merge` based on the names that are intersecting:
# 

# In[59]:

data = pd.merge(pd.merge(ratings, users), movies)
print data[:10]


# #Exercises: 
# 
# **MovieLens database**

# 1- Filter films that have received at least 250 ratings:

# In[60]:

### Your Code HERE


# 2- Obtain the mean ratings for each movie grouped by gender that have at least 250 ratings. 

# In[61]:

### Your Code HERE


# 3- Show films more valued by women.

# In[62]:

### Your Code HERE


# 4- Now we wonder which movies are rated more differently between men and women. Which films have more different rating and are more highly valued by women? And the films preferred by men which doesn't liked women? What are the films that have generated the most discordant ratings, regardless of gender?

# In[63]:

### Your Code HERE


# 5- Calculate the average rating of each user. 

# In[64]:

### Your Code HERE


# What is the highest rated movie in average?

# In[65]:

### Your Code HERE


# 6- Define a function called  <b>top_movies</b> that given a user it returns what movies have the highest rank for this user.
# 
# def top_movies(user)
# 

# **Data from Excel**

# 7- Read data from excel files: `MunicipisCatalunya.xlsx` and `BBDD_1_OCTUBRE_WEB_SÍ.xslx`. Browse their contents and find the % of catalan municipalities that supports a ballot for the self-determination of Catalonia at 1st of October.

# In[66]:

### Your Code HERE


# ** Data from CSV**

# 8- Read data from csv file: `ma-ba.csv`. Count the number of times `Barça` wins `Madrid` and compute the stadistics of % win, % lose and % draw.

# In[ ]:

### Your Code HERE


# #Further Reading
# Pandas has much more functionalities. Check out the (very readable) pandas docs if you want to learn more:
# 
# http://pandas.pydata.org/pandas-docs/stable/
