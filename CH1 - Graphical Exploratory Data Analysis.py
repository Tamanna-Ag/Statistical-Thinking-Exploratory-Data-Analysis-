#!/usr/bin/env python
# coding: utf-8

# CH1 - Graphical Exploratory Data Analysis

# In[3]:


# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


# In[4]:


data = load_iris()
#Methods to convert this data into numpy array/pandas Datframe

                                           #Method1
df = pd.DataFrame(np.column_stack((data.data, data.target)), columns = data.feature_names+['target'])
print(df.head())


# In[5]:


#Method2
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df = sklearn_to_df(data)
print(df.head())
print(df.tail())


# In[34]:


versicolor_petal_length= df.loc[50:99,['petal length (cm)']]
setosa_petal_length= df.loc[0:49,['petal length (cm)']]
virginica_petal_length= df.loc[100:149,['petal length (cm)']]

print(versicolor_petal_length)


# In[36]:


versicolor_petal_length =versicolor_petal_length.reset_index()
setosa_petal_length =setosa_petal_length.reset_index()
virginica_petal_length =virginica_petal_length.reset_index()


# In[32]:


versicolor_petal_length


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Set default Seaborn style
sns.set()

#The "square root rule" is a commonly-used rule of thumb for choosing number of bins: 
#choose the number of bins to be the square root of the number of samples.

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
plt.hist(versicolor_petal_length['petal length (cm)'] ,bins= n_bins, )

# Label axes
plt.xlabel('petal length (cm)')
plt.ylabel('count')

# Show histogram
plt.show();


# In[27]:


# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x='target',y='petal length (cm)', data=df)

# Label the axes
plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()


# In[30]:


#plotting all of our data using Empirical Cumulative distribution Functions
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# In[33]:


# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length['petal length (cm)'])

# Generate plot
plt.plot(x_vers,y_vers,marker='.',linestyle='none')

# Label the axes
plt.ylabel('ECDF')
plt.xlabel('petal length (cm)')

# Display the plot
plt.show()


# In[38]:


# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length['petal length (cm)'])
x_vers, y_vers = ecdf(versicolor_petal_length['petal length (cm)'])
x_virg, y_virg = ecdf(virginica_petal_length['petal length (cm)'])


# Plot all ECDFs on the same plot
plt.plot(x_set,y_set,marker='.',linestyle='none')
plt.plot(x_vers,y_vers,marker='.',linestyle='none')
plt.plot(x_virg,y_virg,marker='.',linestyle='none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


# CH2 - Quantittative Exploratory Data Analysis

# In[39]:


# Compute the mean: mean_length_vers
mean_length_vers=np.mean(versicolor_petal_length['petal length (cm)'])


# Print the result with some nice formatting
print('versicolor mean length:', mean_length_vers, 'cm')


# In[40]:


# Specify array of percentiles: percentiles
percentiles =np.array([2.5,25,50,75,97.5])

# Compute percentiles: ptiles_vers
ptiles_vers =np.percentile(versicolor_petal_length['petal length (cm)'], percentiles)

# Print the result
print(ptiles_vers)


# In[41]:


# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',linestyle='none')

# Show the plot
plt.show()


# In[42]:


# Create box plot with Seaborn's default settings
sns.boxplot(x='target',y='petal length (cm)',data=df)

# Label the axes
plt.xlabel('target')
plt.ylabel('petal length (cm)')


# Show the plot
plt.show()


# In[44]:


# Array of differences to mean: differences
differences = versicolor_petal_length['petal length (cm)'] - np.mean(versicolor_petal_length['petal length (cm)'])

# Square the differences: diff_sq
diff_sq= differences**2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np =np.var(versicolor_petal_length['petal length (cm)'])

# Print the results
print(variance_explicit, variance_np)

# Print the square root of the variance
print(np.sqrt(variance_np))

# Print the standard deviation
print(np.std(versicolor_petal_length['petal length (cm)']))


# In[45]:


versicolor_petal_width = df.loc[50:99,'petal width (cm)']
versicolor_petal_width =versicolor_petal_width.reset_index()

# Make a scatter plot
_ = plt.plot(versicolor_petal_length['petal length (cm)'], versicolor_petal_width['petal width (cm)'], marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('petal width(cm)')

# Show the result
plt.show()


# In[46]:


# Compute the covariance matrix: covariance_matrix
covariance_matrix= np.cov(versicolor_petal_length['petal length (cm)'],versicolor_petal_width['petal width (cm)'])

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]

# Print the length/width covariance
print(petal_cov)


# In[47]:


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)


    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r=pearson_r(versicolor_petal_length['petal length (cm)'],versicolor_petal_width['petal width (cm)'])

# Print the result
print(r)


# In[ ]:




