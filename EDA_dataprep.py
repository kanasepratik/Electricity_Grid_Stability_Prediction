#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sweetviz
from dataprep.eda import plot, plot_correlation, plot_missing

class EDA_dataprep:
    def univariatre_multiple_plots(data):
        return(plot(data)) 
    def univariate_categorical(data,feature):
        return(plot(data,feature))
    def univarite_numerical(data,feature):
        return(plot(data,feature, bins=20))
    def bivariate_numerical_scatterplot(data,feature1,feature2):
        return(plot_correlation(data, x=feature1, y=feature2, k=5))
    def correlation_plot(data):
        return(plot_correlation(data))
    def missing_data_analysis(data):
        return(plot_missing(data))


# In[12]:




