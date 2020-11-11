#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas_profiling import ProfileReport
import pandas_profiling
import sweetviz
from dataprep.eda import plot, plot_correlation, plot_missing

class EDA_reports:
    def pandas_profile_report(data):
        #profile = ProfileReport(data, title='Pandas Profiling Report')
        #profile.to_file("output.html")
        return(data.profile_report())
        
    def sweetviz_report(data,target_feature):
        my_report = sweetviz.analyze(data)
        return(my_report.show_html('Report.html'))

