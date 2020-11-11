import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Graphical_analysis:
    class Numerical_data_analysis: 
        def __init__(self,data):
            self.data=data
        def bivariate_analysis_scatterplot(self,numerical_feature_y):
            for feature in self.data:
                ax=sns.jointplot(x=feature,y=numerical_feature_y,data=self.data,kind='scatter')
                plt.xlabel(feature)
                plt.ylabel(numerical_feature_y)
                plt.title('Bivariate_Analysis')
            
        def univariate_analysis_histogram(self):
            plt.style.use('seaborn')
            self.data.hist(bins=20, figsize=(20,10))
            plt.show()
            
        def bivariate_analysis_scatterplot(self,numerical_feature_y):
            for feature in self.data:
                ax=sns.jointplot(x=feature,y=numerical_feature_y,data=self.data,kind='scatter')
                plt.xlabel(feature)
                plt.ylabel(numerical_feature_y)
                plt.title('Bivariate_Analysis')
                
        def correlation_plot(self):
            plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
            p=sns.heatmap(self.data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
            
                
                
                              
    class Categorical_data_analysis:
        def __init__(self,data):
            self.data=data
        def univariate_analysis_categorical_countplot(self,label):
            plt.figure(figsize=(60,40))
            ax=sns.countplot(y=self.data[label],data=self.data,order = self.data[label].value_counts().index)
            ax.axes.set_title('Bivariate Analysis',fontsize=40)
            ax.set_ylabel(label,fontsize=60)
            ax.set_xlabel('count',fontsize=60)
            ax.tick_params(labelsize=60)
            total = len(self.data[label])
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_width()/total)
                x = p.get_x() + p.get_width() + 0.02
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y),fontsize=60)
            print('Percentage of datapoints present in class : \n\n',(self.data[label].value_counts()/self.data[label].count())*100)   
            plt.show()
            
        def bivarate_analysis_barplot(self,categorical_feature,numerical_feature,hue):
            plt.figure(figsize=(30,20))
            result = self.data.groupby([categorical_feature])[numerical_feature].aggregate(np.median).reset_index().sort_values(numerical_feature)
            b=sns.barplot(y=self.data[categorical_feature],x=self.data[numerical_feature],data=self.data,hue=hue,order=result[categorical_feature])
            b.axes.set_title('Bivariate Analysis',fontsize=40)
            b.set_xlabel(numerical_feature,fontsize=40)
            b.set_ylabel(categorical_feature,fontsize=40)
            b.tick_params(labelsize=30)
            plt.show()
            
            
        def bivariate_analysis_boxplot(self,categorical_feature,numerical_feature):
            sns.set(context='notebook', style='darkgrid',palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
            ax=sns.boxplot(y=self.data[numerical_feature],x=self.data[categorical_feature],data=self.data) 
            plt.ylabel(numerical_feature)
            plt.xlabel(categorical_feature)
            plt.title(categorical_feature)  
            plt.show()
            
            
