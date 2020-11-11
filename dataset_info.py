import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Dataset_inspection:
    
    def __init__(self,data):
        self.data=data
        
    def dataset_info(self):
        print('\nShape of Dataset',self.data.shape)
        print('\nNumber of Rows',self.data.shape[0],'\nNumber of Columns: ',self.data.shape[1])
        print('\nFeature Names : \n',self.data.columns.values)
        print('\nInformation about Datatypes: ')
        print('\n%s'%self.data.info())
        print('\nUnique values per column: \n%s'%self.data.nunique())
        print('\nAny Missing Values in data?: %s'%self.data.isnull().values.any())
        return(self.data.profile_report(minimal=True))
    
    
    def missing_data_analysis(self):
        print('Any missing datapoints in dataset:',self.data.isnull().values.any())
                     
        if self.data.isnull().values.any()==True:
            print('Columnwise missing data present in the dataset')
            missing_data=pd.DataFrame({'total_missing_count':self.data.isnull().sum(),
                                       'percentage_missing':self.data.isnull().sum()/self.data.shape[0]*100,
                                       'datatype':self.data.dtypes})

            print(missing_data[missing_data.total_missing_count>0])
            sns.heatmap(self.data.isnull().values)
            
            
            #Counting cells with missing values:(Total number of NA)
            a=sum(self.data.isnull().values.ravel())
            #Getting total number of cells
            b=np.prod(self.data.shape)
            #Getting percentage of NA in overall data
            print('\n','\n','Total percentage of missing data :',(a/b)*100,' % \n')


            #Calculating Rows affected by NA- Rows having na/ Total number of rows
            #Counting rows that have missing values somewhere:
            a=sum([True for idx,row in self.data.iterrows() if any(row.isnull())])
            #Total Number of rows in data
            b=self.data.shape[0]
            print('\n','Total percentage of rows affected by missing data :',(a/b)*100,'% \n')    
            
        else:
            print('There is no missing datapoints in dataset')
            sns.heatmap(self.data.isnull().values)
            
class Outlier_analysis:
        
    def __init__(self,data):
        self.data=data
        
    def graphical_outlier_analysis(self):
        for col in self.data:
            print('Name of Feature :',col)
            print('Skewness of Feature :',self.data[col].skew())
            plt.figure(figsize=(20,5))
            plt.subplot(1,3,1)
            sns.distplot(self.data[col],bins=20,kde=False)
            plt.subplot(1,3,2)
            stats.probplot(self.data[col],dist='norm',plot=plt)
            plt.subplot(1,3,3)
            sns.boxplot(y=self.data[col])
            plt.show()        
            #calculating the outliers with IQR
        Feature_Name=[]
        Outliers_Number=[]
        Percentage_of_Outliers=[]
        Skewness=[]
            
                
        for col in self.data:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
                
            Outliers=((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            Outliers_Number.append(Outliers)
            Percentage=(Outliers*100)/len(self.data[col])
            Percentage_of_Outliers.append(Percentage)
            Skewness.append(self.data[col].skew())
            Feature_Name.append(col)
            
    
        Outlier_Data=pd.DataFrame({'Feature_name':Feature_Name,'Outliers_Number':Outliers_Number,'Percentage_of_Outliers':Percentage_of_Outliers,
                                  'Skewness':Skewness})

        print('\nSkewness Table: \n%s \n\n\n'%Outlier_Data)


        
