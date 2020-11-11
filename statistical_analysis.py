import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2



class Statistical_analysis:  

    class Significance_tests:

        def __init__(self,data):
            self.data=data
            
        def one_way_anova(self,formula):

            model = ols(formula,data=self.data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print('One way ANOVO test results: \nANOVA table: \n%s'%anova_table)
            p_value=anova_table['PR(>F)'][0]
            alpha=0.05
            print('alpha: %s \np_value: %s '%(alpha,p_value))
            if p_value <= alpha:
                print('Result: \np_value < %s \nnull hypothesis is true : \nall the sample means are equal or the factor did not have any significant effect on the results, reject the alternate hypothesis \n\n'%alpha) 
            else:
                print('Result: \np_value > %s \nalternate hypothesis is true :\nleast one of the sample means is different from another,reject the null hypothesis and conclude that not all of population means are equal \n\n'%alpha)
        
        def two_way_anova(self,formula,feature):
  
            model = ols(formula,data=self.data).fit()
            # Perform ANOVA and print table
            anova_table = sm.stats.anova_lm(model, typ=2) 
            print('Two way ANOVO test results: \nANOVA table: \n%s'%anova_table)
            p_value=anova_table['PR(>F)'][feature]
            alpha=0.05
            print('\nTest Results for feature: %s'%feature)
            print('alpha: %s \np_value for feature: %s: %s '%(alpha,feature,p_value))
            if p_value <= alpha:
                print('p_value < %s \nnull hypothesis is true : \nall the sample means are equal or the factor did not have any significant effect on the results, reject the alternate hypothesis \n\n\n'%alpha) 
            else:
                print('p_value > %s \nalternate hypothesis is true :\nleast one of the sample means is different from another,reject the null hypothesis and conclude that not all of population means are equal \n\n\n'%alpha)
                
        def chi_square_test(self,categorical_feature1,categorical_feature2):
     
            dataset_table=pd.crosstab(self.data[categorical_feature1],self.data[categorical_feature2])
            Observed_Values = dataset_table.values
            val=stats.chi2_contingency(dataset_table)
            Expected_Values=val[3]
            no_of_rows=dataset_table.shape[0]
            no_of_columns=dataset_table.shape[1]
            ddof=(no_of_rows-1)*(no_of_columns-1)
            alpha = 0.05
            chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
            chi_square_statistic=chi_square[0]+chi_square[1]
            critical_value=chi2.ppf(q=1-alpha,df=ddof)
            p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
        
            print('Chi square test between features: ',categorical_feature1,' and ',categorical_feature2)
            print("chi-square statistic:-",chi_square_statistic)
            print('critical_value:',critical_value)
            print('Significance level: ',alpha)
            print('Degree of Freedom: ',ddof)
            print('p-value:',p_value)
            #if chi_square_statistic>=critical_value:
            #print("Reject H0,There is a relationship between 2 categorical variables")
            #else:
            #print("Retain H0,There is no relationship between 2 categorical variables")
            if p_value<=alpha:
                print("p_value < %s \nTest Results  Reject H0,There is a relationship between 2 categorical variables \n\n"%(alpha))
            else:
                print("p_value > %s \nTest Results : Retain H0,There is no relationship between 2 categorical variables \n\n"%(alpha))
    
    class Normality_test:
      
        def __init__(self,data):
            self.data=data
        
        def shapiro_wilk_test(self,feature):
      
            stat,p=stats.shapiro(self.data[feature])
            print('Test Results: \nShapiro wilk normality test for feature: %s'%feature)
            alpha = 0.05
            print('alpha :',alpha,'\ntest_statistics: ',stat,'\np_value: ',p)
            if p > alpha:
                print('p_value > %s \nSample looks Gaussian (fail to reject H0) \n\n'%alpha)
            else:
                print('p_value < %s \nSample does not look Gaussian (reject H0) \n\n'%alpha)
                
        def kolmogorov_smirnov_test(self,feature):
           
            stat,p=stats.kstest(self.data[feature],cdf='norm')
            print('Test Results: \nkolmogorov_smirnov normality test for feature: %s'%feature)
            alpha = 0.05
            print('alpha :',alpha,'\ntest_statistics: ',stat,'\np_value: ',p)
            if p > alpha:
                print('p_value > %s \nSample looks Gaussian (fail to reject H0) \n\n'%alpha)
            else:
                print('p_value < %s \nSample does not look Gaussian (reject H0) \n\n'%alpha)
            
        def anderson_darling_test(self,feature):
           
            result=stats.anderson(self.data[feature],dist='norm')
            print('Test Results: \nanderson_darling_test for feature: %s \n'%feature)
            print('Test Statistic: %s\n'%result.statistic)
            p = 0    
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic < result.critical_values[i]:
                    print('for significance_level: %s \ncrital_values: %s \nResult: data looks normal (fail to reject H0) \n' % (sl, cv))
                else:
                    print('for significance_level: %s \ncrital_values: %s \nResult: data does not look normal (reject H0) \n' % (sl, cv))
    
    

    