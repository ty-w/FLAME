# coding: utf-8

# In[46]:

import numpy as np
import pandas as pd
#import pyodbc
import pickle
import time
import itertools
from joblib import Parallel, delayed

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from operator import itemgetter

import operator
from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from sqlalchemy import create_engine

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[85]:

def data_generation_gradual_decrease(num_control, num_treated, num_cov, exponential = True):
    
    # a data generation function, not used here
    
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov))   # data for conum_treatedrol group
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov))   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.05, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.05, size=num_treated)    # some noise
    
    #dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    if exponential:
        dense_bs = [ 5.*(1./2**(i+1)) for i in range(num_cov) ]
    else:
        dense_bs = [ (5./(i+1)) for i in range(num_cov) ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for control group 
    
    yt = np.dot(xt, np.array(dense_bs)) + 10 #+ errors2    # y for treated group 
        
    df1 = pd.DataFrame(np.hstack([xc]), 
                       columns=range(num_cov))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt]), 
                       columns=range(num_cov ) ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs


# In[49]:

def data_generation_gradual_decrease_imbalance(num_control, num_treated, num_cov):
    
    # a data generation function, not used here
    
    xcs = []
    xts = []
    
    for i in np.linspace(0.1, 0.4, num_cov):
        xcs.append(np.random.binomial(1, i, size=num_control))   # data for conum_treatedrol group
        xts.append(np.random.binomial(1, 1.-i, size=num_treated))   # data for treatmenum_treated group
        
    xc = np.vstack(xcs).T
    xt = np.vstack(xts).T
    
    errors1 = np.random.normal(0, 1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 1, size=num_treated)    # some noise
    
    #dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    dense_bs = [ (1./2)**i for i in range(num_cov) ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    
    yt = np.dot(xt, np.array(dense_bs)) + 10 #+ errors2    # y for treated group 
        
    df1 = pd.DataFrame(np.hstack([xc]), 
                       columns = range(num_cov))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt]), 
                       columns = range(num_cov ) ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs


# In[50]:

def construct_sec_order(arr):
    
    # an intermediate data generation function used for generating second order information
    
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)


# In[51]:

def data_generation_dense_2(num_control, num_treated, num_cov_dense, num_covs_unimportant, 
                            control_m = 0.1, treated_m = 0.9):
    
    # the data generating function that we will use. include second order information
    
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov_dense))   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    #dense_bs = [ np.random.normal(dense_bs_sign[i]* (i+2), 1) for i in range(len(dense_bs_sign)) ]
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec #+ errors2    # y for treated group 

    xc2 = np.random.binomial(1, control_m, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    xt2 = np.random.binomial(1, treated_m, size=(num_treated, num_covs_unimportant))   # unimportant covariates for treated group
        
    df1 = pd.DataFrame(np.hstack([xc, xc2]), 
                       columns = range(num_cov_dense + num_covs_unimportant))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt, xt2]), 
                       columns = range(num_cov_dense + num_covs_unimportant ) ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs, treatment_eff_coef


# In[52]:

def data_generation(num_control, num_treated, num_cov, control_m = 0.3, treated_m = 0.7):
    
    # a data generation function. not used
    
    x1 = np.random.binomial(1, control_m, size=(num_control, num_cov) )   # data for conum_treatedrol group
    x2 = np.random.binomial(1, treated_m, size=(num_treated, num_cov) )   # data for treatmenum_treated group

    errors1 = np.random.normal(0, 0.005, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.005, size=num_treated)    # some noise

    mus = [] 
    sigmas = [] 
    bs = [] 
    for i in range(num_cov):     
        mus.append(i)
        sigmas.append(1./(i**2+1))            # generating weights of covariates for the outcomes

    bs = [np.random.normal(mus[i], sigmas[i]) for i in range(len(sigmas))]  # bs are the weights for the covariates for generating ys

    y1 = np.dot(x1, np.array(bs)) + errors1     # y for control group 
    y2 = np.dot(x2, np.array(bs)) + 1 + errors2    # y for treated group 

    df1 = pd.DataFrame(x1, columns=[i for i in range(num_cov)])
    df1['outcome'] = y1
    df1['treated'] = 0

    df2 = pd.DataFrame(x2, columns=[i for i in range(num_cov)] ) 
    df2['outcome'] = y2
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df