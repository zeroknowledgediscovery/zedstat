import pandas as pd
import numpy as np


def getCHroc(rf0):
    '''
    get convex hull of roc curve
    '''   
    from scipy.spatial import ConvexHull 
    rf=rf0.set_index('fpr')#.drop('threshold',axis=1)
    pts=rf.reset_index().values
    if len(pts)<3:
        rf_=rf.reset_index()
        rf_.columns=['tpr','fpr']
        rf_=rf_.sort_values('fpr')
        rf_=rf_.sort_values('tpr')
        return rf_
    
    hull = ConvexHull(pts)
    rf_=pd.DataFrame(pts[hull.vertices, 0], pts[hull.vertices, 1]).reset_index()
    rf_.columns=['tpr','fpr']
    rf_=rf_.sort_values('fpr')
    rf_=rf_.sort_values('tpr')
    
    #rf_['threshold'] = rf0.threshold
    return rf_

def smoothAndSample(df,VAR='fpr',STEP=0.0001):
    DF=pd.concat([pd.DataFrame(df[df[VAR].between(i,i+STEP)].max()).transpose()
              for i in np.arange(0,1,STEP)]).set_index(VAR)
    #DF.loc[0]=1
    #DF.loc[1]=0
    DF=DF.sort_index()
    return DF.dropna()

def correctPPV(df):
    arr=df.ppv.values
    a_=[]
    for i in np.arange(len(arr)-1):
        if arr[i+1]>arr[i]:
            a_.append(1.05*arr[i+1])
        else:
            a_.append(arr[i])
    a_.append(arr[-1])
    df.ppv=a_
    return df

def cb_func(p,n,n_pos,alpha=.1):
    import scipy.stats as stats
    z=stats. norm. ppf(1 - (alpha/2))
    P=p.copy()
    P['fpr']=z*np.sqrt((p.fpr*(1-p.fpr))/n)
    P['tpr']=z*np.sqrt((p.tpr*(1-p.tpr))/n_pos)
    return P

def getDF_UL(df_,chg,DIR='U'):
    df__=df_.copy()
    if DIR=='U':
        df__.tpr=df__.tpr+chg.tpr
        df__.fpr=df__.fpr-chg.fpr
    if DIR=='L':
        df__.tpr=df__.tpr-chg.tpr
        df__.fpr=df__.fpr+chg.fpr
    df__[df__ < 0] = 0
    df__[df__ > 1.0] = 1.0
    return df__

def getDF(df_,p):
    df__=df_.copy()
    df__['ppv']=1/(1+((df__.fpr/df__.tpr)*((1/p)-1)))
    df__['acc']=p*df__.tpr + (1-p)*(1-df__.fpr)
    df__['npv']=1/(1+((1-df__.tpr)/(1-df__.fpr))*(1/((1/p)-1)))
    df__['LR+']=(df__.tpr)/(df__.fpr)
    df__['LR-']=(1-df__.tpr)/(1-df__.fpr)
    return df__



