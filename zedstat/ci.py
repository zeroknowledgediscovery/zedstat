import pandas as pd
import numpy as np

def ci(df,xvar,yvar,alpha=.05):
    '''Get CI bounds for OLS regresssion
    
    Args:
        df (pandas.DataFrame): input dataframe
        xvar (str): x variable name
        yvar (str): y variable name
        alpha (float): significance level float

    Returns:
        pandas.DataFrame: x, pred,ub,lb
        float: pvalue (f_pvalue of OLS estimator)
        float: aic (aic of OLS estimator)
        float: correlation (numpy corrcoeff with predicted values)
        statsmodels.regression.linear_model.OLSResults
    '''
    import pandas as pd  
    import statsmodels.api as sm
    
    df = df.sort_values(xvar)
    X = sm.add_constant(df[xvar].values)
    ols_model = sm.OLS(df[yvar].values, X)
    est = ols_model.fit()
    out = est.conf_int(alpha=alpha, cols=None)
    y_pred = est.predict(X)
    x_pred = df[xvar].values
    pred = est.get_prediction(X).summary_frame()
    predf=pd.DataFrame({'pred': est.params[0]+x_pred*est.params[1],
                        'lb':pred['mean_ci_lower'].values,
                        'ub':pred['mean_ci_upper'].values},
                       index=x_pred)
    corr=np.corrcoef(predf.pred,df[yvar])[0,1]
    return predf,est.f_pvalue,est.aic,corr,est
