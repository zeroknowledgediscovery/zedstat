import pandas as pd
import numpy as np
from scipy.stats import norm

def rrci(epos,eneg,cpos,cneg,alpha=.05,ZERO_POLICY='ADDONE'):
    '''Get CI bounds for risk ratio
    
    Args:
        epos (int): exposed group with outcome
        eneg (int): exposed group without oucome
        cpos (int): control group with outcome
        cneg (int): control group without oucome
        alpha (float): significance level float
        ZERO_POLICY (str): How to handle zeropositives (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8196730/)

    Returns:
        float: risk ratio
        float: lower confidence bound
        float: wpper confidence bound
    '''
    import pandas as pd  
    import statsmodels.api as sm


    if ZERO_POLICY=="ADDONE":
        if epos == 0:
            epos=epos+1
            eneg=eneg+1
            cneg=cneg+1
            cpos=cpos+1

    if epos > 0:
        rr = (epos/(epos+eneg))/(cpos/(cpos+cneg))
    else:
        rr = (eneg/(epos+eneg))/(cneg/(cpos+cneg))


    
    lrr = np.log(rr)
    
    if epos>0:
        V=np.sqrt(1/(epos+eneg) + 1/(cpos+cneg) + 1/epos + 1/cpos)
    else:
        V=np.sqrt(1/(epos+eneg) + 1/(cpos+cneg) + 1/eneg + 1/cneg )

    
    n_sided=1
    z=norm.ppf(1-alpha/n_sided)
    del_ = z*V

    lrrlb = (lrr-del_)
    lrrub = lrr+del_

    

    return  rr, np.exp(lrrlb), np.exp(lrrub)
