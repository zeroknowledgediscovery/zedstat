import scipy as sp
import numpy as np

def calc_auc(y_true, y_score,cb=0.99):
    '''
    Calculate AUC and confidence bounds / pvalue 
    on AUC using U test correspondance using y_true and y_score
    '''    
    ZALPHA={0.9:1.645,0.95:1.96,.99:2.58,.999:3.27}
    n1 = np.sum(y_true==1)
    n0 = len(y_score)-n1
    order = np.argsort(y_score)
    rank = np.argsort(order)
    rank += 1
    U1 = np.sum(rank[y_true == 1]) - n1*(n1+1)/2
    U0 = np.sum(rank[y_true == 0]) - n0*(n0+1)/2
    AUC1 = U1/ (n1*n0)
    AUC0 = U0/ (n1*n0)
    
    EU1=n0*n1*0.5
    s1=np.sqrt(n0*n1*(n0+n1+1)/12.)
    U1_z= (U1-EU1)/s1
    p = sp.stats.norm.sf(abs(U1_z))*2 #twosided
    
    CF=(ZALPHA[cb]*s1)/(n1*n0)
    
    if AUC1>AUC0:
        return AUC1, p,U1,U1_z,CF
    
    return AUC0, p0,U0,U0_z,CF  
