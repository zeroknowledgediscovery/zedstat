import pandas as pd
import numpy as np
from sklearn.metrics import auc
from scipy.interpolate import UnivariateSpline

class processRoc(object):
    """process ROC datafile"""

    def __init__(self,
                 df=None,
                 fprcol='fpr',
                 tprcol='tpr',
                 thresholdcol='threshold',
                 prevalence=None,
                 order=2,
                 total_samples=None,
                 positive_samples=None,
                 alpha=0.05):

        """Initialization

        Args:
            df (pandas.DataFrame): dataframe with columns tabulating fpr, tpr, and optionally threshold values
            fprcol (str): string name of fpr column
            tprcol (str): string name of tpr column
            thresholdcol (str): string name of threshold column
            prevalence (float): prevalence of positive cases in population (need not be the data ratio)
            order (int): order of polynomial/spline for smoothing
            total_samples (int): total number of samples in the original data
            positive samples (int): number of positive cases in the original data
            alpha (float): significance level e.g. 0.05
        """
        
        #print('local version 1')
        
        if df.index.name==fprcol:
            self.df=df.copy()
        else:
            if fprcol in df.columns:
                self.df=df.set_index(fprcol).copy()
            else:
                raise('fpr not in columns or index')
        self.thresholdcol = thresholdcol
            
        if thresholdcol not in self.df.columns:
            self.thresholdcol=None

        self.df = self.df.sort_values('fpr')
            
        self.fprcol = fprcol
        self.tprcol = tprcol
        self.raw_df = self.df.copy()
        self.prevalence = prevalence
        self.order = order
        self.df_lim = {}
        self._auc = {'U':[],'L':[]}
        self.total_samples = total_samples
        self.positive_samples = positive_samples
        self.alpha = alpha
        
        self.df=self.df.groupby(self.fprcol).max().reset_index()

        
    def get(self):
        '''
        return dataframe currently in class

        Returns:
            pandas.DataFrame 
        '''
        return self.df.copy()
    

    def nominal_auc(self):
        '''
        calculate nominal auc
        '''
        from sklearn.metrics import auc
        self._auc['nominal']=auc(self.df.index.values,self.df.tpr.values)
        return

    
    def auc(self,
            total_samples=None,
            positive_samples=None,
            alpha=None):
        '''
        calculate auc with confidence bounds. As default, the arguments are read from class initializtion.

        Args:
            total_samples (int): total number fo samples, default None
            positive_samples (int): number fo positive samples, default None
            alpha (float): significance level, default None

        Returns:
            float: nominal auc
            float: upper bound
            float: lower bound
        
        '''
        self.nominal_auc()
        self._auc['U']=[]
        self._auc['L']=[]
        
        self.getBounds(total_samples=total_samples,
                positive_samples=positive_samples,
                alpha=alpha)
        self.__auc_cb2(total_samples=total_samples,
                positive_samples=positive_samples,
                alpha=alpha)
        return self._auc['nominal'], self._auc['U'].min(), self._auc['L'].max()

    
    def __convexify(self):
        '''
        compute convex hull of the roc curve
        '''
        #print(self.df)
        from scipy.spatial import ConvexHull
        if self.df.index.name==self.fprcol:
            rf=self.df
        else:
            if self.fprcol in self.df.columns:
                rf=self.df.set_index(self.fprcol)#.drop('threshold',axis=1)
            else:
                raise('fpr not in columns or index')

        rf = rf.reset_index()
        rf=pd.concat([rf,pd.DataFrame({self.fprcol:0, self.tprcol:0},index=[0])])
        rf=pd.concat([rf,pd.DataFrame({self.fprcol:1, self.tprcol:0},index=[0])])
        rf=pd.concat([rf,pd.DataFrame({self.fprcol:1, self.tprcol:1},index=[0])])
        
        rf=rf.drop_duplicates()
        rf=rf.sort_values(self.fprcol)
        rf=rf.sort_values(self.tprcol)

        pts=rf[[self.fprcol,self.tprcol]].values

        #print(pts)
        
        if len(pts)<3:
            rf_=rf.copy()
            rf_.columns=[self.tprcol,self.fprcol]
            rf_=rf_.sort_values(self.fprcol)
            rf_=rf_.sort_values(self.tprcol)
            self.df=rf_.set_index(self.fprcol).copy()
            return #self.df
        
        hull = ConvexHull(pts)
        rf_=pd.DataFrame(pts[hull.vertices, 0], pts[hull.vertices, 1]).reset_index()
        rf_.columns=[self.tprcol,self.fprcol]
        rf_ = rf_.set_index(self.fprcol)
        
        rf_ = rf_.drop(1.0).sort_index()
        rf_.loc[1.0]=1.0
        
        self.df=rf_.copy()
        return 


    def smooth(self,
               STEP=0.0001,
               interpolate=True,
               convexify=True):
        '''
        smooth roc curves and update processRoc.df which is accessible using processRoc.get()

        Args:
            STEP (float): smooting step, default 0.0001
            interpolate (bool): if True, interpolate missing values, default True
            convexify (bool): if True, replace ROC with convex hull, default True
        '''
        self.df=self.raw_df.copy()
        VAR=self.fprcol
        df_=self.df.reset_index()
        DF=pd.concat([pd.DataFrame(
            df_[df_[VAR].between(i,i+STEP)].max()).transpose()
                      for i in np.arange(0,1,STEP)]).set_index(VAR)
        DF=DF.dropna()
        DF.loc[0]=pd.Series([],dtype=float) 
        DF.loc[1]=pd.Series([],dtype=float) 
        DF.loc[0,'tpr']=0
        DF.loc[1,'tpr']=1
        
        DF=DF.sort_index()
        if interpolate:
            DF=DF.interpolate(limit_direction='both',method='spline',order=self.order)
            DF[DF < 0] = 0
            self.df=DF
        if convexify:
            self.__convexify()
        return 

    
    def allmeasures(self,prevalence=None,interpolate=False):
        '''
        compute accuracy, PPV, NPV, positive and negative likelihood ratios, and update processRoc.df, which can be accessed using processRoc.get()

        Args:
            prevalence (float): prevalence od positive cases in population
            interpolate (bool): if True interpolate missing values, default False
        '''
        if prevalence is not None:
            p=prevalence
            self.prevalence=p
        else:
            p=self.prevalence
        if p is None:
            raise('prevalence undefined')
        df__=self.df.copy()
        if self.fprcol == df__.index.name:
            df__=df__.reset_index()
            df__['ppv']=1/(1+((df__.fpr/df__.tpr)*((1/p)-1)))
            df__['acc']=p*df__.tpr + (1-p)*(1-df__.fpr)
            df__['npv']=1/(1+((1-df__.tpr)/(1-df__.fpr))*(1/((1/p)-1)))
            df__['LR+']=(df__.tpr)/(df__.fpr)
            df__['LR-']=(1-df__.tpr)/(1-df__.fpr)
        else:
            assert('set fpr as index')
            
            
        df__=df__.set_index(self.fprcol)

        #display(df__)
        df__=df__.replace(np.inf,np.nan)
        #display(df__)


        
        if interpolate:
            try:
                df__=df__.interpolate(limit_direction='both',method='spline',order=self.order)
            except:
                print('interpolation failed')

 
        #display(df__)
        
        if self.thresholdcol is not None:
            if self.thresholdcol not in df__.columns:
                df__=df__.join(self.raw_df[self.thresholdcol])
        self.df=df__
        self.__correctPPV()
        self.df[self.df < 0] = 0


        self.df=self.df.reset_index().groupby('fpr').max()
        self.df.ppv=UnivariateSpline(self.df.index.values,self.df.ppv.values,k=1,s=None)(self.df.index.values)
        self.df[self.df < 0] = 0

        #display(self.df)
        
        return #self.df


    def __correctPPV(self,df=None):
        '''
        make ppv monotonic
        '''
        if 'ppv' not in self.df.columns:
            return

        if df is None:
            df__=self.df
        else:
            df__=df.copy()
            
        
        #df__.loc[0].ppv=1.0
        #df__.loc[1].ppv=0.0
        arr=df__.ppv.values
        a_=[]
        for i in np.arange(len(arr)-1):
            if arr[i+1]>arr[i]:
                a_.append(1.05*arr[i+1])
            else:
                a_.append(arr[i])
        a_.append(arr[-1])

        
        df__.ppv=a_
        #df__.loc[0].ppv=1.0
        #df__.loc[1].ppv=0.0
        
        
        return df__


    def scoretoprobability(self, score, regen=True, **kwargs):
        '''
        Map computed score to probability of sample being in the positive class.
        This is simply the PPV corresponding to the threshold which equals the score.
        Now supports both single scores and lists/numpy arrays of scores.

        Args:
            score (float or list or numpy.ndarray): computed score(s)
            regen (bool): if True, regenerate roc curve
            kwargs (dict): values passed for regeneration of smoothed roc

        Return:
            float or numpy.ndarray representing probability of being in positive cohort
        '''

        if score is None:
            return None

        if regen:
            STEP = 0.01
            precision = 3
            interpolate = True

            STEP = kwargs.get('STEP', STEP)
            precision = kwargs.get('precision', precision)
            interpolate = kwargs.get('interpolate', interpolate)

            self.smooth(STEP=STEP)
            self.allmeasures(interpolate=interpolate)
            self.usample(precision=precision)

        df = self.get()
        if 'threshold' not in df.columns:
            raise ValueError('Threshold not in columns or index')
        if 'ppv' not in df.columns:
            raise ValueError('PPV not in columns or index')

        def compute_val(score):
            if score is None:
                return None
            if score > df.threshold.max():
                val = df.ppv.values.max()
            else:
                val = df[df.threshold > score].ppv.tail(1).values[0]
            return (val - df.ppv.values.min()) / (df.ppv.values.max() - df.ppv.values.min())

        if isinstance(score, (list, np.ndarray)):
            return np.array([compute_val(s) for s in score])
        else:
            return compute_val(score)

    
    def usample(self,
                df=None,
                precision=3):
        '''
        make performance measures estimated at regular intervals of false positive rate

        Args:
            df (pandas.DataFrame): dataframe woth performance values, fpr as index. default: None, when the dataframe entered at initialization is used
            precision (int): number of digits after decismal point used to sample fpr range
        
        Returns:
            pandas.DataFrame: uniformly sampled performance dataframe
        '''
        step=10**(-precision)
        fpr=[np.round(x,precision) for x in np.arange(0,1+step,step)]
        if df is None:        
            fpr_=[x for x in fpr if x not in self.df.index]
        else:
            fpr_=[x for x in fpr if x not in df.index]

        if df is None:
            df___=self.df.copy()
        else:
            df___=df.copy()
            
        for x in fpr_:
            df___.loc[x]=pd.Series([],dtype=float) 
        df___=df___.sort_index().interpolate()
        
        df___=df___.loc[fpr]        
        if df is None:
            self.df=df___

        return df___


    def __getDelta(self,
                 total_samples=None,
                 positive_samples=None,
                 alpha=None):
        '''
        confidence bounds on specificity and sensitivity using Wald-type approach
        '''
        if total_samples is None:
            total_samples=self.total_samples
        if positive_samples is None:
            positive_samples = self.positive_samples
        if alpha is None:
            alpha=self.alpha
            
        n=total_samples
        n_pos=positive_samples
        
        if self.fprcol not in self.df.columns:
            if self.fprcol != self.df.index.name:
                return
        if self.tprcol not in self.df.columns:
            return
        
        import scipy.stats as stats
        z=stats. norm. ppf(1 - (alpha/2))
        delta_=pd.DataFrame()
        df_=self.df.copy()
        if self.fprcol == df_.index.name:
            df_=df_.reset_index()
            delta_['fprdel']=z*np.sqrt((df_.fpr*(1-df_.fpr))/n)
            delta_['tprdel']=z*np.sqrt((df_.tpr*(1-df_.tpr))/n_pos)
            self.delta_=delta_
        return 

    
    def getBounds(self,
                  total_samples=None,
                  positive_samples=None,
                  alpha=None,
                  prevalence=None):
        '''
        compute confidence bounds on performance measures

        Args:
            total_samples (int): total number fo samples, default None
            positive_samples (int): number fo positive samples, default None
            alpha (float): significance level, default None
            prevalence (float): prevalence of positive cases in population, default None

        '''
        self.__getDelta(total_samples=total_samples,
                 positive_samples=positive_samples,
                      alpha=alpha)

        if prevalence is None:
            p=self.prevalence
        else:
            p=prevalence
        
        if p is None:
            raise('prevalence undefined')

        for direction in ['U','L']:
            df__=self.df.copy().reset_index()

            if direction=='U':
                df__.tpr=df__.tpr+self.delta_.tprdel
                #df__.fpr=df__.fpr-self.delta_.fprdel
            else:
                df__.tpr=df__.tpr-self.delta_.tprdel
                #df__.fpr=df__.fpr+self.delta_.fprdel
                
            df__['ppv']=1/(1+((df__.fpr/df__.tpr)*((1/p)-1)))
            df__['acc']=p*df__.tpr + (1-p)*(1-df__.fpr)
            df__['npv']=1/(1+((1-df__.tpr)/(1-df__.fpr))*(1/((1/p)-1)))
            df__['LR+']=(df__.tpr)/(df__.fpr)
            df__['LR-']=(1-df__.tpr)/(1-df__.fpr)

            df__=df__.replace(np.inf,np.nan)

            
            df__=df__.interpolate(limit_direction='both',method='spline',
                                  order=self.order).set_index(self.fprcol)
            df__[df__ < 0] = 0

            self.df_lim[direction]=self.__correctPPV(df__)
            
            if direction=='U':
                self._auc[direction]=np.array([np.append(self._auc[direction],
                                                auc(df__.index.values,
                                                    df__.tpr.values)).min()])
            if direction=='L':
                self._auc[direction]=np.array([np.append(self._auc[direction],
                                                auc(df__.index.values,
                                                    df__.tpr.values)).max()])


            # adjust datframe to cneter of upper and lowwr bounds    
        #self.df=(self.df_lim['U']+ self.df_lim['L'] )/2
        #self.__correctvalues()
        return 

    def __correctvalues(self):
        #ppv
        if 'ppv' in self.df.columns:
            self.df.ppv[self.df.ppv>1]=1.0
        if 'npv' in self.df.columns:
            self.df.npv[self.df.npv>1]=1.0
        if 'LR-' in self.df.columns:
            self.df['LR-'][self.df['LR-']>1]=1.0
            

    def __auc_cb2(self,
                total_samples=None,
                positive_samples=None,
                alpha=None):
        '''
        compute auc confidence bounds using Danzig bounds

        Args:
            total_samples (int): total number fo samples, default None
            positive_samples (int): number fo positive samples, default None
            alpha (float): significance level, default None
        '''
        if total_samples is None:
            total_samples=self.total_samples
        if positive_samples is None:
            positive_samples = self.positive_samples
        if alpha is None:
            alpha=self.alpha

        n=total_samples
        n_pos=positive_samples


        if 'nominal' not in self._auc.keys():
            assert('calculate nominal auc first')
            
        import scipy.stats as stats
        auc=self._auc['nominal']
        z=stats. norm. ppf(1 - (alpha/2))

        eta=1+(n_pos/(z*z))
        b=(auc-.5)/eta
        auc_U=auc+b+ (1/eta)*np.sqrt((auc-.5)**2 + (auc*(1-auc)*eta))
        auc_L=auc+b- (1/eta)*np.sqrt((auc-.5)**2 + (auc*(1-auc)*eta))

        self._auc['L']=np.array([np.append(self._auc['L'],auc_L).max()])
        self._auc['U']=np.array([np.append(self._auc['U'],auc_U).min()])

        return

    
    def operating_zone(self,
                       n=1,
                       LRplus=10,
                       LRminus=0.6):
        '''
        compute the end points of the operating zone, 
        one for maximizing precions, and one for maximizing sensitivity

        Args:
            n (int): number of operting points per condition returned, default 1
            LRplus (float): lower bound on positive likelihood ratio, default 10.0
            LRminus (float): upper bound on negative likelihood ratio, default 0.6

        '''
        wf=self.df.copy()
        
        opf=pd.concat([wf[(wf['LR+']>LRplus)
                          & (wf['LR-']<LRminus) ]\
                       .sort_values('ppv',ascending=False).head(n),
                       wf[(wf['LR+']>LRplus)
                          & (wf['LR-']<LRminus) ]\
                       .sort_values('tpr',ascending=False).head(n)])

        if opf.empty:
            self._operating_zone=opf.copy()
            return #self._operating_zone.copy()
        self._operating_zone=opf.reset_index()
        self._operating_zone.index=['high precision']*n + ['high sensitivity']*n
        return #self._operating_zone.copy()

    
    def samplesize(self,
                   delta_auc=0.1,
                   target_auc=None,
                   alpha=None):
        '''
        estimate sample size for atataing auc bound under given significance level

        Args:
            delta_auc (float): maximum perturbation from estimated auc, default 0.1
            target_auc (float): if None, using estimate current nominal auc
            alpha (float): significanec level. If None use processRoc.alpha

        Returns:
            float: minimum sample size
        '''
        if alpha is None:
            alpha=self.alpha

        if target_auc is None:
            if 'nominal' not in self._auc.keys():
                self.auc()
            target_auc=self._auc['nominal']
            
        import scipy.stats as stats
        z=stats. norm. ppf(1 - (alpha/2))
        required_npos = (z*z)*target_auc*(1-target_auc)/(delta_auc*delta_auc)

        return required_npos

    def pvalue(self,
               delta_auc=0.1,
               twosided=True):
        '''
        compute p-value for given auc bounds

        Args:
            delta_auc (float): maximum perturbation from estimated auc, default 0.1
            twosided (bool): one sided or twosided confidence bounds

        Returns:
            float: pvalue for the null hypothesis that estimated nominal auc is lower by more than delta_auc
        '''
        if 'nominal' not in self._auc.keys():
            self.auc()
        auc=self._auc['nominal']
            
        import scipy.stats as stats
        z=np.sqrt(self.positive_samples/(auc*(1-auc)/(delta_auc*delta_auc)) )
        pvalue=stats.norm.sf(abs(z))

        if twosided:
            pvalue=2*pvalue
        return pvalue
    
    
    def interpret(self,
                  fpr=0.01,
                  number_of_positives=10,
                  five_yr_survival=None,
                  factor=1):
        '''
        generate simple interpretation of inferred model, based on a number of positive cases

        Args:
            fpr (float): the false psotive rate or 1-specificity of the operating point
            number_of_positives (int): interpret assuming this many positive cases, default 10
            five_yr_survival (float): fraction not experiencing severe event after 5 years (default: None)
            factor (float): fraction of TP who avert the severe outcome due to correct screen
        '''
        wf=self.df.copy()
                
        wf.loc[fpr]=pd.Series([],dtype=float)
        wf=wf.sort_index().interpolate(method='spline',order=self.order,limit_direction='both')
        
        row=wf.loc[fpr]

#        POS=number_of_positives
#        TP=POS*row.tpr
#        FP = TP*((1/row.ppv) -1)
#        NEG=FP/fpr
#        TOTALFLAGS=TP+FP
#        FN=POS-TP
#        TN=POS/self.prevalence

        #factor=0.21*(0.95-0.69) + 0.08*(0.95-0.17)
        #factor=0.33*(0.95-0.17)

        POS=number_of_positives
        NEG=POS*(1-self.prevalence)*(1/self.prevalence)
        TP=POS*row.tpr
        TOTALFLAGS=TP/row.ppv
        FP=TOTALFLAGS-TP
        FN=POS-TP
        TN=NEG-FP
        if five_yr_survival is not None:
            NNS=TOTALFLAGS/(TP*factor*(1- five_yr_survival))
        else:
            NNS=np.nan

            
        resdf=pd.DataFrame.from_dict({"POS":np.round(POS),"TP":np.round(TP),
                                      "FP":np.round(FP),"NEG":np.round(NEG),
                                      "FLAGS":np.round(TOTALFLAGS),"FN":np.round(FN),
                                      "TN":np.round(TN),
                                      "NNS":np.round(NNS),
                                      "FLAGGED_FRACTION":np.round(TOTALFLAGS/(POS+NEG),
                                                                  2)},orient='index',columns=['estimates'])
        
        rf=pd.DataFrame({'pos':np.round(POS),
                      'flags':int(np.round(TOTALFLAGS)),
                      'tp':int(np.round(TP)),
                      'fp':int(np.round(FP)),
                      'fn':int(np.round(FN)),
                      'tn':int(np.round(TN))},index=['numbers'])

        pos=rf.pos.values[0]
        flags=rf['flags'].values[0]
        fp=rf['fp'].values[0]
        tp=rf['tp'].values[0]
        fn=rf['fn'].values[0]

        txt=[f"For every {pos} positive instances",
             f"we raise {flags} flags,",
             f"out of which {tp} are true positives",
             f"{fp} are false alarms",
             f"{fn} cases are missed"]
        if five_yr_survival is not None:
            txt.append(f"Number needed to screen is {NNS}")
        
            

        return rf,txt,resdf




def genroc(df,
           risk='predicted_risk',
           target='target',
           steps=1000,
           TARGET=[1],
           outfile=None):
    '''
    compute roc curve from raw observation of risk-target information on samples

    Args:
        df (pandas.DataFrame): dataframe of raw samples identified as positive or negative with computed risk
        target (str): name of target column
        risk (str): name of risk column
        TARGET (list): list of values of target column that define the positive case, default [1]
        steps (int): steps between max and min of risk value, default 1000
        outfile (str): write datafraem with fpr tpr threshold, default None

    Returns:
        pandas.DataFrame: roc dataframe
        int: total number of samples
        int: total number of positive samples
    '''
    threshold={}
    df_=df[[risk,target]].rename(columns={risk:'risk',target:'target'})
    delta=(df_.risk.max()-df_.risk.min())/steps
    for r in np.arange(df_.risk.min(),df_.risk.max()+delta,delta):
        #print(r)
        fn=df_[(df_.risk<r) & df_.target.isin(TARGET)].index.size
        tp=df_[(df_.risk>=r) & df_.target.isin(TARGET)].index.size
        fp=df_[(df_.risk>=r) & ~df_.target.isin(TARGET)].index.size
        tn=df_[(df_.risk<r) & ~df_.target.isin(TARGET)].index.size
        threshold[r]={'tp':tp,'fp':fp,'tn':tn,'fn':fn}
        
    xf=pd.DataFrame.from_dict(threshold).transpose()
    xf.index.name='threshold'
    
    xf=xf.assign(tpr=(xf.tp)/(xf.tp+xf.fn)).assign(fpr=(xf.fp)/(xf.fp+xf.tn))
    xf=xf[['fpr','tpr']].reset_index()#.set_index('fpr')
    
    if outfile is not None:
        xf.to_csv(outfile)
    return xf,df_.index.size,df_[df_.target.isin(TARGET)].index.size    



def pipeline(df,
            risk='predicted_risk',
            target='target',
            steps=1000,
            TARGET=[1],
            order=3,
            alpha=0.05,
            prevalence=.002,
            precision=3,
            outfile=None):
    rf,total_samples,positive_samples=genroc(df,risk=risk,
                                             target=target,
                                             TARGET=TARGET)
    zt=processRoc(rf,
                  order=order, 
                  total_samples=total_samples,
                  positive_samples=positive_samples,
                  alpha=alpha,
                  prevalence=prevalence)
    
    zt.smooth(STEP=0.001)
    zt.allmeasures(interpolate=True)
    zt.usample(precision=precision)
    zt.getBounds()

    df_=zt.get()
    df_u=zt.df_lim['U']
    df_l=zt.df_lim['L']

    df_=df_.join(df_u,rsuffix='_upper').join(df_l,rsuffix='_lower')

    if outfile is not None:
        df_.to_csv(outfile)

    return df_,zt.auc()


def score_to_probability(scores,df,
                  prevalence,
                  total_samples,
                  positive_samples,
                  alpha=0.05):
    '''
    returns score to probability and upper and lower bound fast 
    and robust, standalone.
    implements: Statist. Med. 2007; 26:3258–3273
    DOI: 10.1002/sim.2812
    Prevalence-dependent diagnostic accuracy measures
    Jialiang Li, Jason P. Fine and Nasia Safdar
    '''
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    se=df.tpr
    sp=1-df.fpr
    var_se = se * (1 - se) / positive_samples
    var_sp = sp * (1 - sp) / (total_samples - positive_samples)
    df['g1']= (1 - sp) / se
    sigma_1 = np.sqrt(total_samples * (1 - sp)**2 * var_se / (se**4) + total_samples * var_sp / (se**2))
    df['ci_g1'] =  z * (sigma_1 / np.sqrt(total_samples))
    df['ppv'] = se * prevalence / (se * prevalence + (1 - sp) * (1 - prevalence))
    df['ppv_lower'] = 1/(1+ ((1-prevalence)/prevalence)*(df['g1']-df['ci_g1']))
    df['ppv_upper'] = 1/(1+ ((1-prevalence)/prevalence)*(df['g1']+df['ci_g1']))
    df=df.dropna().drop(['g1','ci_g1'],axis=1)
    
    return [df[df.threshold>=score].tail(1)[['ppv','ppv_upper','ppv_lower']].round(2).values[0] for score in scores]
