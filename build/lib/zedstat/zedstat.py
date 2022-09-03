import pandas as pd
import numpy as np
from sklearn.metrics import auc

class zedstat(object):
    """zedstat"""

    def __init__(self,
                 df=None,
                 fprcol='fpr',
                 tprcol='tpr',
                 thresholdcol='threshold',
                 prevalence=None,
                 order=2,
                 total_samples=None,
                 positive_samples=None,
                 alpha=0.01,
                 random_state=None):
        
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
        self.random_state = random_state
        self.raw_df = self.df.copy()
        self.prevalence = prevalence
        self.order = order
        self.df_lim = {}
        self._auc = {'U':[],'L':[]}
        self.total_samples = total_samples
        self.positive_samples = positive_samples
        self.alpha = alpha
        
        
    def get(self):
        '''
        return dataframe
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
        calculate auc with confidence bounds
        '''
        self.nominal_auc()
        self._auc['U']=[]
        self._auc['L']=[]
        
        self.getBounds(total_samples=total_samples,
                positive_samples=positive_samples,
                alpha=alpha)
        self.auc_cb2(total_samples=total_samples,
                positive_samples=positive_samples,
                alpha=alpha)
        return self._auc['nominal'], self._auc['U'].min(), self._auc['L'].max()
    
    def convexify(self):
        '''
        compute convex hull of the roc curve
        '''   
        from scipy.spatial import ConvexHull
        if self.df.index.name==self.fprcol:
            rf=self.df
        else:
            if self.fprcol in self.df.columns:
                rf=self.df.set_index(self.fprcol)#.drop('threshold',axis=1)
            else:
                raise('fpr not in columns or index')
        pts=rf.reset_index()[[self.fprcol,self.tprcol]].values
        if len(pts)<3:
            rf_=rf.reset_index()
            rf_.columns=['tpr','fpr']
            rf_=rf_.sort_values('fpr')
            rf_=rf_.sort_values('tpr')
            self.df=rf_.set_index(self.fprcol).copy()
            return #self.df
        
        hull = ConvexHull(pts)
        rf_=pd.DataFrame(pts[hull.vertices, 0], pts[hull.vertices, 1]).reset_index()
        rf_.columns=['tpr','fpr']
        rf_=rf_.sort_values('fpr')
        rf_=rf_.sort_values('tpr')
        self.df=rf_.set_index(self.fprcol).sort_index().copy()
        return #self.df


    def smooth(self,
               STEP=0.0001,
               interpolate=True,
               convexify=True):
        '''
        smooth roc curves
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
            self.convexify()
        return 

    def allmeasures(self,prevalence=None,interpolate=False):
        '''
        compute accuracy, PPV, NPV, positive and negative likelihood ratios
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
        if interpolate:
            df__=df__.interpolate(limit_direction='both',method='spline',order=self.order)
            
        if self.thresholdcol is not None:
            if self.thresholdcol not in df__.columns:
                df__=df__.join(self.raw_df[self.thresholdcol])
        self.df=df__
        self.correctPPV()
        self.df[self.df < 0] = 0
        
        return #self.df


    def correctPPV(self):
        '''
        make ppv monotonic
        '''
        if 'ppv' not in self.df.columns:
            return
        
        self.df.loc[0].ppv=1.0
        self.df.loc[1].ppv=0.0
        arr=self.df.ppv.values
        a_=[]
        for i in np.arange(len(arr)-1):
            if arr[i+1]>arr[i]:
                a_.append(1.05*arr[i+1])
            else:
                a_.append(arr[i])
        a_.append(arr[-1])
        self.df.ppv=a_
        return #self.df

    
    def usample(self,precision=3):
        '''
        make roc curve and other measures estimated at regular intervals of false positive rate
        '''
        step=10**(-precision)
        fpr=[np.round(x,precision) for x in np.arange(0,1+step,step)]
        fpr_=[x for x in fpr if x not in self.df.index]

        df___=self.df.copy()
        for x in fpr_:
            df___.loc[x]=pd.Series([],dtype=float) 
        df___=df___.sort_index().interpolate()
        self.df=df___.loc[fpr]
        return #self.df
    
    def getDelta(self,
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
        '''
        self.getDelta(total_samples=total_samples,
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
                df__.fpr=df__.fpr-self.delta_.fprdel
            else:
                df__.tpr=df__.tpr-self.delta_.tprdel
                df__.fpr=df__.fpr+self.delta_.fprdel
                

            df__['ppv']=1/(1+((df__.fpr/df__.tpr)*((1/p)-1)))
            df__['acc']=p*df__.tpr + (1-p)*(1-df__.fpr)
            df__['npv']=1/(1+((1-df__.tpr)/(1-df__.fpr))*(1/((1/p)-1)))
            df__['LR+']=(df__.tpr)/(df__.fpr)
            df__['LR-']=(1-df__.tpr)/(1-df__.fpr)
            df__=df__.interpolate(limit_direction='both',method='spline',
                                  order=self.order).set_index(self.fprcol)
            df__[df__ < 0] = 0
            self.df_lim[direction]=df__

            if direction=='U':
                self._auc[direction]=np.array([np.append(self._auc[direction],
                                                auc(df__.index.values,
                                                    df__.tpr.values)).min()])
            if direction=='L':
                self._auc[direction]=np.array([np.append(self._auc[direction],
                                                auc(df__.index.values,
                                                    df__.tpr.values)).max()])
        return 



    def auc_cb2(self,
                total_samples=None,
                positive_samples=None,
                alpha=None):
        '''
        compute auc confidence bounds using Danzig bounds
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
                   delta_auc,
                   target_auc=None,
                   alpha=None):
        '''
        estimate sample size for atataing auc bound under given significance level
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
    
    
    def interpret(self,fpr=0.01,number_of_positives=100):
        '''
        generate simple interpretation of inferred model, based on a number of positive cases
        '''
        wf=self.df.copy()
        wf.loc[fpr]=pd.Series([],dtype=float)
        wf=wf.sort_index().interpolate(method='spline',order=self.order)
        row=wf.loc[fpr]

        POS=number_of_positives
        TP=POS*row.tpr
        FP = TP*((1/row.ppv) -1)
        NEG=FP/fpr
        TOTALFLAGS=TP+FP
        FN=POS-TP
        TN=POS/self.prevalence

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

        return rf,txt
