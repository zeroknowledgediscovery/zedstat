import pandas as pd
import numpy as np
DUMMY=False
STRA='L{1in}|L{1.25in}|L{1.25in}|L{1.5in}|L{.3in}|L{.3in}'

def textable(df,tabname='tmp.tex',FORMAT='%1.2f',INDEX=True,DUMMY=DUMMY,USE_l=False,
             TABFORMAT=None,LNTERM='\\\\\\hline\n'):
    '''
        write latex table
    '''
    if DUMMY:
        return
    if INDEX:
        df=df.reset_index()
    columns=df.columns
    df.columns=[x.replace('_','\\_').replace('\_\_','_') for x in columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col]=df[col].str.replace('_','\\_')
    
    if USE_l:
        TABFORMAT='l'*len(df.columns)
    else:
        if TABFORMAT is None:
            TABFORMAT='L{1in}|'*len(df.columns)
            TABFORMAT=TABFORMAT[:-1]
    STR='\\begin{tabular}{'+TABFORMAT+'}\\hline\n'        
    with open(tabname,'w') as f:
        f.write(STR)
    df.to_csv(tabname,float_format=FORMAT,
              line_terminator=LNTERM,
              sep='&',quotechar=' ',index=None,mode='a')
    
    with open(tabname,'a') as f:
        f.write('\\hline\\end{tabular}\n')



def getpm(row,tag):
    return '$'+str(row[tag])[:5]+' \pm '+str(1*row[tag+'_cb'])[:5]+'$'


def tablewithbounds(df,
                    df_upper=None,
                    df_lower=None,
                    df_delta=None,
                    thresholdcolname='threshold',
                    width=5):
    '''
    get datafraem with bounds displayed
    '''

    dfthis=df.copy()
    if df_delta is not None:
        assert (df_upper is None) and (df_lower is None)


    if df_upper is not None:
        assert df_delta is None
        if df_lower is None:
            df_delta=(df_upper-dfthis)/2

    if df_lower is not None:
        assert df_delta is None
        if df_upper is None:
            df_delta=(df_lower-dfthis)/2

    if (df_lower is not None) and (df_upper is not None):
        df_delta = pd.DataFrame()
        for col in df.columns:
            if col == 'tpr':
                df_delta[col]=df[col]-df_lower[col]
                
            if col == 'ppv':
                df_delta[col]=df[col]-df_lower[col]
                
            if col == 'acc':
                df_delta[col]=df[col]-df_lower[col]
                
            if col == 'npv':
                df_delta[col]=df[col]-df_lower[col]
                
            if col == 'LR+':
                df_delta[col]=df[col]-df_lower[col]
                
            if col == 'LR-':
                df_delta[col]=df_lower[col]-df[col]
        

    df_=dfthis.join(df_delta,rsuffix='_cb')

    for col in df.columns:
        if col != thresholdcolname:
            dfthis[col]=df_.apply(getpm,axis=1,tag=col)

    return dfthis
            


        
