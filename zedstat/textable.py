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
