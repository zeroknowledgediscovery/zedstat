import pandas as pd
import numpy as np
#STRA='L{1in}|L{1.25in}|L{1.25in}|L{1.5in}|L{.3in}|L{.3in}'
def textable(df,tabname='tmp.tex',
             FORMAT='%1.2f',
             INDEX=True,DUMMY=False,
             USE_l=False,
             TABFORMAT=None,
             LNTERM='\\\\\\hline\n'):
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
    #df.to_csv(tabname,float_format=FORMAT,
    #          line_terminator=LNTERM,
    #          sep='&',quotechar=' ',index=None,mode='a')

    hf=pd.DataFrame(columns=df.columns)
    csv_string1 = hf.to_csv(sep='&', quotechar=' ', index=False, float_format=FORMAT)
    csv_string1 = csv_string1.replace('\n','\\\\\\hline\n' )
    
    with open(tabname,'w') as f:
        f.write(csv_string1)

    # Convert DataFrame to CSV string
    csv_string = df.to_csv(sep='&', header=None,quotechar=' ', index=False, float_format=FORMAT)
    
    # Replace newline character with your desired line terminator
    csv_string = csv_string.replace('\n', LNTERM)
    
    # Write to file
    with open(tabname, 'a') as file:
        file.write(csv_string)

    
    with open(tabname,'a') as f:
        f.write('\\hline\\end{tabular}\n')


def getpm(row,tag):
    '''
    add confidence bound to table entry
    '''
    return '$'+str(row[tag])[:5]+' \pm '+str(1*row[tag+'_cb'])[:5]+'$'


def tablewithbounds(df,
                    df_upper=None,
                    df_lower=None,
                    df_delta=None,
                    thresholdcolname='threshold',
                    width=5):
    '''
    get dataframe with bounds displayed
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
        df_delta = ((df_upper-df_lower)/2).abs()
        

    df_=dfthis.join(df_delta,rsuffix='_cb')

    for col in df.columns:
        if col != thresholdcolname:
            dfthis[col]=df_.apply(getpm,axis=1,tag=col)

    return dfthis
            


        
