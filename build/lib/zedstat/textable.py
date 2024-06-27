import pandas as pd
import numpy as np
#STRA='L{1in}|L{1.25in}|L{1.25in}|L{1.5in}|L{.3in}|L{.3in}'
def textable(df,tabname='tmp.tex',
             FORMAT='%1.2f',
             INDEX=True,DUMMY=False,
             USE_l=False,
             TABFORMAT=None,
             HEADERCOLOR=None,
             LNTERM='\\\\\\hline\n'):
    '''
    Write a pandas DataFrame to a LaTeX formatted table. This function allows customization of the table format, including the inclusion of indices, column formatting, header color, and LaTeX table format specifications.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data to be written to the LaTeX table.
        tabname (str, optional): Name of the output LaTeX file. Defaults to 'tmp.tex'.
        FORMAT (str, optional): String format for floating point numbers. Defaults to '%1.2f'.
        INDEX (bool, optional): Whether to include the DataFrame index as a column in the table. Defaults to True.
        DUMMY (bool, optional): If True, the function returns immediately without writing anything. Useful for debugging. Defaults to False.
        USE_l (bool, optional): If True, uses 'l' (left-align) for all columns. Defaults to False.
        TABFORMAT (str, optional): Custom LaTeX tabular format string. If not specified, defaults based on other parameters.
        HEADERCOLOR (str, optional): LaTeX code for coloring the header row. Defaults to None.
        LNTERM (str, optional): String to use as a line terminator in the LaTeX table. Defaults to '\\\\hline\n'.

    Returns:
        None: The output is written directly to a file specified by `tabname`.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> textable(df, 'example.tex', FORMAT='%.2f', INDEX=False)
        This will write the DataFrame to 'example.tex' without the index and formatted floating points.
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
    print(hf)
    csv_string1 = hf.to_csv(sep='&', quotechar=' ', index=False, float_format=FORMAT)
    csv_string1 = csv_string1.replace('\n','\\\\\\hline\n' )
    if HEADERCOLOR is not None:
        csv_string1 = HEADERCOLOR + csv_string1
    print(csv_string1)
    
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
    Create a new DataFrame that includes confidence bounds for each entry. This function is useful for preparing data that needs to display error margins or confidence intervals.

    Parameters:
        df (pandas.DataFrame): The base DataFrame with central values.
        df_upper (pandas.DataFrame, optional): DataFrame containing the upper bounds. Must not be used together with df_delta.
        df_lower (pandas.DataFrame, optional): DataFrame containing the lower bounds. Must not be used together with df_delta.
        df_delta (pandas.DataFrame, optional): DataFrame containing half the interval of the bounds around the central values. Must not be used with df_upper or df_lower.
        thresholdcolname (str, optional): Column name in `df` that should not have bounds applied. Defaults to 'threshold'.
        width (int, optional): Unused parameter in the current function version.

    Returns:
        pandas.DataFrame: A new DataFrame with the bounds formatted as strings in LaTeX math mode.

    Example:
        >>> df = pd.DataFrame({'value': [10, 20], 'error': [1, 2]})
        >>> df_bounds = tablewithbounds(df, df_delta=df['error'])
        This will create a DataFrame where each 'value' is shown with its error as a LaTeX-formatted string.
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
            


        
