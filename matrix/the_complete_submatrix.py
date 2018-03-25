import numpy as np

def day_filter(df, n_valid):
    return df.groupby('Stock').filter(lambda x: x.y.count() >= n_valid)

def the_complete_submatrix(df):
    filt = day_filter(df, 246)
    valid = filt.pivot('Day', 'Stock', 'Count')
    thedays = set(valid.T.sum().pipe(lambda x: x[x >= 1955]).index)
    _thestocks = filt[filt.Day.isin(thedays)].groupby('Day').Stock.agg(lambda x: set(x))
    thestocks = np.bitwise_and.reduce(_thestocks.values)
    return filt[filt.Day.isin(thedays) & filt.Stock.isin(thestocks)]