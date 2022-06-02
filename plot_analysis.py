import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

def preprocess(fname='vehicles_cleaned_imputed.csv', drops=['condition', 'VIN']):
    """Preprocesses the data from the csv in fname.

    For the csv provided, the function will drop irrelevant columns,
    correct or convert the values in a few of the relevant columns,
    and add categorical equivalents (grouped by four ranges) to the numeric columns.

    Args:
        fname (str): The filename of the csv file to be preprocessed.
        drops (list): The columns to be dropped.
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    df = pd.read_csv(fname, header=0, index_col=0)

    all_drops = ['url', 'region_url'] + drops
    df.drop(all_drops, inplace=True, axis=1)

    # Correction of previous columns
    df['state'] = df['state'].str.upper()
    df['price'] = df['price'].replace(0, np.nan)
    df = df.dropna(subset=['price'])
    df = df[df['price'] < 300000]

    # Split posting date to year and month
    df['posting_date'] = pd.to_datetime(df['posting_date'], utc=True, infer_datetime_format=True, cache=True)
    df['posting_year'] = pd.DatetimeIndex(df['posting_date']).year
    df['posting_month'] = pd.DatetimeIndex(df['posting_date']).month

    df['price_range'] = pd.cut(df['price'],
                               [-np.inf, 5000, 15000, 25000, np.inf],
                               labels=['<$5000', '$5000-15000', '$15000-25000', '>$25000']
                               )
    df['year_range'] = pd.cut(df['year'],
                               [-np.inf, 2005, 2010, 2015, np.inf],
                               labels=['<2005', '2005-2010', '2010-2015', '>2015']
                               )
    df['odometer_range'] = pd.cut(df['odometer'],
                              [-np.inf, 40000, 80000, 120000, np.inf],
                              labels=['<40000', '40000-80000', '80000-120000', '>120000']
                              )
    return df

def plot_pie(col, lim = 15):
    """Plots the pie chart of the categorical variable col.

    The col variable must be the name of a categorical column of the dataframe.
    From there, it will create a pie chart of the most frequent values up to lim.
    All other values will be assigned to the 'Other' group.

    Args:
        col (str): The name of the categorical column.
        lim (int): The top number of values to group by.
    """
    assert col in cats

    values = df[col].dropna().value_counts().sort_values(ascending=False)
    topk = min(len(values), lim)
    top_values = values.nlargest(topk)
    if len(values) > lim: top_values['other'] = values.nsmallest(values.size - topk).sum()
    title = 'Number of cars by %s' % (col.replace('_', ' '))
    fig = px.pie(values, values=top_values.values, names=top_values.index, title=title)

    fig.show()

def get_groupby_columns(col1, col2='url', agg='count', num_vals=1):
    """ This function returns col2 grouped by col1 with the aggregation function agg.

    Args:
        col1 (str): The column to group by
        col2 (str, optional): The column to return. Defaults to 'url'.
        agg (str, optional): Aggregation function to be applied. Defaults to 'count'.
        num_vals (int, optional): Top n categories of col1 are returned to make plotting easier. Defaults to 1.
    Returns:
        pd.dataframe: columns specified are returned
    """
    df = preprocess()
    assert col2 in df.columns
    assert col1 in df.columns
    valid_aggregations = {'count', 'mean', 'sum'}
    assert agg in valid_aggregations
    # drop nulls
    df_clean = df.dropna(subset=[col2], axis='index')
    # apply aggregation
    if agg == 'count':
        return df_clean[[col1, col2]].groupby(col1)[col2].count().sort_values(ascending=False).reset_index()
    elif agg == 'mean':
        return df_clean[[col1, col2]].groupby(col1)[col2].mean().sort_values(ascending=False).reset_index()
    elif agg == 'sum':
        return df_clean[[col1, col2]].groupby(col1)[col2].sum().sort_values(ascending=False).reset_index()
    # TODO: add topk handling

if __name__ == '__main__':
    df = preprocess()
    cats, nums = df.select_dtypes(['object', 'category']).columns, df.select_dtypes(np.number).columns
    plot_pie('manufacturer')