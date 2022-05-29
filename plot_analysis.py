import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def preprocess(fname, drops=['condition', 'size']):
    '''
    Preprocesses the dataframe by converting posting date to datetime
    and adding ranges to price/year.
    :return:
    '''
    df = pd.read_csv(fname, header=0, index_col=0)

    all_drops = ['url', 'region_url'] + drops
    df.drop(all_drops, inplace=True, axis=1)
    print(len(df['manufacturer'].unique()))

    # Correction of previous columns
    df['state'] = df['state'].str.upper()
    print('state')
    # df['posting_date'] = pd.to_datetime(df['posting_date'], format='%Y-%m-%dT%H:%M:%S', cache=True)
    df['posting_date'] = pd.to_datetime(df['posting_date'], infer_datetime_format=True, cache=True)
    print('date')
    df['price'] = df['price'].replace(0, np.nan)
    print('price')
    df = df.dropna(subset=['price'])

    print(len(df['price'].dropna()), len(df['price']))
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
    df.to_csv('data/processed.csv')
    return df

def plot_pie(col, lim = 15):
    '''
    Plots the pie chart of the categorical variable col.
    :param col: The name of the categorical column.
    :param lim: The top number of values to group by.
    :return:
    '''
    assert col in cats

    values = df[col].dropna().value_counts().sort_values(ascending=False)
    topk = min(len(values), lim)
    top_values = values.nlargest(topk)
    if topk > lim: top_values['other'] = values.nsmallest(values.size - topk).sum()
    title = 'Number of cars by %s' % (col.replace('_', ' '))
    fig = px.pie(values, values=top_values.values, names=top_values.index, title=title)

    fig.show()

def plot_two(col1, col2): pass
    # assert col1 in df.columns and col2 in df.columns
    # df_nonull = df.dropna(subset=[col2], axis='index')
    # p = df_nonull.groupby([col1]).mean()[col2]
    # statewise_prices = pd.DataFrame({'states': p.index, 'avg_price': p.values})
    # statewise_prices['states'] = statewise_prices['states'].str.upper()
    #
    # # TODO - split into groups of price values or percentiles to distinguish easily
    # fig = px.choropleth(statewise_prices, locations='states', locationmode="USA-states",
    #                     color='avg_price',
    #                     # color_continuous_scale="Viridis",
    #                     scope="usa",
    #                     color_continuous_scale=px.colors.sequential.Plasma
    #                     )
    # fig.show()

def get_groupby_columns(col1, col2='url', agg='count', num_vals=1):
    """ This function returns col2 grouped by col1 with aggregation function applied

    Args:
        col1 (str): column to groupby
        col2 (str, optional): column to return. Defaults to 'url'
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
    df = preprocess('data/vehicles_no_url_des_filled.csv')
    cats, nums = df.select_dtypes(['object', 'category']).columns, df.select_dtypes(np.number).columns
    plot_pie('price_range')