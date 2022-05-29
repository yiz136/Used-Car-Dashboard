import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from textwrap import dedent as d
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.offline import plot
import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Sign
#from sklearn.externals import joblib
import joblib
# from util import dynamic_predict
import matplotlib.pyplot as plt
# from visualization import analysis  
# from visualization import plots
# from data import plot_analysis

external_stylesheets = ['assets/style.css'] #https://codepen.io/chriddyp/pen/bWLwgP.css
app = dash.Dash(__name__, external_stylesheets=external_stylesheets) #external_stylesheets=external_stylesheets

#app.scripts.config.serve_locally = True

app.config['suppress_callback_exceptions'] = True

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
tab_style = {
    'borderTop': '1px solid #292841',
    'borderBottom': '1px solid #292841',
    'borderLeft': '1px solid #bcbfc2',
    'borderRight': '1px solid #bcbfc2',
    'fontWeight': 'bold',
    'font-size': '20px',
    'font-family': 'Optima, sans-serif',
    'color': '#b8bbbe',
    'backgroundColor': '#292841',
    'padding': '8px',
}
# vis_tab_style = {
#     'borderBottom': '1px solid #d6d6d6',
#     'padding': '12px',
# }
tab_selected_style = {
    'borderTop': '2px solid #bed2ff',
    'borderBottom': '1px solid #292841',
    'borderLeft': '1px solid #bcbfc2',
    'borderRight': '1px solid #bcbfc2',
    'backgroundColor': '#292841',
    'color': 'white',
    'font-size': '22px',
    'font-family': 'Optima, sans-serif',
    'font-weight': 'bold',
    'padding': '8px',
}

H3_style = {
    'font-family': 'Optima, sans-serif',
    'font-size': '20px',
    'color': '#bed2ff',
    'font-weight': 'bold',
    # 'float': 'left',
    # 'text-align': 'center',
}

label_style = {
    'font-family': 'Optima, sans-serif',
    'font-size': '20px',
    'color': '#bed2ff',
    'font-weight': 'bold',
    'float': 'bottom',
    'margin-left': '1em',
    # 'text-align': 'center',
}

app.layout = html.Div(
    children = [
    html.H1("Used Car Sales Recommendation Dashboard", style={'text-align': 'center', 'font-family': 'Optima, sans-serif',
                                                              'padding-top': '0.4em', 'padding-bottom': '0.4em',
                                                              'margin-top': '0em', 'margin-bottom': '0em',
                                                              'font-weight': 'bold', 'color': '#bed2ff', 'font-size': '40px',
                                                              'backgroundColor':'#292841'}), #bed2ff, 2ddfb3
    dcc.Tabs(id="tabs", value='tab-1', style={'margin-top': '0em', 'margin-bottom': '0em'},
        children=[
        dcc.Tab(label='Car Sales Information & Trends', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Cars Recommendation Dashboard', value='tab-new', style=tab_style, selected_style=tab_selected_style),
    ]),
    html.Div(id='tabs-content')
]) # style={'backgroundColor':'#cbd4ff'}

# layout_tab_1  = html.Div(children = [
#     dcc.Tabs(id = "vis-tabs", value = "vistab", vertical=True, parent_style={'float': 'left','width': '40'},children =[
#         dcc.Tab(label='Prices Information', value='tab-3', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='Car price among states', value='tab-4', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='Conditions', value='tab-5', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='Odometer', value='tab-6', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='Region', value='tab-7', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='...', value='tab-8', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='...', value='tab-9', style=vis_tab_style, selected_style=tab_selected_style),
#         dcc.Tab(label='...', value='tab-10', style=vis_tab_style, selected_style=tab_selected_style),
#     ]),
#     html.Div(id='vis-tabs-content', style={'float': 'left'})
# ])

#*****************************


df = pd.read_csv('data/processed1.csv', header=0, index_col=0)

cats, nums = df.select_dtypes(['object', 'category']).columns, df.select_dtypes(np.number).columns
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
    fig = px.pie(values, values=top_values.values, names=top_values.index, title=title, color_discrete_sequence=px.colors.sequential.RdBu)

    return fig


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
    df = pd.read_csv('data/processed1.csv', header=0, index_col=0)
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

# if __name__ == '__main__':
#     df = preprocess()
#     cats, nums = df.select_dtypes(['object', 'category']).columns, df.select_dtypes(np.number).columns
#     plot_pie('price_range')

def state_average():
    fig = px.choropleth(get_groupby_columns('state', 'price', 'mean'), locations='state', locationmode="USA-states",
                        color='price',
                        # color_continuous_scale="Viridis",
                        scope="usa",
                        color_continuous_scale=px.colors.sequential.Plasma
                        )
    return fig
#***************************** DropDown 


educational_Level_vis = html.Div(
    children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = state_average()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center' }),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = plot_pie('price_range')
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])


from interactive_plots import preprocess, interactive_plots_preprocess, price_trendency_plot, count_plot
import holoviews as hv
hv.extension('bokeh', 'matplotlib')
import hvplot.pandas
from bokeh.plotting import show
from holoviews.plotting.plotly.dash import to_dash

df = pd.read_csv('data/processed2.csv', header=0, index_col=0)
df, edata = interactive_plots_preprocess(df)
def dropdown_select(k):
    components = to_dash(app, [price_trendency_plot(edata, k)])
    plot = html.Div(components.children)
    return plot

# count_plot(df, "manufacturer")

def dropdown_count(k):
    components = to_dash(app, [count_plot(df, k)])
    plot = html.Div(components.children)
    return plot

# income_vis1 = html.Div(children =[

#             html.Div([
#             html.Div(children =[
#                 dcc.Graph(
#                 id = "marital status",
#                 figure = price_trendency_plot(edata, "year")
#             ) ],
#             style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

#             html.Div(children =[
#                 dcc.Graph(
#                 id = "marital prob",
#                 figure = count_plot(df, "manufacturer")
#             )],
#             style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
#             ]),

# ])

contact_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = state_average()
            ) ],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = state_average()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

loan_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = state_average()
            ) ],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = state_average()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])
house_vis = html.Div(children =[            
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = state_average()
            ) ],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = state_average()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])


prediction_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "prediction_pie_chart",
                figure = state_average()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "predicted_prob_hist",
                figure = state_average()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])

age_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "prediction_pie_chart",
                figure = state_average()
            )],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "predicted_prob_hist",
                figure = state_average()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])


@app.callback(Output('vis-tabs-content', 'children'),
              [Input('vis-tabs', 'value')])
def render_content(tab):
    if tab == 'tab-3':
        return marital_status_vis
    elif tab == 'tab-4':
        return educational_Level_vis
    # elif tab == 'tab-5':
    #     return income_vis 
    elif tab == 'tab-6':
        return contact_vis 
    elif tab == 'tab-7':
        return loan_vis 
    elif tab == 'tab-8':
        return house_vis 
    elif tab == 'tab-9':
        return age_vis
    elif tab == "tab-10":
        return prediction_vis
    else:
        return marital_status_vis


layout_tab_2 = html.Div(children =[
             
             html.Div(dash_table.DataTable(
                         columns=[
                             {'name': 'Customer ID', 'id': 'customer_id', 'type': 'numeric', 'editable': False},
                             {'name': 'Age', 'id': 'age', 'type': 'numeric', 'editable': False},
                             {'name': 'Income', 'id': 'job_transformed', 'type': 'text', 'editable': False},
                             {'name': 'Previously Contacted', 'id': 'poutcome', 'type': 'text', 'editable': False},
                             {'name': 'Probability of Success', 'id': 'prob_1', 'type': 'numeric', 'editable': False, 'format': FormatTemplate.percentage(1)},
                             {'name': 'Call Result', 'id': 'is_called', 'type': 'any', 'editable': True, 'presentation': 'dropdown'}
                         ],
                         data=df.to_dict('records'),
                         filter_action='native',
                         dropdown={
                             'is_called': {
                                 'options': [
                                     {'label': i, 'value': i}
                                     for i in ['Not Called', 'Success', 'Failure']
                                 ]
                             }
                         },                
                         style_table={
                             'maxHeight': '50ex',
                             'overflowY': 'scroll',
                             'width': '100%',
                             'minWidth': '100%',
                         },
                         style_data={
                             'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
                             'overflow': 'hidden',
                             'textOverflow': 'ellipsis',
                         },
                         style_cell = {
                             'font_family': 'arial',
                             'font_size': '16px',
                             'text_align': 'center'
                         },
#                         style_cell_conditional=[
#                            {
#                                'if': {'column_id': c},
#                                'textAlign': 'left'
#                            } for c in ['customer_id', 'job_transformed', 'poutcome', 'is_called']
#                        ],
                         style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }, {
                                'if': {
                                    'column_id': 'is_called',
                                    'filter_query': '{is_called} eq "Not Called"'
                                },
                                'backgroundColor': '#E0E280'
                            }, {
                                'if': {
                                    'column_id': 'is_called',
                                    'filter_query': '{is_called} eq "Success"'
                                },
                                'backgroundColor': '#8CE280'
                            }, {
                                'if': {
                                    'column_id': 'is_called',
                                    'filter_query': '{is_called} eq "Failure"'
                                },
                                'backgroundColor': '#E28080'
                            }
                        ],
                         style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                         page_action="native",
                         page_current= 0,
                         sort_action="native",
                         sort_mode="multi"
                         )                    
                     )                
                
        ])

layout_tab_new = html.Div(children =[
    html.Div(children =[
    html.Div(children =[
    html.Label('Enter number of employees (quarterly indicator): ', style=label_style),
    dcc.Input(id='nremployed', placeholder='# employees', type='number')],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the outcome of the previous marketing campaign: ', style=label_style),
    dcc.Input(id='poutcome_success', placeholder='prev.', type='number', min = 0, max = 1, step = 1)],
              style={'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the employment variation rate - quarterly indicator: ', style=label_style),
    dcc.Input(id='emp', placeholder='emp. variation rate', type='number')],
              style={'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the number of days since the last call (999 if NA): ', style=label_style),
    dcc.Input(id='pdays', placeholder='# days since last call', type='number')],
              style={'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the consumer confidence index (monthly indicator): ', style=label_style),
    dcc.Input(id='consconfidx', placeholder='consumer conf. index', type='number')],
              style={'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the euribor 3 month rate (daily indicator): ', style=label_style),
    dcc.Input(id='euribor3m', placeholder='euribor rate', type='number')],
              style={'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the no income indicator, 1 if the customer job retired, student or unemployed: ', style=label_style),
    dcc.Input(id='job_transformed_no_income', placeholder='inc', type='number', min = 0, max = 1, step = 1)],
              style={'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),
    
    ]),
   
    html.Div(children=[
        html.H1(children='Probability of Success: ', style=H3_style),
        html.Div(id='pred-output')
    ], style={'textAlign': 'center', 'justify-content': 'center'}),
])
@app.callback(
    Output('pred-output', 'children'),
    [Input('nremployed', 'value'),
     Input('poutcome_success', 'value'),
     Input('emp', 'value'),
     Input('pdays', 'value'),
     Input('consconfidx', 'value'),
     Input('euribor3m', 'value'),
     Input('job_transformed_no_income', 'value')])
def show_success_probability(nr_employed, poutcome_success, emp_var_rate, pdays, cons_conf, euribor, no_income):
    if not nr_employed: 
        nr_employed = 0
    if not poutcome_success: 
        poutcome_success = 0
    if not emp_var_rate: 
        emp_var_rate = 0
    if not pdays: 
        pdays = 0
    if not cons_conf: 
        cons_conf = 0
    if not euribor: 
        euribor = 0
    if not no_income:
        no_income = 0
        
        #raise PreventUpdate
    #else:
    # prob = dynamic_predict(nr_employed, poutcome_success, emp_var_rate, pdays, cons_conf, euribor, no_income)[0]*100
    return html.Div(children =[
        html.H1(children=f'{round(0.5, ndigits = 2)}'+"%")
    ])



### DropDown ###

dd_1 = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "prediction_pie_chart",
                figure = price_trendency_plot(edata, "year")
            )],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "predicted_prob_hist",
                figure = price_trendency_plot(edata, "manufacturer")
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])

dropdown_plot = html.Div(children =[
    dcc.Dropdown(['year', 'state', 'manufacturer', 'model', 'condition', 'odometer', 'fuel', 'size', 'type', 'cylinders', 'drive', 'transmission', 'paint_color', 'price_range','year_range', 'odometer_range'], 
    id='demo-dropdown', placeholder='Select an attribute...', style={'width': '80%'}),
    html.Div(children =[
    html.Div(id='dd-output-container',style={'float': 'left', 'display': 'flex', 'justify-content': 'center'})]),],
    style={'height': 380,'width': '280', 'float': 'left', 'justify-content': 'center', 'margin-right':'2em'}) # bottom of H3 / no flex

@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    if not value:
        return dropdown_select("year")
    return dropdown_select(value)

# dd_count = html.Div(children =[
#     dcc.Dropdown(['year', 'state', 'manufacturer', 'model', 'condition', 'odometer', 'fuel', 'size', 'type', 'cylinders', 'drive', 'transmission', 'paint_color', 'price_range','year_range', 'odometer_range'], 
#     id='count-dropdown', placeholder='Select an attribute...', style={'width': '80%'}),
#     html.Div(children =[
#     html.Div(id='count-output-container',style={'float': 'left', 'display': 'flex', 'justify-content': 'center'})]),],
#     style={'height': 380,'width': '280', 'float': 'left', 'justify-content': 'center', 'margin-right':'2em'})

# @app.callback(
#     Output('count-output-container', 'children'),
#     Input('count-dropdown', 'value')
# )
# def update_output(value):
#     if not value:
#         return dropdown_count("year")
#     return dropdown_count(value)

marital_status_vis = html.Div(style={'margin-left':'2em', 'margin-right':'2em'},
    children =[
            html.Div([
            html.H3("Prices:", style=H3_style),
            html.Div(children =[
                # html.H3("Prices:", style=H3_style),
                dcc.Graph(
                id = "marital status",
                figure = plot_pie('price_range')
            )],
            style={'height': 380,'width': '280', 'float': 'left', 'display': 'flex', 'justify-content': 'center', 'margin-right':'2em'}),

            html.H3("States Overview:", style=H3_style),
            html.Div(children =[
                # html.H3("States Overview:", style=H3_style),
                dcc.Graph(
                id = "marital prob",
                figure = state_average()
            )],
            style={'height': 380,'width': '280', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.H3("Price Trendency:", style=H3_style),
            dropdown_plot,

            html.H3("Years:", style=H3_style),
            html.Div(children =[
                # html.H3("Prices:", style=H3_style),
                dcc.Graph(
                id = "marital status",
                figure = plot_pie('year_range')
            )],
            style={'height': 380,'width': '280', 'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),

            html.H3("Next...:", style=H3_style),
            html.Div(children =[
                # html.H3("Prices:", style=H3_style),
                dcc.Graph(
                id = "marital status",
                figure = plot_pie('year_range')
            )],
            style={'height': 380,'width': '280', 'float': 'left', 'display': 'flex', 'justify-content': 'center', 'margin-right':'2em'}),

            html.H3("Next...:", style=H3_style),
            html.Div(children =[
                # html.H3("Prices:", style=H3_style),
                dcc.Graph(
                id = "marital status",
                figure = plot_pie('year_range')
            )],
            style={'height': 380,'width': '280', 'float': 'bottom', 'display': 'flex', 'justify-content': 'center'}),

            ]),
            # dropdown_plot,
        ])
        
layout_whole = html.Div(children = [marital_status_vis, educational_Level_vis, dropdown_plot, contact_vis])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return marital_status_vis #layout_whole
    elif tab == "tab-new":
        return layout_tab_new

server = app.server

if __name__ == '__main__':
    # model = joblib.load("LR_prediction.joblib")
    #app.run_server(debug=True)
    #application.run_server(host='0.0.0.0', port=8050, debug=True)
    #application.run(debug=True, port=8080)
    #application.run_server(host='0.0.0.0')
    app.run_server(host="0.0.0.0")