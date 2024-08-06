import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import os
import plotly.express as px
import plotly.graph_objects as go
import sgis as sg
import sys
import geopandas as gpd
import json
from plotly.subplots import make_subplots

sys.path.append("../functions")
import kommune
import visualisations 

def run_dash_app(aar, timeseries_knn_kommune, histogram_data, knn_data, timeseries_knn_agg, koordinates):

    # Load data
    aar = 2021
    # timeseries_knn_kommune, histogram_data, knn_data, timeseries_knn_agg, foretak_varendel, foretak_pub, koordinates = visualisations.gather_visualisation_data(aar)

    # Getting environment variables for proper integration with JupyterHub
    port = 8063
    service_prefix = os.getenv("JUPYTERHUB_SERVICE_PREFIX", "/")
    domain = os.getenv("JUPYTERHUB_HTTP_REFERER", None)

    # Define plotting functions (as before)
    def plot_time_dash(df, n3, variable, chart_type):
        filtered_df = df[df['n3'] == n3]
        fig = None
        if chart_type == 'Line Chart':
            fig = px.line(filtered_df, x='year', y=variable, title=f'{variable} over Years for {n3}')
        elif chart_type == 'Bar Chart':
            fig = px.bar(filtered_df, x='year', y=variable, title=f'{variable} over Years for {n3}')
        elif chart_type == 'Scatter Plot':
            fig = px.scatter(filtered_df, x='year', y=variable, title=f'{variable} over Years for {n3}')
        fig.update_layout(xaxis_title='Year', yaxis_title=variable, template='plotly_white')
        return fig  

    def plot_n2_dash(df, n2, variable, chart_type):
        filtered_df = df[df['n2'] == n2]
        fig = None
        if chart_type == 'Line Chart':
            fig = px.line(filtered_df, x='year', y=variable, color='n3', markers=True, title=f'{variable} over Years for n2={n2}')
        elif chart_type == 'Bar Chart':
            fig = px.bar(filtered_df, x='year', y=variable, color='n3', title=f'{variable} over Years for n2={n2}')
        elif chart_type == 'Area Chart':
            wide_df = filtered_df.pivot(index='year', columns='n3', values=variable).fillna(0)
            fig = px.area(wide_df, title=f'{variable} over Years for n2={n2}')
        fig.update_layout(xaxis_title='Year', yaxis_title=variable, legend_title='n3', template='plotly_white')
        return fig

    def plot_heatmap_dash(df, n2, variable):
        filtered_df = df[df['n2'] == n2]
        heatmap_data = filtered_df.pivot(index='year', columns='n3', values=variable)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlGnBu',
            hoverongaps=False,
            text=heatmap_data.values,
            hoverinfo="text",
            texttemplate="%{text}",
            showscale=True
        ))
        fig.update_layout(
            title=f'Heatmap of {variable} over time for all n3 under n2={n2}',
            xaxis_title='n3 Categories',
            yaxis_title='Year',
            template='plotly_white'
        )
        return fig

    def cumulative_histogram(df, variable, naring):
        data = df[df['n3'] == naring]
        sorted_data = data.sort_values(by=variable, ascending=False).reset_index(drop=True)
        sorted_data['cumulative'] = sorted_data[variable].cumsum()
        sorted_data['cumulative_pct'] = 100 * sorted_data['cumulative'] / sorted_data[variable].sum()
        sorted_data['rank'] = range(1, len(sorted_data) + 1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_data['rank'],
            y=sorted_data[variable],
            name=f'{variable} per Business',
            marker=dict(color='blue'),
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=sorted_data['rank'],
            y=sorted_data['cumulative_pct'],
            name='Cumulative %',
            marker=dict(color='red'),
            mode='lines+markers'
        ))
        fig.update_layout(
            title=f"Cumulative Distribution of {variable} for {naring}",
            xaxis_title="Business Rank",
            yaxis=dict(
                title="Cumulative Percentage (%)",
                range=[0, 100]
            ),
            yaxis2=dict(
                title=f"{variable} Value",
                overlaying='y',
                side='right',
                showgrid=False,
            ),
            hovermode='x'
        )
        return fig

    def linked_plots_dash(df, naring):
        data = df[df['n3'] == naring]
        variables = ['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss']
        rows = (len(variables) + 2) // 3
        fig = make_subplots(rows=rows, cols=3, subplot_titles=variables)
        row, col = 1, 1
        for index, var in enumerate(variables):
            fig.add_trace(
                go.Scatter(x=data['year'], y=data[var], mode='lines+markers', name=var),
                row=row, col=col
            )
            col += 1
            if col > 3:
                col = 1
                row += 1
        fig.update_layout(
            height=300 * rows,
            hovermode='closest',
            title_text=f"Data Overview for {naring}",
            showlegend=False
        )
        return fig

    def parallel_coordinates_dash(df, selected_year):
        n3_unique = sorted(df['n3'].unique())
        n3_to_num = {n3: i for i, n3 in enumerate(n3_unique)}
        df['n3_num'] = df['n3'].map(n3_to_num)
        colors = px.colors.qualitative.Set3
        color_scale = [color for color in colors[:len(n3_to_num)]]
        filtered_data = df[df['year'] == selected_year]

        fig = px.parallel_coordinates(
            filtered_data,
            dimensions=['forbruk', 'oms', 'drkost', 'salgsint', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'],
            color='n3_num',
            labels={
                "forbruk": "Forbruk",
                "oms": "Oms",
                "drkost": "Driftskost",
                "salgsint": "Salgsint",
                "lonn": "Lønn",
                "syss": "Sysselsetting",
                "resultat": "Resultat",
                "lonn_pr_syss": "Lønn per Sysselsetting",
                "oms_pr_syss": "Oms per Sysselsetting"
            },
            color_continuous_scale=color_scale,
            title=f"Parallel Coordinates Plot for Year: {selected_year}"
        )
        fig.update_layout(
            height=800,
            coloraxis_colorbar=dict(
                title='N3 Category',
                tickvals=list(n3_to_num.values()),
                ticktext=list(n3_to_num.keys())
            )
        )
        return fig

    def bubble_plot_dash(df, years, kommunenrs, n3s, x_axis, y_axis, size):
        df_filtered = df[
            (df['year'].isin(years) if years != ['All'] and isinstance(years, list) else df['year'].notnull()) &
            (df['kommunenr'].isin(kommunenrs) if kommunenrs != ['All'] and isinstance(kommunenrs, list) else df['kommunenr'].notnull()) &
            (df['n3'].isin(n3s) if n3s != ['All'] and isinstance(n3s, list) else df['n3'].notnull())
        ]
        fig = px.scatter(
            df_filtered,
            x=x_axis,
            y=y_axis,
            size=size,
            color=size,
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=60,
            title=f"Bubble Plot of {y_axis} vs {x_axis}"
        )
        fig.update_layout(
            xaxis_title=x_axis.replace('_', ' ').title(),
            yaxis_title=y_axis.replace('_', ' ').title()
        )
        return fig

    def animated_barchart_dash(df, value_column):
        df['rank'] = df.groupby('year')[value_column].rank("dense", ascending=False)
        df_sorted = df.sort_values(by=['year', 'rank'], ascending=[True, True])
        color_map = {n3: f"#{hash(n3) & 0xFFFFFF:06x}" for n3 in df['n3'].unique()}
        fig = px.bar(
            df_sorted,
            x=value_column,
            y='n3',
            animation_frame='year',
            range_x=[0, df_sorted[value_column].max() + 10],
            color='n3',
            color_discrete_map=color_map,
            orientation='h'
        )
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
        fig.update_yaxes(categoryorder='total ascending')
        return fig

    def scatter_3d_dash(df, selected_year):
        filtered_data = df[df['year'] == selected_year]
        filtered_data['profit'] = filtered_data['resultat'] > 0
        fig = px.scatter_3d(filtered_data, x='drkost', y='oms', z='lonn_pr_syss',
                            color='n3',
                            symbol='profit',
                            size='syss',
                            title=f"3D Scatter of Turnover, Consumption, and Wages by Industry for {selected_year}")
        fig.update_layout(
            width=1000,
            height=800,
            margin=dict(l=10, r=10, b=10, t=30)
        )
        fig.update_traces(marker=dict(size=filtered_data['syss'] * 10))
        return fig

    def calculate_percentage_change_norge(df, variable, year, n3):
        if year == 'All' or n3 == 'All':
            return 0
        temp = df.copy()
        temp['year'] = temp['year'].astype(int)
        year = int(year)
        previous_year = year - 1

        # Filter the DataFrame for the current and previous year for the given N3 code
        current_year_data = temp[(temp['year'] == year) & (temp['n3'] == n3)]
        previous_year_data = temp[(temp['year'] == previous_year) & (temp['n3'] == n3)]

        # Calculate the total for the given variable in the current and previous year
        current_total = current_year_data[variable].sum()
        previous_total = previous_year_data[variable].sum()

        if previous_total == 0:
            return 0  # To avoid division by zero

        # Calculate the percentage change
        percentage_change = ((current_total - previous_total) / previous_total) * 100

        return percentage_change, current_total


    def calculate_percentage_change(df, variable, year, kommune, n3):
        if year == 'All' or n3 == 'All':
            return 0
        temp_1 = df.copy()
        temp_1['year'] = temp_1['year'].astype(int)
        year = int(year)
        previous_year = year - 1

        # Filter the DataFrame for the current and previous year for the given kommune and N3 code
        current_year_data = temp_1[(temp_1['year'] == year) & (temp_1['kommunenr'] == kommune) & (temp_1['n3'] == n3)]
        previous_year_data = temp_1[(temp_1['year'] == previous_year) & (temp_1['kommunenr'] == kommune) & (temp_1['n3'] == n3)]

        # Calculate the total for the given variable in the current and previous year
        current_total = current_year_data[variable].sum()
        previous_total = previous_year_data[variable].sum()

        if previous_total == 0:
            return 0  # To avoid division by zero

        # Calculate the percentage change
        percentage_change = ((current_total - previous_total) / previous_total) * 100

        return percentage_change, current_total


    # Create a Dash application
    app = dash.Dash(
        __name__,
        requests_pathname_prefix=f"{service_prefix}proxy/{port}/",
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    # Application layout
    app.layout = html.Div([
        # Header
        dbc.NavbarSimple(
            brand="SSB - Statistikk for Varehandel",
            brand_href="#",
            color="green",
            dark=True,
        ),

        # Main content
        dbc.Container([
            # Dropdowns for common filters
            dbc.Row([
                dbc.Col([
                    html.Label('Select Year:'),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': year, 'value': year} for year in sorted(timeseries_knn_agg['year'].unique())],
                        value=sorted(timeseries_knn_agg['year'].unique())[-1],
                        clearable=False
                    ),
                ], width=2),
                dbc.Col([
                    html.Label('Select Kommune:'),
                    dcc.Dropdown(
                        id='kommune-dropdown',
                        options=[{'label': knr, 'value': knr} for knr in sorted(timeseries_knn_kommune['kommunenr'].unique())],
                        value='0301',
                        clearable=False
                    ),
                ], width=2),
                dbc.Col([
                    html.Label('Select NACE Code:'),
                    dcc.Dropdown(
                        id='nace-dropdown',
                        options=[{'label': n, 'value': n} for n in sorted(timeseries_knn_agg['n3'].unique())],
                        value=sorted(timeseries_knn_agg['n3'].unique())[0],
                        clearable=False
                    ),
                ], width=2),
                dbc.Col([
                    html.Label('Select N2 Code:'),
                    dcc.Dropdown(
                        id='n2-dropdown',
                        options=[{'label': n2, 'value': n2} for n2 in sorted(timeseries_knn_agg['n2'].unique())],
                        value='47',
                        clearable=False
                    ),
                ], width=2),
                dbc.Col([
                    html.Label('Select Variable:'),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': var, 'value': var} for var in timeseries_knn_agg.columns if var not in ['year', 'kommunenr', 'n3', 'n2']],
                        value='oms',
                        clearable=False
                    ),
                ], width=2),
            ], className="mb-4"),

            # Key Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Focus metric for Norway", className="card-title"),
                            html.Div(id='key-metric-norge', className="card-text"),
                        ])
                    ], color="light", outline=True),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Focus metric for chosen kommune", className="card-title"),
                            html.Div(id='key-metric-kommune', className="card-text"),
                        ])
                    ], color="light", outline=True),
                ], width=6),
            ], className="mb-4"),

            # Visualizations
            dbc.Row([
                dbc.Col([
                    html.Label('Select Chart Type:'),
                    dcc.Dropdown(
                        id='chart-type-time-dropdown',
                        options=[{'label': 'Line Chart', 'value': 'Line Chart'}, {'label': 'Bar Chart', 'value': 'Bar Chart'}, {'label': 'Scatter Plot', 'value': 'Scatter Plot'}],
                        value='Line Chart',
                        clearable=False
                    ),
                    html.Div([
                        dcc.Graph(id='plotly-time-chart')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=6),
                dbc.Col([
                    html.Label('Select Chart Type:'),
                    dcc.Dropdown(
                        id='chart-type-n2-dropdown',
                        options=[{'label': 'Line Chart', 'value': 'Line Chart'}, {'label': 'Bar Chart', 'value': 'Bar Chart'}, {'label': 'Area Chart', 'value': 'Area Chart'}],
                        value='Bar Chart',
                        clearable=False
                    ),
                    html.Div([
                        dcc.Graph(id='plotly-n2-chart')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='heatmap-graph')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='cumulative-histogram-chart')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='parallel-coordinates-graph')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Select X Axis Variable:'),
                        dcc.Dropdown(
                            id='x-axis-dropdown',
                            options=[{'label': var, 'value': var} for var in timeseries_knn_agg.columns if var not in ['year', 'kommunenr', 'n3', 'n2']],
                            value='lonn',
                            clearable=False
                        ),
                        html.Label('Select Y Axis Variable:'),
                        dcc.Dropdown(
                            id='y-axis-dropdown',
                            options=[{'label': var, 'value': var} for var in timeseries_knn_agg.columns if var not in ['year', 'kommunenr', 'n3', 'n2']],
                            value='resultat',
                            clearable=False
                        ),
                        html.Label('Select Bubble Size Variable:'),
                        dcc.Dropdown(
                            id='size-dropdown',
                            options=[{'label': var, 'value': var} for var in timeseries_knn_agg.columns if var not in ['year', 'kommunenr', 'n3', 'n2']],
                            value='syss',
                            clearable=False
                        ),
                        dcc.Graph(id='bubble-plot-graph')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='animated-barchart-graph')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='3d-scatter-plot')
                    ], style={'background-color': 'white', 'padding': '10px', 'border-radius': '5px'}),
                ], width=12),
            ], className="mb-4"),
        ], fluid=True),

        # Footer
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.Footer("© 2024 SSB", className="text-center"),
                    className="py-4"
                )
            ),
            style={'background-color': 'green', 'color': 'white'},
            fluid=True
        )
    ])

    # Callbacks for updating the graphs and metrics
    @app.callback(
        Output('key-metric-norge', 'children'),
        [Input('variable-dropdown', 'value'), Input('year-dropdown', 'value'), Input('nace-dropdown', 'value')]
    )
    def update_key_metric_norge(variable, year, n3):
        percentage_change, current_total = calculate_percentage_change_norge(timeseries_knn_agg, variable, year, n3)
        arrow = "↑" if percentage_change > 0 else "↓"
        color = "green" if percentage_change > 0 else "red"
        return html.Div([
            html.H3(f"{current_total:,.0f}", style={"font-weight": "bold", "color": "black"}),
            html.P(f"{arrow} {percentage_change:.2f}%", style={"color": color, "font-size": "18px"}),
            html.P(f"Total {variable} for Norway", style={"color": "black"})

        ])

    @app.callback(
        Output('key-metric-kommune', 'children'),
        [Input('variable-dropdown', 'value'), Input('year-dropdown', 'value'), Input('kommune-dropdown', 'value'), Input('nace-dropdown', 'value')]
    )
    def update_key_metric_kommune(variable, year, kommune, n3):
        percentage_change, current_total = calculate_percentage_change(timeseries_knn_kommune, variable, year, kommune, n3)
        arrow = "↑" if percentage_change > 0 else "↓"
        color = "green" if percentage_change > 0 else "red"
        return html.Div([
            html.H3(f"{current_total:,.0f}", style={"font-weight": "bold", "color": "black"}),
            html.P(f"{arrow} {percentage_change:.2f}%", style={"color": color, "font-size": "18px"}),
            html.P(f"Total {variable} for Kommune {kommune}", style={"color": "black"})
        ])

    @app.callback(
        Output('plotly-time-chart', 'figure'),
        [Input('nace-dropdown', 'value'), Input('variable-dropdown', 'value'), Input('chart-type-time-dropdown', 'value')]
    )
    def update_time_chart(n3, variable, chart_type):
        if not n3 or not variable:
            raise PreventUpdate
        return plot_time_dash(timeseries_knn_agg, n3, variable, chart_type)

    @app.callback(
        Output('plotly-n2-chart', 'figure'),
        [Input('n2-dropdown', 'value'), Input('variable-dropdown', 'value'), Input('chart-type-n2-dropdown', 'value')]
    )
    def update_n2_chart(n2, variable, chart_type):
        return plot_n2_dash(timeseries_knn_agg, n2, variable, chart_type)

    @app.callback(
        Output('heatmap-graph', 'figure'),
        [Input('n2-dropdown', 'value'), Input('variable-dropdown', 'value')]
    )
    def update_heatmap(n2, variable):
        if not n2 or not variable:
            raise PreventUpdate
        return plot_heatmap_dash(timeseries_knn_agg, n2, variable)

    @app.callback(
        Output('cumulative-histogram-chart', 'figure'),
        [Input('variable-dropdown', 'value'),
         Input('nace-dropdown', 'value')]
    )
    def update_cumulative_histogram_chart(variable, n3):
        if not variable or not n3:
            raise PreventUpdate
        return cumulative_histogram(histogram_data, variable, n3)

    @app.callback(
        Output('parallel-coordinates-graph', 'figure'),
        [Input('year-dropdown', 'value')]
    )
    def update_parallel_coordinates(selected_year):
        return parallel_coordinates_dash(timeseries_knn_agg, selected_year)

    @app.callback(
        Output('bubble-plot-graph', 'figure'),
        [
            Input('year-dropdown', 'value'),
            Input('kommune-dropdown', 'value'),
            Input('nace-dropdown', 'value'),
            Input('x-axis-dropdown', 'value'),
            Input('y-axis-dropdown', 'value'),
            Input('size-dropdown', 'value')
        ]
    )
    def update_bubble_plot(years, kommunenrs, n3s, x_axis, y_axis, size):
        if 'All' in years:
            years = timeseries_knn_kommune['year'].unique().tolist()
        if 'All' in kommunenrs:
            kommunenrs = timeseries_knn_kommune['kommunenr'].unique().tolist()
        if 'All' in n3s:
            n3s = timeseries_knn_kommune['n3'].unique().tolist()

        if not x_axis or not y_axis or not size:
            raise PreventUpdate

        return bubble_plot_dash(timeseries_knn_kommune, years, kommunenrs, n3s, x_axis, y_axis, size)

    @app.callback(
        Output('animated-barchart-graph', 'figure'),
        [Input('variable-dropdown', 'value')]
    )
    def update_animated_barchart(variable):
        if not variable:
            raise PreventUpdate
        return animated_barchart_dash(timeseries_knn_agg, variable)

    @app.callback(
        Output('3d-scatter-plot', 'figure'),
        [Input('year-dropdown', 'value')]
    )
    def update_3d_scatter(selected_year):
        if not selected_year:
            raise PreventUpdate
        return scatter_3d_dash(timeseries_knn_agg, selected_year)
    
    return app, port, service_prefix, domain

    # if __name__ == "__main__":
    #     app.run(debug=True, port=port, jupyter_server_url=domain, jupyter_mode="tab", use_reloader=False)
