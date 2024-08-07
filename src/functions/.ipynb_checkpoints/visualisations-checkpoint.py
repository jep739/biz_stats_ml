import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
import getpass
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import sgis as sg
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from pyjstat import pyjstat
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, Dropdown, interactive_output, Button
from IPython.display import display, clear_output
import ipywidgets as widgets



import sys

sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
import kommune_translate
import kommune

fs = FileClient.get_gcs_file_system()
import numpy as np


import warnings

warnings.filterwarnings("ignore")


def gather_visualisation_data(year):

    # Fetch and process time series data similar to the steps above
    fil_path = [f for f in fs.glob(f"gs://ssb-prod-noeku-data-produkt/temp/timeseries_knn.parquet") if f.endswith(".parquet")]
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()
    timeseries_knn_agg = table.to_pandas()
    timeseries_knn_agg['n2'] = timeseries_knn_agg['n3'].str[:2]
    timeseries_knn_agg['lonn_pr_syss'] = timeseries_knn_agg['lonn'] / timeseries_knn_agg['syss']
    timeseries_knn_agg['oms_pr_syss'] = timeseries_knn_agg['oms'] / timeseries_knn_agg['syss']

    # Repeat for other data sources
    fil_path = [f for f in fs.glob(f"gs://ssb-prod-noeku-data-produkt/temp/knn_varehandel_cleaned.parquet") if f.endswith(".parquet")]
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()
    knn_data = table.to_pandas()

    # Process and filter histogram data
    histogram_data = knn_data.copy()
    histogram_data['n2'] = histogram_data['n3'].str[:2]
    histogram_data = histogram_data[(histogram_data['n2'] == '45') | (histogram_data['n2'] == '46') | (histogram_data['n2'] == '47')]

    # Fetch and process municipality level time series data
    fil_path = [f for f in fs.glob(f"gs://ssb-prod-noeku-data-produkt/temp/timeseries_knn_kommune.parquet") if f.endswith(".parquet")]
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()
    timeseries_knn_kommune = table.to_pandas()
    timeseries_knn_kommune['lonn_pr_syss'] = timeseries_knn_kommune['lonn'] / timeseries_knn_kommune['syss']
    timeseries_knn_kommune['oms_pr_syss'] = timeseries_knn_kommune['oms'] / timeseries_knn_kommune['syss']
    
    # get coordinates:
    
    koordinates = kommune.get_coordinates(knn_data)
    
    koordinates['n2'] = koordinates['n3'].str[:2]

    # filter histogram data so that n2 is only 45, 46 or 47
    koordinates = koordinates[(koordinates['n2'] == '45') | (koordinates['n2'] == '46') | (koordinates['n2'] == '47')]

    # Attempt to convert 'b_sysselsetting_syss' to numeric, coercing errors to NaN
    koordinates['b_sysselsetting_syss'] = pd.to_numeric(koordinates['b_sysselsetting_syss'], errors='coerce')


    # Return all prepared DataFrames for visualization or further analysis
    return timeseries_knn_kommune, histogram_data, knn_data, timeseries_knn_agg, koordinates

#timeseries_knn_agg
def plots_time(df):

    def plot_variable(n3, variable, chart_type):
        filtered_df = df[df['n3'] == n3]

        if chart_type == 'Line Chart':
            fig = px.line(filtered_df, x='year', y=variable, markers=True, title=f'{variable} over Years for {n3}')
        elif chart_type == 'Bar Chart':
            fig = px.bar(filtered_df, x='year', y=variable, title=f'{variable} over Years for {n3}')
        elif chart_type == 'Scatter Plot':
            fig = px.scatter(filtered_df, x='year', y=variable, title=f'{variable} over Years for {n3}')

        fig.update_layout(xaxis_title='Year',
                          yaxis_title=variable,
                          template='plotly_white',
                          xaxis=dict(showgrid=True),
                          yaxis=dict(showgrid=True))
        fig.show()

    # Dropdown menu for the type of chart
    chart_type_selector = Dropdown(options=['Line Chart', 'Bar Chart', 'Scatter Plot'], value='Line Chart', description='Chart Type:')

    # Interactive widget setup
    interact(plot_variable, 
             n3=sorted(df['n3'].unique()), 
             variable=['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'], 
             chart_type=chart_type_selector)


#timeseries_knn_agg
def plot_all_time(df):

    def plot_all_n3(variable, chart_type):
        if chart_type == 'Line Chart':
            fig = px.line(df, x='year', y=variable, color='n3', markers=True,
                          title=f'Trend of {variable} over Years for all n3 categories')
        elif chart_type == 'Bar Chart':
            fig = px.bar(df, x='year', y=variable, color='n3',
                         title=f'Trend of {variable} over Years for all n3 categories')
        elif chart_type == 'Area Chart':
            wide_df = df.pivot_table(index='year', columns='n3', values=variable, aggfunc='sum').fillna(0)
            fig = px.area(wide_df, labels={'value': variable, 'year': 'Year'},
                          title=f'Trend of {variable} over Years for all n3 categories')

        # Adjust the layout
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title=variable,
            template='plotly_white',
            height=800,  # Increased height to accommodate legend
            width=1000,  # Increased width to accommodate legend
            legend_title='n3',
            legend=dict(
                x=1,  # Adjust the x position of the legend (1 is at the right end of the plot area)
                xanchor='auto',  # Anchor point for the legend x position
                y=1,  # Adjust the y position of the legend (1 is at the top of the plot area)
                yanchor='auto',  # Anchor point for the legend y position
                tracegroupgap=0,
                title_font=dict(size=14),
                font=dict(size=12, color="black"),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=1
            )
        )

        # Optionally add an orientation to the legend if needed
        # fig.update_layout(legend_orientation="h")

        fig.show()

    chart_type_selector = Dropdown(options=['Line Chart', 'Bar Chart', 'Area Chart'], value='Line Chart', description='Chart Type:')
    interact(plot_all_n3, 
             variable=['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'],
             chart_type=chart_type_selector)


#timeseries_knn_agg
def plot_n2(df):   
    def plot_n3_by_n2(n2, chart_type):
        # Filter the DataFrame based on selected 'n2'
        filtered_df = df[df['n2'] == n2]

        # Create the plot according to the selected chart type
        if chart_type == 'Line Chart':
            fig = px.line(filtered_df, x='year', y='oms', color='n3', markers=True,
                          title=f'OMS over Years for n2={n2}')
        elif chart_type == 'Bar Chart':
            fig = px.bar(filtered_df, x='year', y='oms', color='n3',
                         title=f'OMS over Years for n2={n2}')
        elif chart_type == 'Area Chart':
            # Pivot data for area chart
            wide_df = filtered_df.pivot(index='year', columns='n3', values='oms').fillna(0)
            fig = px.area(wide_df, title=f'OMS over Years for n2={n2}')

        # Update layout for prettiness and interactivity
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='OMS',
            legend_title='n3',
            template='plotly_white',
            height=600,  # Height of the plot
            width=950   # Width of the plot
        )
        fig.update_xaxes(tickangle=-45)  # Improve x-axis label readability
        fig.show()

    # Dropdown menu for the type of chart
    chart_type_selector = Dropdown(options=['Line Chart', 'Bar Chart', 'Area Chart'], value='Line Chart', description='Chart Type:')
    n2_selector = Dropdown(options=sorted(df['n2'].unique()), description='Select n2:')

    # Interactive widget setup
    interact(plot_n3_by_n2, 
             n2=n2_selector,
             chart_type=chart_type_selector)
    
#timeseries_knn_agg
def heatmap(df):

    def plot_heatmap(n2, variable):
        # Filter data for the selected n2 category
        filtered_df = df[df['n2'] == n2]

        # Pivot data to get 'year' as index and 'n3' as columns
        heatmap_data = filtered_df.pivot(index='year', columns='n3', values=variable)

        # Create a heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlGnBu',
            hoverongaps=False,
            text=heatmap_data.values,
            hoverinfo="text",  # Show the values on hover
            texttemplate="%{text}",  # Display text in each cell
            showscale=True  # Optional: turn on/off the color scale bar
        ))

        # Update layout for a better visual experience
        fig.update_layout(
            title=f'Heatmap of {variable} over time for all n3 under n2={n2}',
            xaxis_title='n3 Categories',
            yaxis_title='Year',
            template='plotly_white',
            height=600,
            width=800,
            xaxis_nticks=len(heatmap_data.columns),
            yaxis_nticks=len(heatmap_data.index)
        )

        fig.show()

    # Dropdown menu for selecting n2 category and the variable to display
    n2_selector = Dropdown(options=sorted(df['n2'].unique()), description='Select n2:')
    variable_selector = Dropdown(options=['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'], 
                                 value='oms', description='Variable:')

    # Interactive widget setup
    interact(plot_heatmap, n2=n2_selector, variable=variable_selector)

def thematic_kommune(df):
    
    # Convert the 'year' column to int
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    
    def update_map(variable, naring, year):
        
        # Filter the main DataFrame for the selected year and possibly other conditions
        filtered_data = df[df['year'] == year]
        
        # Filter the DataFrame for the selected year and other conditions
        kommuner = kommune.kommune(variable, naring, year, filtered_data)
        # Create and display the thematic map
        m = sg.ThematicMap(kommuner, column=variable, size=15)
        m.title = "Selected Business Metric by Kommune"
        m.plot()

    # Define the interactive map function
    def interactive_map(variable, naring, year):
        clear_output(wait=True)
        update_map(variable, naring, year)

    # Create dropdown widgets
    year_dropdown = widgets.Dropdown(
        options=sorted(df['year'].unique()), 
        description='Year:'
    )
    naring_dropdown = widgets.Dropdown(
        options=sorted(df['n3'].unique()), 
        description='Naring:'
    )
    column_dropdown = widgets.Dropdown(
        options=['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'],
        description='Variable:'
    )

    # Interactive widget to control the map
    @widgets.interact(variable=column_dropdown, naring=naring_dropdown, year=year_dropdown)
    def interactive_map(variable, naring, year):
        clear_output(wait=True)
        update_map(variable, naring, year)

    display(interactive_map)

              
# timeseries_knn_kommune
def animated_thematic_kommune(df):

    # Convert the 'year' column to int
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    def update_map(variable, naring, year):
        # Filter the main DataFrame for the selected year and possibly other conditions
        filtered_data = df[df['year'] == year]

        kommuner = kommune.kommune(variable, naring, year, filtered_data)

        # Create a new map instance with updated data
        m = sg.ThematicMap(kommuner, column=variable, size=15)
        m.title = "Selected Business Metric by Kommune Through the Years"
        m.plot()

    # Widgets
    year_slider = widgets.IntSlider(
        value=min(df['year']),
        min=min(df['year']),
        max=max(df['year']),
        step=1,
        description='Year:',
        continuous_update=False
    )

    play = widgets.Play(
        value=min(df['year']),
        min=min(df['year']),
        max=max(df['year']),
        step=1,
        interval=1000,  # Change interval to control speed (in milliseconds)
        description="Press play",
    )

    widgets.jslink((play, 'value'), (year_slider, 'value'))  # Link play and slider

    naring_dropdown = widgets.Dropdown(
        options=sorted(df['n3'].unique()),
        description='Næring:'
    )

    column_dropdown = widgets.Dropdown(
        options=['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'],
        description='Variable:'
    )

    ui = widgets.HBox([play, year_slider])  # Arrange widgets horizontally

    # Interactive widget to control the map
    @widgets.interact(variable=column_dropdown, naring=naring_dropdown)
    def interactive_map(variable, naring, year=year_slider):
        clear_output(wait=True)
        update_map(variable, naring, year)

    display(ui)  # Display the play button and slider together

    
# histogram_data
def cumulative_histogram(df):


    def update_plot(variable, naring, source_data):
        # Filter and prepare data
        data = source_data[source_data['n3'] == naring]
        sorted_data = data.sort_values(by=variable, ascending=False).reset_index(drop=True)
        sorted_data['cumulative'] = sorted_data[variable].cumsum()
        sorted_data['cumulative_pct'] = 100 * sorted_data['cumulative'] / sorted_data[variable].sum()
        sorted_data['rank'] = range(1, len(sorted_data) + 1)

        # Plot configuration
        fig = go.Figure()

        # Add bar chart for variable values
        fig.add_trace(go.Bar(
            x=sorted_data['rank'],
            y=sorted_data[variable],
            name=f'{variable} per Business',
            marker=dict(color='blue'),
            yaxis='y2'
        ))

        # Add line chart for cumulative percentage
        fig.add_trace(go.Scatter(
            x=sorted_data['rank'],
            y=sorted_data['cumulative_pct'],
            name='Cumulative %',
            marker=dict(color='red'),
            mode='lines+markers'
        ))

        # Layout with dual y-axes and larger size
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
                showgrid=False,  # Optionally hide the gridlines for the secondary y-axis
            ),
            hovermode='x',
            width=900,  # Width of the plot in pixels
            height=600   # Height of the plot in pixels
        )

        fig.show()

    # Widgets for interactive selection
    variable_selector = widgets.Dropdown(
        options=['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'],
        value='oms',
        description='Variable:'
    )

    sorted_n3_options = sorted(df['n3'].unique())

    naring_selector = widgets.Dropdown(
        options=sorted_n3_options,
        description='Naring:'
    )

    # Integration with widgets
    interact(update_plot,
             variable=variable_selector,
             naring=naring_selector,
             source_data=widgets.fixed(df))
    
    
# timeseries_knn_agg
def linked_plots(df):

    def create_linked_plots(naring, source_data):
        # Filter data for the selected naring
        data = source_data[source_data['n3'] == naring]

        variables = ['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss']

        # Create subplots with 1 row and the number of variables columns, initially not linked
        rows = (len(variables) + 2) // 3  # Calculate rows needed for 3 columns layout
        fig = make_subplots(rows=rows, cols=3, subplot_titles=variables)

        # Position tracking for subplot placement
        row = 1
        col = 1

        for index, var in enumerate(variables):
            # Plot each variable in a subplot
            fig.add_trace(
                go.Scatter(x=data['year'], y=data[var], mode='lines+markers', name=var),
                row=row, col=col
            )

            # Advance subplot position
            col += 1
            if col > 3:
                col = 1
                row += 1

        # Ensure all x-axes are linked by setting all to the domain of the first x-axis
        for i in range(1, rows+1):
            for j in range(1, 4):
                axis_name = f'xaxis{3*(i-1)+j}'  # generate x-axis name dynamically
                fig.layout[axis_name].update(matches='x1')

        # Update layout to adjust the appearance and add hover data
        fig.update_layout(
            height=300 * rows,  # Adjust height based on the number of rows
            width=800,
            hovermode='closest',
            title_text=f"Data Overview for {naring}",
            showlegend=False
        )

        fig.show()

    # Widgets for interactive selection
    naring_selector = widgets.Dropdown(
        options=sorted(df['n3'].unique()),  
        description='Naring:'
    )

    def update_plot(naring):
        create_linked_plots(naring, df)

    interact(update_plot, naring=naring_selector)

# timeseries_knn_agg
def parallel_coordinates(df):

    # Map 'n3' categories to a numerical scale
    n3_unique = sorted(df['n3'].unique())
    n3_to_num = {n3: i for i, n3 in enumerate(n3_unique)}
    df['n3_num'] = df['n3'].map(n3_to_num)

    # Define colors for each unique number mapped from 'n3'
    colors = px.colors.qualitative.Set3  # Adjust the color set based on the number of unique n3 categories
    color_scale = {num: color for num, color in zip(n3_to_num.values(), colors)}

    def plot_parallel_coordinates(selected_year):
        # Filter data for the selected year
        filtered_data = df[df['year'] == selected_year]

        # Create parallel coordinates plot with numerical 'n3' mapped to colors
        fig = px.parallel_coordinates(
            filtered_data,
            dimensions=['forbruk', 'oms', 'drkost', 'salgsint', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss'],
            color='n3_num',  # Use the numerical 'n3' mapping for coloring
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
            color_continuous_scale=colors,  # Use the defined color scale
            title=f"Parallel Coordinates Plot for Year: {selected_year}"
        )

        # Adjust the plot height
        fig.update_layout(
            height=800,  # Set the height of the plot (in pixels)
            coloraxis_colorbar=dict(
                tickvals=list(n3_to_num.values()),
                ticktext=list(n3_to_num.keys())
            )
        )

        fig.show()

    # Widget for selecting the year
    year_selector = widgets.Dropdown(
        options=sorted(df['year'].unique()),
        value=sorted(df['year'].unique())[0],
        description='Select Year:',
        disabled=False
    )

    # Set up interaction
    interact(plot_parallel_coordinates, selected_year=year_selector)

# koordinates
def geomapping(df):

    # Initialize the map update function with the slider range
    def update_map(variable, naring, syssel_range):
        filtered_df = df[(df['n3'] == naring) & (df['b_sysselsetting_syss'] >= syssel_range[0]) & (df['b_sysselsetting_syss'] <= syssel_range[1])]
        sg.explore(filtered_df, variable)

    # Define the slider
    syssel_slider = widgets.IntRangeSlider(
        value=[df['b_sysselsetting_syss'].min(), df['b_sysselsetting_syss'].max()],
        min=df['b_sysselsetting_syss'].min(),
        max=df['b_sysselsetting_syss'].max(),
        step=1,
        description='Employment Range:',
        continuous_update=False
    )

    # Define dropdowns
    naring_dropdown = widgets.Dropdown(
        options=sorted(df['n3'].unique()),
        description='naring:'
    )
    column_dropdown = widgets.Dropdown(
        options=['oms', 'bedr_forbruk', 'bedr_salgsint', 'new_drkost', 'b_sysselsetting_syss'],
        description='Variable:'
    )

    # Setup the interactive map widget
    @widgets.interact(variable=column_dropdown, naring=naring_dropdown, syssel_range=syssel_slider)
    def interactive_map(variable, naring, syssel_range):
        clear_output(wait=True)
        update_map(variable, naring, syssel_range)

        

# timeseries_knn_kommune
def bubble_plot(df):


    # Numerical columns available for selection
    num_columns = ['oms', 'forbruk', 'salgsint', 'drkost', 'lonn', 'syss', 'resultat', 'lonn_pr_syss', 'oms_pr_syss']

    # Adding 'All' option to the filters
    years = ['All'] + list(df['year'].unique())
    kommunenrs = ['All'] + list(df['kommunenr'].unique())
    n3s = ['All'] + list(df['n3'].unique())

    # Interactive widgets
    year_widget = widgets.SelectMultiple(
        options=years,
        value=['All'],
        description='Year:',
        disabled=False
    )
    kommunenr_widget = widgets.SelectMultiple(
        options=kommunenrs,
        value=['All'],
        description='Kommunenr:',
        disabled=False
    )
    n3_widget = widgets.SelectMultiple(
        options=n3s,
        value=['All'],
        description='NACE Code:',
        disabled=False
    )

    x_axis_widget = widgets.Dropdown(
        options=num_columns,
        value='oms_pr_syss',
        description='X Axis:',
        disabled=False
    )
    y_axis_widget = widgets.Dropdown(
        options=num_columns,
        value='resultat',
        description='Y Axis:',
        disabled=False
    )
    size_widget = widgets.Dropdown(
        options=num_columns,
        value='syss',
        description='Bubble Size:',
        disabled=False
    )

    # Toggle button for zoom
    toggle_button = Button(description='Activate Zoom')

    # Function to update the plot
    def plot_bubble(years, kommunenrs, n3s, x_axis, y_axis, size, toggle_button):
        df_filtered = df[
            ((df['year'].isin(years)) | ('All' in years)) &
            ((df['kommunenr'].isin(kommunenrs)) | ('All' in kommunenrs)) &
            ((df['n3'].isin(n3s)) | ('All' in n3s))
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df_filtered[x_axis],
            df_filtered[y_axis],
            s=df_filtered[size] * 10,  # Adjust bubble size scale as necessary
            c=df_filtered[size],
            cmap='viridis',
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5
        )
        plt.colorbar(scatter, label='Number of Employees')
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel(y_axis.replace('_', ' ').title())
        ax.set_title(f'Interactive Bubble Chart: {y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}')
        ax.grid(True)

        # Rectangle selector for zoom functionality
        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            ax.set_xlim(min(x1, x2), max(x1, x2))
            ax.set_ylim(min(y1, y2), max(y1, y2))
            plt.draw()

        selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)

        def toggle_selector(event):
            if toggle_button.description == 'Activate Zoom':
                selector.set_active(True)
                toggle_button.description = 'Deactivate Zoom'
            else:
                selector.set_active(False)
                toggle_button.description = 'Activate Zoom'

        toggle_button.on_click(toggle_selector)

        plt.show()

    # Create interactive output
    output = interactive_output(plot_bubble, {
        'years': year_widget, 
        'kommunenrs': kommunenr_widget, 
        'n3s': n3_widget, 
        'x_axis': x_axis_widget, 
        'y_axis': y_axis_widget, 
        'size': size_widget,
        'toggle_button': widgets.fixed(toggle_button)
    })

    # Display widgets and output
    ui = widgets.VBox([widgets.HBox([year_widget, kommunenr_widget, n3_widget]), 
                       widgets.HBox([x_axis_widget, y_axis_widget, size_widget]),
                       toggle_button])
    display(ui, output)

    

# timeseries_knn_agg
import plotly.express as px
from IPython.display import display
import ipywidgets as widgets

def animated_barchart(df):

    # Function to compute and sort data based on the ranks
    def prepare_data(df, value_column):
        # Compute ranks within each year group
        df['rank'] = df.groupby('year')[value_column].rank("dense", ascending=False)
        # Sort by year and rank for correct plotting order
        return df.sort_values(by=['year', 'rank'], ascending=[True, True])

    # Create a color map for each unique 'n3' value
    color_map = {n3: f"#{hash(n3) & 0xFFFFFF:06x}" for n3 in df['n3'].unique()}

    # Dropdown widget for selecting the numerical variable
    dropdown = widgets.Dropdown(
        options=[col for col in df.columns if col not in ['year', 'n3']],
        value='oms',  # Default selection
        description='Variable:',
        disabled=False,
    )

    # Initial plotting function
    def initial_plot(value_column):
        ranked_df = prepare_data(df.copy(), value_column)
        fig = px.bar(
            ranked_df,
            x=value_column,
            y='n3',  # Use 'n3' for y-axis
            animation_frame='year',
            range_x=[0, ranked_df[value_column].max() + 10],
            color='n3',
            color_discrete_map=color_map,
            orientation='h',
            height=700,  # Increased height for better visibility
            width=1200   # Increased width
        )
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
        fig.update_yaxes(categoryorder='total ascending')  # Ensure y-axis categories are sorted by total ascending
        return fig

    # Update function for the dropdown
    def update_graph(change):
        # Clear the current output and display the updated plot
        output.clear_output(wait=True)
        with output:
            fig = initial_plot(change.new)
            fig.show()

    # Observe changes in the dropdown
    dropdown.observe(update_graph, names='value')

    # Display the dropdown and the initial plot
    output = widgets.Output()
    display(dropdown)
    with output:
        fig = initial_plot(dropdown.value)
        fig.show()
    display(output)



#timeseries_knn_agg
def scatter_3d(df):

    # Add a 'profit' column based on the 'resultat' column

    df['profit'] = df['resultat'] > 0


    # Function to plot the 3D scatter plot for a selected year
    def plot_3d_scatter(selected_year):
        # Filter the DataFrame for the selected year
        filtered_data = df[df['year'] == selected_year]

        # Create the 3D scatter plot, adjust size and size_max for better visibility
        fig = px.scatter_3d(filtered_data, x='drkost', y='oms', z='lonn_pr_syss',
                            color='n3',  # Color by industry code
                            symbol='profit',  # Use the profit symbol
                            size='syss',  # Size by number of employees
                            # size_max=50,  # Max size of markers
                            title=f"3D Scatter of Turnover, Consumption, and Wages by Industry for {selected_year}")

        # Update layout to adjust the size of the plot
        fig.update_layout(
            width=1000,  # Width of the plot in pixels
            height=800,  # Height of the plot in pixels
            margin=dict(l=10, r=10, b=10, t=30)  # Adjust margins if needed
        )

        # # Optionally, you can manually adjust marker sizes in the traces
        fig.update_traces(marker=dict(size=filtered_data['syss'] * 10))  # Scale marker size

        fig.show()

    # Create a dropdown for year selection
    year_dropdown = widgets.Dropdown(
        options=sorted(df['year'].unique()),
        description='Select Year:',
        disabled=False
    )

    # Display the interactive plot with the dropdown
    interact(plot_3d_scatter, selected_year=year_dropdown)
    
    
    
def guage(df):
    
    import plotly.graph_objects as go
    from ipywidgets import interact, Dropdown

    # Function to determine the color based on the percentage difference from the target
    def determine_color(current, target):
        diff_percent = ((current - target) / target) * 100
        if diff_percent > 0.5:
            return "red"
        elif diff_percent >= -2:
            return "green"
        else:
            return "yellow"

    # Function to create gauge charts with overflow indication
    def create_gauge_with_overflow(n3_f, variable):
        df_filtered = df[df['n3_f'] == n3_f]
        if variable == 'oms':
            target = df_filtered['foretak_omsetning'].values[0]
            current = df_filtered['oms'].values[0]
        elif variable == 'new_drkost':
            target = df_filtered['foretak_driftskostnad'].values[0]
            current = df_filtered['new_drkost'].values[0]
        elif variable == 'bedr_forbruk':
            target = df_filtered['forbruk'].values[0]
            current = df_filtered['bedr_forbruk'].values[0]
        elif variable == 'bedr_salgsint':
            target = df_filtered['salgsint'].values[0]
            current = df_filtered['bedr_salgsint'].values[0]

        color = determine_color(current, target)
        # overflow = current > target
        overflow = (current - target) / target > 0.02
        overflow_value = current - target if overflow else 0

        steps = [
            {'range': [0, 0.2 * target], 'color': "red"},
            {'range': [0.2 * target, 0.4 * target], 'color': "orange"},
            {'range': [0.4 * target, 0.6 * target], 'color': "yellow"},
            {'range': [0.6 * target, 0.8 * target], 'color': "lightgreen"},
            {'range': [0.8 * target, target], 'color': "green"}
        ]

        if overflow:
            steps.append({'range': [target, target * 1.5], 'color': "darkred"})

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current,
            title={'text': f"{n3_f} - {variable} (Total: {target:,})"},
            gauge={
                'axis': {'range': [0, target * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color, 'thickness': 0.2},  # Adjust thickness for a better pointer look
                'steps': steps,
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': target
                }
            }
        ))

        if overflow:
            fig.add_annotation(x=0.5, y=0.1,
                               text=f"Overflow: {overflow_value:,}",
                               showarrow=False,
                               font=dict(size=12, color="red"))

        fig.update_layout(
            font={'size': 18},
            margin=dict(t=100, b=50, l=50, r=50),  # Adjust margins for a cleaner look
        )
        fig.show()

    # Dropdown widgets for interactivity
    n3_f_dropdown = Dropdown(options=df['n3_f'].unique(), description='n3_f')
    variable_dropdown = Dropdown(options=['oms', 'new_drkost', 'bedr_forbruk', 'bedr_salgsint'], description='Variable')

    # Interactive widget
    interact(create_gauge_with_overflow, n3_f=n3_f_dropdown, variable=variable_dropdown)

    
    
def thermometer(df):
    
    # Function to determine the color based on the percentage difference from the target
    def determine_color(current, target):
        diff_percent = ((current - target) / target) * 100
        if diff_percent > 0.5:
            return "red"
        elif diff_percent >= -2:
            return "green"
        else:
            return "yellow"

    # Function to create thermometer charts for a specific n3_f
    def create_thermometer_charts(n3_f):
        df_filtered = df[df['n3_f'] == n3_f]
        metrics = ['oms', 'new_drkost', 'bedr_forbruk', 'bedr_salgsint']
        targets = {
            'oms': df_filtered['foretak_omsetning'].values[0],
            'new_drkost': df_filtered['foretak_driftskostnad'].values[0],
            'bedr_forbruk': df_filtered['forbruk'].values[0],
            'bedr_salgsint': df_filtered['salgsint'].values[0]
        }

        fig = go.Figure()

        for i, metric in enumerate(metrics):
            target = targets[metric]
            current = df_filtered[metric].values[0]
            color = determine_color(current, target)

            fig.add_trace(go.Bar(
                x=[metric],
                y=[current],
                marker_color=color,
                name=f"{metric} (Current)",
                hoverinfo='y+text',
                text=f"Current: {current}<br>Target: {target}",
                textposition='auto',
                width=0.3
            ))

            # Add target line
            fig.add_trace(go.Scatter(
                x=[metric],
                y=[target],
                mode='markers+text',
                marker=dict(size=10, color='black'),
                name=f"{metric} (Target)",
                text=[f"Target: {target}"],
                textposition='top center',
                showlegend=False
            ))

        fig.update_layout(
            title_text=f"Thermometer Chart for {n3_f}",
            barmode='group',
            yaxis=dict(range=[0, max(targets.values()) * 1.1]),
            height=500,
            width=800,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        fig.show()

    # Dropdown widget for interactivity
    n3_f_dropdown = Dropdown(options=df['n3_f'].unique(), description='n3_f')

    # Interactive widget
    interact(create_thermometer_charts, n3_f=n3_f_dropdown)