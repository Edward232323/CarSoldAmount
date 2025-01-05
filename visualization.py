import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

class Plotting:
    def __init__(self):
        self.width = 1000
        self.height = 1000


    def get_bar_chart(self, data: pd.DataFrame, x_column: str, y_column: str, color_column: str = None):
        """
        Create a bar chart using Plotly.

        Parameters:
        data (pandas.DataFrame): The input DataFrame
        x_column (str): The column name for the x-axis
        y_column (str): The column name for the y-axis
        color_column (str, optional): The column name for grouping by color

        Returns:
        plotly.graph_objs._figure.Figure: The Plotly figure object
        """
        if color_column:
            fig = px.bar(data_frame=data, x=x_column, y=y_column, color=color_column, width=self.width,
                         height=self.height)
        else:
            fig = px.bar(data_frame=data, x=x_column, y=y_column, width=self.width, height=self.height)
        return fig

    def get_scatterplot(self, data, variables: list):
        """
        Create a scatter plot for multiple variables in a DataFrame.

        Parameters:
        variables (list): A list of variables to plot

        Returns:
        None
        """
        # Create subplots with multiple scatter plots
        fig = make_subplots(rows=len(variables), cols=1, subplot_titles=variables)

        # Add scatter plots for each variable
        for i, variable in enumerate(variables):
            fig.add_trace(go.Scatter(x=data['x'], y=data[variable], mode='markers'), row=i + 1, col=1)

            # Update the layout
        fig.update_layout(height=self.height, width=self.width, showlegend=False)

        # Show the scatter plot
        return fig
    

    def create_correlation_grid(data: pd.DataFrame, target_var: str):  
        # Separate numerical and categorical columns  
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()  
        categorical_cols = data.select_dtypes(include=[object, 'category']).columns.tolist()  
        
        # Remove the target variable from the list of numerical columns  
        numerical_cols.remove(target_var)  
        
        # Total number of plots  
        total_plots = len(numerical_cols) + len(categorical_cols)  
        
        # Determine the number of rows and columns for the subplot grid  
        cols = 3  # Define the number of columns you want  
        rows = (total_plots + cols - 1) // cols  # Calculate number of rows needed  
        
        # Create subplot figure  
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=numerical_cols + categorical_cols)  
        
        # Add numerical plots  
        plot_num = 0  
        for col in numerical_cols:  
            row = (plot_num // cols) + 1  
            col_index = (plot_num % cols) + 1  
            scatter = px.scatter(data, x=col, y=target_var)  
            for trace in scatter['data']:  
                fig.add_trace(trace, row=row, col=col_index)  
            plot_num += 1  
        
        # Add categorical plots  
        for col in categorical_cols:  
            row = (plot_num // cols) + 1  
            col_index = (plot_num % cols) + 1  
            box = px.box(data, x=col, y=target_var)  
            for trace in box['data']:  
                fig.add_trace(trace, row=row, col=col_index)  
            plot_num += 1  
    
        # Update layout to improve appearance  
        fig.update_layout(height=300*rows, width=1000, title_text=f'Correlation Grid Plot with {target_var}')  
        
        return fig
