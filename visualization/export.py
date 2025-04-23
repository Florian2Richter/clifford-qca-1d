"""
High-resolution export functionality for QCA visualizations.
"""

import plotly.graph_objects as go
from config import COLOR_MAPPING
from visualization.converters import pauli_strings_to_numeric

def generate_hires_plot(pauli_strings, width=1920, height=1080):
    """
    Generate a high-resolution plot for wallpaper export.
    
    Parameters:
    -----------
    pauli_strings : list of strings
        List of Pauli strings representing the state at each time step.
    width : int
        Width of the image in pixels.
    height : int
        Height of the image in pixels.
        
    Returns:
    --------
    bytes
        PNG image as bytes.
    """
    try:
        import io
        from plotly.io import to_image
        
        # Get dimensions from the input
        time_steps = len(pauli_strings)
        if time_steps == 0:
            return None
        cell_count = len(pauli_strings[0])
        
        # Convert Pauli strings to numeric representation
        numeric_data = pauli_strings_to_numeric(pauli_strings)
        
        # Create a new figure for high-resolution export
        fig = go.Figure(data=go.Heatmap(
            z=numeric_data,
            x=list(range(cell_count)),
            y=list(range(time_steps)), 
            colorscale=COLOR_MAPPING,
            zmin=0,
            zmax=3,
            showscale=True,
            colorbar=dict(
                title='Pauli Operator',
                tickvals=[0, 1, 2, 3],
                ticktext=['I', 'X', 'Z', 'Y'],
                thickness=25,
                outlinewidth=1,
                outlinecolor='black'
            )
        ))
        
        # Configure layout for high resolution
        fig.update_layout(
            title='1D Clifford QCA Evolution',
            xaxis_title='Cell Position',
            yaxis_title='Time Step',
            yaxis_autorange='reversed',
            font=dict(
                family="Arial, sans-serif",
                size=18,
                color="black"
            ),
            width=width,
            height=height,
            margin=dict(l=80, r=50, t=100, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Set axis properties for cleaner look
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        
        # Convert to PNG image
        img_bytes = to_image(fig, format='png')
        return img_bytes
        
    except Exception as e:
        print(f"Error generating high-resolution plot: {e}")
        return None 