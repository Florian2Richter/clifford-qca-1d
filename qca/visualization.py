import numpy as np
import plotly.graph_objects as go

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    numeric = [mapping.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

def make_empty_figure(cell_count, total_time_steps):
    """
    Create an empty plotly figure with the heatmap structure but initialized with all 'I' operators.
    This creates the figure only once, which can then be updated efficiently.
    
    Parameters:
    -----------
    cell_count : int
        Number of cells in the QCA.
    total_time_steps : int
        Total number of time steps to show in the plot.
    """
    # Initialize empty data arrays
    data = np.zeros((total_time_steps, cell_count), dtype=int)  # All 'I' operators
    customdata = [['I'] * cell_count for _ in range(total_time_steps)]
    
    # Define the color scale
    colorscale = [
        [0.0, 'white'],
        [0.25, 'white'],
        [0.25, '#008080'],
        [0.5, '#008080'],
        [0.5, '#FF7F50'],
        [0.75, '#FF7F50'],
        [0.75, '#4A4A4A'],
        [1.0, '#4A4A4A']
    ]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=list(range(cell_count)),
        y=list(range(total_time_steps)),
        colorscale=colorscale,
        zmin=0,
        zmax=3,
        showscale=True,
        colorbar=dict(
            title='Pauli Operator',
            tickvals=[0.5, 1.5, 2.5, 3.5],
            ticktext=['I', 'X', 'Z', 'Y'],
            lenmode='pixels',
            len=200,
            yanchor='top',
            y=1
        ),
        hovertemplate="Time: %{y}<br>Cell: %{x}<br>Operator: %{customdata}<extra></extra>",
        customdata=customdata
    ))
    
    # Set layout
    fig.update_layout(
        title='1D Clifford QCA Spacetime Diagram',
        xaxis_title='Cell Position',
        yaxis_title='Time Step',
        yaxis_autorange='reversed',
        xaxis=dict(tickmode='linear', dtick=max(1, cell_count // 15)),
        yaxis=dict(tickmode='linear', dtick=max(1, total_time_steps // 15)),
        width=800,
        height=500,
        autosize=False
    )
    
    return fig

def update_figure(fig, pauli_strings):
    """
    Update an existing plotly figure with new data without recreating the entire figure.
    Optimized for performance with large datasets.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The existing figure to update.
    pauli_strings : list of strings
        List of Pauli strings representing the state at each calculated time step.
    
    Returns:
    --------
    The updated figure (same object reference).
    """
    # Get dimensions
    current_time_steps = len(pauli_strings)
    if current_time_steps == 0:
        return fig
    
    cell_count = len(pauli_strings[0])
    total_time_steps = len(fig.data[0].z)
    
    # Performance optimization: Only update what's needed
    # Create empty arrays only for the data we need
    data = np.zeros((current_time_steps, cell_count), dtype=np.int8)
    
    # Convert the Pauli strings to numeric values in bulk
    # This is much faster than doing it one by one
    mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    
    # Use vectorized operations for better performance
    for t, s in enumerate(pauli_strings):
        for i, ch in enumerate(s):
            data[t, i] = mapping.get(ch, 0)
    
    # Only update the changed portion of the heatmap
    # Instead of recreating the entire 2D array
    if current_time_steps < total_time_steps:
        # Use restyle for partial updates (much faster than updating the entire z property)
        fig.update_traces(
            z=[data[t] for t in range(current_time_steps)],
            selector=dict(type='heatmap'),
            row=1, col=1
        )
        
        # Update customdata efficiently
        customdata_update = [[ch for ch in s] for s in pauli_strings]
        fig.update_traces(
            customdata=customdata_update,
            selector=dict(type='heatmap'),
            row=1, col=1
        )
    else:
        # If we have more data than can fit, create a new full dataset
        full_data = np.zeros((total_time_steps, cell_count), dtype=np.int8)
        full_data[:current_time_steps] = data[:total_time_steps]
        
        # Create customdata with the same efficient indexing
        customdata = [['I'] * cell_count for _ in range(total_time_steps)]
        for t, s in enumerate(pauli_strings[:total_time_steps]):
            customdata[t] = list(s)
        
        # Update the entire heatmap at once
        fig.data[0].z = full_data
        fig.data[0].customdata = customdata
    
    return fig