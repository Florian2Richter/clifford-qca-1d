import numpy as np
import plotly.graph_objects as go
import time

# Global constants for better performance
PAULI_MAPPING = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    numeric = [PAULI_MAPPING.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

def pauli_strings_to_numeric(pauli_strings):
    """
    Vectorized conversion of multiple Pauli strings to numeric arrays.
    
    Parameters:
    -----------
    pauli_strings : list of strings
        List of Pauli strings to convert.
        
    Returns:
    --------
    np.ndarray of shape (len(pauli_strings), len(pauli_strings[0]))
    """
    if not pauli_strings:
        return np.array([])
    
    n_strings = len(pauli_strings)
    string_len = len(pauli_strings[0])
    
    # Pre-allocate the result array
    result = np.zeros((n_strings, string_len), dtype=np.int8)
    
    # Create a character array from the strings
    char_array = np.array([list(s) for s in pauli_strings])
    
    # Vectorized mapping using NumPy
    for char, value in PAULI_MAPPING.items():
        result[char_array == char] = value
    
    return result

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
    The updated figure (same object reference) and timing dictionary.
    """
    timing = {}
    total_start = time.time()
    
    # Get dimensions
    current_time_steps = len(pauli_strings)
    if current_time_steps == 0:
        return fig, {'total': 0}
    
    cell_count = len(pauli_strings[0])
    total_time_steps = len(fig.data[0].z)
    
    # Measure array allocation time
    alloc_start = time.time()
    data = np.zeros((total_time_steps, cell_count), dtype=np.int8)
    timing['allocation'] = time.time() - alloc_start
    
    # Measure vectorized conversion time
    conversion_start = time.time()
    if current_time_steps > 0:
        # Convert all strings to numeric using vectorized function
        numeric_data = pauli_strings_to_numeric(pauli_strings)
        
        # Update the data array with available time steps
        max_steps = min(current_time_steps, total_time_steps)
        data[:max_steps] = numeric_data[:max_steps]
    timing['numeric_conversion'] = time.time() - conversion_start
    
    # Measure customdata creation time
    customdata_start = time.time()
    customdata = [['I'] * cell_count for _ in range(total_time_steps)]
    for t, s in enumerate(pauli_strings):
        if t < total_time_steps:
            customdata[t] = list(s)
    timing['customdata_creation'] = time.time() - customdata_start
    
    # Measure the actual figure update time
    update_start = time.time()
    fig.data[0].z = data
    fig.data[0].customdata = customdata
    timing['figure_update'] = time.time() - update_start
    
    timing['total'] = time.time() - total_start
    
    return fig, timing