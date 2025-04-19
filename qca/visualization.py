import numpy as np
import plotly.graph_objects as go

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

def pauli_to_rgba(data):
    """
    Convert Pauli operator numeric data (0-3) to RGBA values for WebGL rendering.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array of Pauli operator numeric values (0-3).
        
    Returns:
    --------
    numpy.ndarray of shape (*data.shape, 4) with RGBA values as uint8.
    """
    # Create empty RGBA array
    rows, cols = data.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    
    # Define our color mapping (same as current colorscale)
    colors = {
        0: [255, 255, 255, 255],  # 'I' -> white
        1: [0, 128, 128, 255],    # 'X' -> teal (#008080)
        2: [255, 127, 80, 255],   # 'Z' -> coral (#FF7F50)
        3: [74, 74, 74, 255]      # 'Y' -> dark gray (#4A4A4A)
    }
    
    # Apply colors to each value
    for val, color in colors.items():
        mask = (data == val)
        for i in range(4):  # Apply each RGBA channel
            rgba[mask, i] = color[i]
            
    return rgba

def add_custom_colorbar(fig):
    """
    Add a custom colorbar to the figure for Pauli operators.
    Since go.Image doesn't have a built-in colorbar, we add a dummy trace.
    """
    # Add a dummy trace for the colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale=[
                [0, 'white'], [0.25, 'white'],
                [0.25, '#008080'], [0.5, '#008080'],
                [0.5, '#FF7F50'], [0.75, '#FF7F50'],
                [0.75, '#4A4A4A'], [1.0, '#4A4A4A']
            ],
            showscale=True,
            cmin=0, cmax=3,
            colorbar=dict(
                title='Pauli Operator',
                tickvals=[0.5, 1.5, 2.5, 3.5],
                ticktext=['I', 'X', 'Z', 'Y'],
                lenmode='pixels',
                len=200,
                yanchor='top',
                y=1
            )
        ),
        showlegend=False
    ))

def make_empty_figure(cell_count, total_time_steps):
    """
    Create an empty plotly figure for the QCA simulation using go.Image for WebGL rendering.
    
    Parameters:
    -----------
    cell_count : int
        Number of cells in the QCA.
    total_time_steps : int
        Total number of time steps to show in the plot.
    """
    # Initialize empty data array (all 'I' operators = 0)
    data = np.zeros((total_time_steps, cell_count), dtype=np.int8)
    
    # Convert to RGBA
    rgba = pauli_to_rgba(data)
    
    # Create figure with go.Image
    fig = go.Figure(go.Image(
        z=rgba,
        x0=0, dx=1,  # Map to cell indices
        y0=0, dy=1   # Map to time steps
    ))
    
    # Add custom colorbar
    add_custom_colorbar(fig)
    
    # Update layout
    fig.update_layout(
        title='1D Clifford QCA Spacetime Diagram',
        xaxis_title='Cell Position',
        yaxis_title='Time Step',
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            dtick=max(1, cell_count // 10),
            range=[0, cell_count]  # Set fixed range for x-axis
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            dtick=max(1, total_time_steps // 10),
            autorange='reversed',
            range=[total_time_steps, 0]  # Set fixed range for y-axis
        ),
        width=800,
        height=500,
        dragmode='zoom',
        hovermode=False,  # Disable hover for now
        margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uirevision='static'  # Preserve view on updates
    )
    
    # Configure interactivity
    config = {
        'displayModeBar': True,
        'scrollZoom': True,     # Enable scroll wheel zooming
        'displaylogo': False,
        'responsive': True,
        'staticPlot': False,    # Allow interactivity
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'qca_simulation',
            'height': 500,
            'width': 800,
            'scale': 1
        },
        'modeBarButtonsToRemove': [
            'select2d', 'lasso2d',  # Keep zoom buttons
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines', 'toggleHover', 'resetViewMapbox'
        ]
    }
    
    # Attach the config to the figure for use in Streamlit
    fig._config = config
    
    return fig

def update_figure(fig, pauli_strings):
    """
    Update an existing plotly figure with new data using WebGL Image rendering.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The existing figure to update.
    pauli_strings : list of strings
        List of Pauli strings representing the state at each calculated time step.
    
    Returns:
    --------
    The updated figure.
    """
    # Check if there's data to update
    if not pauli_strings:
        return fig
    
    # Get the total dimensions from the figure's layout
    yaxis_range = fig.layout.yaxis.range
    if yaxis_range and len(yaxis_range) == 2:
        total_time_steps = int(max(yaxis_range))
    else:
        # Default if not set
        total_time_steps = 250
    
    # Convert strings to numeric array
    numeric_data = pauli_strings_to_numeric(pauli_strings)
    current_time_steps = numeric_data.shape[0]
    cell_count = numeric_data.shape[1]
    
    # Create full-sized data array with zeros
    full_data = np.zeros((total_time_steps, cell_count), dtype=np.int8)
    
    # Copy the actual data into the beginning of the array
    full_data[:current_time_steps, :] = numeric_data
    
    # Convert to RGBA
    rgba = pauli_to_rgba(full_data)
    
    # Update the image
    fig.update_traces(z=rgba, selector=dict(type='image'))
    
    return fig