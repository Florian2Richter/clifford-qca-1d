import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import streamlit.components.v1 as components
import streamlit as st

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    numeric = [mapping.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

# New function using Plotly
def plot_spacetime_plotly(pauli_strings):
    """Plot the spacetime diagram using Plotly for interactivity."""
    time_steps = len(pauli_strings)
    if time_steps == 0:
        return go.Figure()
    cell_count = len(pauli_strings[0])
    
    # Convert Pauli strings to numeric data
    data = np.array([pauli_to_numeric(s) for s in pauli_strings])
    
    # Define colorscale for Plotly (matching Matplotlib)
    # Note: Plotly colorscales map values 0-1. We need to define boundaries.
    colorscale = [
        [0.0, 'white'],   # I (corresponds to value 0)
        [0.25, 'white'],
        [0.25, '#008080'], # X (corresponds to value 1)
        [0.5, '#008080'],
        [0.5, '#FF7F50'], # Z ‚(corresponds to value 2)
        [0.75, '#FF7F50'],
        [0.75, '#4A4A4A'], # Y (corresponds to value 3)
        [1.0, '#4A4A4A']
    ]
    
    # Create the heatmap trace
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=list(range(cell_count)),
        y=list(range(time_steps)),
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
        # Custom hover text
        hovertemplate="Time: %{y}<br>Cell: %{x}<br>Operator: %{customdata}<extra></extra>",
        customdata=[[pauli_strings[y][x] for x in range(cell_count)] for y in range(time_steps)]
    ))
    
    # Update layout
    fig.update_layout(
        title='1D Clifford QCA Spacetime Diagram (Plotly)',
        xaxis_title='Cell Position',
        yaxis_title='Time Step',
        yaxis_autorange='reversed', # Time flows downwards
        xaxis=dict(tickmode='linear', dtick=max(1, cell_count // 15)),
        yaxis=dict(tickmode='linear', dtick=max(1, time_steps // 15)),
        width=800,
        height=500,
        autosize=True  # Enable autosizing
    )
    
    
    return fig