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

def plot_spacetime_plotly(pauli_strings, total_time_steps=None):
    """
    Plot the spacetime diagram using Plotly for interactivity.
    
    Parameters:
    -----------
    pauli_strings : list of strings
        List of Pauli strings representing the state at each calculated time step.
    total_time_steps : int, optional
        Total number of time steps to show in the plot. If None, uses len(pauli_strings).
        This is used for progressive visualization to keep axes consistent.
    """
    current_time_steps = len(pauli_strings)
    if current_time_steps == 0:
        return go.Figure()
    
    cell_count = len(pauli_strings[0])
    
    # Set total time steps (for fixed y-axis)
    if total_time_steps is None or total_time_steps < current_time_steps:
        total_time_steps = current_time_steps
    
    # Convert Pauli strings to numeric data for calculated steps
    data = np.zeros((total_time_steps, cell_count), dtype=int)
    for t, s in enumerate(pauli_strings):
        data[t] = pauli_to_numeric(s)
    
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
        # Custom hover text for only the calculated steps
        hovertemplate="Time: %{y}<br>Cell: %{x}<br>Operator: %{customdata}<extra></extra>",
        customdata=[[pauli_strings[y][x] if y < current_time_steps else 'I' 
                    for x in range(cell_count)] 
                    for y in range(total_time_steps)]
    ))
    
    # Add progress indicator if still calculating
    if current_time_steps < total_time_steps:
        progress_pct = int(100 * current_time_steps / total_time_steps)
        fig.add_annotation(
            text=f"Calculating: {current_time_steps}/{total_time_steps} steps ({progress_pct}%)",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=14, color="red")
        )
    
    # Update layout with fixed axes
    fig.update_layout(
        title='1D Clifford QCA Spacetime Diagram',
        xaxis_title='Cell Position',
        yaxis_title='Time Step',
        yaxis_autorange='reversed', # Time flows downwards
        xaxis=dict(tickmode='linear', dtick=max(1, cell_count // 15)),
        yaxis=dict(tickmode='linear', dtick=max(1, total_time_steps // 15)),
        width=800,
        height=500,
        autosize=False  # Enable autosizing
    )
    
    return fig