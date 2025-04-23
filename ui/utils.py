"""
UI utilities for the 1D Clifford QCA Simulator.
"""

import streamlit as st

def parse_matrix_row(row_str):
    """
    Parse a comma-separated string of 0s and 1s into a list of integers.
    
    Parameters:
    -----------
    row_str : str
        Comma-separated string of 0s and 1s.
    
    Returns:
    --------
    list
        List of integers (0s and 1s).
    """
    try:
        # Remove spaces and split by commas
        values = [int(x.strip()) for x in row_str.replace(' ', '').split(',')]
        
        # Validate that all values are 0 or 1
        for val in values:
            if val not in [0, 1]:
                st.sidebar.error("Only 0 and 1 are allowed.")
                return None
        return values
    except Exception as e:
        st.sidebar.error("Error parsing the row: " + str(e))
        return None 