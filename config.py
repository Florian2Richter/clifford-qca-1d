"""
Centralized configuration settings for the 1D Clifford QCA Simulator.
"""

# Simulation constants
BATCH_SIZE = 5  # Number of simulation steps to process before updating the display

# Visualization constants
PAULI_MAPPING = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}  # Mapping of Pauli operators to numeric values

# Color mappings for visualization
COLOR_MAPPING = [
   [0, 'white'],       # I (value 0) start
    [0.249, 'white'],   # I (value 0) end
    [0.25, '#008080'],  # X (value 1) start
    [0.499, '#008080'], # X (value 1) end
    [0.5, '#FF7F50'],   # Z (value 2) start
    [0.749, '#FF7F50'], # Z (value 2) end
    [0.75, '#4A4A4A'],  # Y (value 3) start
    [1.0, '#4A4A4A']    # Y (v
]

# Default figure dimensions
DEFAULT_FIG_WIDTH = 800
DEFAULT_FIG_HEIGHT = 500

# High resolution export options
EXPORT_RESOLUTIONS = {
    "1920x1080 (baseline)": (1920, 1080),
    "2560x1440 (QHD)": (2560, 1440),
    "3840x2160 (4K)": (3840, 2160),
    "3440x1440 (ultrawide)": (3440, 1440),
    "1080x1920 (mobile/portrait)": (1080, 1920)
} 