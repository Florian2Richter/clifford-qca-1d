import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    numeric = [mapping.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

def plot_spacetime(pauli_strings, cell_count):
    """
    Plot the spacetime diagram of the QCA simulation.
    
    - pauli_strings: list of strings (each of length cell_count) representing the state at each time step.
    """
    time_steps = len(pauli_strings)
    data = np.zeros((time_steps, cell_count), dtype=int)
    for t, s in enumerate(pauli_strings):
        data[t, :] = pauli_to_numeric(s)
    
    # Define a discrete colormap: I: white, X: red, Z: blue, Y: green.
    cmap = ListedColormap(["white", "red", "blue", "green"])
    
    plt.figure(figsize=(0.5 * cell_count, 0.5 * time_steps))
    plt.imshow(data, cmap=cmap, interpolation="nearest", aspect="auto")
    plt.xlabel("Cell position")
    plt.ylabel("Time step")
    plt.title("1D Clifford QCA Spacetime Diagram")
    plt.colorbar(ticks=[0, 1, 2, 3], label="Pauli")
    plt.show()
