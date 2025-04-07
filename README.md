# 1D Clifford QCA Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clifford-qca-1d-wqukva9js5377m8nz48nd8.streamlit.app/)

An interactive web application for simulating and visualizing 1-Dimensional Clifford Quantum Cellular Automata (QCA). This simulator demonstrates the evolution of Pauli operators through a 1D lattice over time, providing insights into quantum information propagation.

![Streamlit App Screenshot](docs/images/app_screenshot.png)

## Features

- Interactive simulation of 1D Clifford QCA
- Real-time visualization of quantum state evolution
- Customizable simulation parameters:
  - Number of cells (lattice size)
  - Number of time steps
  - Local update rules
- Multiple initial state options:
  - Single active cell
  - Random configuration
  - Manual Pauli string input
- Beautiful spacetime diagram visualization
- Intuitive web interface built with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Florian2Richter/clifford-qca-1d.git
cd clifford-qca-1d
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development
Run the Streamlit app locally:
```bash
streamlit run app.py
```

The app will open in your default web browser. You can then:
1. Adjust the number of cells and time steps
2. Set the local rule matrix (2x6 over F2)
3. Choose an initial state configuration
4. Watch the QCA evolution in the spacetime diagram

### Streamlit Cloud Deployment

To deploy this app on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your forked repository and branch (main)
6. Set the main file path as `app.py`
7. Click "Deploy"

The app will be automatically built and deployed. Streamlit Cloud will:
- Install the required Python packages from `requirements.txt`
- Use the configuration from `.streamlit/config.toml`
- Make the app available at a public URL

## Mathematical Background

The simulator implements a 1D Clifford Quantum Cellular Automaton, where:
- Each cell state is represented by a Pauli operator (I, X, Z, or Y)
- The evolution is governed by a local update rule in the form of a 2x6 matrix over F2
- The global update preserves the Clifford group structure
- Periodic boundary conditions are applied

### Relation to Classical Cellular Automata

This quantum cellular automaton can be seen as a quantum generalization of [elementary cellular automata](https://en.wikipedia.org/wiki/Elementary_cellular_automaton). While classical elementary cellular automata operate with binary states (0 or 1) and update rules based on three neighboring cells, our quantum version:
- Uses four possible states (I, X, Z, Y) represented by pairs of bits in F2
- Preserves the quantum mechanical properties through Clifford operations
- Updates each cell based on its left and right neighbors similar to classical CAs
- Uses matrix operations over F2 instead of boolean functions

The key difference is that our QCA preserves the algebraic structure of Pauli operators, making it a suitable model for quantum information propagation while classical CAs focus on binary state evolution.

### Local Rule Matrix Structure

The local rule is specified by a 2×6 matrix over F2 (binary field), which can be understood as three 2×2 blocks:
```
[A_left | A_center | A_right]
```
where each block determines how a cell's new state depends on its left neighbor (A_left), itself (A_center), and its right neighbor (A_right).

For example, the identity transformation that leaves each cell unchanged would use:
```
[0 0 | 1 0 | 0 0]  First row
[0 0 | 0 1 | 0 0]  Second row
```
Here:
- A_left = [0 0; 0 0]: No contribution from left neighbor
- A_center = [1 0; 0 1]: Identity matrix, preserves current state
- A_right = [0 0; 0 0]: No contribution from right neighbor

In contrast, the default simulation rule:
```
[1 0 | 1 1 | 0 1]  First row
[0 1 | 0 1 | 1 0]  Second row
```
creates interesting propagation patterns as each cell's state depends on both its neighbors and itself.

Each cell's state is encoded as a pair of bits (x,z) representing Pauli operators:
- I = (0,0)
- X = (1,0)
- Z = (0,1)
- Y = (1,1)

The matrix multiplication is performed modulo 2, ensuring the output remains in F2.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this simulator in your research, please cite:
```bibtex
@software{clifford_qca_1d,
  author = {Richter, Florian},
  title = {1D Clifford QCA Simulator},
  year = {2024},
  url = {https://github.com/Florian2Richter/clifford-qca-1d}
}
```