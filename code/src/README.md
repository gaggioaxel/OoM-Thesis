# Python File Function - Clustering.py

This file contains a set of functions related to clustering techniques used in various files within this project.

- **clustering.py**: This file contains the following functions used across different parts of the project:
  - `cluster_table_by_joints(table: pd.DataFrame) -> pd.DataFrame`: Function to group a table by joints.
  - `joints_array_to_xyz_columns(table: pd.DataFrame) -> pd.DataFrame`: Function to convert joints array to XYZ columns.
  - `table_to_list_xyz_tables(table: pd.DataFrame, into="xyz")`: Function to convert a table into XYZ lists.
  - `xyz_tables_to_xyz_columns(tablesList)`: Function to convert XYZ tables to XYZ columns.
  - `movmean(x, w)`: Function to calculate moving mean.
  - `movmedian(x, w)`: Function to calculate moving median.
  - `smoothing(x: pd.DataFrame) -> pd.DataFrame`: Function to perform smoothing on a DataFrame.
  - `compute_derivatives(x: pd.DataFrame, dt: float, smooth=True) -> pd.DataFrame`: Function to compute derivatives of motion data.
  - `shi_malik_spectral_clustering(weightMatrix: np.ndarray) -> np.array`: Function to perform spectral clustering using Shi-Malik algorithm.
  - `myeigs(S: np.ndarray, unused_k=2)`: Function to compute eigenvalues and eigenvectors of a matrix.

## Usage

These functions can be imported and utilized in other files within this project by importing the `clustering.py` module. For example:

```python
from clustering import cluster_table_by_joints, joints_array_to_xyz_columns, table_to_list_xyz_tables
# Import other functions as needed
```

# Python Notebook - ClusteringOoM.ipynb
This file contains the implementation within this project.

# Python Notebook - ML_approach.ipynb
This file contains the ML part of our project

# Python Notebook - MWPM.ipynb
This file contains experiments of the Maximum Weight Perfect Match

# Python Notebook - MarkersCompression.ipynb
This file contains functions to compress Qualisys ".tsv" files of 64 or 41 markerset, into the ".csv" with 20 markers.

# Python Notebook - ExampleThesis.ipynb
This file contains examples used for our thesis.