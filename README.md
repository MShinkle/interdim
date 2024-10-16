# InterDim
[![Docs and Tests](https://github.com/MShinkle/interdim/actions/workflows/docs.yml/badge.svg?branch=main&label=tests)](https://github.com/MShinkle/interdim/actions)
![Python Versions](https://img.shields.io/pypi/pyversions/interdim.svg)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://MShinkle.github.io/interdim)

### Interactive Dimensionality Reduction, Clustering, and Visualization

InterDim is a Python package for interactive exploration of latent data dimensions. It wraps existing tools for dimensionality reduction, clustering, and data visualization in a streamlined interface, allowing for quick and intuitive analysis of high-dimensional data.

## Features

- Easy-to-use pipeline for dimensionality reduction, clustering, and visualization
- Interactive 3D scatter plots for exploring reduced data
- Support for various dimensionality reduction techniques (PCA, t-SNE, UMAP, etc.)
- Multiple clustering algorithms (K-means, DBSCAN, etc.)
- Customizable point visualizations for detailed data exploration

## Installation

You can install from [PyPI](https://pypi.org/project/interdim/) via `pip` (recommended):

```bash
pip install interdim
```

Or from source:

```bash
git clone https://github.com/MShinkle/interdim.git
cd interdim
pip install .
```

## Quick Start

Here's a basic example using the Iris dataset:

```python
from sklearn.datasets import load_iris
from interdim import InterDimAnalysis

iris = load_iris()
analysis = InterDimAnalysis(iris.data, true_labels=iris.target)
analysis.reduce(method='tsne', n_components=3)
analysis.cluster(method='kmeans', n_clusters=3)
analysis.show(n_components=3, point_visualization='bar')
```

![3D Scatter Plot with Interactive Bar Charts](https://raw.githubusercontent.com/MShinkle/interdim/refs/heads/main/docs/images/iris_plot.png)

This will reduce the Iris dataset to 3 dimensions using t-SNE, clusters the data using K-means, and displays an interactive 3D scatter plot with bar charts for each data point as you hover over them.

However, this is just a small example of what you can do with InterDim. You can use it to explore all sorts of data, including high-dimensional data like language model embeddings!

## Demo Notebooks

For more in-depth examples and use cases, check out our demo notebooks:

1. [Iris Species Analysis](https://github.com/MShinkle/interdim/blob/main/notebooks/IRIS_visualization.ipynb): Basic usage with the classic Iris dataset.
![Iris Species Analysis](https://raw.githubusercontent.com/MShinkle/interdim/refs/heads/main/docs/images/iris_plot_pretty.png)

2. [DNN Latent Space Exploration](https://github.com/MShinkle/interdim/blob/main/notebooks/DNN_latents.ipynb): Visualizing deep neural network activations.
![DNN Latent Space Exploration](https://raw.githubusercontent.com/MShinkle/interdim/refs/heads/main/docs/images/fashion_plot.png)

3. [LLM Token Analysis](https://github.com/MShinkle/interdim/blob/main/notebooks/LLM_token_embeddings.ipynb): Exploring language model token embeddings and layer activations.
![LLM Token Analysis](https://raw.githubusercontent.com/MShinkle/interdim/refs/heads/main/docs/images/llm_weevil.png)

## Documentation

For detailed API documentation and advanced usage, visit our [GitHub Pages](https://MShinkle.github.io/interdim).

## Contributing

We welcome discussion and contributions!

## License

InterDim is released under the BSD 3-Clause License. See the [LICENSE](https://github.com/MShinkle/interdim/blob/main/LICENSE) file for details.

## Contact

For questions and feedback, please [open an issue](https://github.com/MShinkle/interdim/issues) on GitHub.
