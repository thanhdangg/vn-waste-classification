# VN Waste Classification Project

This project aims to classify waste images into different categories using deep learning models. The project includes data preprocessing, model training, evaluation, and visualization of training metrics.

## Project Structure

- **data/**: Contains the dataset used for training and evaluation.
  - **README.md**: Documentation related to the dataset, including sources, structure, and preprocessing steps.

- **models/**: Contains documentation about the models used in the project.
  - **README.md**: Descriptions of each model architecture and their intended use cases.

- **notebooks/**: Contains Jupyter notebooks for interactive development.
  - **train_and_evaluate.ipynb**: Notebook for training and evaluating the models, including code for loading data, training each model separately, saving the trained models, and visualizing the training process.

- **src/**: Contains the source code for the project.
  - **\_\_init\_\_.py**: Marks the directory as a Python package.
  - **config.py**: Configuration settings for the project, including paths and hyperparameters.
  - **dataset.py**: Defines the dataset class for loading and preprocessing data.
  - **train.py**: Contains the training logic for the models.
  - **evaluate.py**: Functions for evaluating the trained models and calculating metrics.
  - **utils.py**: Utility functions used throughout the project.
  - **plot.py**: Functions for plotting training metrics.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd vn-waste-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset as described in `data/README.md`.

4. Use the Jupyter notebook `notebooks/train_and_evaluate.ipynb` to train and evaluate the models.

## Usage

- To train the models, run the cells in the `train_and_evaluate.ipynb` notebook.
- The trained models will be saved in the specified directory.
- Visualizations of the training process will be displayed within the notebook.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.