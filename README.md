### Overview
This project utilizes machine learning techniques with Keras, Sci-Keras, and TensorFlow to build, train, and optimize a neural network model. The model is trained on data stored in the **`dataset.arff`** file and includes techniques for handling imbalanced datasets, such as SMOTE (Synthetic Minority Over-sampling Technique) and Random Under-sampling. It also features grid search for hyperparameter optimization and several evaluation metrics to assess model performance.

### Files

- **`notebook.ipynb`**: Main Jupyter notebook file containing the code for data preprocessing, model building, training, and evaluation.
- **`notebookO.ipynb`**: Another Jupyter notebook (specific purpose TBD).
- **`dataset.arff`**: The dataset used for training and testing the machine learning model.

### Key Features
- **Neural Network**: Model built using TensorFlow and Keras.
- **Resampling Techniques**: SMOTE and Random Under-sampling to handle class imbalance.
- **Grid Search**: Hyperparameter optimization using `GridSearchCV`.
- **Data Visualization**: Visualizations with `Matplotlib`, `Seaborn`, and `Plotly` for exploring the dataset and evaluating the model.
- **Performance Metrics**: Use of confusion matrix, classification report, and other evaluation metrics.

### Dependencies

The project requires the following Python libraries:
- `pandas`
- `numpy`
- `tensorflow`
- `keras`
- `scipy`
- `scikit-learn`
- `scikeras`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`
- `plotly`

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

### Dataset
The dataset is in **`dataset.arff`** format. It is loaded using the `scipy.io.arff` module, which provides tools to handle ARFF files. The dataset is processed to split features and labels for training and testing.

### Usage
To run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter notebook (`notebook.ipynb` or `notebookO.ipynb`) and run the cells to preprocess data, build the model, train it, and evaluate the results.

### Model Overview
The model utilizes a neural network built with Keras, using regularizers and grid search for hyperparameter tuning. The code also includes SMOTE and under-sampling to balance the dataset before training.

### Evaluation
After training, the model is evaluated using:
- **Confusion Matrix**
- **Classification Report**
- **Accuracy, Precision, Recall, F1 Score**

Additionally, visualizations of model performance are generated using `Matplotlib`, `Seaborn`, and `Plotly`.
