# Iris Species Classifier using k-Nearest Neighbors

## Introduction
This project is an implementation of a machine learning model to classify iris species. Utilizing the Iris dataset, it provides a beginner-friendly introduction to the workflow of a machine learning project in Python, with an emphasis on using the scikit-learn library.

## Technologies
- Python 3.8
- scikit-learn
- pandas
- numpy

## Setup and Installation
To run this project, Python 3.8 and the following libraries are required:
- scikit-learn
- pandas
- numpy

You can install them using pip:
```bash
pip install numpy pandas scikit-learn
```
Note: You may need to use `pip3` as follow:
```bash
pip3 install numpy pandas scikit-learn
```

## Methodology
1. **Data Loading:** The Iris dataset is loaded using scikit-learn.
2. **Data Splitting:** The dataset is split into training and testing sets to evaluate the model's performance.
3. **Model Training:** A k-Nearest Neighbors (kNN) classifier is trained with `n_neighbors=3`.

Note: Usually before selecting ML method and setting configurations of the model, we needed to study and "Know Our Data", but we pass this step because it is a known problem and the reason is to showcasing the ability of kNN. Also seting `n_neighbors=3` needs prior data visualaziation and for the same reason I skipped this step as well.

4. **Making Predictions:** The trained model is used to make predictions on the test set.
5. **Model Evaluation:** The model's accuracy is evaluated against the test set.

## Code Example
1. **Data Loading:** Scikit-learn comes with a few standard datasets, including the Iris dataset. Here's how you can load it:
```python
from sklearn.datasets import load_iris
iris = load_iris()
```
Before jumping into modeling, it's important to understand your data:
```python
# Print the names of the features
print("Features:", iris.feature_names)

# Print the label type of iris('setosa' 'versicolor' 'virginica')
print("Labels:", iris.target_names)

# Print data(feature)shape
print("Data shape:", iris.data.shape)

# Print the data features
print("First five rows of data:\n", iris.data[:5])
```
Result:
```plaintext
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Labels: ['setosa' 'versicolor' 'virginica']
Data shape: (150, 4)
First five rows of data:
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
```

2. **Data Splitting:** 
## Results
The kNN model achieved an accuracy of (insert accuracy), showcasing its effectiveness in classifying iris species based on the provided features.

## Conclusion
This project demonstrates the basic steps involved in a machine learning task using Python and scikit-learn. The kNN algorithm, with its simplicity and efficacy, provides a strong foundation for classification problems.

## Author
Vahid Keshmiri
