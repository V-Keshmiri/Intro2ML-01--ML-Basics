# Intro to Machine Learning: Iris Species Classifier using k-Nearest Neighbors

## Introduction
This project is an introduction into Machnie Learning by implementation of a machine learning model to classify iris species. Utilizing the Iris dataset, it provides a beginner-friendly introduction to the workflow of a machine learning project in Python, with an emphasis on using the scikit-learn library.

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

2. **Data Splitting:** Divide the data into a training set and a test set:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
```
### Understanding train_test_split
`train_test_split` is a function provided by scikit-learn that splits arrays or matrices into random train and test subsets. It's crucial in machine learning for evaluating the performance of your model.

Here's a breakdown of its components:

***Input Data (Features) - X:*** In machine learning, the input data (often denoted as X) represents the "features" or "predictors". In the case of the Iris dataset, these are measurements like sepal length, sepal width, petal length, and petal width.
***Output Data (Labels) - Y:*** The output data (often denoted as Y) represents the "labels" or "targets". For the Iris dataset, this would be the species of the iris (setosa, versicolor, or virginica).
***Splitting the Data:***
    - Training Set: Used to train the machine learning model. It learns from this data.
    - Test Set: Used to evaluate the performance of the model. It's important that this data is not used in the training phase to ensure that the evaluation of the model's performance is fair and unbiased.
### Why Separate X and Y, Train and Test Sets?
***`X_train` and `X_test`:*** These are the subsets of your features (X). `X_train` contains a portion of your data used for training the model. `X_test` is used to test the model after training. It's crucial that the model has never seen `X_test` during training to accurately assess its performance.
***`y_train` and `y_test`:*** These are the corresponding subsets of labels (Y) for `X_train` and `X_test`. `y_train` contains the labels for the training data, and `y_test` contains the labels for the test data.
By having these separate datasets, you can train your model on one set of data (`X_train`, `y_train`) and then test its effectiveness on a different set (`X_test`, `y_test`). This approach gives a more accurate measure of how well your model generalizes to new, unseen data, which is a critical aspect of machine learning.

### What is `random_state`?
**Randomness in Splitting:** When we split a dataset into training and testing sets, we often want to do this randomly. This randomness ensures that the split is unbiased and that both sets are representative of the whole dataset. However, this random process can lead to different splits each time you run your code, which can make your results inconsistent and hard to reproduce.
**Setting a Seed for Randomness:** The `random_state` parameter is essentially a way to "set the seed" for this random splitting process. A seed is a starting point for the sequence of random numbers that will be generated. If you use the same seed (or `random_state`) every time, it ensures that you get the exact same split each time you run your code.
### Why is Reproducibility Important?
**Consistent Results:** Using `random_state` helps in getting consistent results. If you're tuning your model and you want to compare its performance under different parameters or configurations, it's crucial that the training and testing data remain the same across these different runs. This consistency allows for a fair comparison.
Collaborative Work and Sharing: If you're working in a team or sharing your work with others (like in a GitHub repository), setting a `random_state` means others can replicate your exact train-test split and validate your findings or build upon them.
**Scientific Rigor:** In any scientific experiment, being able to reproduce the results is a cornerstone of the scientific method. Similarly, in machine learning, reproducibility ensures the validity of your model's performance evaluations.
### Example
Consider you're splitting the Iris dataset:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
If you run this code multiple times with `random_state=42`, you'll always get the same `X_train`, `X_test`, `y_train`, and `y_test`. However, if you remove `random_state` or change its value, the split will be different each time you run the code.

In summary, the `random_state` parameter is a tool for ensuring that the random processes in your code (like splitting a dataset) are reproducible. This is a key aspect of creating reliable and verifiable machine learning models.

3. **Model Training:** We'll use the k-Nearest Neighbors algorithm:
### What is k-Nearest Neighbors (kNN)?
Basic Idea: kNN is a simple, intuitive machine learning algorithm used for classification (and regression). It classifies a new data point based on the majority vote of its 'k' nearest neighbors in the training data.
How it Works:
For a given new data point, the algorithm finds 'k' closest data points (neighbors) in the training set.
It then looks at the labels of these neighbors and assigns the most common label to the new data point.
### Implementing kNN in Python
In Python, you can use scikit-learn's `KNeighborsClassifier` to implement the kNN algorithm. Here’s a basic setup:
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```
### Why Choose `n_neighbors = 3`?
**Trial and Error:** The choice of 'k' (in our case, 3) is often made based on trial and error. You might start with a certain value and adjust it based on the model's performance.
**Balance Between Overfitting and Underfitting:**
    ***- Overfitting:*** A very small value of 'k' (like 1 or 2) can make the model too sensitive to noise in the training data, leading to overfitting.
    ***- Underfitting:*** Conversely, a very large 'k' might make the model too general, potentially ignoring important patterns, leading to underfitting.
***Odd Number to Avoid Ties:*** In classification, using an odd number for 'k' can be helpful to avoid ties, i.e., situations where an equal number of neighbors belong to different classes.
**Dataset Characteristics:** For smaller datasets like Iris, a smaller 'k' is often more appropriate. The Iris dataset is relatively simple and well-behaved, so a small value like 3 can work well.
### Fine-Tuning 'k'
It's important to note that there's no one-size-fits-all value for 'k'. The optimal number of neighbors depends on the specifics of the dataset:
***Cross-Validation:*** One way to find the best value of 'k' is through cross-validation, where you test the model with different 'k' values and see which one performs best.
***Consider Data Complexity:*** If your dataset is more complex or has a lot of noise, you might need to adjust 'k' accordingly.

In summary, building the model with kNN involves setting up the classifier with a chosen number of neighbors (in this case, 3), and then fitting the model to your training data. The choice of 'k' is a key decision that can significantly affect the model's performance, and it's often determined through experimentation and considering the nature of your dataset.

4. **Making Predictions:** Now, let's use our model to make predictions on the test data:
**The `predict` Function**

Once you've trained your kNN model, you can use it to make predictions on new data. This is where the predict function comes into play.

Here's how it works:
1. ***Input to `predict`:*** The input to the predict function is the set of new data points for which you want to predict the output. In our case, this is X_test, which is the subset of your original dataset that you've set aside for testing.
2. ***How `predict` Works:***
    * The kNN algorithm looks at each data point in `X_test`.
    * For each data point, it finds the 'k' nearest neighbors in the training data (`X_train`).
    * It then looks at the labels of these 'k' nearest neighbors.
    * The most common label among these neighbors is assigned as the predicted label for the data point.
3. ***Output of predict:*** The output is a series of predicted labels, one for each data point in your test set. These labels are in the same order as the data points in X_test.

**Example Code**
```python
# Making predictions on the test data
predictions = knn.predict(X_test)

# Printing the predictions
print("Predictions:", predictions)
```
In this code snippet, `knn` is our trained kNN model. We're calling its `predict` method and passing `X_test` as the argument. The model then predicts the labels for each data point in `X_test`, returning them as predictions.

**Understanding the Predictions**
* The `predictions` array contains the predicted species (as integers) for each iris in the test set.
* These predictions can then be compared to the actual labels (`y_test`) to evaluate how well our model is performing.
### Significance of Making Predictions
* ***Testing Model Performance:*** Making predictions on the test data is a crucial step in the machine learning workflow. It helps in evaluating how well your model performs on unseen data.
* ***Real-world Applications:*** In practical scenarios, this step simulates how the model would perform when deployed in a real-world environment, making predictions on new data.

5. **Model Evaluation:** Finally, evaluate the accuracy of your model:
**The Concept of Model Evaluation**
Model evaluation in machine learning involves assessing how well your model's predictions match up against the actual outcomes. It's a key step to understand the accuracy and effectiveness of your model.

**Using the `score` Method in kNN**
For classification tasks, like our Iris species prediction, one common way to evaluate the model is by calculating its accuracy, which is the proportion of correct predictions among the total number of cases evaluated.

Scikit-learn's kNN classifier provides a `score` method, which conveniently does this for us:
```python
# Evaluating the model
accuracy = knn.score(X_test, y_test)

# Printing the accuracy
print("Test set accuracy: {:.2f}".format(accuracy))
```
**How Does the `score` Method Work?**
1. ***Input:*** The method takes two arguments: the test dataset (`X_test`) and the true labels for this dataset (`y_test`).
2. ***Process:*** The `score` method first uses the trained model to make predictions on `X_test`. It then compares these predictions with the true labels (`y_test`).
3. ***Output:*** It returns the accuracy, which is the fraction of correct predictions out of all predictions made.

**Accuracy as a Metric**
* ***Definition:*** Accuracy is calculated as the number of correct predictions divided by the total number of predictions.
* ***Interpretation:*** An accuracy of 1.0 means perfect accuracy, while an accuracy of 0.0 means the model failed to make any correct predictions.
* ***Limitations:*** While accuracy is intuitive and widely used, it's not always the best measure, especially for imbalanced datasets where one class is much more frequent than others. In such cases, other metrics like precision, recall, and the F1 score are also considered.

### Importance of Model Evaluation
* ***Model Validation:*** This step validates how well your model performs on data it hasn't seen during training (generalization).
* ***Model Improvement:*** By understanding your model's performance, you can make informed decisions about how to improve it, such as adjusting parameters, choosing a different model, or collecting more data.


----
## Results
Here is the results:
```plaintext
Predictions: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
 0 0 0 2 1 1 0 0]
 ```
The kNN model achieved an accuracy of `1.00`, showcasing its effectiveness in classifying iris species based on the provided features.
```plaintext
Test set accuracy: 1.00
```
## Missing Step!
Visualizing the data before deciding on the model parameters, including 'k' in the k-Nearest Neighbors (kNN) algorithm, is often a wise and insightful approach. Let's explore this in more detail:

### Data Visualization Before Model Building
**Why Visualize Data First?**
1. ***Understanding Data Structure:*** Visualization helps in understanding the structure, trends, and relationships within your data. This can be particularly useful in a kNN context, where the choice of 'k' might be influenced by how closely grouped or spread out the data points are.
2. ***Identifying Patterns and Anomalies:*** Through plots and graphs, you can identify patterns, correlations, or even anomalies and outliers in your dataset, which can significantly impact your model choice and parameters.
3. ***Feature Relationships:*** Visualizing how features relate to each other and to the target variable can provide insights into which features are important and how they might influence the model.

Example: Visualizing the Iris Dataset
For the Iris dataset, scatter plots can be particularly useful. You might plot different pairs of features against each other and color the points according to the iris species. This can give you a sense of how well the species are separated based on different feature combinations.

Impact on Choosing 'k'
* Clustered Data: If the data points for each class are tightly clustered, a smaller 'k' might work well since each data point's immediate neighbors are likely to belong to the same class.
* Overlapping Data: If there's considerable overlap between classes, a larger 'k' might help in smoothing out the noise, as the model considers more neighbors for making a decision.
**Is It Always Necessary?**
* ***Context-Dependent:*** In some scenarios, especially with familiar datasets like Iris, one might start with a standard approach (like `k=3` for kNN) as a baseline. This is often due to prior knowledge about the dataset's behavior.
* ***Exploratory vs. Confirmatory Analysis:*** Sometimes, the goal is more exploratory, where you're experimenting with different models and parameters to see what works best. Other times, you might have a more confirmatory approach, where you're testing specific hypotheses or models based on prior insights or theories.

In summary, while starting with a default or heuristic choice for 'k' can be a practical approach, especially in educational or straightforward examples, in real-world applications, a more data-driven approach that includes initial data visualization can provide deeper insights and lead to more informed modeling decisions.

## Conclusion
This project demonstrates the basic steps involved in a machine learning task using Python and scikit-learn. The kNN algorithm, with its simplicity and efficacy, provides a strong foundation for classification problems.

## Author
Vahid Keshmiri
