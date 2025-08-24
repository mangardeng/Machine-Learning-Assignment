Machine Learning Assignments

This repository contains two assignments demonstrating the application of Supervised Learning (Logistic Regression) and Unsupervised Learning (K-Means Clustering) using Python and Scikit-learn.

📂 Project Overview

Assignment 1 – Logistic Regression

Objective: Build a classification model to predict customer loan status (Load_status) based on age and loan amount.

Techniques used: Logistic Regression, Confusion Matrix, Accuracy Score, ROC Curve, AUC.

Assignment 2 – K-Means Clustering

Objective: Apply clustering to segment customers based on income and spending behavior.

Techniques used: Elbow Method, K-Means Clustering, Cluster Visualization, Cluster Centroids.

⚙️ Requirements

Install the required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
🚀 How to Run

Download the datasets:

Implementing Logistic Recursition.csv

K-Mean Clustering Spending.csv

Update the file_path in each script to match your dataset location.

Run each Python script:
python logistic_regression_model.py
python kmeans_clustering.py
📊 Assignment 1: Logistic Regression
🔎 Workflow

Load dataset and select features (Customer_Age, Customer_Load_Amount).

Split dataset into training and testing sets.

Train Logistic Regression model.

Evaluate model using:

Accuracy Score

Confusion Matrix

ROC Curve & AUC

📌 Example Outputs

Accuracy on Test Set: Displayed in console.

Confusion Matrix: Displays TP, FP, TN, FN.

ROC Curve: Plotted with AUC value.

✅ Conclusion

Logistic Regression successfully classifies loan status with measurable performance using evaluation metrics.

📊 Assignment 2: K-Means Clustering
🔎 Workflow

Load dataset with Income and Spending.

Visualize data distribution.

Apply Elbow Method to find optimal cluster number.

Train K-Means model and assign cluster labels.

Visualize customer groups using Seaborn.

Extract cluster centroids.

📌 Example Outputs

Elbow Curve: Helps select number of clusters.

Clustered Plot: Groups customers into clusters.

Cluster Centers: Printed as array values.

✅ Conclusion

K-Means clustering effectively segments customers into groups, which can be used for targeted marketing and business insights.

🎯 Interview Questions
Logistic Regression

What is Logistic Regression and how does it differ from Linear Regression?

What is the role of the sigmoid function?

What are True Positive, False Positive, True Negative, and False Negative?

What is ROC Curve and AUC, and why are they important?

What assumptions does Logistic Regression make?

K-Means Clustering

What is K-Means and how does it work?

How do you determine the number of clusters?

What is the role of inertia in K-Means?

What are limitations of K-Means?

How do cluster centroids help in interpretation?
✅ Final Note

These two assignments demonstrate the application of both Supervised Learning (classification)
and Unsupervised Learning (clustering) techniques. They provide practical examples of how
machine learning can be applied to real-world datasets for prediction and segmentation tasks.
