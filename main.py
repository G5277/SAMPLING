import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

Z = 1.96  
p = 0.5   
E = 0.05  

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

# Step 2: Balance the dataset
# Assuming the target column is named 'Class'
majority = data[data['Class'] == 0]
minority = data[data['Class'] == 1]

# Upsample minority class
minority_upsampled = resample(minority, 
                               replace=True, 
                               n_samples=len(majority), 
                               random_state=42)

balanced_data = pd.concat([majority, minority_upsampled])

# Calculate number of strata
num_strata = balanced_data['Class'].nunique()

# Calculate sample sizes
simple_random_size = int((Z**2 * p * (1 - p)) / (E**2))
stratified_size = int((Z**2 * p * (1 - p)) / ((E / num_strata)**2))
cluster_size = int((Z**2 * p * (1 - p)) / ((E / 5)**2))  

# Sampling Techniques
simple_random_sample = balanced_data.sample(n=simple_random_size, random_state=0)
stratified_sample = balanced_data.groupby('Class', group_keys=False).apply(
    lambda x: x.sample(min(int(stratified_size / num_strata), len(x)), random_state=0)
)
clusters = random.sample(list(balanced_data['Class'].unique()), 2)
cluster_sample = balanced_data[balanced_data['Class'].isin(clusters)]

# Systematic Sampling
step = len(balanced_data) // simple_random_size
systematic_sample = balanced_data.iloc[::step, :]

# Bootstrap Sampling
bootstrap_sample = balanced_data.sample(n=simple_random_size, replace=True, random_state=0)

# Models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "SVM": SVC(random_state=0),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Naive Bayes": GaussianNB()
}

# Store results
results = []

samples = {
    "Simple Random Sampling": simple_random_sample,
    "Stratified Sampling": stratified_sample,
    "Cluster Sampling": cluster_sample,
    "Systematic Sampling": systematic_sample,
    "Bootstrap Sampling": bootstrap_sample
}

# Train models on each sampling method
for sampling_name, sample in samples.items():
    X_sample = sample.drop(columns='Class')
    y_sample = sample['Class']
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=0)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            "Sampling Technique": sampling_name,
            "Model": model_name,
            "Accuracy": accuracy * 100
        })

# Convert results to a DataFrame and display as a table
results_df = pd.DataFrame(results)
results_table = results_df.pivot(index="Model", columns="Sampling Technique", values="Accuracy")
print(results_table)

# Save results as CSV
results_table.to_csv("sampling_results.csv")