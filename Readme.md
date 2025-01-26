# Sampling Techniques for Credit Card Fraud Detection

## Overview

This project demonstrates different sampling techniques applied to a dataset for credit card fraud detection. It includes the following sampling methods:

1. **Simple Random Sampling**
2. **Stratified Sampling**
3. **Cluster Sampling**
4. **Systematic Sampling**
5. **Bootstrap Sampling**

The goal of this project is to evaluate various sampling techniques and assess their performance on classification models using a credit card transaction dataset.

## Sampling Methods

### 1. Simple Random Sampling
Simple Random Sampling involves randomly selecting a sample from the entire dataset, without regard to the class labels (fraud vs non-fraud).

### 2. Stratified Sampling
Stratified Sampling divides the data into subgroups or strata (in this case, based on the class label: fraud or non-fraud), and then samples from each subgroup in a way that ensures the proportion of samples from each class is maintained.

### 3. Cluster Sampling
Cluster Sampling divides the data into distinct groups or clusters (here, based on the class label), and then selects a random sample of clusters. All data points within the selected clusters are included in the sample.

### 4. Systematic Sampling
Systematic Sampling involves selecting every `k`-th data point from the dataset, starting from a random point.

### 5. Bootstrap Sampling
Bootstrap Sampling involves randomly selecting data points with replacement to create a sample of the same size as the original dataset.

## Models Used

The following classification models are used to evaluate the performance of each sampling technique:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Naive Bayes**

Each model is trained on samples created by different sampling techniques, and the accuracy of each model is computed.

## Installation

To run this code on your local machine, make sure to have Python 3.x installed. You will also need to install the following libraries:

```bash
pip install pandas numpy random scikit-learn
```

## Dataset

The dataset used in this project is a **credit card transaction dataset** available at:

[Credit Card Data on GitHub](https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv)

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/G5277/SAMPLING.git
```

2. Navigate to the project directory:
```bash
cd SAMPLING
```

3. Run the main.py script to perform the sampling techniques and model training:
2. Navigate to the project directory:
```bash
python main.py
```

4. The results of the sampling techniques and model performances will be displayed, and a CSV file (sampling_results.csv) will be saved with the accuracies of each model and sampling technique.

## Results
The performance of each model for each sampling technique is displayed in a table, and the results are saved in a CSV file for further analysis.

## License
This project is licensed under the MIT License - see the 
LICENSE file for details.