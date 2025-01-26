import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

url = "https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv"
data = pd.read_csv(url)
print(data)

print(data[(data['Class'] == 1)])