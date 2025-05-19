#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#upload a CSV file
from google.colab import files
uploaded = files.upload()

# reading dataset
data = pd.read_csv("diabetes.csv")

# show first 5 row
data.head()

# shape
data.shape

# dataframe info
data.info()

# check duplicates
data.duplicated().sum()

# check nulls
data.isna().sum()

# check outliers
data.describe().round(2)

# Make a copy of the data
data = data.copy()

# Create a set to store indices of all rows with outliers
outlier_indices = set()

# Loop through all columns except the target column 'Outcome'
for col in data.columns:
    if col == 'Outcome':
        continue
    
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Get indices of outliers
    outliers = data[(data[col] < lower) | (data[col] > upper)].index
    outlier_indices.update(outliers)

# Drop all rows with any outliers
data = data.drop(index=outlier_indices)

print(f"New shape of data after removing outliers from all columns: {data.shape}")

# Distribution of Glucose
sns.histplot(data['Glucose'], kde=True, color="blue")
plt.title('Distribution of Glucose')
plt.show()

sns.histplot(data['BMI'], kde=True)
plt.title('Distribution of BMI')
plt.show()

sns.boxplot(x='Outcome', y='Glucose', data=data)
plt.title('Glucose by Diabetes Outcome')
plt.show()

sns.pairplot(data[['Glucose', 'BMI', 'Age', 'Outcome']], hue='Outcome')
plt.show()

sns.countplot(x='Outcome', data=data)
plt.title('Diabetes Outcome Count')
plt.show()

sns.violinplot(x='Outcome', y='BMI', data=data)
plt.title('BMI Distribution by Outcome')
plt.show()

# Age vs Glucose with Outcome
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=data)
plt.title('Age vs Glucose with Outcome')
plt.show()

# Calculate the correlation matrix
corr = data.corr()

# Plot the heatmap without annotation numbers
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.title('Correlation Heatmap (without numbers)')
plt.show()
