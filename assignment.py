# =============================================
# Data Analysis & Visualization Assignment
# Using pandas + matplotlib with the Iris dataset
# =============================================

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Task 1: Load and Explore the Dataset
# ---------------------------

# Load dataset (Iris dataset from seaborn or CSV file)
# If you have iris.csv locally, replace with: pd.read_csv("iris.csv")
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("First 5 rows of the dataset:")
print(df.head())   # preview dataset

print("\nDataset info:")
print(df.info())   # data types and null values

print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean dataset (no missing values in iris, but if they existed we’d handle them)
df = df.dropna()  # or df.fillna(...)

# ---------------------------
# Task 2: Basic Data Analysis
# ---------------------------

print("\nBasic statistical summary:")
print(df.describe())   # summary statistics

# Grouping example: Average petal_length by species
grouped = df.groupby("species")["petal_length"].mean()
print("\nAverage petal_length per species:")
print(grouped)

# Observation example
print("\nObservations:")
print("- Setosa generally has the smallest petal length, while Virginica has the largest.")

# ---------------------------
# Task 3: Data Visualization
# ---------------------------

# Line chart (not time series in Iris, so we simulate with index trend of sepal_length)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal_length"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(6,4))
grouped.plot(kind="bar", color=["skyblue", "orange", "green"])
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(6,4))
plt.hist(df["sepal_width"], bins=15, color="purple", alpha=0.7, edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(6,4))
plt.scatter(df["sepal_length"], df["petal_length"], alpha=0.7, c="red")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
