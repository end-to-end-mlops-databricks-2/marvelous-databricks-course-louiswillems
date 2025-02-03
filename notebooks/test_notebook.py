# Databricks notebook source

# MAGIC %md
# MAGIC # Hotel Reservation Prediction Exercise
# MAGIC
# MAGIC This notebook demonstrates how to predict Hotel Reservation Status using the Hotel Reservation dataset. We'll go through the process of loading data, preprocessing, model creation, and visualization of results.
# MAGIC
# MAGIC ## Importing Required Libraries
# MAGIC
# MAGIC First, let's import all the necessary libraries.

# COMMAND ----------
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Only works in a Databricks environment if the data is there
filepath = "/Volumes/mlops_dev/louiswil/wine_quality_data/wine-quality-white-and-red.csv"
# Load the data
df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
filepath = "../data/wine-quality-white-and-red.csv"
# Load the data
df = pd.read_csv(filepath)
df.head(2)

# COMMAND ----------

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print(config.get("catalog_name"))
num_features = config.get("num_features")
print(num_features)


# MAGIC ## Preprocessing

# COMMAND ----------

# Remove rows with missing target

# Handle missing values and convert data types as needed
df["fixed acidity"] = pd.to_numeric(df["fixed acidity"], errors="coerce")

median_no_of_previous_cancellations = df["alcohol"].median()
df["alcohol"].fillna(median_no_of_previous_cancellations, inplace=True)

# Handle numeric features
num_features = config.get("num_features")
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values with mean or default values
df.fillna(
    {
        "citric acid": df["citric acid"].mean(),
        # "type_of_meal_plan": "None",
        "sulphates": 0,
    },
    inplace=True,
)

# Convert categorical features to the appropriate type
cat_features = config.get("cat_features")
for cat_col in cat_features:
    df[cat_col] = df[cat_col].astype("category")

# Extract target and relevant features
target = config.get("target")
# relevant_columns = cat_features + num_features + [target]

df["Id"] = range(1, len(df) + 1)
relevant_columns = cat_features + num_features + [target] + ["Id"]
print(relevant_columns)

df = df[relevant_columns]
df["Id"] = df["Id"].astype("str")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

df.head(2)
