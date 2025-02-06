# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/louiswil/wine_quality_data/XXXXXX-0.0.1-py3-none-any.whl

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------
import pandas as pd
import yaml

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))


# Initialize DataProcessor
# filepath = "/Volumes/mlops_dev/louiswil/wine_quality_data/wine-quality-white-and-red.csv"

# Works both locally and in a Databricks environment
filepath = "../data/wine-quality-white-and-red.csv"

# Load the data
pandas_df = pd.read_csv(filepath)
# Initialize DataProcessor
data_processor = DataProcessor(pandas_df, config)

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# COMMAND ----------

if "spark" not in locals():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

data_processor.save_to_catalog(X_train, X_test, spark)


# COMMAND ----------
