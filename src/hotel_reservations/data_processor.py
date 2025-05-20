"""Data preprocessing module."""

import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservations.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the hotel reservations dataset."""
        # Convert arrival fields to datetime
        self.df["arrival_date"] = pd.to_numeric(self.df["arrival_date"], errors="coerce")
        self.df["arrival_month"] = pd.to_numeric(self.df["arrival_month"], errors="coerce")
        self.df["arrival_year"] = pd.to_numeric(self.df["arrival_year"], errors="coerce")

        self.df["arrival_datetime"] = pd.to_datetime(
            dict(year=self.df["arrival_year"],
                 month=self.df["arrival_month"],
                 day=self.df["arrival_date"]),
            errors="coerce"
        )

        # Drop original date columns
        self.df.drop(columns=["arrival_year", "arrival_month", "arrival_date"], inplace=True)

        # Ensure numeric features are in the right type
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing numeric values with the mean
        for col in num_features:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Convert categorical features
        cat_features = self.config.cat_features
        for col in cat_features:
            self.df[col] = self.df[col].astype("category")

        # Filter only relevant columns
        relevant_columns = cat_features + num_features + [self.config.target, "Booking_ID"]
        self.df = self.df[relevant_columns]
        self.df["Booking_ID"] = self.df["Booking_ID"].astype("str")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )