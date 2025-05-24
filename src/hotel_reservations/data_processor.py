"""Data preprocessing module."""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

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
        """Preprocess features."""
        self.df["arrival_date"] = pd.to_numeric(self.df["arrival_date"], errors="coerce")
        self.df["arrival_month"] = pd.to_numeric(self.df["arrival_month"], errors="coerce")
        self.df["arrival_year"] = pd.to_numeric(self.df["arrival_year"], errors="coerce")

        # Create datetime from components
        self.df["arrival_datetime"] = pd.to_datetime(
            {"year": self.df["arrival_year"], "month": self.df["arrival_month"], "day": self.df["arrival_date"]},
            errors="coerce",
        )

        # Extract datetime features
        self.df["arrival_dayofweek"] = self.df["arrival_datetime"].dt.dayofweek
        self.df["arrival_is_weekend"] = self.df["arrival_dayofweek"].isin([5, 6]).astype(int)
        self.df["arrival_month_num"] = self.df["arrival_datetime"].dt.month  # Already had month, but reconfirm

        # Drop original date columns
        self.df.drop(columns=["arrival_year", "arrival_month", "arrival_date"], inplace=True)

        # Ensure numeric features are in the right type
        num_features = self.config.num_features.copy()

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

        # self.df.drop(columns=["arrival_datetime"], inplace=True, errors="ignore")

    def split_data(self, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets with a temporal split using arrival_datetime.

        The training set comprises the oldest (1 - test_size) fraction of records, while the test
        set consists of the most recent test_size fraction. This is ideal for time-dependent data to
        prevent information leakage from the future.

        :param test_size: The proportion of the dataset to use as the test set.
        :return: A tuple containing the training and test DataFrames.
        """
        # Sort the DataFrame by arrival_datetime
        df_sorted = self.df.sort_values(by="arrival_datetime")

        # Determine the split index. The training set is the earliest portion.
        split_index = int(len(df_sorted) * (1 - test_size))

        # Make temporal splits ensuring no overlap between train and test sets
        train_set = df_sorted.iloc[:split_index].copy()
        test_set = df_sorted.iloc[split_index:].copy()

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

        train_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
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
