"""PySpark ingestion pipeline.

This version loads a single prepared CSV:
    data/airbnb_listings.csv

Expected columns:
    listing_price, room_type, room_shared, room_private, person_capacity,
    host_is_superhost, multi, biz, cleanliness_rating, guest_satisfaction_overall,
    bedrooms, city_center_dist, metro_dist, city, weekend, n_bookings

It then:
    - standardizes column names to snake_case
    - casts booleans to 0/1
    - writes parquet to data/processed/airbnb_listings.parquet and consolidated.parquet
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

class AirbnbSparkPipeline:
    def __init__(
        self,
        data_path: str,
        spark: Optional[SparkSession] = None,
    ):
        self.input_csv_path = data_path
        self.spark = spark or (
            SparkSession.builder.appName("AirbnbDynamicPricing").config("spark.driver.memory", "4g").getOrCreate()
        )

    def load_airbnb_listings(self):
        csv_path = Path(self.input_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")

        return self.spark.read.option("header", "true").option("inferSchema", "true").csv(str(csv_path))

    @staticmethod
    def _to_snake(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()

    @staticmethod
    def _sanitize_category_value(value: str) -> str:
        value = (value or "").strip().lower()
        value = re.sub(r"[^0-9a-z]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "unknown"

    @staticmethod
    def _cast_bool_to_int(col_expr):
        # Handles True/False, "true"/"false", 1/0, "1"/"0"
        s = F.lower(F.col(col_expr) if isinstance(col_expr, str) else col_expr.cast("string"))
        return (
            F.when(s.isin(["true", "1", "t", "yes", "y"]), F.lit(1))
            .when(s.isin(["false", "0", "f", "no", "n"]), F.lit(0))
            .otherwise(F.lit(None))
            .cast("int")
        )

    def clean_and_transform(self, df):
        # 1) snake_case column names
        rename_expr = [F.col(c).alias(self._to_snake(c)) for c in df.columns]
        df = df.select(*rename_expr)

        # 2) required casts (tolerant if column missing)
        numeric_double = [
            "listing_price",
            "person_capacity",
            "cleanliness_rating",
            "guest_satisfaction_overall",
            "city_center_dist",
            "metro_dist",
            "bedrooms",
        ]
        for c in numeric_double:
            if c in df.columns:
                df = df.withColumn(c, F.col(c).cast(DoubleType()))

        if "n_bookings" in df.columns:
            df = df.withColumn("n_bookings", F.col("n_bookings").cast(IntegerType()))

        # 3) booleans -> 0/1
        bool_cols = ["room_shared", "room_private", "host_is_superhost", "weekend"]
        for c in bool_cols:
            if c in df.columns:
                df = df.withColumn(c, self._cast_bool_to_int(c))

        # multi/biz sometimes are already 0/1; normalize if needed
        for c in ["multi", "biz"]:
            if c in df.columns:
                df = df.withColumn(c, F.col(c).cast(IntegerType()))

        return df

    def run_pipeline(self):
        df = self.load_airbnb_listings()
        df = self.clean_and_transform(df)

        # Write parquet outputs
        project_root = Path.cwd()
        for p in [project_root] + list(project_root.parents):
            if (p / "src").exists():
                project_root = p
                break

        processed_dir = project_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        out_main = processed_dir / "airbnb_listings.parquet"
        out_alias = processed_dir / "consolidated.parquet"

        df.write.mode("overwrite").parquet(str(out_main))
        df.write.mode("overwrite").parquet(str(out_alias))

        print("Saved:", out_main)
        print("Saved (alias for notebooks):", out_alias)
        return df

if __name__ == "__main__":
    p = AirbnbSparkPipeline()
    p.run_pipeline()