from __future__ import annotations

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Iterable, Optional, List

class SparkFeatureEngineer:
    """Spark-native feature engineering utilities.

    Notes
    -----
    - The Airbnb dataset in this repo is cross-sectional (no booking_date column),
      so true seasonality/time-series lags are not available. This class focuses on:
        - price/quality/distance interactions
        - competitive indices within market segments
        - geo bucketing (grid cells)
        - aggregated segment features for optimization
    """

    def __init__(self, spark=None):
        self.spark = spark

    @staticmethod
    def _ensure_cols(df, required: Iterable[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _price_col(df) -> str:
        if 'listing_price' in df.columns:
            return 'listing_price'
        if 'realsum' in df.columns:
            return 'realsum'
        raise ValueError("No price column found (expected 'listing_price' or 'realsum').")

    @staticmethod
    def _dist_col(df) -> Optional[str]:
        if 'city_center_dist' in df.columns:
            return 'city_center_dist'
        if 'metro_dist' in df.columns:
            return 'metro_dist'
        return None
    
    @staticmethod
    def encode_categoricals(df, categorical_cols: Optional[Iterable[str]] = None, drop_originals: bool = True):
        """One-hot encode categorical columns using Spark SQL (big-data compatible).
        
        Args:
            df: Spark DataFrame
            categorical_cols: columns to encode; if None, auto-detects string columns
            drop_originals: if True (default), drop original categorical columns after encoding
        
        Returns:
            Spark DataFrame with one-hot encoded categoricals (originals dropped if drop_originals=True)
        
        Example:
            df = SparkFeatureEngineer.encode_categoricals(df, ['room_type', 'city'], drop_originals=True)
            # Output columns: cat_room_type_private_room, cat_city_amsterdam (room_type, city dropped)
        """
        from pyspark.sql import functions as F
        
        if categorical_cols is None:
            # Auto-detect string columns (likely categorical)
            categorical_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == 'string']
        
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        
        if not categorical_cols:
            return df
        
        out = df
        
        for col in categorical_cols:
            # Get distinct values using Spark SQL (no RDD lambdas)
            distinct_vals = [row[0] for row in out.select(col).distinct().collect()]
            distinct_vals = sorted([v for v in distinct_vals if v is not None])
            
            # Create binary columns for each category
            for val in distinct_vals:
                safe_val = (val or "unknown").strip().lower()
                safe_val = re.sub(r"[^0-9a-z]+", "_", safe_val).strip("_")
                new_col_name = f"cat_{col}_{safe_val}"
                
                out = out.withColumn(
                    new_col_name,
                    F.when(F.col(col) == F.lit(val), F.lit(1)).otherwise(F.lit(0)).cast('int')
                )
            
            # Drop original categorical column if requested
            if drop_originals:
                out = out.drop(col)
        
        return out

    @staticmethod
    def add_listing_features(df):
        """Add listing-level engineered features using Spark SQL functions.
        
        Features created:
        - log_price: log(1 + price) for price normalization
        - price_per_person: price / person_capacity (per-guest cost)
        - price_per_bedroom: price per bedroom
        - capacity_bin: categorical binning of person_capacity
        - quality_score: composite satisfaction/cleanliness/distance score (0-100)
        - price_x_satisfaction: price × satisfaction interaction
        - price_per_dist_km: price per km from city center
        - log_metro_dist: log distance to metro
        - host_is_superhost: binary (cast to int)
        - multi, biz: property feature indicators
        - log_bookings: log(n_bookings) demand signal
        - is_weekend: binary weekend indicator (weekend column kept for segmentation)
        """
        from pyspark.sql import functions as F

        price_col = SparkFeatureEngineer._price_col(df)
        dist_col = SparkFeatureEngineer._dist_col(df)

        out = df

        # 1) Price transformations
        out = out.withColumn('log_price', F.log1p(F.col(price_col).cast('double')))

        # 2) Price normalization by capacity
        if set([price_col, 'person_capacity']).issubset(out.columns):
            out = out.withColumn(
                'price_per_person',
                F.col(price_col).cast('double') / F.greatest(F.col('person_capacity').cast('double'), F.lit(1.0)),
            )

        # 3) Estimate per-unit space from bedrooms (proxy for property size / scale)
        if set([price_col, 'bedrooms']).issubset(out.columns):
            out = out.withColumn(
                'price_per_bedroom',
                F.col(price_col).cast('double') / F.greatest(F.col('bedrooms').cast('double'), F.lit(1.0)),
            )

        # 4) Capacity binning
        if 'person_capacity' in out.columns:
            out = out.withColumn(
                'capacity_bin',
                F.when(F.col('person_capacity') <= 1, F.lit('1'))
                .when(F.col('person_capacity') == 2, F.lit('2'))
                .when(F.col('person_capacity') == 3, F.lit('3'))
                .when(F.col('person_capacity') == 4, F.lit('4'))
                .otherwise(F.lit('5')),
            )

        # 5) Quality score: weighted composite of satisfaction, cleanliness, proximity
        if dist_col and set(['guest_satisfaction_overall', 'cleanliness_rating', dist_col]).issubset(out.columns):
            out = out.withColumn(
                'quality_score',
                F.least(
                    F.lit(100.0),
                    F.greatest(
                        F.lit(0.0),
                        (F.col('guest_satisfaction_overall').cast('double') * F.lit(0.5))
                        + (F.col('cleanliness_rating').cast('double') * F.lit(0.3))
                        + (F.greatest(F.lit(0.0), F.lit(5.0) - F.col(dist_col).cast('double')) * F.lit(4.0)),
                    ),
                ),
            )

        # 6) Price × satisfaction interaction (higher satisfaction justifies premium)
        if set([price_col, 'guest_satisfaction_overall']).issubset(out.columns):
            out = out.withColumn(
                'price_x_satisfaction',
                F.col(price_col).cast('double') * F.col('guest_satisfaction_overall').cast('double') / F.lit(100.0),
            )

        # 7) Price per distance (accessibility premium)
        if dist_col and set([price_col, dist_col]).issubset(out.columns):
            out = out.withColumn(
                'price_per_dist_km',
                F.col(price_col).cast('double') / (F.col(dist_col).cast('double') + F.lit(0.1)),
            )

        # 8) Metro accessibility as separate feature
        if 'metro_dist' in out.columns:
            out = out.withColumn(
                'log_metro_dist',
                F.log1p(F.col('metro_dist').cast('double')),
            )

        # 9) Host quality (superhost premium)
        if 'host_is_superhost' in out.columns:
            out = out.withColumn('host_is_superhost', F.col('host_is_superhost').cast('int'))

        # 10) Property features (business/multi-room indicators)
        for col in ['multi', 'biz']:
            if col in out.columns:
                out = out.withColumn(col, F.col(col).cast('int'))

        # 11) Booking demand signal
        if 'n_bookings' in out.columns:
            out = out.withColumn(
                'log_bookings',
                F.log1p(F.col('n_bookings').cast('double')),
            )
        
        # 12) Weekend premium indicator (keep weekend for now; will drop after segmentation)
        if 'weekend' in out.columns:
            out = out.withColumn('is_weekend', F.col('weekend').cast('int'))

        return out

    @staticmethod
    def add_market_competition_features(df, segment_cols: Optional[Iterable[str]] = None):
        """Join segment-level market stats back onto listings for competitive pricing indices.
        
        Computed statistics per segment (city/day_type/room_type):
        - segment_listing_count: supply in the segment
        - segment_avg_price, segment_median_price: competitive benchmarks
        - segment_price_range: p90 - p10 (market volatility)
        
        Derived features:
        - price_vs_segment_median: relative positioning (>1 = premium)
        - price_vs_segment_avg: average relative positioning
        - segment_price_spread: volatility in competitive set
        """
        from pyspark.sql import functions as F

        if segment_cols is None:
            segment_cols = [c for c in ['city', 'weekend'] if c in df.columns]

        price_col = SparkFeatureEngineer._price_col(df)
        SparkFeatureEngineer._ensure_cols(df, list(segment_cols) + [price_col])

        seg_stats = (
            df.groupBy(*segment_cols)
            .agg(
                F.count('*').alias('segment_listing_count'),
                F.mean(price_col).alias('segment_avg_price'),
                F.expr(f'percentile_approx({price_col}, 0.5)').alias('segment_median_price'),
                F.expr(f'percentile_approx({price_col}, 0.9)').alias('segment_p90_price'),
                F.expr(f'percentile_approx({price_col}, 0.1)').alias('segment_p10_price'),
                F.stddev(price_col).alias('segment_price_std'),
            )
            .withColumn('segment_price_spread', F.col('segment_p90_price') - F.col('segment_p10_price'))
        )

        out = df.join(seg_stats, on=list(segment_cols), how='left')
        
        # Relative positioning indices
        out = out.withColumn(
            'price_vs_segment_median',
            F.col(price_col).cast('double') / F.greatest(F.col('segment_median_price'), F.lit(1.0)),
        )
        out = out.withColumn(
            'price_vs_segment_avg',
            F.col(price_col).cast('double') / F.greatest(F.col('segment_avg_price'), F.lit(1.0)),
        )
        
        # Relative spread (how volatile is this segment)
        out = out.withColumn(
            'relative_price_volatility',
            F.col('segment_price_spread') / F.greatest(F.col('segment_median_price'), F.lit(1.0)),
        )
        
        return out

    @staticmethod
    def build_segment_dataset(df, segment_cols: Optional[Iterable[str]] = None):
        """Build a segment-level dataset for optimization and compact modeling.
        
        Segments listings by city and weekend indicator, aggregating key statistics:
        - listing_count: supply size proxy
        - avg_price, median_price: typical listing cost
        - std_price, p10/p90_price: price distribution
        - avg_satisfaction, avg_cleanliness: quality indicators
        - superhost_rate, biz_rate, multi_rate: host/property characteristics
        - avg_bookings: demand proxy
        - log_demand: log(listing_count) for modeling
        """
        from pyspark.sql import functions as F

        if segment_cols is None:
            segment_cols = [c for c in ['city', 'weekend'] if c in df.columns]

        price_col = SparkFeatureEngineer._price_col(df)
        SparkFeatureEngineer._ensure_cols(df, list(segment_cols) + [price_col])

        agg_exprs = [
            F.count('*').alias('listing_count'),
            F.mean(price_col).alias('avg_price'),
            F.expr(f'percentile_approx({price_col}, 0.5)').alias('median_price'),
            F.stddev(price_col).alias('std_price'),
            F.expr(f'percentile_approx({price_col}, 0.1)').alias('p10_price'),
            F.expr(f'percentile_approx({price_col}, 0.9)').alias('p90_price'),
        ]

        # Quality metrics
        optional_means = {
            'guest_satisfaction_overall': 'avg_satisfaction',
            'cleanliness_rating': 'avg_cleanliness',
            'city_center_dist': 'avg_city_center_dist',
            'metro_dist': 'avg_metro_dist',
            'person_capacity': 'avg_capacity',
            'bedrooms': 'avg_bedrooms',
            'price_per_person': 'avg_price_per_person',
            'quality_score': 'avg_quality_score',
        }
        for col_name, alias in optional_means.items():
            if col_name in df.columns:
                agg_exprs.append(F.mean(col_name).alias(alias))

        # Host/property characteristics
        for col in ['host_is_superhost', 'biz', 'multi']:
            if col in df.columns:
                alias = f'{col}_rate'
                agg_exprs.append(F.mean(F.col(col).cast('double')).alias(alias))

        # Demand signal
        if 'n_bookings' in df.columns:
            agg_exprs.append(F.mean('n_bookings').alias('avg_bookings'))

        segment_df = (
            df.groupBy(*segment_cols)
            .agg(*agg_exprs)
            .withColumn('log_demand', F.log1p(F.col('listing_count')))
        )

        return segment_df

    @staticmethod
    def build_feature_dataset(df, segment_cols: Optional[Iterable[str]] = None, encode_categoricals: bool = True):
        """Full Spark feature engineering pipeline: encode → add listing features → add market features → aggregate.
        
        Args:
            df: input Spark DataFrame (raw listings from consolidated.parquet)
            segment_cols: columns to segment by (city, weekend); if None, auto-detect
            encode_categoricals: if True, one-hot encode room_type and city before engineering
        
        Returns:
            (listing_features_df, segment_dataset_df)
            - listing_features_df: full listing-level engineered features with one-hot encoded categoricals
            - segment_dataset_df: compact city × weekend segment aggregates for optimization
        
        Example:
            listing_df, segment_df = SparkFeatureEngineer.build_feature_dataset(
                df, segment_cols=['city', 'weekend'], encode_categoricals=True
            )
        """
        from pyspark.sql import functions as F
        
        out = df
        
        # 1) Encode categoricals (Spark native, big-data style)
        # Keep originals for segmentation downstream (drop_originals=False)
        if encode_categoricals:
            cat_cols_to_encode = [c for c in ['room_type', 'city'] if c in out.columns]
            if cat_cols_to_encode:
                out = SparkFeatureEngineer.encode_categoricals(out, cat_cols_to_encode, drop_originals=False)
        
        # 2) Add listing-level engineered features (keeps weekend for segmentation)
        out = SparkFeatureEngineer.add_listing_features(out)
        
        # 3) Add market competition features (uses original city/weekend columns)
        if segment_cols is None:
            segment_cols = [c for c in ['city', 'weekend'] if c in df.columns]
        
        out = SparkFeatureEngineer.add_market_competition_features(out, segment_cols)
        
        # 4) Build segment dataset for optimization
        segment_df = SparkFeatureEngineer.build_segment_dataset(out, segment_cols)
        
        # 5) Drop originals now that we're done with segmentation
        cols_to_drop = [c for c in ['room_type','city','weekend'] if c in out.columns]
        if cols_to_drop:
            out = out.drop(*cols_to_drop)
        
        return out, segment_df


class FeatureEngineer:
    """Pandas-based feature engineering for downstream modeling pipelines.
    
    Use this for final train/val/test preparation after Spark feature generation.
    Handles scaling, categorical encoding, and train/val/test splits with scikit-learn utilities.
    """

    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []

    def encode_categoricals(self, X: pd.DataFrame, categorical_cols: Optional[List[str]] = None, fit: bool = True) -> pd.DataFrame:
        """One-hot encode categorical columns.
        
        Args:
            X: input dataframe
            categorical_cols: columns to encode; if None, auto-detects object dtype columns
            fit: if True, learn encoder params; if False, apply existing encoder
        
        Returns:
            DataFrame with one-hot encoded categoricals (original columns dropped)
        """
        X = X.copy()
        
        if categorical_cols is None:
            categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
        
        categorical_cols = [c for c in categorical_cols if c in X.columns]
        
        if not categorical_cols:
            return X
        
        if fit:
            # One-hot encode and store the resulting column names
            X_encoded = pd.get_dummies(X[categorical_cols], prefix='cat', drop_first=False)
            self.categorical_cols = categorical_cols
            self.encoders['categorical_columns'] = X_encoded.columns.tolist()
            X = X.drop(columns=categorical_cols)
            X = pd.concat([X, X_encoded], axis=1)
        else:
            if 'categorical_columns' not in self.encoders:
                raise RuntimeError("Categorical encoder not fitted. Call encode_categoricals with fit=True first.")
            # One-hot encode with the same columns as training
            X_encoded = pd.get_dummies(X[categorical_cols], prefix='cat', drop_first=False)
            # Align with training columns (handle unseen categories)
            expected_cols = self.encoders['categorical_columns']
            for col in expected_cols:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[[col for col in expected_cols if col in X_encoded.columns]]
            X = X.drop(columns=categorical_cols)
            X = pd.concat([X, X_encoded], axis=1)
        
        return X

    def scale_features(self, X: pd.DataFrame, feature_cols: Optional[List[str]] = None, fit: bool = True) -> pd.DataFrame:
        """Standardize numeric features to mean=0, std=1.
        
        Args:
            X: input dataframe
            feature_cols: columns to scale; if None, uses all numeric columns
            fit: if True, learn scaler params; if False, apply existing scaler
        
        Returns:
            DataFrame with scaled columns
        """
        X = X.copy()
        
        if feature_cols is None:
            feature_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
        
        feature_cols = [c for c in feature_cols if c in X.columns]
        
        if fit:
            self.scalers['numeric'] = StandardScaler()
            X[feature_cols] = self.scalers['numeric'].fit_transform(X[feature_cols])
            self.numeric_cols = feature_cols
        else:
            if 'numeric' not in self.scalers:
                raise RuntimeError("Scaler not fitted. Call scale_features with fit=True first.")
            X[feature_cols] = self.scalers['numeric'].transform(X[feature_cols])
        
        return X

    def prepare_datasets(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split features and target into train/val/test.
        
        Args:
            X: feature matrix (already encoded/scaled)
            y: target vector
            test_size: fraction for test split (0.2 = 20%)
            val_size: fraction of remaining for validation (applied to 80%)
            random_state: seed for reproducibility
        
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.feature_columns = X.columns.tolist()

        # train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # train/val split from remainder
        val_frac = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_pipeline(
        self, df: pd.DataFrame, target_col: str = 'listing_price'
    ) -> Tuple[Tuple, pd.DataFrame]:
        """Execute full pandas feature pipeline: encode categoricals → scale → split.
        
        Args:
            df: input dataframe (should be output from Spark pipeline)
            target_col: column name for prediction target
        
        Returns:
            ((X_train, X_val, X_test, y_train, y_val, y_test), X_processed)
        """
        print("Encoding categorical features...")
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        X_encoded = self.encode_categoricals(X, fit=True)

        print("Scaling numeric features...")
        X_scaled = self.scale_features(X_encoded, fit=True)

        print("Preparing train/val/test split...")
        datasets = self.prepare_datasets(X_scaled, y)

        print(f"Feature columns: {len(self.feature_columns)}")
        print(f"  Categorical cols (one-hot): {len([c for c in self.feature_columns if c.startswith('cat_')])}")
        print(f"  Numeric cols (scaled): {len(self.numeric_cols)}")
        print(
            f"Train size: {len(datasets[0])}, "
            f"Val size: {len(datasets[1])}, "
            f"Test size: {len(datasets[2])}"
        )

        return datasets, X_scaled



if __name__ == "__main__":
    df = pd.read_parquet("data/processed/consolidated.parquet")
    engineer = FeatureEngineer()
    datasets, processed_df = engineer.run_pipeline(df)
