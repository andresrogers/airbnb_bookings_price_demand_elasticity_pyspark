# Airbnb Pricing & Demand Forecasting (PySpark)

End-to-end PySpark project for exploring Airbnb listing data, engineering features, and training demand-forecasting models. Companion notebooks **01–03** cover EDA → feature engineering → model training/selection, with artifacts saved under [`outputs/`](outputs/) and trained models under [`models/`](models/).

## Project contents (focus: notebooks 01–03)

- **EDA**: [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)  
  Produces summary tables/figures (e.g., price stats, deciles by city) and saves reproducible reporting artifacts to [`outputs/`](outputs/).
- **Feature engineering**: [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)  
  Builds a model-ready feature set from raw/cleaned inputs and prepares data for training.
- **Demand forecasting / modeling**: [`notebooks/03_demand_forecasting.ipynb`](notebooks/03_demand_forecasting.ipynb)  
  Trains and compares models (sorted by RMSE), selects the best performer (e.g., `GBTRegressor` in the notebook run), and demonstrates an optional TensorFlow baseline.

## Repository structure

Key paths:

- Data: [`data/`](data/) and processed outputs in [`data/processed/`](data/processed/)
- Notebooks: [`notebooks/`](notebooks/)
- Spark pipeline: [`src/data_processing/spark_pipeline.py`](src/data_processing/spark_pipeline.py)
- Training entry point: [`src/train.py`](src/train.py)
- Dashboard: [`streamlit_app.py`](streamlit_app.py)
- Saved models: [`models/`](models/)
- Generated artifacts: [`outputs/`](outputs/)

## Dataset schema

Documented columns (see table below) are used throughout notebooks 01–03 for analysis and modeling.

| Column name | Description |
|---|---|
| listing_price | The total price of the Airbnb listing. (Numeric) |
| room_type | The type of room being offered (e.g. private, shared, etc.). (Categorical) |
| room_shared | Whether the room is shared or not. (Boolean) |
| room_private | Whether the room is private or not. (Boolean) |
| person_capacity | The maximum number of people that can stay in the room. (Numeric) |
| host_is_superhost | Whether the host is a superhost or not. (Boolean) |
| multi | Whether the listing is for multiple rooms or not. (Boolean) |
| biz | Whether the listing is for business purposes or not. (Boolean) |
| cleanliness_rating | The cleanliness rating of the listing. (Numeric) |
| guest_satisfaction_overall | The overall guest satisfaction rating of the listing. (Numeric) |
| bedrooms | The number of bedrooms in the listing. (Numeric) |
| city_center_dist | The distance from the city centre. (Numeric) |
| metro_dist | The distance from the nearest metro station. (Numeric) |
| weekend | Booking was weekday or weekend. (Boolean) |
| n_bookings | Total bookings per year. (Numeric) |

## Setup

### Requirements

- Python (notebooks show 3.10.x)
- Java (required by Spark)
- PySpark (installed via [`requirements.txt`](requirements.txt))

Install dependencies:

Install steps (Windows, concise)

Create & activate venv:

```bash
python -m venv venv
.\venv\Scripts\activate
Install PyTorch wheel first (choose one):
```

CUDA 12.4:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 -f https://download.pytorch.org/whl/cu124/torch_stable.html
```

CPU-only:

```bash
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

Then install remaining requirements:

```bash
pip install -r requirements.txt
```

Windows PySpark setup (required)
Download winutils.exe (Hadoop 3.3.1):
https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.1/bin/winutils.exe

## Notebooks workflow (recommended)
1. Run notebooks/01_data_exploration.ipynb to generate EDA tables/plots and validate distributions.
2. Run notebooks/02_feature_engineering.ipynb to create model features and a training-ready dataset.
2. Run notebooks/03_demand_forecasting.ipynb to train/compare models and persist the best model under models/.

## Outputs & artifacts
Reporting artifacts (CSVs/PNGs/JSON) are written to `outputs/`.

## Notes on reproducibility
Notebooks are designed to save “report-ready” outputs (tables/figures) to disk so results can be reviewed without rerunning full Spark jobs.

For consistent results, keep the same Python/Spark versions (Python 3.10.11, PySpark 3.5.0).
