# Airbnb Pricing & Demand Forecasting (PySpark)

End-to-end PySpark project for exploring Airbnb listing data, engineering features, and training demand-forecasting models. Companion notebooks **01–04** cover EDA → Feature engineering → Model training/selection → Dynamic Price Optimization, with artifacts saved under [`outputs/`](outputs/) and trained models under [`models/`](models/).

## Project contents (focus: notebooks 01–04)

- **EDA**: [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)  
  Produces summary tables/figures (e.g., price stats, deciles by city) and saves reproducible reporting artifacts to [`outputs/`](outputs/).
- **Feature engineering**: [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)  
  Builds a model-ready feature set from raw/cleaned inputs and prepares data for training.
- **Demand forecasting / modeling**: [`notebooks/03_demand_forecasting.ipynb`](notebooks/03_demand_forecasting.ipynb)  
  Trains and compares models (sorted by RMSE), selects the best performer (e.g., `GBTRegressor` in the notebook run), and demonstrates Neural Networks using PyTorch and TensorFlow Keras.
- **Dynamic pricing optimization**: [`notebooks/04_price_optimization.ipynb`](notebooks/04_price_optimization.ipynb)  
  Uses Mixed-Integer Linear Programming (**PuLP**) to maximize annual revenue by optimizing prices across 6 segments (3 seasons × 2 day types). Combines ML-predicted baseline occupancy with regime-aware elasticity curves to generate demand forecasts. Outputs optimal pricing schedules, revenue lift analysis, and demand curve visualizations.

## Repository structure

Key paths:

- Data: [`data/`](data/) and processed outputs in [`data/processed/`](data/processed/)
- Notebooks: [`notebooks/`](notebooks/)
- auxiliary functions: [`src/`](src/)
- Saved models: [`models/`](models/)
- Generated artifacts: [`outputs/`](outputs/)

## Dataset schema

Documented columns (see table below) are used throughout notebooks 01–03 for analysis and modeling.

| Column name | Description |
|---|---|
| listing_id | the unique identifier for each Airbnb property (Numeric) |
| listing_price | The total price of the Airbnb listing. (Numeric) |
| room_type | The type of room being offered (e.g. private, shared, etc.). (Categorical) |
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
| season | Booking was High, Shoulder or Low season. (Categorical) |
| days_in_period | total days in period composed of season + weekend/weekdays (Numeric) |
|	days_booked | Total days booked from specific period (Numeric) |
| occupancy_rate | Percent of days booked per period (Numeric) |

## Setup

### Requirements

- Python (notebooks show 3.10.x)
- Java (required by Spark)
- PySpark (installed via [`requirements.txt`](requirements.txt))
- PuLP (for MILP optimization in Notebook 04)

Install dependencies:

Install steps (Windows)

Create & activate venv:

```bash
python -m venv venv
.\venv\Scripts\activate
```
Install PyTorch wheel first (choose one):

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
1. Run `notebooks/01_data_exploration.ipynb` to generate EDA tables/plots, calculate price elasticity by segment, and validate distributions.
2. Run `notebooks/02_feature_engineering.ipynb` to create model features (including elasticity slopes) and a training-ready dataset.
3. Run `notebooks/03_demand_forecasting.ipynb` to train/compare models and persist the best model under `models/`.
4. Run `notebooks/04_price_optimization.ipynb` to optimize pricing strategy using MILP with regime-aware elasticity and generate revenue lift recommendations.

## Outputs & artifacts
Reporting artifacts (CSVs/PNGs/JSON) are written to `outputs/`, including:
- **Notebook 01**: `elasticities_by_segment.json` (price elasticity curves with breakpoints)
- **Notebook 02**: `listing_features.parquet` (engineered features with elasticity slopes)
- **Notebook 03**: Model performance metrics, feature importance plots
- **Notebook 04**: `optimal_prices_paris.csv` (pricing schedule), `price_optimization_summary.json` (business recommendations), demand curve visualizations

## Notes on reproducibility
Notebooks are designed to save “report-ready” outputs (tables/figures) to disk so results can be reviewed without rerunning full Spark jobs.

For consistent results, keep the same Python/Spark versions (Python 3.10.11, PySpark 3.5.0).
