# NexGen Sustainability Tracker - Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Function Reference](#function-reference)
6. [Data Schema](#data-schema)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Overview
The NexGen Sustainability Tracker is a Streamlit-based web application for tracking and optimizing carbon emissions in logistics operations. It processes data from orders, routes, vehicles, and costs to provide insights, recommendations, and simulations for reducing environmental impact.

Key components:
- **app.py**: Main Streamlit application with UI and dashboards.
- **data_processing.py**: Core logic for data loading, emissions calculation, analytics, and AI features.
- **datasets/**: Directory containing CSV files for input data.

## Architecture
The application follows a modular architecture:
- **Data Layer**: CSV files loaded via `load_data()` in `data_processing.py`.
- **Processing Layer**: Functions for computations (e.g., `compute_emissions`, `recommend_greener_operations`).
- **Presentation Layer**: Streamlit UI in `app.py` for visualization and interaction.
- **Caching**: Uses `@st.cache_data` for performance optimization.

Dependencies: Streamlit, Pandas, Plotly, scikit-learn, etc. (see `requirements.txt`).

## Installation
1. Ensure Python 3.8+ is installed.
2. Clone the repository: `git clone <repo-url>`
3. Navigate to the directory: `cd nexgen-sustainability-tracker`
4. Install dependencies: `pip install -r requirements.txt`
5. Place CSV datasets in `datasets/` folder.
6. Run: `streamlit run app.py`

## Usage
### Running the App
- Launch with `streamlit run app.py`.
- Access via browser at `http://localhost:8501`.

### User Interface
- **Sidebar Filters**: Adjust date range, origin, destination, vehicle type, status.
- **Tabs**: AI Assistant, Dashboard, Emissions, Fleet, Routes, Recommendations, Compare.
- **Charts**: Interactive Plotly visualizations for trends, hotspots, etc.
- **Downloads**: Export CSV/HTML reports.

### Key Workflows
1. Load data and view KPIs.
2. Apply filters for targeted analysis.
3. Explore recommendations and simulations.
4. Use AI Assistant for queries.

## Function Reference
Detailed reference for functions in `data_processing.py`.

### Data Loading
- `load_data(base_dir=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - Loads orders, routes, vehicles, costs CSVs.
  - Standardizes columns to snake_case.
  - Returns: (orders_df, routes_df, vehicles_df, costs_df)

### Emissions Calculation
- `compute_emissions(orders, routes, emission_factor_kg_per_l=2.68) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]`
  - Computes CO₂ per order using fuel or distance fallback.
  - Returns: (orders_with_emissions, route_summary, kpis)

### Recommendations
- `recommend_greener_operations(orders_with_emissions, vehicles) -> pd.DataFrame`
  - Suggests vehicle switches and operational tips.
  - Returns: DataFrame with recommendations and savings.

### Analytics
- `monthly_emission_trend(orders_with_emissions) -> pd.DataFrame`
  - Aggregates emissions by month.
- `forecast_emissions(trend_df, periods=1) -> pd.DataFrame`
  - Predicts future emissions using linear regression.
- `simulate_emission_scenarios(...) -> Dict[str, float]`
  - Simulates savings from EV switches, replacements, etc.

### AI Features
- `generate_ai_suggestions(orders_with_emissions, vehicles) -> list[str]`
  - Returns list of eco tips.
- `assistant_answer(question, context) -> str`
  - Provides contextual responses to queries.

### Utilities
- `eco_score(orders_with_emissions, vehicles) -> Dict[str, float]`
  - Computes composite eco score (0-100).
- `carbon_offset_estimate(total_kg, cost_per_ton=1500.0) -> float`
  - Estimates offset cost in INR.

## Data Schema
### orders.csv
- `order_id` (str): Unique order identifier.
- `order_date` (str): Date in YYYY-MM-DD.
- `origin` (str): Starting city.
- `destination` (str): Ending city.
- `product_category` (str): Category (e.g., Electronics).
- `priority` (str): Priority level.

### routes_distance.csv
- `order_id` (str): Links to orders.
- `distance_km` (float): Distance traveled.
- `fuel_consumption_l` (float): Fuel used (optional).
- `route` (str): Route identifier.

### vehicle_fleet.csv
- `vehicle_id` (str): Unique vehicle ID.
- `vehicle_type` (str): Type (e.g., Small_Van).
- `co2_emissions_kg_per_km` (float): Emission intensity.
- `age_years` (float): Vehicle age.
- `status` (str): Available/Maintenance.
- `current_location` (str): City.

### cost_breakdown.csv
- `order_id` (str): Links to orders.
- Various cost columns (e.g., `fuel_cost`, `maintenance_cost`).

## Configuration
- Emission factor: Default 2.68 kg/L (adjust in functions).
- Geocoding: Hardcoded city coords in `CITY_COORDS` (extend for more cities).
- Caching: Enabled via `@st.cache_data` for data loading.

## Troubleshooting
- **Data Loading Errors**: Ensure CSVs are in `datasets/` with correct columns.
- **Missing Dependencies**: Run `pip install -r requirements.txt`.
- **Performance Issues**: Reduce data size or disable caching temporarily.
- **Charts Not Loading**: Check Plotly version in `requirements.txt`.

## Contributing
- Fork the repo and create a feature branch.
- Follow PEP 8 for code style.
- Add docstrings to new functions.
- Test changes locally before PR.


