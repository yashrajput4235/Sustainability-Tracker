import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.linear_model import LinearRegression
import numpy as np


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to snake_case for consistency."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.lower()
    )
    return df


def load_data(base_dir: Path | str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets with basic validation and consistent schemas.

    Returns: orders_df, routes_df, vehicles_df, costs_df
    """
    base_path = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    ds_path = base_path / "datasets"

    orders = pd.read_csv(ds_path / "orders.csv")
    routes = pd.read_csv(ds_path / "routes_distance.csv")
    vehicles = pd.read_csv(ds_path / "vehicle_fleet.csv")
    costs = pd.read_csv(ds_path / "cost_breakdown.csv")

    # Standardize columns
    orders = _standardize_columns(orders)
    routes = _standardize_columns(routes)
    vehicles = _standardize_columns(vehicles)
    costs = _standardize_columns(costs)

    # Parse dtypes and sanity checks
    if "order_id" not in orders.columns:
        raise ValueError("orders.csv must contain 'Order_ID'")
    if not {"order_id", "distance_km", "fuel_consumption_l"}.issubset(routes.columns):
        raise ValueError("routes_distance.csv must contain 'Order_ID', 'Distance_KM', 'Fuel_Consumption_L'")
    if not {"vehicle_id", "vehicle_type", "co2_emissions_kg_per_km"}.issubset(vehicles.columns):
        raise ValueError("vehicle_fleet.csv must contain 'Vehicle_ID', 'Vehicle_Type', 'CO2_Emissions_Kg_per_KM'")

    # Numeric coercion
    for c in ["distance_km", "fuel_consumption_l"]:
        routes[c] = pd.to_numeric(routes[c], errors="coerce")

    vehicles["co2_emissions_kg_per_km"] = pd.to_numeric(
        vehicles["co2_emissions_kg_per_km"], errors="coerce"
    )

    return orders, routes, vehicles, costs


def compute_emissions(
    orders: pd.DataFrame,
    routes: pd.DataFrame,
    emission_factor_kg_per_l: float = 2.68,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Compute order-level emissions using available fields.

    Strategy:
    - Prefer fuel-based emissions when Fuel_Consumption_L is present: fuel_l * EF (kg/L)
    - Fallback to a conservative distance-based intensity of 0.3 kg/km when fuel is missing

    Returns: (orders_with_emissions, route_summary, kpis)
    """
    df = routes.merge(orders[["order_id", "order_date", "origin", "destination"]], on="order_id", how="left")
    df = df.copy()

    # Distance-based fallback intensity (kg CO2 per km) when fuel is unknown
    fallback_intensity_kg_per_km = 0.30

    fuel_emissions = df["fuel_consumption_l"] * emission_factor_kg_per_l
    distance_emissions = df["distance_km"] * fallback_intensity_kg_per_km

    df["co2_emission_kg"] = fuel_emissions.fillna(distance_emissions)

    # Route identifier for grouping
    if "route" in df.columns:
        df["route_id"] = df["route"].fillna(df["origin"].astype(str) + "-" + df["destination"].astype(str))
    else:
        df["route_id"] = df["origin"].astype(str) + "-" + df["destination"].astype(str)

    # Summaries
    route_summary = (
        df.groupby("route_id", as_index=False)
        .agg(co2_emission_kg=("co2_emission_kg", "sum"), distance_km=("distance_km", "sum"), orders=("order_id", "count"))
        .sort_values("co2_emission_kg", ascending=False)
    )

    # KPIs
    total_emissions = float(df["co2_emission_kg"].sum())
    total_distance = float(df["distance_km"].sum())
    avg_intensity = (total_emissions / total_distance) if total_distance > 0 else 0.0
    kpis = {
        "total_emissions_kg": total_emissions,
        "total_distance_km": total_distance,
        "avg_kg_per_km": avg_intensity,
        "num_orders": int(df["order_id"].nunique()),
    }

    return df, route_summary, kpis


def potential_vehicle_scenarios(
    routes: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate emissions per route across vehicle options using vehicle per-km intensities.
    This is a what-if analysis, not reflecting actual assignments.
    """
    base = routes[["order_id", "distance_km"]].copy()
    vehicles_slim = vehicles[["vehicle_id", "vehicle_type", "co2_emissions_kg_per_km"]].copy()

    # Cartesian join for scenario modeling
    base["key"] = 1
    vehicles_slim["key"] = 1
    scen = base.merge(vehicles_slim, on="key").drop(columns=["key"])
    scen["scenario_emission_kg"] = scen["distance_km"] * scen["co2_emissions_kg_per_km"]
    return scen


def recommend_greener_operations(
    orders_with_emissions: pd.DataFrame,
    vehicles: pd.DataFrame,
    emission_factor_kg_per_l: float = 2.68,
) -> pd.DataFrame:
    """
    Recommend greener vehicle choices and operational tips per order.

    Assumptions:
    - Current emissions are estimated from provided fuel consumption (preferred) or distance fallback.
    - Best-case vehicle is the one with the lowest per-km CO2 intensity.
    - This is an advisory: does not consider capacity or availability constraints.
    """
    df = orders_with_emissions[[
        "order_id", "route_id", "distance_km", "co2_emission_kg", "traffic_delay_minutes"
    ]].copy()
    if "traffic_delay_minutes" not in df.columns:
        df["traffic_delay_minutes"] = pd.NA

    # Pre-sort vehicles by intensity
    veh_sorted = vehicles[[
        "vehicle_id", "vehicle_type", "co2_emissions_kg_per_km", "current_location", "status"
    ]].dropna(subset=["co2_emissions_kg_per_km"]).copy()
    veh_sorted = veh_sorted.sort_values("co2_emissions_kg_per_km", ascending=True)

    recommended_ids = []
    recommended_types = []
    recommended_intensity = []
    notes = []

    for _, row in df.join(orders_with_emissions[["order_id", "origin"]].set_index("order_id"), on="order_id").iterrows():
        origin = row.get("origin")
        # Prefer available vehicles at the same origin
        pool = veh_sorted
        available_at_origin = veh_sorted[
            (veh_sorted["status"].str.lower() == "available") & (veh_sorted["current_location"] == origin)
        ]
        if len(available_at_origin) > 0:
            pool = available_at_origin

        greenest = pool.iloc[0]
        second_best = pool.iloc[1] if len(pool) > 1 else greenest

        recommended_ids.append(greenest["vehicle_id"])
        recommended_types.append(greenest["vehicle_type"])
        recommended_intensity.append(float(greenest["co2_emissions_kg_per_km"]))

        suggestions = []
        if row.get("traffic_delay_minutes") and pd.notna(row["traffic_delay_minutes"]) and row["traffic_delay_minutes"] > 60:
            suggestions.append("Shift to off-peak windows to cut idle fuel burn")
        if row["distance_km"] <= 200 and (veh_sorted["vehicle_type"].eq("Express_Bike").any()):
            suggestions.append("For short-haul, consider bikes/e-van micro-fulfillment")
        # Add vehicle switch suggestion; scenario savings computed after loop
        suggestions.append(
            f"Switch to {greenest['vehicle_type']} ({greenest['co2_emissions_kg_per_km']:.3f} kg/km)"
        )
        notes.append("; ".join(suggestions))

    df["recommended_vehicle_id"] = recommended_ids
    df["recommended_vehicle_type"] = recommended_types
    df["recommended_intensity_kg_per_km"] = recommended_intensity

    # Potential emissions under recommendation
    df["scenario_emission_kg"] = df["distance_km"] * df["recommended_intensity_kg_per_km"]
    df["potential_savings_kg"] = (df["co2_emission_kg"] - df["scenario_emission_kg"]).clip(lower=0)

    df["recommendation_note"] = notes

    # Sort by highest potential savings
    df = df.sort_values("potential_savings_kg", ascending=False)
    return df


def compute_costs(costs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost components into total cost per order."""
    cost_cols = [c for c in costs.columns if c != "order_id"]
    agg = costs.copy()
    agg["total_cost"] = agg[cost_cols].sum(axis=1)
    return agg[["order_id", "total_cost"]]


def cost_emission_analytics(orders_with_emissions: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """
    Combine emissions and costs; compute metrics and Pareto frontier (cost vs CO2).
    """
    costs_total = compute_costs(costs)
    df = orders_with_emissions.merge(costs_total, on="order_id", how="left")
    df["cost_per_km"] = df["total_cost"] / df["distance_km"].replace(0, pd.NA)
    df["cost_per_kg_co2"] = df["total_cost"] / df["co2_emission_kg"].replace(0, pd.NA)

    # Pareto frontier: minimize (total_cost, co2_emission_kg)
    temp = df[["order_id", "total_cost", "co2_emission_kg"]].dropna().sort_values(["total_cost", "co2_emission_kg"], ascending=[True, True])
    best_emission = float("inf")
    efficient_ids = set()
    for _, r in temp.iterrows():
        if r["co2_emission_kg"] < best_emission:
            best_emission = r["co2_emission_kg"]
            efficient_ids.add(r["order_id"])
    df["is_pareto_efficient"] = df["order_id"].isin(efficient_ids)
    return df


# --- Benchmarks, trends, locations, and narratives ---

CITY_COORDS = {
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Bengaluru": (12.9716, 77.5946),
    "Singapore": (1.3521, 103.8198),
    "Bangkok": (13.7563, 100.5018),
    "Dubai": (25.2048, 55.2708),
    "Hong Kong": (22.3193, 114.1694),
}


def percent_fleet_under_years(vehicles: pd.DataFrame, years: int = 3) -> float:
    v = vehicles.copy()
    v["age_years"] = pd.to_numeric(v["age_years"], errors="coerce")
    total = len(v)
    if total == 0:
        return 0.0
    pct = (v["age_years"] < years).sum() / total * 100.0
    return float(pct)


def top_efficient_vehicle_types(vehicles: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    v = vehicles.copy()
    v = v.sort_values("co2_emissions_kg_per_km").groupby("vehicle_type", as_index=False).agg(
        avg_kg_per_km=("co2_emissions_kg_per_km", "mean"), count=("vehicle_id", "count")
    )
    return v.sort_values("avg_kg_per_km").head(n)


def locations_highest_emissions(orders_with_emissions: pd.DataFrame, field: str = "origin", n: int = 5) -> pd.DataFrame:
    field = field if field in ["origin", "destination"] else "origin"
    agg = orders_with_emissions.groupby(field, as_index=False)["co2_emission_kg"].sum()
    return agg.sort_values("co2_emission_kg", ascending=False).head(n)


def monthly_emission_trend(orders_with_emissions: pd.DataFrame) -> pd.DataFrame:
    df = orders_with_emissions.copy()
    # Gracefully handle missing dates
    if "order_date" not in df.columns:
        return pd.DataFrame({"month": pd.to_datetime([]), "co2_emission_kg": []})
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    trend = df.groupby("month", as_index=False)["co2_emission_kg"].sum().sort_values("month")
    return trend


def geocode_city(city: str) -> Tuple[float, float] | Tuple[None, None]:
    return CITY_COORDS.get(city, (None, None))


def city_emission_map(orders_with_emissions: pd.DataFrame, by: str = "origin") -> pd.DataFrame:
    field = by if by in ["origin", "destination"] else "origin"
    agg = orders_with_emissions.groupby(field, as_index=False)["co2_emission_kg"].sum()
    agg[["lat", "lon"]] = agg[field].apply(lambda c: pd.Series(geocode_city(c)))
    agg = agg.dropna(subset=["lat", "lon"]).rename(columns={field: "city"})
    return agg


def narrative_insights(
    orders_with_emissions: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> Dict[str, str]:
    trend = monthly_emission_trend(orders_with_emissions)
    delta_txt = "Insufficient data for trend"
    if len(trend) >= 2:
        last = trend.iloc[-1]["co2_emission_kg"]
        prev = trend.iloc[-2]["co2_emission_kg"]
        if prev > 0:
            change = (last - prev) / prev * 100.0
            arrow = "decreased" if change < 0 else "increased"
            delta_txt = f"This period, CO₂ {arrow} by {abs(change):.1f}% versus previous period."

    # Vehicle efficiency callout
    v = vehicles.sort_values("co2_emissions_kg_per_km")
    if len(v) > 0:
        best = v.iloc[0]
        best_txt = f"Greenest fleet segment is {best['vehicle_type']} at {best['co2_emissions_kg_per_km']:.3f} kg/km."
    else:
        best_txt = "No vehicle data available."

    return {
        "emission_trend": delta_txt,
        "fleet_efficiency": best_txt,
    }


def monthly_savings_vs_prev(orders_with_emissions: pd.DataFrame) -> float:
    """Compute percent change in total CO₂ from previous month to current month."""
    trend = monthly_emission_trend(orders_with_emissions)
    if len(trend) < 2:
        return 0.0
    last = trend.iloc[-1]["co2_emission_kg"]
    prev = trend.iloc[-2]["co2_emission_kg"]
    if prev == 0:
        return 0.0
    return float((prev - last) / prev * 100.0)


def flag_inefficient_vehicles(vehicles: pd.DataFrame, age_years: int = 6, kg_per_km: float = 0.5) -> pd.DataFrame:
    """Vehicles older than threshold and above intensity threshold."""
    v = vehicles.copy()
    v["age_years"] = pd.to_numeric(v["age_years"], errors="coerce")
    v["co2_emissions_kg_per_km"] = pd.to_numeric(v["co2_emissions_kg_per_km"], errors="coerce")
    mask = (v["age_years"] > age_years) & (v["co2_emissions_kg_per_km"] > kg_per_km)
    return v.loc[mask, ["vehicle_id", "vehicle_type", "age_years", "co2_emissions_kg_per_km", "current_location", "status"]]


def flag_inefficient_routes(orders_with_emissions: pd.DataFrame, intensity_threshold: float = 0.5) -> pd.DataFrame:
    """Routes with average kg/km above threshold."""
    df = orders_with_emissions.copy()
    df["intensity_kg_per_km"] = df["co2_emission_kg"] / df["distance_km"].replace(0, pd.NA)
    agg = df.groupby("route_id", as_index=False).agg(
        avg_intensity=("intensity_kg_per_km", "mean"),
        total_co2=("co2_emission_kg", "sum"),
        distance_km=("distance_km", "sum"),
        orders=("order_id", "count"),
    )
    return agg[agg["avg_intensity"] > intensity_threshold].sort_values("avg_intensity", ascending=False)


def vehicle_status_map(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Return vehicle locations with lat/lon for mapping."""
    v = vehicles.copy()
    v[["lat", "lon"]] = v["current_location"].apply(lambda c: pd.Series(geocode_city(str(c))))
    v = v.dropna(subset=["lat", "lon"]) 
    return v[["vehicle_id", "vehicle_type", "status", "current_location", "lat", "lon"]]


def summarize_actions(
    orders_with_emissions: pd.DataFrame,
    vehicles: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> Dict[str, str]:
    inefficient_veh = flag_inefficient_vehicles(vehicles)
    inefficient_routes = flag_inefficient_routes(orders_with_emissions)
    potential_reduction = float(recommendations["potential_savings_kg"].sum()) if "potential_savings_kg" in recommendations.columns else 0.0

    actions = {
        "next_3_actions": (
            "1) Replace high-intensity trucks (>0.5 kg/km, age >6y). "
            "2) Shift short-haul (<200 km) to bikes/e-vans where possible. "
            "3) Retime routes with >60 min delays to off-peak."
        ),
        "fleet_optimization": (
            f"Replace {len(inefficient_veh)} inefficient vehicles; prioritize {inefficient_routes['route_id'].head(3).tolist()} routes for re-optimization."
            if len(inefficient_routes) > 0 else f"Replace {len(inefficient_veh)} inefficient vehicles; no high-intensity routes flagged."
        ),
        "potential_reduction_kg_month": f"~{potential_reduction:,.0f}",
    }
    return actions


def simulate_emission_scenarios(
    orders_with_emissions: pd.DataFrame,
    vehicles: pd.DataFrame,
    pct_switch_to_ev: float = 0.2,
    pct_replace_old: float = 0.2,
    pct_distance_reduction: float = 0.1,
    pct_idle_reduction: float = 0.1,
) -> Dict[str, float]:
    """
    Simple what-if simulation returning projected emissions and savings.

    - Distance reduction scales all emissions linearly.
    - EV switch assumes EV intensity of 0.05 kg/km for the switched share.
    - Replace-old assumes vehicles >6y and >0.5 kg/km could be improved to fleet median intensity.
    - Idle reduction applies to orders with traffic delays (>0), scaling their emissions modestly.
    """
    df = orders_with_emissions.copy()
    baseline_total = float(df["co2_emission_kg"].sum())
    if baseline_total <= 0:
        return {"baseline_total": 0.0, "projected_total": 0.0, "savings_kg": 0.0, "savings_pct": 0.0}

    # Distance reduction
    after_distance = baseline_total * (1.0 - max(0.0, min(1.0, pct_distance_reduction)))

    # EV switch
    avg_intensity = float((df["co2_emission_kg"].sum() / df["distance_km"].sum())) if df["distance_km"].sum() > 0 else 0.3
    ev_intensity = 0.05
    ev_gain_per_km = max(0.0, avg_intensity - ev_intensity)
    total_km = float(df["distance_km"].sum())
    ev_savings = ev_gain_per_km * total_km * max(0.0, min(1.0, pct_switch_to_ev))

    # Replace old
    v = vehicles.copy()
    v["age_years"] = pd.to_numeric(v["age_years"], errors="coerce")
    v["co2_emissions_kg_per_km"] = pd.to_numeric(v["co2_emissions_kg_per_km"], errors="coerce")
    old = v[(v["age_years"] > 6) & (v["co2_emissions_kg_per_km"] > 0.5)]
    fleet_median_intensity = float(v["co2_emissions_kg_per_km"].median()) if len(v) else 0.35
    old_excess = (old["co2_emissions_kg_per_km"] - fleet_median_intensity).clip(lower=0)
    # Approximate portion of activity influenced by old vehicles
    influence = min(1.0, len(old) / max(1, len(v)))
    replace_share = max(0.0, min(1.0, pct_replace_old))
    replace_savings = float(old_excess.mean() * total_km * influence * replace_share) if len(old_excess) else 0.0

    # Idle reduction
    if "traffic_delay_minutes" in df.columns:
        delayed = df[df["traffic_delay_minutes"].fillna(0) > 0]
        # Assume idle contributes ~10% of emission for delayed orders; reduce fractionally
        idle_component = float(delayed["co2_emission_kg"].sum()) * 0.10
        idle_savings = idle_component * max(0.0, min(1.0, pct_idle_reduction))
    else:
        idle_savings = 0.0

    projected_total = max(0.0, after_distance - ev_savings - replace_savings - idle_savings)
    savings_kg = max(0.0, baseline_total - projected_total)
    savings_pct = (savings_kg / baseline_total) * 100.0
    return {
        "baseline_total": baseline_total,
        "projected_total": projected_total,
        "savings_kg": savings_kg,
        "savings_pct": savings_pct,
    }


def compare_periods(trend_df: pd.DataFrame, a, b) -> Dict[str, float]:
    """Compare two periods by total CO2; a and b are pandas timestamps or strings in trend index."""
    t = trend_df.copy()
    t["month"] = pd.to_datetime(t["month"])
    va = float(t.loc[t["month"] == pd.to_datetime(a), "co2_emission_kg"].sum())
    vb = float(t.loc[t["month"] == pd.to_datetime(b), "co2_emission_kg"].sum())
    delta = va - vb
    pct = (delta / vb * 100.0) if vb > 0 else 0.0
    return {"period_a": va, "period_b": vb, "delta": delta, "delta_pct": pct}


# --- Predictive & Intelligence helpers ---

def forecast_emissions(trend_df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Fit a simple linear regression on monthly totals and project forward 'periods' months.
    Returns a DataFrame with observed and predicted values.
    """
    df = trend_df.copy()
    df = df.dropna(subset=["co2_emission_kg"]).reset_index(drop=True)
    if len(df) < 2:
        return df.assign(predicted=np.nan)

    df["month"] = pd.to_datetime(df["month"])  # ensure datetime
    df = df.sort_values("month")
    # Numeric index as feature
    df["t"] = np.arange(len(df))

    X = df[["t"]].values
    y = df["co2_emission_kg"].values
    model = LinearRegression().fit(X, y)
    df["predicted"] = model.predict(X)

    # Future points
    last_t = df["t"].iloc[-1]
    futures = []
    for i in range(1, periods + 1):
        t = last_t + i
        yhat = float(model.predict(np.array([[t]])))
        month_future = (df["month"].iloc[-1] + pd.offsets.MonthBegin(i))
        futures.append({"month": month_future, "co2_emission_kg": np.nan, "predicted": yhat, "t": t})
    if futures:
        df = pd.concat([df, pd.DataFrame(futures)], ignore_index=True)
    return df.drop(columns=["t"]) 


def vehicle_assignment_ai(orders_with_emissions: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    """
    Suggest eco-efficient vehicle types per order using per-km intensity and distance rules.
    """
    df = orders_with_emissions[["order_id", "route_id", "distance_km", "origin", "destination"]].copy()
    v = vehicles[["vehicle_type", "co2_emissions_kg_per_km"]].groupby("vehicle_type", as_index=False).agg(
        avg_kg_per_km=("co2_emissions_kg_per_km", "mean")
    ).sort_values("avg_kg_per_km")
    if len(v) == 0:
        return df.assign(recommended_type=np.nan, rationale="No vehicle data")

    def choose_type(dist_km: float) -> str:
        # Simple heuristic: short-haul -> smallest intensity types, long-haul -> medium over large when beneficial
        if dist_km <= 200 and (v["vehicle_type"].eq("Small_Van").any()):
            return "Small_Van"
        if 200 < dist_km <= 600 and (v["vehicle_type"].eq("Medium_Truck").any()):
            return "Medium_Truck"
        return v.iloc[0]["vehicle_type"]

    df["recommended_type"] = df["distance_km"].apply(choose_type)
    df = df.merge(v.rename(columns={"avg_kg_per_km": "type_kg_per_km"}), left_on="recommended_type", right_on="vehicle_type", how="left")
    df["insight"] = df.apply(lambda r: f"For {int(r['distance_km'])} km to {r['destination']}, use {r['recommended_type']} to minimize kg/km.", axis=1)
    return df[["order_id", "route_id", "distance_km", "recommended_type", "type_kg_per_km", "insight"]]


def idle_emission_estimate(orders_with_emissions: pd.DataFrame) -> Dict[str, float]:
    df = orders_with_emissions.copy()
    delayed = df[df["traffic_delay_minutes"].fillna(0) > 0] if "traffic_delay_minutes" in df.columns else df.iloc[0:0]
    baseline_idle = float(delayed["co2_emission_kg"].sum()) * 0.10 if len(delayed) else 0.0
    potential_20pct = baseline_idle * 0.20
    return {"idle_component_kg": baseline_idle, "savings_20pct_kg": potential_20pct}


def sustainability_score_orders(orders_with_emissions: pd.DataFrame) -> pd.DataFrame:
    """Score 0-100 per route based on emission intensity."""
    df = orders_with_emissions.copy()
    intensity = df["co2_emission_kg"] / df["distance_km"].replace(0, np.nan)
    df["intensity_kg_per_km"] = intensity
    # Normalize: 0.0 intensity -> 100 score, >=0.6 -> 0 score
    df["score"] = (1 - (df["intensity_kg_per_km"].clip(0, 0.6) / 0.6)) * 100
    return df[["order_id", "route_id", "intensity_kg_per_km", "score"]]


def emission_cost_correlation(ce_df: pd.DataFrame) -> Dict[str, object]:
    df = ce_df.dropna(subset=["total_cost", "co2_emission_kg"]).copy()
    corr = float(df[["total_cost", "co2_emission_kg"]].corr().iloc[0, 1]) if len(df) >= 2 else 0.0
    high_cost = df["total_cost"] > df["total_cost"].median()
    high_em = df["co2_emission_kg"] > df["co2_emission_kg"].median()
    outliers = df[high_cost & high_em]
    return {"corr": corr, "outliers": outliers}


def carbon_offset_estimate(total_kg: float, cost_per_ton: float = 1500.0) -> float:
    """Estimate offset cost in INR given cost per ton CO2."""
    return float((total_kg / 1000.0) * cost_per_ton)


def order_impact_breakdown(orders_with_emissions: pd.DataFrame, orders_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = orders_with_emissions.merge(orders_df[["order_id", "product_category", "priority"]], on="order_id", how="left")
    by_category = df.groupby("product_category", as_index=False)["co2_emission_kg"].sum().sort_values("co2_emission_kg", ascending=False)
    by_priority = df.groupby("priority", as_index=False)["co2_emission_kg"].sum().sort_values("co2_emission_kg", ascending=False)
    return {"by_category": by_category, "by_priority": by_priority}


def load_warehouse_inventory(base_dir: Path | str = None) -> pd.DataFrame:
    base_path = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    ds_path = base_path / "datasets"
    file = ds_path / "warehouse_inventory.csv"
    if not file.exists():
        return pd.DataFrame()
    inv = pd.read_csv(file)
    inv = _standardize_columns(inv)
    return inv


def warehouse_energy_estimate(inventory: pd.DataFrame, kg_co2_per_unit: float = 0.2) -> pd.DataFrame:
    if inventory is None or len(inventory) == 0:
        return pd.DataFrame()
    cols = inventory.columns
    # Accept multiple possible quantity fields
    qty_candidates = [
        "quantity",
        "current_stock_units",
        "stock_units",
        "units",
    ]
    qty_col = next((c for c in qty_candidates if c in cols), None)
    wh_col = "warehouse" if "warehouse" in cols else ("location" if "location" in cols else ("warehouse_id" if "warehouse_id" in cols else None))
    if not qty_col:
        return pd.DataFrame()
    df = inventory.copy()
    df["quantity"] = pd.to_numeric(df[qty_col], errors="coerce")
    df["energy_co2_kg"] = df["quantity"] * kg_co2_per_unit
    if wh_col:
        return df.groupby(wh_col, as_index=False)["energy_co2_kg"].sum().rename(columns={wh_col: "warehouse"})
    return df[["energy_co2_kg"]]


# --- AI suggestions, eco score, challenges, offsets, assistant ---

def eco_score(
    orders_with_emissions: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute a composite eco score (0-100) using:
    - Intensity (kg/km): lower is better
    - Trend improvement MoM: reductions are better
    - Fleet efficiency: median kg/km vs 0.6 cap
    """
    df = orders_with_emissions.copy()
    total_km = float(df["distance_km"].sum()) if "distance_km" in df.columns else 0.0
    total_co2 = float(df["co2_emission_kg"].sum()) if "co2_emission_kg" in df.columns else 0.0
    # Dynamic normalization reference from vehicle distribution (95th percentile) with sensible fallback
    veh_intens = pd.to_numeric(vehicles.get("co2_emissions_kg_per_km", pd.Series(dtype=float)), errors="coerce")
    max_ref = float(np.nanpercentile(veh_intens, 95)) if len(veh_intens) else 0.7
    if not np.isfinite(max_ref) or max_ref <= 0:
        max_ref = 0.7

    def to_score(value: float, ref: float) -> float:
        if not np.isfinite(value) or ref <= 0:
            return 50.0
        value = min(max(value, 0.0), ref)
        return float((1 - (value / ref)) * 100)

    intensity = (total_co2 / total_km) if total_km > 0 else np.nan
    intensity_score = to_score(intensity, max_ref)

    trend = monthly_emission_trend(df) if "order_date" in df.columns else pd.DataFrame()
    improv_score = 50.0
    if len(trend) >= 2:
        last = trend.iloc[-1]["co2_emission_kg"]
        prev = trend.iloc[-2]["co2_emission_kg"]
        if prev > 0:
            change = (prev - last) / prev  # positive if reduced
            improv_score = min(max(change * 100 + 50, 0), 100)

    fleet_median = float(pd.to_numeric(vehicles.get("co2_emissions_kg_per_km", pd.Series(dtype=float)), errors="coerce").median()) if len(vehicles) else np.nan
    if not np.isfinite(fleet_median):
        fleet_median = max_ref * 0.5  # neutral fallback
    fleet_score = to_score(fleet_median, max_ref)

    # Weighted composite
    composite = 0.5 * intensity_score + 0.3 * improv_score + 0.2 * fleet_score
    return {
        "eco_score": float(round(composite, 1)),
        "intensity_score": float(round(intensity_score, 1)),
        "improvement_score": float(round(improv_score, 1)),
        "fleet_score": float(round(fleet_score, 1)),
    }


def generate_ai_suggestions(orders_with_emissions: pd.DataFrame, vehicles: pd.DataFrame) -> list[str]:
    df = orders_with_emissions.copy()
    suggestions: list[str] = []
    avg_dist = float(df["distance_km"].mean()) if len(df) else 0
    if avg_dist <= 200:
        suggestions.append("Switch to eco-mode and smaller vehicles for short-haul trips (<200 km).")
    if "traffic_delay_minutes" in df.columns and df["traffic_delay_minutes"].fillna(0).mean() > 30:
        suggestions.append("Reducing idling by 5 min/order can save ~2 kg CO₂/week across delayed routes.")
    if (vehicles["vehicle_type"].eq("Large_Truck").sum() > vehicles["vehicle_type"].eq("Medium_Truck").sum()):
        suggestions.append("Right-size: prefer Medium_Truck over Large_Truck for mid-range deliveries (~250–600 km).")
    suggestions.append("Consider consolidated loads/carpooling twice a week — potential 10–20% CO₂ reduction.")
    return suggestions


def weekly_challenges(orders_with_emissions: pd.DataFrame) -> list[str]:
    challenges = [
        "Maintain idle time under 5 minutes per stop this week.",
        "Target a 5% CO₂ reduction versus last week.",
        "Shift 30% of <200 km trips to vans/e-bikes.",
    ]
    return challenges


def recommend_offsets_programs(location: str | None, total_kg: float) -> pd.DataFrame:
    programs = pd.DataFrame([
        {"program": "Tree Planting (India)", "provider": "Grow-Trees", "cost_per_ton_inr": 1500},
        {"program": "Renewable Energy (APAC)", "provider": "Gold Standard", "cost_per_ton_inr": 2200},
        {"program": "Clean Cookstoves (Asia)", "provider": "Verra", "cost_per_ton_inr": 1800},
    ])
    programs["estimated_cost_inr"] = (total_kg / 1000.0) * programs["cost_per_ton_inr"]
    return programs


def weather_traffic_tips(orders_with_emissions: pd.DataFrame) -> list[str]:
    tips: list[str] = []
    if "weather_impact" in orders_with_emissions.columns:
        weather_counts = orders_with_emissions["weather_impact"].fillna("None").value_counts()
        if weather_counts.get("Heavy_Rain", 0) > 0:
            tips.append("Avoid peak hours on heavy-rain routes; pre-position to covered docks.")
        if weather_counts.get("Fog", 0) > 0:
            tips.append("Foggy routes: reduce speed and plan daylight departures to lower risk/idling.")
    if "traffic_delay_minutes" in orders_with_emissions.columns and orders_with_emissions["traffic_delay_minutes"].fillna(0).mean() > 30:
        tips.append("Traffic congestion high: retime departures and use dynamic routing to cut idle emissions.")
    return tips


def assistant_answer(question: str, context: Dict[str, object]) -> str:
    q = (question or "").lower()
    if "reduce" in q and "emission" in q:
        return "Cut idle time, right-size vehicles, and consolidate loads; consider EV for urban routes."
    if "most polluting" in q or "worst route" in q:
        worst = context.get("top_route", "your highest-emission route")
        return f"{worst} currently emits the most CO₂. Optimize timing and vehicle type there first."
    if "compare" in q and "last month" in q:
        delta = context.get("delta_pct", 0.0)
        dirword = "up" if delta > 0 else "down"
        return f"Your emissions are {dirword} {abs(delta):.1f}% vs last month."
    return "Try: 'How can I reduce my emissions today?' or 'Compare my performance to last month.'"
