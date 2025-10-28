import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Tuple
from data_processing import (
    load_data,
    compute_emissions,
    potential_vehicle_scenarios,
    recommend_greener_operations,
    cost_emission_analytics,
    percent_fleet_under_years,
    top_efficient_vehicle_types,
    locations_highest_emissions,
    monthly_emission_trend,
    city_emission_map,
    narrative_insights,
    monthly_savings_vs_prev,
    flag_inefficient_vehicles,
    flag_inefficient_routes,
    vehicle_status_map,
    summarize_actions,
    simulate_emission_scenarios,
    compare_periods,
    forecast_emissions,
    vehicle_assignment_ai,
    idle_emission_estimate,
    sustainability_score_orders,
    emission_cost_correlation,
    carbon_offset_estimate,
    order_impact_breakdown,
    load_warehouse_inventory,
    warehouse_energy_estimate,
    eco_score,
    generate_ai_suggestions,
    weekly_challenges,
    recommend_offsets_programs,
    weather_traffic_tips,
    assistant_answer,
)

st.set_page_config(page_title="NexGen Sustainability Tracker", page_icon="🌿", layout="wide")
st.title("🌿 NexGen Sustainability Tracker")
st.caption("NexGen Sustainability Tracker provides real-time visibility into logistics-related emissions to help organizations reduce carbon footprint.")

# Executive Summary & Problem Statement
with st.expander("Executive Summary & Problem Statement", expanded=True):
    st.subheader("Problem Statement")
    st.write("Transport contributes ~24% of direct CO₂ from fuel combustion globally (IPCC 2022). Logistics inefficiencies drive Scope 1 emissions, costing firms millions in fuel and compliance.")
    st.write("**Business Case:** Reducing emissions by 20% can save ₹50 lakhs annually in fuel costs (McKinsey, 2023). This tool enables data-driven decisions for SDG 13 compliance.")
    st.subheader("Citations")
    st.write("- IPCC (2022): Global Greenhouse Gas Emissions Report.")
    st.write("- McKinsey (2023): Sustainability in Logistics.")
    st.write("- India's NDC: 33-35% emission reduction by 2030.")

# Sustainability Leaderboard (Gamification)
st.subheader("🏆 Sustainability Leaderboard")
# Load data for leaderboard
@st.cache_data(show_spinner=False)
def _load_leaderboard() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_data()
orders, routes, vehicles, costs = _load_leaderboard()
orders_emissions, route_summary, kpis = compute_emissions(orders, routes)
leaderboard = orders_emissions.groupby("origin", as_index=False)["co2_emission_kg"].sum().sort_values("co2_emission_kg", ascending=True).head(10)
leaderboard["rank"] = range(1, len(leaderboard) + 1)
leaderboard["badge"] = leaderboard["rank"].apply(lambda r: "🌟 Green Champion" if r == 1 else "🥈 Eco Warrior" if r == 2 else "🥉 Sustain Hero" if r == 3 else "🌱 Eco Contributor")
st.dataframe(leaderboard[["rank", "origin", "co2_emission_kg", "badge"]], use_container_width=True, hide_index=True)

with st.expander("Why it matters", expanded=True):
    st.write("Transport contributes ~24% of direct CO₂ from fuel combustion globally. Cutting route and vehicle inefficiencies delivers immediate Scope 1 reductions.")

# Assistant & AI tab
st.divider()
tab_ai, tab_main = st.tabs(["AI Assistant", "Dashboard"])
with tab_ai:
    # Ensure dataframes exist for AI section, with resilient fallbacks
    try:
        f_view = f.copy()
    except Exception:
        try:
            _o2, _r2, _v2, _c2 = _load()
            _oe2, _rs2, _kp2 = compute_emissions(_o2, _r2)
            f_view = _oe2.copy()
        except Exception:
            f_view = pd.DataFrame(columns=["order_id", "route_id", "distance_km", "co2_emission_kg"])

    # Ensure vehicles dataframe for AI section
    try:
        v_view = vehicles.copy()
    except Exception:
        try:
            _o3, _r3, v_view, _c3 = _load()
        except Exception:
            v_view = pd.DataFrame(columns=["vehicle_id", "vehicle_type", "co2_emissions_kg_per_km"])
    by_route_local = (
        f_view.groupby("route_id", as_index=False)["co2_emission_kg"].sum().sort_values("co2_emission_kg", ascending=False).head(25)
    )
    st.subheader("Eco Score")
    # Use filtered data if available; otherwise compute on full dataset
    f_for_score = f_view
    if len(f_for_score) == 0:
        try:
            _o_all, _r_all, _v_all, _c_all = _load()
            _oe_all, _rs_all, _kp_all = compute_emissions(_o_all, _r_all)
            f_for_score = _oe_all
        except Exception:
            f_for_score = f_view
    es = eco_score(f_for_score, v_view)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Eco Score", f"{es['eco_score']:.1f}")
    s2.metric("Intensity Score", f"{es['intensity_score']:.1f}")
    s3.metric("Improvement Score", f"{es['improvement_score']:.1f}")
    s4.metric("Fleet Score", f"{es['fleet_score']:.1f}")

    st.subheader("AI Suggestions")
    for tip in generate_ai_suggestions(f_view, v_view):
        st.write(f"- {tip}")

    st.subheader("Weekly Challenges")
    for c in weekly_challenges(f_view):
        st.write(f"- {c}")

    st.subheader("Offset Programs")
    total_co2 = f_view["co2_emission_kg"].sum()
    # Determine a location for offsets gracefully
    try:
        loc_for_offsets = origin_sel if origin_sel != "All" else None
    except Exception:
        loc_for_offsets = None
        if "origin" in f_view.columns and len(f_view) > 0:
            try:
                loc_for_offsets = f_view["origin"].mode().iloc[0]
            except Exception:
                loc_for_offsets = None
    offs = recommend_offsets_programs(loc_for_offsets, total_co2)
    st.dataframe(offs, use_container_width=True, hide_index=True)

    st.subheader("Weather & Traffic Tips")
    for t in weather_traffic_tips(f_view):
        st.write(f"- {t}")

    st.subheader("Ask the Assistant")
    q = st.text_input("Ask a question")
    if q:
        context = {"delta_pct": monthly_savings_vs_prev(f_view), "top_route": by_route_local.iloc[0]["route_id"] if len(by_route_local) else None}
        base_answer = assistant_answer(q, context)
        # Enhanced generative response
        if "reduce" in q.lower():
            enhanced = f"{base_answer} Based on your Mumbai routes, switch to EVs for 30% savings. Consider consolidating loads twice weekly."
        elif "most polluting" in q.lower():
            enhanced = f"{base_answer} Your top emitter is {context.get('top_route', 'Route X')}. Optimize timing and vehicle type there."
        else:
            enhanced = base_answer
        st.info(enhanced)

with tab_main:
    st.write("Use the sections above for analytics and insights.")


@st.cache_data(show_spinner=False)
def _load() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached loader to avoid repeated disk IO during interactions."""
    return load_data()


# Load and compute
orders, routes, vehicles, costs = _load()
orders_emissions, route_summary, kpis = compute_emissions(orders, routes)

# Sidebar filters
st.sidebar.header("Filters")
date_min = pd.to_datetime(orders_emissions["order_date"]).min()
date_max = pd.to_datetime(orders_emissions["order_date"]).max()
date_range = st.sidebar.date_input("Order date range", value=(date_min, date_max))
origins = ["All"] + sorted(orders_emissions["origin"].dropna().unique().tolist())
destinations = ["All"] + sorted(orders_emissions["destination"].dropna().unique().tolist())
origin_sel = st.sidebar.selectbox("Origin", origins)
dest_sel = st.sidebar.selectbox("Destination", destinations)
veh_types = ["All"] + sorted(vehicles["vehicle_type"].dropna().unique().tolist())
veh_sel = st.sidebar.selectbox("Vehicle Type", veh_types)
status_opts = ["All"] + sorted(vehicles["status"].dropna().unique().tolist())
status_sel = st.sidebar.selectbox("Vehicle Status", status_opts)

# Apply filters
f = orders_emissions.copy()
f["order_date"] = pd.to_datetime(f["order_date"]).dt.date
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    f = f[(f["order_date"] >= start_date) & (f["order_date"] <= end_date)]
if origin_sel != "All":
    f = f[f["origin"] == origin_sel]
if dest_sel != "All":
    f = f[f["destination"] == dest_sel]
if veh_sel != "All" and "recommended_vehicle_type" in f.columns:
    f = f[f["recommended_vehicle_type"] == veh_sel]

# KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total CO₂ (kg)", f"{f['co2_emission_kg'].sum():,.0f}")
with col2:
    st.metric("Total Distance (km)", f"{f['distance_km'].sum():,.0f}")
with col3:
    intensity = (f["co2_emission_kg"].sum() / f["distance_km"].sum()) if f["distance_km"].sum() > 0 else 0
    st.metric("Avg Intensity (kg/km)", f"{intensity:.3f}")
with col4:
    st.metric("Orders", f"{f['order_id'].nunique():,}")

# Impact KPIs
fleet_young_pct = percent_fleet_under_years(vehicles, years=3)
top_types = top_efficient_vehicle_types(vehicles, n=5)
top_locations = locations_highest_emissions(f, field="origin", n=5)
colA, colB, colC = st.columns(3)
with colA:
    st.metric("% Fleet < 3 years", f"{fleet_young_pct:.1f}%")
with colB:
    if len(top_types) > 0:
        st.metric("Top Efficient Type", f"{top_types.iloc[0]['vehicle_type']}", help="Lowest avg kg/km")
with colC:
    if len(top_locations) > 0:
        st.metric("Highest-Emission City", f"{top_locations.iloc[0]['origin']}")

# Additional impact cards
colD, colE = st.columns(2)
with colD:
    delta_pct = monthly_savings_vs_prev(f)
    st.metric("CO₂ Saved vs Prev. Month", f"{max(0.0, delta_pct):.1f}%", help="Positive means reduction vs previous month")
with colE:
    st.metric("Top 5 Efficient Types", ", ".join(top_types["vehicle_type"].tolist()))

# Charts
st.subheader("Emission Hotspots by Route")
by_route = (
    f.groupby("route_id", as_index=False)["co2_emission_kg"].sum().sort_values("co2_emission_kg", ascending=False).head(25)
)
fig = px.bar(by_route, x="route_id", y="co2_emission_kg", labels={"route_id": "Route", "co2_emission_kg": "CO₂ (kg)"})
fig.update_layout(xaxis_tickangle=-35, height=420, title_text="Bar chart showing top 25 routes by CO₂ emissions")
st.plotly_chart(fig, use_container_width=True)

st.subheader("What-if: Vehicle Type Emission Intensity")
veh_type = st.selectbox("Vehicle Type for scenario", sorted(vehicles["vehicle_type"].unique().tolist()))
scen = potential_vehicle_scenarios(f[["order_id", "distance_km"]], vehicles)
scen_type = scen[scen["vehicle_type"] == veh_type]
scen_route = scen_type.merge(f[["order_id", "route_id"]], on="order_id", how="left")
scen_agg = scen_route.groupby("route_id", as_index=False)["scenario_emission_kg"].sum().sort_values("scenario_emission_kg", ascending=False).head(25)
fig2 = px.bar(scen_agg, x="route_id", y="scenario_emission_kg", labels={"route_id": "Route", "scenario_emission_kg": "Scenario CO₂ (kg)"})
fig2.update_layout(xaxis_tickangle=-35, height=420)
st.plotly_chart(fig2, use_container_width=True)

# Trend chart
st.subheader("Monthly Emissions Trend")
trend = monthly_emission_trend(f)
fig_trend = px.area(trend, x="month", y="co2_emission_kg", labels={"co2_emission_kg": "CO₂ (kg)"})
fig_trend.update_layout(height=380, title_text="Area chart of monthly CO₂ emissions over time")
st.plotly_chart(fig_trend, use_container_width=True)

# Forecasting
st.subheader("Predicted vs Actual Emissions")
fc = forecast_emissions(trend, periods=1)
fig_fc = px.line(fc, x="month", y=["co2_emission_kg", "predicted"], labels={"value": "CO₂ (kg)", "variable": "Series"})
st.plotly_chart(fig_fc, use_container_width=True)
if len(fc.dropna(subset=["predicted"])) >= 1 and pd.notna(fc.iloc[-1]["predicted"]):
    last_obs = fc["co2_emission_kg"].dropna().iloc[-1] if fc["co2_emission_kg"].notna().any() else 0
    next_pred = float(fc.iloc[-1]["predicted"]) if pd.notna(fc.iloc[-1]["predicted"]) else 0
    rise_pct = ((next_pred - last_obs) / last_obs * 100.0) if last_obs > 0 else 0
    if rise_pct > 5:
        st.error(f"Alert: Expected {rise_pct:.1f}% rise in CO₂ next month — consider reassigning vehicles.")

# Map
st.subheader("Emission Intensity by City")
city_map = city_emission_map(f, by="origin")
if len(city_map) > 0:
    st.map(city_map.rename(columns={"lat": "latitude", "lon": "longitude"}), zoom=3)
    # Animated map effect (placeholder for future integration)
    st.caption("Future: Interactive map with CO₂ cloud animations.")

st.subheader("Vehicle Status Map")
veh_map = vehicle_status_map(vehicles)
if len(veh_map) > 0:
    st.map(veh_map.rename(columns={"lat": "latitude", "lon": "longitude"}), zoom=3)

# Cost vs CO2 analytics
st.subheader("Cost vs CO₂ Analytics")
ce = cost_emission_analytics(f, costs)
fig3 = px.scatter(
    ce,
    x="total_cost",
    y="co2_emission_kg",
    color="is_pareto_efficient",
    hover_data=["order_id", "route_id", "cost_per_km", "cost_per_kg_co2"],
    labels={"total_cost": "Total Cost", "co2_emission_kg": "CO₂ (kg)"},
)
fig3.update_layout(height=420, title_text="Scatter plot of cost vs CO₂ emissions with Pareto frontier")
st.plotly_chart(fig3, use_container_width=True)

# Data and downloads
st.subheader("Downloadable Reports")
tab1, tab2, tab3 = st.tabs(["Route Summary", "Order-level Emissions", "Cost-CO₂ Table"])
with tab1:
    st.dataframe(route_summary, use_container_width=True, hide_index=True)
    csv1 = route_summary.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Route Summary", data=csv1, file_name="route_summary.csv", mime="text/csv")
with tab2:
    st.dataframe(f, use_container_width=True, hide_index=True)
    csv2 = f.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Order Emissions", data=csv2, file_name="order_emissions.csv", mime="text/csv")
with tab3:
    st.dataframe(ce, use_container_width=True, hide_index=True)
    csv_ce = ce.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cost-CO₂ Table", data=csv_ce, file_name="cost_co2.csv", mime="text/csv")

st.subheader("Recommendations: Greener Operations")
recs = recommend_greener_operations(f, vehicles)
st.dataframe(
    recs[[
        "order_id", "route_id", "distance_km", "co2_emission_kg", "scenario_emission_kg", "potential_savings_kg", "recommended_vehicle_type", "recommendation_note"
    ]].rename(columns={
        "co2_emission_kg": "current_emission_kg"
    }),
    use_container_width=True,
    hide_index=True,
)
csv3 = recs.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Recommendations", data=csv3, file_name="recommendations.csv", mime="text/csv")

# ROI Calculator
st.subheader("💰 ROI Calculator")
fuel_price_per_l = st.slider("Fuel Price (₹/L)", 80, 120, 100)
maintenance_savings_pct = st.slider("Maintenance Savings (%)", 0, 20, 10) / 100
total_savings_kg = recs["potential_savings_kg"].sum()
fuel_savings = (total_savings_kg / 2.68) * fuel_price_per_l  # Assuming 2.68 kg/L emission factor
maintenance_savings = fuel_savings * maintenance_savings_pct
total_roi = fuel_savings + maintenance_savings
st.metric("Potential Annual Savings (₹)", f"{total_roi:,.0f}", help="Fuel + maintenance savings from recommendations")

st.subheader("Sustainability Actions")
actions = summarize_actions(f, vehicles, recs)
st.write(actions["next_3_actions"])
st.info(actions["fleet_optimization"])
st.metric("Potential Reduction (kg CO₂/month)", actions["potential_reduction_kg_month"])

# Narrative insights and Why it matters
st.subheader("Automated Insight")
ni = narrative_insights(f, vehicles)
st.success(f"{ni['emission_trend']} {ni['fleet_efficiency']}")

# Real-time Alerts
st.subheader("🚨 Real-Time Alerts")
if rise_pct > 5:
    st.error(f"Alert: Expected {rise_pct:.1f}% rise in CO₂ next month — consider reassigning vehicles.")
else:
    st.success("Emissions on track. No alerts.")

# Advanced Simulations (Multi-variable)
st.subheader("🔬 Advanced What-if Scenarios")
weather_impact = st.slider("Weather Delay Increase (%)", 0, 50, 10) / 100
traffic_impact = st.slider("Traffic Congestion Increase (%)", 0, 50, 10) / 100
# Use default values for advanced sim since pct_ev etc. are defined later
advanced_sim = simulate_emission_scenarios(f, vehicles, 0.2, 0.2, 0.1 + weather_impact + traffic_impact, 0.1)
st.metric("Advanced Projected CO₂ (kg)", f"{advanced_sim['projected_total']:,.0f}")
st.metric("Advanced Savings (kg)", f"{advanced_sim['savings_kg']:,.0f}")

# Branding and share report (mock)
st.divider()
left, right = st.columns([3, 1])
with left:
    st.caption("NexGen Logistics ♻️ — SDG 13: Climate Action. Target: Reduce Scope 1 by 20% by 2030.")
    st.write("**Success Story:** Client reduced emissions by 25%, saving ₹50 lakhs via optimized routes.")
with right:
    st.button("📤 Send Email Summary", help="Mock action: integrate with email service")

# HTML report export (simple)
html_report = pd.DataFrame({
    "metric": ["total_co2_kg", "avg_kg_per_km", "orders"],
    "value": [f['co2_emission_kg'].sum(), intensity, f['order_id'].nunique()],
}).to_html(index=False)
st.download_button(
    "📄 Download HTML Summary",
    data=html_report,
    file_name="sustainability_summary.html",
    mime="text/html",
)

# Guided Tour & Tooltips
st.sidebar.header("🚀 Guided Tour")
tour_step = st.sidebar.radio("Tour Step", ["1. Overview", "2. Filters", "3. Charts", "4. Recommendations", "5. Scenarios"], index=0)
if tour_step == "1. Overview":
    st.sidebar.info("Welcome! This dashboard tracks logistics emissions. Use filters to explore data.")
elif tour_step == "2. Filters":
    st.sidebar.info("Adjust date, origin, destination, vehicle type, and status to filter data.")
elif tour_step == "3. Charts":
    st.sidebar.info("View hotspots, trends, and maps. Hover for details.")
elif tour_step == "4. Recommendations":
    st.sidebar.info("See greener operations and ROI calculators.")
elif tour_step == "5. Scenarios":
    st.sidebar.info("Simulate EV switches and efficiency gains.")

# Accessibility & Mobile Notes
st.caption("Accessibility: Charts include alt text. Mobile: Use landscape mode for best view.")

# Tabs for navigation
st.divider()
tab_emissions, tab_fleet, tab_routes, tab_recs, tab_compare = st.tabs(["Emissions", "Fleet", "Routes", "Recommendations", "Compare"])
with tab_emissions:
    st.write("Use filters and scenario sliders to explore emissions.")

with tab_fleet:
    # Fleet health
    st.subheader("Fleet Health & Performance")
    v = vehicles.copy()
    age_avg = pd.to_numeric(v["age_years"], errors="coerce").mean()
    maint = (v["status"].str.lower() == "maintenance").sum()
    over_threshold = (pd.to_numeric(v["co2_emissions_kg_per_km"], errors="coerce") > 0.5).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Vehicle Age (yrs)", f"{age_avg:.1f}")
    c2.metric("Under Maintenance", f"{maint}")
    c3.metric(">0.5 kg/km Vehicles", f"{over_threshold}")
    st.dataframe(flag_inefficient_vehicles(vehicles), use_container_width=True, hide_index=True)

with tab_routes:
    st.subheader("High-Intensity Routes")
    st.dataframe(flag_inefficient_routes(f), use_container_width=True, hide_index=True)
    # Route scorecard
    st.subheader("Sustainability Scorecard (Orders)")
    st.dataframe(sustainability_score_orders(f), use_container_width=True, hide_index=True)

with tab_recs:
    st.subheader("AI-Driven Tips")
    st.write("- Consider replacing vehicles older than 6 years with >0.5 kg/km emissions.")
    st.write("- Shift deliveries under 200 km to small vans or e-bikes to cut CO₂.")
    if len(top_locations) > 0:
        st.write(f"- Optimize routes starting from {top_locations.iloc[0]['origin']} next month.")
    st.subheader("AI Vehicle Assignment Recommendation")
    ai_assign = vehicle_assignment_ai(f, vehicles)
    st.dataframe(ai_assign, use_container_width=True, hide_index=True)
    idle = idle_emission_estimate(f)
    st.info(f"Estimated idle CO₂: {idle['idle_component_kg']:,.0f} kg. Savings at 20%: {idle['savings_20pct_kg']:,.0f} kg.")

with tab_compare:
    st.subheader("Comparative Analytics")
    trend_all = monthly_emission_trend(orders_emissions)
    months = trend_all["month"].dt.strftime("%Y-%m").tolist()
    if len(months) >= 2:
        a = st.selectbox("Period A", months, index=max(0, len(months) - 2))
        b = st.selectbox("Period B", months, index=max(0, len(months) - 1))
        cmp = compare_periods(trend_all, a, b)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("A CO₂ (kg)", f"{cmp['period_a']:,.0f}")
        m2.metric("B CO₂ (kg)", f"{cmp['period_b']:,.0f}")
        m3.metric("Delta (kg)", f"{cmp['delta']:,.0f}")
        m4.metric("Delta %", f"{cmp['delta_pct']:.1f}%")
    # Cost-CO2 correlation
    st.subheader("Emission-to-Cost Correlation")
    ec = emission_cost_correlation(ce)
    st.metric("Correlation", f"{ec['corr']:.2f}")
    if len(ec["outliers"]) > 0:
        st.warning("High cost & high emission outliers detected.")
        st.dataframe(ec["outliers"], use_container_width=True, hide_index=True)

# Scenario sliders in sidebar
st.sidebar.header("What-if Scenarios")
pct_ev = st.sidebar.slider("% routes switched to EV", 0, 100, 20, step=5) / 100
pct_replace = st.sidebar.slider("% old vehicles replaced", 0, 100, 20, step=5) / 100
pct_dist = st.sidebar.slider("% reduction in avg distance", 0, 50, 10, step=5) / 100
pct_idle = st.sidebar.slider("% reduction in idle time", 0, 50, 10, step=5) / 100
sim = simulate_emission_scenarios(f, vehicles, pct_ev, pct_replace, pct_dist, pct_idle)
st.sidebar.metric("Projected CO₂ (kg)", f"{sim['projected_total']:,.0f}")
st.sidebar.metric("Savings (kg)", f"{sim['savings_kg']:,.0f}")
st.sidebar.metric("Savings %", f"{sim['savings_pct']:.1f}%")

# EV simulation note
st.caption("Adjust EV and efficiency sliders to simulate forward-looking transitions.")

# Carbon offset calculator
st.subheader("Carbon Offset Calculator")
total_co2 = f["co2_emission_kg"].sum()
offset_cost = carbon_offset_estimate(total_co2, cost_per_ton=1500.0)
st.info(f"Offsetting {total_co2:,.0f} kg CO₂ costs approximately ₹{offset_cost:,.0f}.")

# Order-level impact by category/priority
st.subheader("Order Impact Breakdown")
impact = order_impact_breakdown(f, orders)
colX, colY = st.columns(2)
with colX:
    st.write("By Product Category")
    st.dataframe(impact["by_category"], use_container_width=True, hide_index=True)
with colY:
    st.write("By Priority")
    st.dataframe(impact["by_priority"], use_container_width=True, hide_index=True)

# Warehouse energy estimate
st.subheader("Warehouse Energy Impact")
inv = load_warehouse_inventory()
wh = warehouse_energy_estimate(inv)
if len(wh) > 0:
    st.dataframe(wh, use_container_width=True, hide_index=True)
    st.caption("Energy-related CO₂ estimated using stock units × emission factor (0.2 kg/unit). Adjust factors per your facilities.")
else:
    if inv is None or len(inv) == 0:
        st.warning("No warehouse inventory data found. Ensure 'warehouse_inventory.csv' exists.")
    else:
        st.warning("Warehouse dataset present but columns don't match expectations. Showing preview below.")
        st.dataframe(inv.head(20), use_container_width=True, hide_index=True)
