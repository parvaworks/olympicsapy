```python
# streamlit_app.py (updated)

"""
Streamlit dashboard for Olympic Medal Analysis (with host mapping & recent year weighting)

Enhancements:
- Treat Year as string for display (but keep numeric for modeling)
- Weighted trendlines to emphasize recent years
- Host city-to-country mapping (1984–2016)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

try:
    import pmdarima as pm
    PM_AVAILABLE = True
except:
    PM_AVAILABLE = False

sns.set(style="whitegrid")

# -----------------------------
# Host City → Country mapping
# -----------------------------
HOST_MAPPING = {
# Summer Olympic Host Cities
summer_olympics_hosts = {
    1984: ("Los Angeles", "USA"),
    1988: ("Seoul", "South Korea"),
    1992: ("Barcelona", "Spain"),
    1996: ("Atlanta", "USA"),
    2000: ("Sydney", "Australia"),
    2004: ("Athens", "Greece"),
    2008: ("Beijing", "China"),
    2012: ("London", "Great Britain"),
    2016: ("Rio de Janeiro", "Brazil")
}

# Winter Olympic Host Cities
winter_olympics_hosts = {
    1984: ("Sarajevo", "Yugoslavia (now Bosnia and Herzegovina)"),
    1988: ("Calgary", "Canada"),
    1992: ("Albertville", "France"),
    1994: ("Lillehammer", "Norway"),
    1998: ("Nagano", "Japan"),
    2002: ("Salt Lake City", "USA"),
    2006: ("Turin", "Italy"),
    2010: ("Vancouver", "Canada"),
    2014: ("Sochi", "Russia")
}

}

# -----------------------------
# Load & preprocess data
# -----------------------------
def load_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df['Medal'] = df['Medal'].replace({'NA': np.nan, '': np.nan})
    df['medal_type'] = df['Medal']
    df['medal'] = df['medal_type'].map({'Gold': 1, 'Silver': 1, 'Bronze': 1}).fillna(0).astype(int)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['Year_str'] = df['Year'].astype(str)
    return df

# -----------------------------
# Aggregation
# -----------------------------
def aggregate_medals(df: pd.DataFrame) -> pd.DataFrame:
    df['is_gold'] = (df['medal_type'] == 'Gold').astype(int)
    df['is_silver'] = (df['medal_type'] == 'Silver').astype(int)
    df['is_bronze'] = (df['medal_type'] == 'Bronze').astype(int)
    agg = df.groupby(['Year','NOC','Sport'], dropna=False).agg(
        medals=('medal','sum'),
        golds=('is_gold','sum'),
        silvers=('is_silver','sum'),
        bronzes=('is_bronze','sum')
    ).reset_index()
    return agg

# -----------------------------
# Forecasting
# -----------------------------
def arima_forecast(series, n_periods=4):
    if PM_AVAILABLE:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = pm.auto_arima(series, seasonal=False, error_action='ignore')
        return model.predict(n_periods=n_periods)
    else:
        model = sm.tsa.ARIMA(series, order=(1,0,0)).fit()
        return model.forecast(steps=n_periods)

def rf_forecast_country(df_agg, country, n_periods=4):
    cdf = df_agg[df_agg['NOC']==country].groupby('Year').sum().reset_index()
    for lag in range(1,5):
        cdf[f'lag_{lag}'] = cdf['medals'].shift(lag).fillna(0)
    X, y = cdf[[f'lag_{i}' for i in range(1,5)]], cdf['medals']
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    last_lags = list(X.iloc[-1])
    preds = []
    for _ in range(n_periods):
        p = model.predict([last_lags[-4:]])[0]
        preds.append(max(0,float(p)))
        last_lags.append(p)
    return preds

# -----------------------------
# Hosting Effect
# -----------------------------
def hosting_effect(df, host_country="USA"):
    panel = df.groupby(['Year','NOC']).agg(total_medals=('medal','sum')).reset_index()
    if panel[panel['NOC']==host_country].empty:
        return "No data for host country"

    # Use mapped host years
    host_years = [y for y in HOST_MAPPING.keys() if y in panel['Year'].unique()]
    if not host_years:
        return "No matching host years in dataset"

    min_host_year = min(host_years)
    panel['post'] = (panel['Year']>=min_host_year).astype(int)
    panel['treated'] = (panel['NOC']==host_country).astype(int)
    panel['did'] = panel['post']*panel['treated']
    model = sm.OLS(panel['total_medals'], sm.add_constant(panel[['treated','post','did']])).fit()
    return model.summary().as_text()

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🏅 Olympic Medal Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Olympic Excel file", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    agg = aggregate_medals(df)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Sport summary
    st.subheader("Sport-wise Medal Summary")
    sport_summary = agg.groupby('Sport').agg(total_medals=('medals','sum'), golds=('golds','sum')).reset_index().sort_values('total_medals', ascending=False)
    st.dataframe(sport_summary)

    # Country selector
    countries = sorted(df['NOC'].dropna().unique())
    country = st.selectbox("Select country (NOC):", countries, index=countries.index('USA') if 'USA' in countries else 0)

    # Medal trend plot with recent weighting
    st.subheader(f"Medal Trend for {country}")
    cdf = agg[agg['NOC']==country].groupby('Year').sum().reset_index()
    weights = np.linspace(0.5, 2, len(cdf))  # more weight to recent years
    fig, ax = plt.subplots()
    ax.plot(cdf['Year'], cdf['medals'], marker='o')
    z = np.polyfit(cdf['Year'], cdf['medals'], 1, w=weights)
    p = np.poly1d(z)
    ax.plot(cdf['Year'], p(cdf['Year']), "--", color="red", label="Weighted Trend")
    ax.legend()
    ax.set_title(f"Medals over time: {country}")
    st.pyplot(fig)

    # Top countries
    st.subheader("Top Countries in Most Recent Year")
    recent_year = int(df['Year'].dropna().max())
    year_df = agg[agg['Year']==recent_year].groupby('NOC').sum().reset_index().sort_values('medals', ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=year_df, x='medals', y='NOC', ax=ax)
    ax.set_title(f"Top 10 countries in {recent_year}")
    st.pyplot(fig)

    # Forecasting
    st.subheader(f"Forecasting Medals for {country}")
    series = agg[agg['NOC']==country].groupby('Year')['medals'].sum()
    arima_fc = arima_forecast(series, 4)
    rf_fc = rf_forecast_country(agg, country, 4)
    st.write("ARIMA forecast (next 4 Games):", arima_fc)
    st.write("RandomForest forecast (next 4 Games):", rf_fc)

    # Hosting Effect
    st.subheader("Hosting Effect Evaluation")
    result = hosting_effect(df, country)
    st.text(result)

    # Show host city-country mapping table
    st.subheader("Host Cities and Countries (1984–2016)")
    host_df = pd.DataFrame([
        {"Year": y, "Host City": c[0], "Host Country": c[1]} for y,c in HOST_MAPPING.items()
    ])
    st.dataframe(host_df)
else:
    st.info("Please upload an Olympic Excel file to start analysis.")
```
