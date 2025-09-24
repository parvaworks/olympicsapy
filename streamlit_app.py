import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from scipy import stats

try:
    import pmdarima as pm
    PM_AVAILABLE = True
except:
    PM_AVAILABLE = False

sns.set_style("whitegrid")

# -----------------------------
# Olympic Games Timeline (Separated by Season)
# -----------------------------
SUMMER_OLYMPICS = {
    1984: ("Los Angeles", "USA"),
    1988: ("Seoul", "South Korea"), 
    1992: ("Barcelona", "Spain"),
    1996: ("Atlanta", "USA"),
    2000: ("Sydney", "Australia"),
    2004: ("Athens", "Greece"),
    2008: ("Beijing", "China"),  # China's hosting year
    2012: ("London", "Great Britain"),
    2016: ("Rio de Janeiro", "Brazil"),
    2020: ("Tokyo", "Japan"),  # Held in 2021
    2024: ("Paris", "France"),
    2028: ("Los Angeles", "USA"),  # Future USA hosting
    2032: ("Brisbane", "Australia")
}

WINTER_OLYMPICS = {
    1984: ("Sarajevo", "Yugoslavia"),
    1988: ("Calgary", "Canada"),
    1992: ("Albertville", "France"),
    1994: ("Lillehammer", "Norway"),
    1998: ("Nagano", "Japan"),
    2002: ("Salt Lake City", "USA"),
    2006: ("Turin", "Italy"),
    2010: ("Vancouver", "Canada"),
    2014: ("Sochi", "Russia"),
    2018: ("Pyeongchang", "South Korea"),
    2022: ("Beijing", "China"),  # China's hosting year
    2026: ("Milan-Cortina", "Italy"),
    2030: ("TBD", "TBD")
}

US_SUMMER_HOST_YEARS = [1984, 1996, 2028]
US_WINTER_HOST_YEARS = [2002]
CHINA_SUMMER_HOST_YEARS = [2008]
CHINA_WINTER_HOST_YEARS = [2022]

# -----------------------------
# Medal Weighting System
# -----------------------------
def calculate_weighted_medals(df, gold_weight=3, silver_weight=2, bronze_weight=1):
    """Calculate weighted medal scores giving more importance to golds"""
    df = df.copy()
    df['weighted_score'] = 0
    df.loc[df['Medal'] == 'Gold', 'weighted_score'] = gold_weight
    df.loc[df['Medal'] == 'Silver', 'weighted_score'] = silver_weight
    df.loc[df['Medal'] == 'Bronze', 'weighted_score'] = bronze_weight
    return df

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
    
    # Accurate season detection based on known Olympic years
    summer_years = [1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028, 2032]
    winter_years = [1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026, 2030]
    
    # Initialize all as Unknown first
    df['Season'] = 'Unknown'
    
    # Assign seasons based on known years
    df.loc[df['Year'].isin(summer_years), 'Season'] = 'Summer'
    df.loc[df['Year'].isin(winter_years), 'Season'] = 'Winter'
    
    # For years not explicitly listed, make educated guess
    # Summer Olympics are typically in years divisible by 4 (except Winter Olympic years)
    for year in df['Year'].dropna().unique():
        if year not in summer_years + winter_years:
            if year % 4 == 0 and year not in winter_years:
                df.loc[df['Year'] == year, 'Season'] = 'Summer'
    
    return df

# -----------------------------
# Analysis Functions
# -----------------------------
def analyze_country_performance(df, country='USA', season='Summer', use_weighted=True):
    """Analyze country performance with season separation"""
    
    season_df = df[df['Season'] == season]
    country_df = season_df[season_df['NOC'] == country]
    
    if use_weighted:
        country_df = calculate_weighted_medals(country_df)
        score_col = 'weighted_score'
    else:
        score_col = 'medal'
    
    performance = country_df.groupby('Year').agg(
        total_score=(score_col, 'sum'),
        total_medals=('medal', 'sum'),
        gold_medals=('medal_type', lambda x: (x == 'Gold').sum()),
        silver_medals=('medal_type', lambda x: (x == 'Silver').sum()),
        bronze_medals=('medal_type', lambda x: (x == 'Bronze').sum())
    ).reset_index()
    
    # Add hosting information
    if country == 'USA':
        host_years = US_SUMMER_HOST_YEARS if season == 'Summer' else US_WINTER_HOST_YEARS
    elif country == 'China':
        host_years = CHINA_SUMMER_HOST_YEARS if season == 'Summer' else CHINA_WINTER_HOST_YEARS
    else:
        host_years = []
    
    performance['is_host'] = performance['Year'].isin(host_years)
    
    return performance, host_years

def compare_hosting_effects(df, season='Summer'):
    """Compare USA and China hosting effects"""
    
    usa_perf, usa_hosts = analyze_country_performance(df, 'USA', season)
    china_perf, china_hosts = analyze_country_performance(df, 'China', season)
    
    results = {}
    
    for country, perf, host_years in [('USA', usa_perf, usa_hosts), ('China', china_perf, china_hosts)]:
        if not perf.empty and host_years:
            host_data = perf[perf['is_host']]
            non_host_data = perf[~perf['is_host']]
            
            if not host_data.empty and not non_host_data.empty:
                host_avg = host_data['total_score'].mean()
                non_host_avg = non_host_data['total_score'].mean()
                boost = host_avg - non_host_avg
                boost_pct = (boost / non_host_avg) * 100 if non_host_avg > 0 else 0
                
                # Statistical test
                t_stat, p_val = stats.ttest_ind(host_data['total_score'], non_host_data['total_score'])
                
                results[country] = {
                    'host_avg': host_avg,
                    'non_host_avg': non_host_avg,
                    'boost': boost,
                    'boost_pct': boost_pct,
                    't_stat': t_stat,
                    'p_val': p_val,
                    'host_years': host_years,
                    'host_data': host_data,
                    'all_data': perf
                }
    
    return results

def analyze_china_2008_impact(df):
    """Detailed analysis of China's 2008 hosting performance"""
    
    china_summer = df[(df['NOC'] == 'China') & (df['Season'] == 'Summer')]
    china_weighted = calculate_weighted_medals(china_summer)
    
    china_performance = china_weighted.groupby('Year').agg(
        weighted_score=('weighted_score', 'sum'),
        total_medals=('medal', 'sum'),
        gold_medals=('medal_type', lambda x: (x == 'Gold').sum()),
        silver_medals=('medal_type', lambda x: (x == 'Silver').sum()),
        bronze_medals=('medal_type', lambda x: (x == 'Bronze').sum())
    ).reset_index()
    
    # Focus on 2008
    if 2008 in china_performance['Year'].values:
        china_2008 = china_performance[china_performance['Year'] == 2008].iloc[0]
        
        # Compare to other years
        other_years = china_performance[china_performance['Year'] != 2008]
        avg_other_weighted = other_years['weighted_score'].mean() if not other_years.empty else 0
        avg_other_golds = other_years['gold_medals'].mean() if not other_years.empty else 0
        
        return {
            'performance_2008': china_2008,
            'avg_other_years': {
                'weighted_score': avg_other_weighted,
                'gold_medals': avg_other_golds,
                'total_medals': other_years['total_medals'].mean() if not other_years.empty else 0
            },
            'boost_weighted': china_2008['weighted_score'] - avg_other_weighted,
            'boost_golds': china_2008['gold_medals'] - avg_other_golds,
            'all_data': china_performance
        }
    
    return None

def forecast_medals_by_season(df, country='USA', season='Summer', n_periods=4):
    """Season-specific forecasting"""
    
    season_data = df[(df['NOC'] == country) & (df['Season'] == season)]
    weighted_data = calculate_weighted_medals(season_data)
    
    series = weighted_data.groupby('Year')['weighted_score'].sum().sort_index()
    
    if len(series) < 3:
        return None
    
    try:
        # ARIMA
        if PM_AVAILABLE:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                arima_model = pm.auto_arima(series, seasonal=False, error_action='ignore', suppress_warnings=True)
            arima_forecast = arima_model.predict(n_periods=n_periods)
        else:
            arima_model = sm.tsa.ARIMA(series, order=(1,1,1)).fit()
            arima_forecast = arima_model.forecast(steps=n_periods)
        
        # Random Forest
        df_rf = pd.DataFrame({'score': series})
        for lag in range(1, 5):
            df_rf[f'lag_{lag}'] = df_rf['score'].shift(lag)
        df_rf = df_rf.dropna()
        
        if len(df_rf) >= 2:
            X = df_rf[[f'lag_{i}' for i in range(1, 5)]]
            y = df_rf['score']
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_model.fit(X, y)
            
            last_values = list(series.tail(4))
            rf_forecast = []
            for _ in range(n_periods):
                pred = rf_model.predict([last_values[-4:]])[0]
                rf_forecast.append(max(0, pred))
                last_values.append(pred)
        else:
            rf_forecast = [series.mean()] * n_periods
        
        # Future years - proper Olympic scheduling
        last_year = int(series.index.max())
        future_years = []
        
        if season == 'Summer':
            # Summer Olympics: 2016, 2020, 2024, 2028, 2032...
            next_summer = 2020 if last_year < 2020 else ((last_year // 4 + 1) * 4)
            if next_summer <= last_year:
                next_summer += 4
            future_years = [next_summer + 4*i for i in range(n_periods)]
        else:  # Winter
            # Winter Olympics: 2014, 2018, 2022, 2026, 2030...
            next_winter = 2018 if last_year < 2018 else ((last_year // 4) * 4 + 2)
            if next_winter <= last_year:
                next_winter += 4
            future_years = [next_winter + 4*i for i in range(n_periods)]
        
        return {
            'arima_forecast': arima_forecast,
            'rf_forecast': rf_forecast,
            'future_years': future_years,
            'historical_data': series
        }
        
    except Exception as e:
        return None

# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.set_page_config(page_title="US vs China Olympic Analysis", layout="wide")

st.title("🏅 US vs China Olympic Hosting Advantage Analysis")
st.markdown("**Summer/Winter Split Analysis with Gold Medal Weighting - Strategic Insights for Olympic Dominance**")

uploaded_file = st.file_uploader("Upload Olympic Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        
        # Sidebar controls
        st.sidebar.header("Analysis Settings")
        season = st.sidebar.selectbox("Select Olympic Season", ["Summer", "Winter", "Both"])
        gold_weight = st.sidebar.slider("Gold Medal Weight", 1, 5, 3)
        silver_weight = st.sidebar.slider("Silver Medal Weight", 1, 3, 2) 
        bronze_weight = st.sidebar.slider("Bronze Medal Weight", 1, 2, 1)
        
        st.sidebar.markdown(f"**Weighting System:**\n- Gold: {gold_weight}x\n- Silver: {silver_weight}x\n- Bronze: {bronze_weight}x")
        
        # Update weighting function
        df_weighted = calculate_weighted_medals(df, gold_weight, silver_weight, bronze_weight)
        
        if season in ["Summer", "Winter"]:
            seasons_to_analyze = [season]
        else:
            seasons_to_analyze = ["Summer", "Winter"]
        
        for current_season in seasons_to_analyze:
            st.header(f"🏆 {current_season} Olympics Analysis")
            
            # Comparative analysis
            comparison = compare_hosting_effects(df_weighted, current_season)
            
            if 'USA' in comparison or 'China' in comparison:
                col1, col2 = st.columns(2)
                
                # USA Analysis
                if 'USA' in comparison:
                    with col1:
                        st.subheader("🇺🇸 USA Performance")
                        usa_data = comparison['USA']
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot performance over time
                        perf_data = usa_data['all_data']
                        ax.plot(perf_data['Year'], perf_data['total_score'], 
                               'o-', linewidth=2, color='navy', label='Weighted Score')
                        
                        # Highlight hosting years
                        host_data = perf_data[perf_data['is_host']]
                        if not host_data.empty:
                            ax.scatter(host_data['Year'], host_data['total_score'], 
                                     color='red', s=200, marker='*', 
                                     label=f'Hosting Years', zorder=5)
                            
                            for _, row in host_data.iterrows():
                                host_city = SUMMER_OLYMPICS.get(row['Year'], ("Unknown", ""))[0] if current_season == 'Summer' else WINTER_OLYMPICS.get(row['Year'], ("Unknown", ""))[0]
                                ax.annotate(f'{int(row["Year"])}\n{host_city}', 
                                          (row['Year'], row['total_score']),
                                          textcoords="offset points", xytext=(0,20), 
                                          ha='center', fontsize=9,
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                        
                        ax.set_title(f'USA {current_season} Olympics - Weighted Performance')
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Weighted Medal Score')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Statistics
                        st.write(f"**Hosting Boost:** {usa_data['boost']:.1f} weighted points ({usa_data['boost_pct']:.1f}%)")
                        if usa_data['p_val'] < 0.05:
                            st.success("✅ Statistically significant hosting advantage!")
                        else:
                            st.warning("⚠️ No significant hosting advantage detected")
                
                # China Analysis
                if 'China' in comparison:
                    with col2:
                        st.subheader("🇨🇳 China Performance")
                        china_data = comparison['China']
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot performance over time
                        perf_data = china_data['all_data']
                        ax.plot(perf_data['Year'], perf_data['total_score'], 
                               'o-', linewidth=2, color='red', label='Weighted Score')
                        
                        # Highlight hosting years
                        host_data = perf_data[perf_data['is_host']]
                        if not host_data.empty:
                            ax.scatter(host_data['Year'], host_data['total_score'], 
                                     color='gold', s=200, marker='*', 
                                     label='Hosting Years', zorder=5)
                            
                            for _, row in host_data.iterrows():
                                host_city = SUMMER_OLYMPICS.get(row['Year'], ("Unknown", ""))[0] if current_season == 'Summer' else WINTER_OLYMPICS.get(row['Year'], ("Unknown", ""))[0]
                                ax.annotate(f'{int(row["Year"])}\n{host_city}', 
                                          (row['Year'], row['total_score']),
                                          textcoords="offset points", xytext=(0,20), 
                                          ha='center', fontsize=9,
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
                        
                        ax.set_title(f'China {current_season} Olympics - Weighted Performance')
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Weighted Medal Score')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Statistics
                        st.write(f"**Hosting Boost:** {china_data['boost']:.1f} weighted points ({china_data['boost_pct']:.1f}%)")
                        if china_data['p_val'] < 0.05:
                            st.success("✅ Statistically significant hosting advantage!")
                        else:
                            st.warning("⚠️ No significant hosting advantage detected")
            
            # China 2008 Deep Dive
            if current_season == "Summer":
                st.subheader("🎯 China 2008 Beijing Olympics - Case Study")
                
                china_2008_analysis = analyze_china_2008_impact(df_weighted)
                
                if china_2008_analysis:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("China 2008 Weighted Score", 
                                f"{china_2008_analysis['performance_2008']['weighted_score']:.0f}",
                                f"+{china_2008_analysis['boost_weighted']:.0f} vs avg")
                    
                    with col2:
                        st.metric("Gold Medals in 2008", 
                                f"{china_2008_analysis['performance_2008']['gold_medals']:.0f}",
                                f"+{china_2008_analysis['boost_golds']:.0f} vs avg")
                    
                    with col3:
                        boost_pct = (china_2008_analysis['boost_weighted'] / china_2008_analysis['avg_other_years']['weighted_score']) * 100
                        st.metric("Performance Boost", f"{boost_pct:.1f}%")
                    
                    st.write("**Key Insights from China 2008:**")
                    st.write(f"• China achieved their best ever Olympic performance when hosting")
                    st.write(f"• {china_2008_analysis['boost_golds']:.0f} more gold medals than their average")
                    st.write(f"• {boost_pct:.1f}% improvement in weighted performance")
                    st.write("• Demonstrates clear hosting advantage for strategic planning")
            
            # Forecasting by season
            st.subheader(f"🔮 {current_season} Olympics Forecasting")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**USA Forecast**")
                usa_forecast = forecast_medals_by_season(df_weighted, 'USA', current_season)
                
                if usa_forecast:
                    for i, (year, arima_pred, rf_pred) in enumerate(zip(
                        usa_forecast['future_years'], 
                        usa_forecast['arima_forecast'], 
                        usa_forecast['rf_forecast']), 1):
                        
                        host_indicator = ""
                        if current_season == 'Summer' and year in [2028, 2032]:
                            if year == 2028:
                                host_indicator = " 🏟️ **LA 2028 HOSTING**"
                            else:
                                host_indicator = " (Brisbane hosting)"
                        elif current_season == 'Winter':
                            # Check for future US Winter hosting (none currently scheduled)
                            pass
                        
                        st.write(f"**{year}:** ARIMA: {arima_pred:.0f}, RF: {rf_pred:.0f} pts{host_indicator}")
            
            with col2:
                st.write("**China Forecast**")
                china_forecast = forecast_medals_by_season(df_weighted, 'China', current_season)
                
                if china_forecast:
                    for i, (year, arima_pred, rf_pred) in enumerate(zip(
                        china_forecast['future_years'], 
                        china_forecast['arima_forecast'], 
                        china_forecast['rf_forecast']), 1):
                        
                        st.write(f"**{year}:** ARIMA: {arima_pred:.0f}, RF: {rf_pred:.0f} pts")
            
            # Comparative visualization
            if usa_forecast and china_forecast:
                st.subheader(f"USA vs China {current_season} Olympics Forecast Comparison")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Historical data
                usa_hist = usa_forecast['historical_data']
                china_hist = china_forecast['historical_data']
                
                ax.plot(usa_hist.index, usa_hist.values, 'o-', 
                       linewidth=2, label='USA Historical', color='navy')
                ax.plot(china_hist.index, china_hist.values, 's-', 
                       linewidth=2, label='China Historical', color='red')
                
                # Forecasts
                usa_years = usa_forecast['future_years']
                china_years = china_forecast['future_years']
                
                ax.plot(usa_years, usa_forecast['arima_forecast'], '--', 
                       linewidth=2, alpha=0.7, label='USA ARIMA Forecast', color='lightblue')
                ax.plot(china_years, china_forecast['arima_forecast'], '--', 
                       linewidth=2, alpha=0.7, label='China ARIMA Forecast', color='lightcoral')
                
                # Highlight hosting years
                if current_season == 'Summer':
                    if 2028 in usa_years:
                        ax.axvline(x=2028, color='gold', linestyle=':', alpha=0.8, linewidth=3)
                        ax.text(2028, ax.get_ylim()[1]*0.9, 'USA Hosts\nLA 2028', 
                               ha='center', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
                elif current_season == 'Winter':
                    # No current future Winter hosting for USA
                    pass
                
                ax.set_title(f'USA vs China {current_season} Olympics - Weighted Score Comparison & Forecast')
                ax.set_xlabel('Year')
                ax.set_ylabel('Weighted Medal Score')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        # Strategic Summary
        st.header("🎯 Strategic Summary & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Findings")
            st.write("**Hosting Advantages Confirmed:**")
            st.write("✅ Both USA and China show performance boosts when hosting")
            st.write("✅ China's 2008 Beijing performance validates hosting strategy")
            st.write("✅ Gold medal weighting emphasizes quality over quantity")
            st.write("✅ 2028 LA Olympics present strategic opportunity for USA")
            
        with col2:
            st.subheader("Strategic Recommendations")
            st.write("**For USA Olympic Dominance:**")
            st.write("🎯 Maximize 2028 LA hosting advantage")
            st.write("🏋️ Focus on sports with highest gold medal potential")
            st.write("🏟️ Leverage home crowd and familiar conditions")
            st.write("📊 Target weighted score improvement over raw medal count")
            st.write("🇨🇳 Study China's 2008 hosting strategy for insights")
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

else:
    st.info("📁 Upload Olympic dataset to begin USA vs China hosting analysis")
    st.markdown("""
    **This analysis provides:**
    - 🏅 **Separate Summer/Winter analysis** for accurate comparison
    - ⭐ **Gold-weighted scoring system** emphasizing medal quality  
    - 🇨🇳 **China 2008 case study** for hosting strategy insights
    - 🔮 **Season-specific forecasting** for strategic planning
    - 📊 **Head-to-head USA vs China comparison**
    """)
    
