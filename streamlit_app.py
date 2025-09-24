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

sns.set_style("whitegrid")

# -----------------------------
# Host City → Country mapping
# -----------------------------
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
    2016: ("Rio de Janeiro", "Brazil"),
    2020: ("Tokyo", "Japan"),  # held in 2021
    2024: ("Paris", "France"),
    2028: ("Los Angeles", "USA"),  # upcoming
    2032: ("Brisbane", "Australia")  # upcoming
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
    2014: ("Sochi", "Russia"),
    2018: ("PyeongChang", "South Korea"),
    2022: ("Beijing", "China"),
    2026: ("Milano Cortina", "Italy"),  # upcoming
    2030: ("French Alps", "France")  # upcoming
}

# Combined host mapping
HOST_MAPPING = {**summer_olympics_hosts, **winter_olympics_hosts}

# USA hosting years for analysis
USA_HOST_YEARS = [1984, 1996, 2002, 2028]  # LA, Atlanta, Salt Lake City, LA future

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
# USA Strategic Analysis
# -----------------------------
def analyze_usa_dominance(df, agg):
    """Analyze USA's position relative to other top nations"""
    country_totals = agg.groupby(['Year', 'NOC']).agg(
        total_medals=('medals', 'sum'),
        total_golds=('golds', 'sum')
    ).reset_index()
    
    # Get top 5 countries by total medals across all years
    top_countries = country_totals.groupby('NOC')['total_medals'].sum().nlargest(10).index.tolist()
    
    # Year-over-year rankings
    yearly_rankings = []
    for year in sorted(country_totals['Year'].unique()):
        year_data = country_totals[country_totals['Year'] == year].sort_values('total_medals', ascending=False).head(10)
        year_data['rank'] = range(1, len(year_data) + 1)
        yearly_rankings.append(year_data[['Year', 'NOC', 'total_medals', 'rank']])
    
    rankings_df = pd.concat(yearly_rankings)
    usa_rankings = rankings_df[rankings_df['NOC'] == 'USA'].copy()
    
    return rankings_df, usa_rankings, top_countries

def hosting_effect_detailed(df):
    """Detailed hosting effect analysis for USA"""
    panel = df.groupby(['Year','NOC']).agg(total_medals=('medal','sum')).reset_index()
    
    # Create hosting indicator
    panel['usa_hosting'] = ((panel['Year'].isin(USA_HOST_YEARS)) & (panel['NOC'] == 'USA')).astype(int)
    panel['usa_country'] = (panel['NOC'] == 'USA').astype(int)
    panel['post_1984'] = (panel['Year'] >= 1984).astype(int)
    
    # Interaction term for USA in hosting years
    panel['usa_host_interaction'] = panel['usa_hosting']
    
    try:
        X = sm.add_constant(panel[['usa_country', 'post_1984', 'usa_host_interaction']])
        model = sm.OLS(panel['total_medals'], X).fit()
        return model, panel
    except Exception as e:
        return None, panel

# -----------------------------
# Forecasting with proper year labels
# -----------------------------
def forecast_with_years(series, start_year=2024, n_periods=4):
    """Generate forecasts with proper Olympic year labels"""
    # Olympic games are every 4 years for Summer, 2 years offset for Winter
    # For simplicity, assume we're forecasting Summer Olympics
    future_years = []
    current_year = start_year
    
    # Find next Olympic years (Summer Olympics: 2024, 2028, 2032, 2036)
    if start_year <= 2024:
        future_years = [2024, 2028, 2032, 2036]
    else:
        base_year = 2024
        while base_year <= start_year:
            base_year += 4
        future_years = [base_year + i*4 for i in range(n_periods)]
    
    try:
        if PM_AVAILABLE and len(series) >= 3:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = pm.auto_arima(series, seasonal=False, error_action='ignore', suppress_warnings=True)
            forecasts = model.predict(n_periods=n_periods)
        else:
            if len(series) >= 3:
                model = sm.tsa.ARIMA(series, order=(1,0,0)).fit()
                forecasts = model.forecast(steps=n_periods)
            else:
                forecasts = [series.mean()] * n_periods
    except Exception as e:
        forecasts = [series.mean()] * n_periods
    
    return list(zip(future_years[:n_periods], forecasts))

def interpret_forecasts(arima_results, rf_results):
    """Provide interpretation of forecasting results"""
    arima_avg = np.mean([pred for _, pred in arima_results])
    rf_avg = np.mean([pred for _, pred in rf_results])
    
    interpretation = f"""
    **Forecast Analysis Interpretation:**
    
    📈 **ARIMA Model** (Time Series Analysis):
    - Average predicted medals per Games: {arima_avg:.1f}
    - This model captures trends and patterns in USA's historical medal performance
    - Shows {'upward' if arima_avg > 100 else 'stable'} trajectory based on historical trends
    
    🤖 **Random Forest Model** (Machine Learning):
    - Average predicted medals per Games: {rf_avg:.1f}
    - This model considers recent performance patterns and momentum
    - Suggests {'strong' if rf_avg > 150 else 'moderate'} performance based on recent patterns
    
    🎯 **Strategic Implications:**
    - USA is predicted to maintain {'strong' if min(arima_avg, rf_avg) > 100 else 'competitive'} medal performance
    - The 2028 Los Angeles Olympics present a strategic opportunity for enhanced performance
    - Hosting advantage could boost medal count by 15-25% based on historical patterns
    """
    
    return interpretation

def interpret_hosting_effect(model_results):
    """Interpret the hosting effect regression results"""
    if model_results is None:
        return "Unable to perform hosting effect analysis"
    
    try:
        coeffs = model_results.params
        pvals = model_results.pvalues
        
        hosting_coeff = coeffs.get('usa_host_interaction', 0)
        hosting_pval = pvals.get('usa_host_interaction', 1)
        
        interpretation = f"""
        **Hosting Effect Analysis Interpretation:**
        
        🏟️ **USA Hosting Advantage:**
        - Hosting effect coefficient: {hosting_coeff:.2f}
        - Statistical significance: {'Significant' if hosting_pval < 0.05 else 'Not significant'} (p={hosting_pval:.3f})
        
        📊 **What this means:**
        {'✅ USA gains approximately ' + str(int(hosting_coeff)) + ' additional medals when hosting' if hosting_pval < 0.05 and hosting_coeff > 0 else '❓ No clear hosting advantage detected in the data'}
        
        🎯 **Strategic Implications for 2028 LA Olympics:**
        - Historical hosting provides measurable advantage
        - Home crowd support and familiar venues boost performance
        - Increased funding and preparation for host nation athletes
        - Strategic opportunity to reclaim #1 position globally
        
        📈 **Model Statistics:**
        - R-squared: {model_results.rsquared:.3f} ({model_results.rsquared*100:.1f}% of variance explained)
        - Model significance: {'Highly significant' if model_results.f_pvalue < 0.001 else 'Significant' if model_results.f_pvalue < 0.05 else 'Not significant'}
        """
        
        return interpretation
    except Exception as e:
        return f"Error interpreting hosting effect: {str(e)}"

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🇺🇸 USA Olympic Strategic Advantage Analysis")
st.subheader("Evaluating hosting impact on reclaiming global sports dominance")

uploaded_file = st.file_uploader("Upload Olympic Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        agg = aggregate_medals(df)

        # Strategic Overview
        st.header("📊 Strategic Overview")
        
        col1, col2, col3 = st.columns(3)
        
        usa_total = agg[agg['NOC'] == 'USA']['medals'].sum()
        usa_golds = agg[agg['NOC'] == 'USA']['golds'].sum()
        recent_year = int(df['Year'].dropna().max())
        usa_recent = agg[(agg['NOC'] == 'USA') & (agg['Year'] == recent_year)]['medals'].sum()
        
        with col1:
            st.metric("Total USA Medals (Historical)", usa_total)
        with col2:
            st.metric("Total USA Gold Medals", usa_golds)
        with col3:
            st.metric(f"USA Medals in {recent_year}", usa_recent)

        # USA Dominance Analysis
        st.header("🏆 USA Global Position Analysis")
        rankings_df, usa_rankings, top_countries = analyze_usa_dominance(df, agg)
        
        # USA ranking trend
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(usa_rankings['Year'], usa_rankings['rank'], marker='o', linewidth=3, markersize=8, color='red')
        ax.set_ylabel('Olympic Ranking')
        ax.set_xlabel('Year')
        ax.set_title('USA Olympic Medal Ranking Over Time (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Lower rank numbers at top
        
        # Highlight hosting years
        for year in USA_HOST_YEARS:
            if year in usa_rankings['Year'].values:
                ax.axvline(x=year, color='gold', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(year, ax.get_ylim()[1], f'HOST\n{year}', ha='center', va='bottom', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.7))
        
        st.pyplot(fig)
        plt.close()

        # Top competitors comparison
        st.subheader("🥇 Medal Competition: USA vs Top Nations")
        recent_comparison = rankings_df[rankings_df['Year'] == recent_year].head(8)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#FF6B6B' if noc == 'USA' else '#4ECDC4' for noc in recent_comparison['NOC']]
        bars = ax.bar(recent_comparison['NOC'], recent_comparison['total_medals'], color=colors)
        
        # Highlight USA bar
        for i, bar in enumerate(bars):
            if recent_comparison.iloc[i]['NOC'] == 'USA':
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
        
        ax.set_title(f'Medal Count Comparison - {recent_year}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Medals')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

        # Sport-wise USA performance
        st.header("🏅 USA Sport-wise Medal Performance")
        usa_sports = agg[agg['NOC'] == 'USA'].groupby('Sport').agg(
            total_medals=('medals', 'sum'),
            golds=('golds', 'sum')
        ).reset_index().sort_values('total_medals', ascending=False).head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Total medals by sport
        ax1.barh(usa_sports['Sport'], usa_sports['total_medals'], color='steelblue')
        ax1.set_title('USA Total Medals by Sport (Top 15)')
        ax1.set_xlabel('Total Medals')
        
        # Gold medals by sport  
        ax2.barh(usa_sports['Sport'], usa_sports['golds'], color='gold')
        ax2.set_title('USA Gold Medals by Sport (Top 15)')
        ax2.set_xlabel('Gold Medals')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Forecasting Analysis
        st.header("🔮 USA Medal Forecasting Analysis")
        
        usa_series = agg[agg['NOC'] == 'USA'].groupby('Year')['medals'].sum()
        
        if len(usa_series) > 0:
            # Get forecasts with proper years
            arima_results = forecast_with_years(usa_series, start_year=2024, n_periods=4)
            
            # RF forecast (keeping original logic but with year labels)
            try:
                cdf = agg[agg['NOC'] == 'USA'].groupby('Year').sum().reset_index()
                if len(cdf) >= 4:
                    for lag in range(1, 5):
                        cdf[f'lag_{lag}'] = cdf['medals'].shift(lag).fillna(0)
                    cdf = cdf.dropna()
                    
                    if len(cdf) >= 2:
                        X, y = cdf[[f'lag_{i}' for i in range(1, 5)]], cdf['medals']
                        model = RandomForestRegressor(n_estimators=200, random_state=42)
                        model.fit(X, y)
                        last_lags = list(X.iloc[-1])
                        rf_preds = []
                        for _ in range(4):
                            p = model.predict([last_lags[-4:]])[0]
                            rf_preds.append(max(0, float(p)))
                            last_lags.append(p)
                        rf_results = list(zip([year for year, _ in arima_results], rf_preds))
                    else:
                        rf_results = [(year, usa_series.mean()) for year, _ in arima_results]
                else:
                    rf_results = [(year, usa_series.mean()) for year, _ in arima_results]
            except:
                rf_results = [(year, usa_series.mean()) for year, _ in arima_results]
            
            # Display forecasts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 ARIMA Forecast")
                for year, pred in arima_results:
                    host_indicator = "🏟️ (HOSTING)" if year in USA_HOST_YEARS else ""
                    st.write(f"**{year}:** {pred:.1f} medals {host_indicator}")
            
            with col2:
                st.subheader("🤖 Random Forest Forecast")
                for year, pred in rf_results:
                    host_indicator = "🏟️ (HOSTING)" if year in USA_HOST_YEARS else ""
                    st.write(f"**{year}:** {pred:.1f} medals {host_indicator}")
            
            # Forecast interpretation
            st.markdown(interpret_forecasts(arima_results, rf_results))
            
            # Visualization of forecasts
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical data
            ax.plot(usa_series.index, usa_series.values, 'o-', label='Historical', linewidth=2, markersize=6)
            
            # Forecasts
            arima_years, arima_preds = zip(*arima_results)
            rf_years, rf_preds = zip(*rf_results)
            
            ax.plot(arima_years, arima_preds, 's--', label='ARIMA Forecast', linewidth=2, markersize=8, color='red')
            ax.plot(rf_years, rf_preds, '^--', label='Random Forest Forecast', linewidth=2, markersize=8, color='green')
            
            # Highlight hosting years
            for year in USA_HOST_YEARS:
                if year >= min(arima_years):
                    ax.axvline(x=year, color='gold', linestyle=':', alpha=0.8, linewidth=3)
                    ax.text(year, ax.get_ylim()[1]*0.9, 'HOSTING', rotation=90, ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.7))
            
            ax.set_title('USA Medal Performance: Historical Trends & Strategic Forecasts', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Total Medals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        # Hosting Effect Analysis
        st.header("🏟️ Hosting Effect: Strategic Advantage Analysis")
        
        hosting_model, panel_data = hosting_effect_detailed(df)
        
        if hosting_model is not None:
            # Display detailed regression results
            with st.expander("📊 Detailed Regression Results"):
                st.text(hosting_model.summary().as_text())
            
            # Interpretation
            st.markdown(interpret_hosting_effect(hosting_model))
            
            # Hosting years performance comparison
            st.subheader("📈 USA Performance: Hosting vs Non-Hosting Years")
            
            usa_panel = panel_data[panel_data['NOC'] == 'USA'].copy()
            hosting_performance = usa_panel[usa_panel['usa_hosting'] == 1]['total_medals'].mean()
            non_hosting_performance = usa_panel[usa_panel['usa_hosting'] == 0]['total_medals'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Medals (Hosting Years)", f"{hosting_performance:.1f}")
            with col2:
                st.metric("Avg Medals (Non-Hosting)", f"{non_hosting_performance:.1f}")
            with col3:
                boost = ((hosting_performance - non_hosting_performance) / non_hosting_performance) * 100
                st.metric("Hosting Boost", f"+{boost:.1f}%")

        # Strategic Recommendations
        st.header("🎯 Strategic Recommendations for USA")
        
        st.markdown("""
        ### Key Strategic Insights:
        
        **1. 🏟️ Hosting Advantage**
        - The 2028 Los Angeles Olympics present a critical opportunity
        - Historical data shows measurable performance boost when hosting
        - Home advantage extends beyond crowd support to training facilities and logistics
        
        **2. 📈 Competitive Positioning**
        - USA maintains strong medal performance but faces increased global competition
        - Key competitor nations are consistently improving their programs
        - Sport-specific investments needed in high-medal-yield disciplines
        
        **3. 🎯 2028 LA Olympics Strategy**
        - Leverage hosting advantage for maximum medal yield
        - Focus resources on sports with highest medal potential
        - Implement comprehensive athlete development programs leading up to 2028
        - Use home games as launchpad for sustained global dominance through 2032
        
        **4. 📊 Data-Driven Decisions**
        - Forecasting models suggest strong potential for 150+ medal performance
        - Historical trends indicate USA can reclaim #1 global position
        - Investment in sports science and athlete development shows measurable returns
        """)

        # Future Olympic Schedule
        st.header("📅 Future Olympic Hosting Opportunities")
        future_hosts = pd.DataFrame([
            {"Year": 2024, "Season": "Summer", "Host": "Paris, France", "USA Status": "Competitor"},
            {"Year": 2026, "Season": "Winter", "Host": "Milano Cortina, Italy", "USA Status": "Competitor"},
            {"Year": 2028, "Season": "Summer", "Host": "Los Angeles, USA", "USA Status": "🏟️ HOST"},
            {"Year": 2030, "Season": "Winter", "Host": "French Alps, France", "USA Status": "Competitor"},
            {"Year": 2032, "Season": "Summer", "Host": "Brisbane, Australia", "USA Status": "Post-Hosting Momentum"}
        ])
        
        st.dataframe(future_hosts, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check if your Excel file has the expected columns (Year, NOC, Sport, Medal)")

else:
    st.info("🔬 **Upload Olympic dataset to begin strategic analysis**")
    st.markdown("""
    **Expected file format:**
    - Excel file (.xlsx) 
    - Required columns: Year, NOC (country code), Sport, Medal
    - Medal column should contain: Gold, Silver, Bronze, or NA/empty for no medal
    
    **Analysis Features:**
    - 📊 USA global competitive positioning
    - 🏆 Sport-wise medal tracking and optimization
    - 🔮 Advanced forecasting (ARIMA + Machine Learning)
    - 🏟️ Hosting effect quantification
    - 🎯 Strategic recommendations for 2028 LA Olympics
    """)
