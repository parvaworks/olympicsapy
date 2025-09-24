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
# Olympic Games Timeline
# -----------------------------
# Recent Olympic Games with host countries
OLYMPIC_TIMELINE = {
    1984: ("Los Angeles", "USA", "Summer"),
    1988: ("Seoul", "South Korea", "Summer"), 
    1992: ("Barcelona", "Spain", "Summer"),
    1996: ("Atlanta", "USA", "Summer"),
    2000: ("Sydney", "Australia", "Summer"),
    2004: ("Athens", "Greece", "Summer"),
    2008: ("Beijing", "China", "Summer"),
    2012: ("London", "Great Britain", "Summer"),
    2016: ("Rio de Janeiro", "Brazil", "Summer"),
    2020: ("Tokyo", "Japan", "Summer"),  # Held in 2021
    2024: ("Paris", "France", "Summer"),
    2028: ("Los Angeles", "USA", "Summer"),  # Future
    2032: ("Brisbane", "Australia", "Summer")  # Future
}

US_HOST_YEARS = [1984, 1996, 2028]  # US hosting years

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
    return df

# -----------------------------
# Analysis Functions
# -----------------------------
def analyze_usa_hosting_advantage(df):
    """Comprehensive analysis of USA hosting advantage"""
    
    # Get USA medal counts by year
    usa_medals = df[df['NOC'] == 'USA'].groupby('Year').agg(
        total_medals=('medal', 'sum'),
        gold_medals=('medal_type', lambda x: (x == 'Gold').sum()),
        silver_medals=('medal_type', lambda x: (x == 'Silver').sum()),
        bronze_medals=('medal_type', lambda x: (x == 'Bronze').sum())
    ).reset_index()
    
    # Add hosting information
    usa_medals['is_host'] = usa_medals['Year'].isin(US_HOST_YEARS)
    usa_medals['host_info'] = usa_medals['Year'].map(
        lambda x: f"{OLYMPIC_TIMELINE.get(x, ('Unknown', 'Unknown', 'Unknown'))[0]}" 
        if x in US_HOST_YEARS else "Not Hosting"
    )
    
    return usa_medals

def calculate_hosting_effect_detailed(df):
    """Detailed hosting effect analysis with proper interpretation"""
    
    # Create panel data
    panel = df.groupby(['Year', 'NOC']).agg(
        total_medals=('medal', 'sum')
    ).reset_index()
    
    # USA specific analysis
    usa_data = panel[panel['NOC'] == 'USA'].copy()
    
    if usa_data.empty:
        return None, "No USA data found"
    
    # Create treatment variables
    usa_data['is_host_year'] = usa_data['Year'].isin(US_HOST_YEARS)
    
    # Calculate differences
    host_performance = usa_data[usa_data['is_host_year']]['total_medals'].mean()
    non_host_performance = usa_data[~usa_data['is_host_year']]['total_medals'].mean()
    hosting_boost = host_performance - non_host_performance
    
    # Statistical test
    host_medals = usa_data[usa_data['is_host_year']]['total_medals']
    non_host_medals = usa_data[~usa_data['is_host_year']]['total_medals']
    
    if len(host_medals) > 0 and len(non_host_medals) > 0:
        t_stat, p_value = stats.ttest_ind(host_medals, non_host_medals)
    else:
        t_stat, p_value = 0, 1
    
    # Difference-in-differences analysis
    panel['treated'] = (panel['NOC'] == 'USA').astype(int)
    panel['post_1984'] = (panel['Year'] >= 1984).astype(int)
    panel['did'] = panel['treated'] * panel['post_1984']
    
    try:
        X = sm.add_constant(panel[['treated', 'post_1984', 'did']])
        model = sm.OLS(panel['total_medals'], X).fit()
        did_result = model
    except:
        did_result = None
    
    return {
        'usa_data': usa_data,
        'host_performance': host_performance,
        'non_host_performance': non_host_performance,
        'hosting_boost': hosting_boost,
        't_stat': t_stat,
        'p_value': p_value,
        'did_model': did_result
    }, None

def forecast_usa_medals_detailed(df):
    """Enhanced forecasting with proper year labeling"""
    
    usa_medals = df[df['NOC'] == 'USA'].groupby('Year')['medal'].sum().sort_index()
    
    if len(usa_medals) < 3:
        return None, "Insufficient data for forecasting"
    
    # ARIMA forecast
    try:
        if PM_AVAILABLE:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                arima_model = pm.auto_arima(usa_medals, seasonal=False, error_action='ignore', suppress_warnings=True)
            arima_forecast = arima_model.predict(n_periods=4)
        else:
            arima_model = sm.tsa.ARIMA(usa_medals, order=(1,1,1)).fit()
            arima_forecast = arima_model.forecast(steps=4)
    except:
        arima_forecast = [usa_medals.mean()] * 4
    
    # Random Forest forecast
    try:
        # Create lag features
        df_rf = pd.DataFrame({'medals': usa_medals})
        for lag in range(1, 5):
            df_rf[f'lag_{lag}'] = df_rf['medals'].shift(lag)
        
        df_rf = df_rf.dropna()
        
        if len(df_rf) >= 2:
            X = df_rf[[f'lag_{i}' for i in range(1, 5)]]
            y = df_rf['medals']
            
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_model.fit(X, y)
            
            # Generate predictions
            last_values = list(usa_medals.tail(4))
            rf_forecast = []
            
            for _ in range(4):
                pred = rf_model.predict([last_values[-4:]])[0]
                rf_forecast.append(max(0, pred))
                last_values.append(pred)
        else:
            rf_forecast = [usa_medals.mean()] * 4
    except:
        rf_forecast = [usa_medals.mean()] * 4
    
    # Future Olympic years
    last_year = usa_medals.index.max()
    future_years = []
    
    # Determine future Olympic years (every 4 years from last known)
    for i in range(1, 5):
        next_year = last_year + (4 * i)
        future_years.append(next_year)
    
    return {
        'arima_forecast': arima_forecast,
        'rf_forecast': rf_forecast,
        'future_years': future_years,
        'historical_data': usa_medals
    }, None

# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.set_page_config(page_title="US Olympic Hosting Advantage Analysis", layout="wide")

st.title("🇺🇸 US Olympic Hosting Advantage Analysis")
st.markdown("**Evaluating if hosting the Games provides the US with a strategic advantage in reclaiming top spot and global influence in international sports**")

uploaded_file = st.file_uploader("Upload Olympic Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        
        # Sidebar for analysis controls
        st.sidebar.header("Analysis Controls")
        show_detailed_stats = st.sidebar.checkbox("Show Detailed Statistics", True)
        show_sport_breakdown = st.sidebar.checkbox("Show Sport-wise Analysis", True)
        
        # Main Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🏆 USA Olympic Performance Analysis")
            
            # USA performance over time
            usa_analysis = analyze_usa_hosting_advantage(df)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Medal count over time with hosting years highlighted
            ax1.plot(usa_analysis['Year'], usa_analysis['total_medals'], 
                    marker='o', linewidth=2, markersize=8, color='navy', label='Total Medals')
            
            # Highlight hosting years
            host_data = usa_analysis[usa_analysis['is_host']]
            ax1.scatter(host_data['Year'], host_data['total_medals'], 
                       color='red', s=200, alpha=0.7, marker='*', 
                       label='Hosting Years', zorder=5)
            
            ax1.set_title('USA Olympic Medal Performance (1984-2016)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Total Medals')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add annotations for hosting years
            for _, row in host_data.iterrows():
                ax1.annotate(f'{int(row["Year"])}\n{row["host_info"]}', 
                           (row['Year'], row['total_medals']),
                           textcoords="offset points", xytext=(0,20), ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Plot 2: Medal type breakdown
            ax2.bar(usa_analysis['Year'], usa_analysis['gold_medals'], 
                   label='Gold', color='gold', alpha=0.8)
            ax2.bar(usa_analysis['Year'], usa_analysis['silver_medals'], 
                   bottom=usa_analysis['gold_medals'], label='Silver', color='silver', alpha=0.8)
            ax2.bar(usa_analysis['Year'], usa_analysis['bronze_medals'], 
                   bottom=usa_analysis['gold_medals'] + usa_analysis['silver_medals'], 
                   label='Bronze', color='#CD7F32', alpha=0.8)
            
            ax2.set_title('USA Medal Composition by Type', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Number of Medals')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.header("📊 Key Metrics")
            
            # Calculate key statistics
            host_years_data = usa_analysis[usa_analysis['is_host']]
            non_host_data = usa_analysis[~usa_analysis['is_host']]
            
            if not host_years_data.empty and not non_host_data.empty:
                host_avg = host_years_data['total_medals'].mean()
                non_host_avg = non_host_data['total_medals'].mean()
                boost_percentage = ((host_avg - non_host_avg) / non_host_avg) * 100
                
                st.metric("Avg Medals (Hosting)", f"{host_avg:.1f}", 
                         f"+{host_avg - non_host_avg:.1f}")
                st.metric("Avg Medals (Non-hosting)", f"{non_host_avg:.1f}")
                st.metric("Hosting Boost", f"{boost_percentage:.1f}%")
                
                # Show hosting years performance
                st.subheader("🏟️ USA Hosting Performance")
                for _, row in host_years_data.iterrows():
                    year = int(row['Year'])
                    medals = int(row['total_medals'])
                    city = row['host_info']
                    st.write(f"**{year}** ({city}): **{medals}** medals")
        
        # Detailed Hosting Effect Analysis
        st.header("📈 Hosting Effect Statistical Analysis")
        
        hosting_analysis, error = calculate_hosting_effect_detailed(df)
        
        if hosting_analysis and not error:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statistical Summary")
                st.write(f"**Average medals when hosting:** {hosting_analysis['host_performance']:.1f}")
                st.write(f"**Average medals when not hosting:** {hosting_analysis['non_host_performance']:.1f}")
                st.write(f"**Hosting advantage:** {hosting_analysis['hosting_boost']:.1f} medals")
                st.write(f"**T-statistic:** {hosting_analysis['t_stat']:.3f}")
                st.write(f"**P-value:** {hosting_analysis['p_value']:.3f}")
                
                if hosting_analysis['p_value'] < 0.05:
                    st.success("✅ **Statistically significant hosting advantage detected!**")
                else:
                    st.warning("⚠️ **No statistically significant hosting advantage detected**")
            
            with col2:
                st.subheader("Interpretation")
                if hosting_analysis['hosting_boost'] > 0:
                    st.write("🔍 **Key Findings:**")
                    st.write(f"• USA wins {hosting_analysis['hosting_boost']:.1f} more medals on average when hosting")
                    st.write(f"• This represents a {(hosting_analysis['hosting_boost']/hosting_analysis['non_host_performance']*100):.1f}% performance boost")
                    
                    if hosting_analysis['p_value'] < 0.05:
                        st.write("• The advantage is statistically significant")
                        st.write("• **Strategic implication:** Hosting provides measurable competitive advantage")
                    else:
                        st.write("• The advantage may be due to random variation")
                
        # Forecasting Analysis
        st.header("🔮 USA Medal Forecasting for Future Olympics")
        
        forecast_results, forecast_error = forecast_usa_medals_detailed(df)
        
        if forecast_results and not forecast_error:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ARIMA Forecast")
                for i, (year, pred) in enumerate(zip(forecast_results['future_years'], forecast_results['arima_forecast']), 1):
                    host_indicator = " 🏟️ **HOSTING**" if year in [2028, 2032] else ""
                    st.write(f"**{year}:** {pred:.0f} medals{host_indicator}")
                
                st.subheader("Random Forest Forecast")
                for i, (year, pred) in enumerate(zip(forecast_results['future_years'], forecast_results['rf_forecast']), 1):
                    host_indicator = " 🏟️ **HOSTING**" if year in [2028, 2032] else ""
                    st.write(f"**{year}:** {pred:.0f} medals{host_indicator}")
            
            with col2:
                st.subheader("Forecast Visualization")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Historical data
                historical_years = list(forecast_results['historical_data'].index)
                historical_medals = list(forecast_results['historical_data'].values)
                
                ax.plot(historical_years, historical_medals, 'o-', 
                       linewidth=2, markersize=6, label='Historical', color='navy')
                
                # Forecasts
                future_years = forecast_results['future_years']
                ax.plot(future_years, forecast_results['arima_forecast'], 's--', 
                       linewidth=2, markersize=8, label='ARIMA Forecast', color='red', alpha=0.7)
                ax.plot(future_years, forecast_results['rf_forecast'], '^--', 
                       linewidth=2, markersize=8, label='RF Forecast', color='green', alpha=0.7)
                
                # Highlight 2028 (USA hosting)
                if 2028 in future_years:
                    idx_2028 = future_years.index(2028)
                    ax.axvline(x=2028, color='gold', linestyle=':', alpha=0.7, linewidth=3)
                    ax.text(2028, max(forecast_results['arima_forecast'] + forecast_results['rf_forecast']), 
                           'USA Hosts\nLA 2028', ha='center', va='bottom', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
                
                ax.set_title('USA Olympic Medal Forecasting', fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel('Medals')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        # Strategic Implications
        st.header("🎯 Strategic Implications for US Olympic Dominance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Findings")
            if hosting_analysis:
                if hosting_analysis['hosting_boost'] > 0:
                    st.success(f"✅ **Hosting Advantage Confirmed:** USA gains {hosting_analysis['hosting_boost']:.0f} additional medals when hosting")
                    st.info("🏟️ **2028 Los Angeles Olympics:** Prime opportunity for medal dominance")
                    st.write("📈 **Recommended Strategy:** Maximize home field advantage through:")
                    st.write("• Enhanced athlete preparation and support")
                    st.write("• Crowd support and familiar conditions")
                    st.write("• Strategic event scheduling optimization")
                
        with col2:
            st.subheader("Global Competition Context")
            
            # Top competitors analysis
            recent_year = df['Year'].max()
            top_countries = df[df['Year'] == recent_year].groupby('NOC')['medal'].sum().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['red' if country == 'USA' else 'lightblue' for country in top_countries.index]
            top_countries.plot(kind='bar', ax=ax, color=colors)
            ax.set_title(f'Top Medal Winners - {recent_year} Olympics')
            ax.set_ylabel('Total Medals')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        # Sport-wise breakdown if requested
        if show_sport_breakdown:
            st.header("🏅 Sport-wise Performance Analysis")
            
            usa_sports = df[df['NOC'] == 'USA'].groupby(['Sport', 'Year']).agg(
                medals=('medal', 'sum'),
                golds=('medal_type', lambda x: (x == 'Gold').sum())
            ).reset_index()
            
            # Top sports for USA
            top_sports = usa_sports.groupby('Sport')['medals'].sum().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            top_sports.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('USA Top 10 Sports by Total Medals')
            ax.set_ylabel('Total Medals')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
            
            st.dataframe(usa_sports.pivot_table(index='Sport', columns='Year', values='medals', fill_value=0))
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please ensure your Excel file contains columns: Year, NOC, Sport, Medal")

else:
    st.info("📁 Please upload an Olympic dataset (Excel format) to begin the analysis")
    st.markdown("""
    **Expected dataset format:**
    - Excel file (.xlsx) 
    - Required columns: **Year, NOC, Sport, Medal**
    - Medal values: Gold, Silver, Bronze, or NA/empty
    - Years: 1984-2016 (or broader range)
    """)
    
    # Show preview of analysis capabilities
    st.subheader("🔍 This analysis will provide:")
    st.write("✅ **Hosting Effect Quantification** - Statistical evidence of home advantage")
    st.write("✅ **Medal Forecasting** - ARIMA and ML predictions for future Olympics")  
    st.write("✅ **Strategic Insights** - Actionable recommendations for 2028 LA Olympics")
    st.write("✅ **Sport-wise Breakdown** - Identify strongest competitive areas")
    st.write("✅ **Global Competition Context** - USA's position vs other nations")
