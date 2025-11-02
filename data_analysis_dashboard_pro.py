import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Pro Data Analysis & Visualization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "A professional data analysis dashboard built with Streamlit and Plotly."
    }
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #B2BEB5;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- State Management for Data ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = None

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Loads CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("🚫 **Error:** The uploaded file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"🚫 **Error reading CSV:** {e}")
        return None

def get_column_types(df):
    """Separates columns into numeric and categorical types."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols

def generate_descriptive_stats(df):
    """Generates descriptive statistics and includes NumPy calculations."""
    stats_df = df.describe().T
    num_cols = df.select_dtypes(include=np.number).columns
    
    np_stats = pd.DataFrame({
        'mean_np': [np.mean(df[col].dropna()) for col in num_cols],
        'median_np': [np.median(df[col].dropna()) for col in num_cols],
        'std_np': [np.std(df[col].dropna()) for col in num_cols]
    }, index=num_cols)
    
    return stats_df.join(np_stats, how='left')

# --- Sidebar Navigation ---

st.sidebar.markdown("# ⚙️ Main Navigation")
menu_selection = st.sidebar.radio(
    "Select an Action",
    ["📂 Data Intake", "🔬 Data Exploration", "📈 Advanced Visualization", "🩹 Data Cleaning", "📥 Export Report"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")

# --- Main App Sections ---

# 1. Data Intake (Upload)
if menu_selection == "📂 Data Intake":
    st.title("📂 Data Intake: Upload & Inspect")
    st.markdown("Upload your **CSV dataset** to begin the automated analysis.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV Dataset", 
        type=["csv"],
        help="Only CSV files are accepted for processing."
    )
    
    if uploaded_file:
        df_new = load_data(uploaded_file)
        if df_new is not None:
            st.session_state['original_df'] = df_new.copy() 
            st.session_state['df'] = df_new
            st.success("✅ Dataset loaded successfully! Proceed to **Data Exploration**.")

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        st.markdown("### 🔍 Current Dataset Overview")
        
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Structure & Types", "Input Validation"])

        with tab1:
            st.dataframe(df.head(10))
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", df.shape[0])
                st.metric("Total Columns", df.shape[1])
            with col2:
                st.subheader("Column Data Types")
                dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.dataframe(dtypes_df)

        with tab3:
            numeric_cols, categorical_cols = get_column_types(df)
            if not numeric_cols:
                st.warning("⚠️ **Warning:** No numeric columns found. Visualization options may be limited.")
            else:
                st.success(f"✅ Found **{len(numeric_cols)}** numeric columns.")
            
            if not categorical_cols:
                st.warning("⚠️ **Warning:** No categorical columns found.")
            else:
                st.success(f"✅ Found **{len(categorical_cols)}** categorical columns.")

# 2. Data Exploration (Summary)
elif menu_selection == "🔬 Data Exploration":
    st.title("🔬 Data Exploration & Summary")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        tab1, tab2 = st.tabs(["Descriptive Statistics", "Unique Value Counts"])

        with tab1:
            st.markdown("### 📊 Descriptive Statistics (Numeric & NumPy Calculations)")
            stats_df = generate_descriptive_stats(df)
            st.dataframe(stats_df.style.highlight_max(axis=0, color='lightblue'))
        
        with tab2:
            st.markdown("### 🔢 Unique Value Counts (Categorical Analysis)")
            categorical_cols = get_column_types(df)[1]
            if categorical_cols:
                selected_cat = st.selectbox("Select Categorical Column", categorical_cols)
                if selected_cat:
                    count_df = df[selected_cat].value_counts().reset_index()
                    count_df.columns = [selected_cat, 'Count']
                    st.dataframe(count_df)
            else:
                st.info("No categorical columns found for unique value analysis.")

    else:
        st.info("Please upload a dataset first in the **Data Intake** section.")

# 3. Advanced Visualization
elif menu_selection == "📈 Advanced Visualization":
    st.title("📈 Interactive Data Visualization")
    st.markdown("Use the sidebar controls to generate **modern, interactive Plotly charts**.")

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        numeric_cols, categorical_cols = get_column_types(df)
        
        st.sidebar.header("🎨 Plot Generator")
        plot_type = st.sidebar.selectbox(
            "Select Plot Type",
            ["Histogram", "Bar Chart", "Box Plot", "Violin Plot", "Line Plot", "Scatter Plot", "Pie Chart", "Correlation Heatmap", "Pairplot"]
        )
        
        st.subheader(f"Plot Type: **{plot_type}**")

        # --- Plotly Express Logic ---
        
        try:
            if plot_type == "Histogram" and numeric_cols:
                col = st.sidebar.selectbox("Numeric Column (X-axis)", numeric_cols)
                fig = px.histogram(df, x=col, marginal="box", title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Bar Chart" and categorical_cols:
                col = st.sidebar.selectbox("Categorical Column (X-axis)", categorical_cols)
                fig = px.bar(df[col].value_counts().reset_index(), 
                             x=df[col].value_counts().index, 
                             y=col, 
                             title=f'Frequency of {col}')
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Scatter Plot" and len(numeric_cols) >= 2:
                x_var = st.sidebar.selectbox("X-Axis", numeric_cols)
                y_var = st.sidebar.selectbox("Y-Axis", [c for c in numeric_cols if c != x_var])
                color_var = st.sidebar.selectbox("Color By (Optional)", [None] + categorical_cols)
                fig = px.scatter(df, x=x_var, y=y_var, color=color_var, title=f'{x_var} vs {y_var}')
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Line Plot" and len(numeric_cols) >= 2:
                if 'Date' in df.columns or df.select_dtypes(include=['datetime']).empty == False:
                     st.info("Best used with ordered or time-series data.")
                x_var = st.sidebar.selectbox("X-Axis (Order/Time)", df.columns)
                y_var = st.sidebar.selectbox("Y-Axis (Value)", numeric_cols)
                fig = px.line(df, x=x_var, y=y_var, title=f'{y_var} Over {x_var}')
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Box Plot" and numeric_cols:
                y_var = st.sidebar.selectbox("Numeric Column", numeric_cols)
                x_var = st.sidebar.selectbox("Group By (Optional)", [None] + categorical_cols)
                fig = px.box(df, x=x_var, y=y_var, title=f'Box Plot of {y_var}')
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Violin Plot" and numeric_cols:
                y_var = st.sidebar.selectbox("Numeric Column", numeric_cols)
                x_var = st.sidebar.selectbox("Group By (Optional)", [None] + categorical_cols)
                fig = px.violin(df, x=x_var, y=y_var, box=True, title=f'Violin Plot of {y_var}')
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Pie Chart" and categorical_cols:
                col = st.sidebar.selectbox("Categorical Column", categorical_cols)
                fig = px.pie(df, names=col, title=f'Proportion of {col}')
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
                st.markdown("### Correlation Matrix (Seaborn)")
                corr_matrix = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', ax=ax)
                st.pyplot(fig)

            elif plot_type == "Pairplot" and len(numeric_cols) >= 2:
                st.markdown("### Pairwise Relationships (Seaborn)")
                selected_cols = st.sidebar.multiselect("Select Numeric Columns (Max 5)", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
                if selected_cols:
                    # Sample for performance on large datasets
                    sample_df = df[selected_cols].sample(n=min(1000, len(df)), random_state=42)
                    st.info("Displaying Pairplot on a sample of the data for performance.")
                    fig = sns.pairplot(sample_df.dropna())
                    st.pyplot(fig)
                else:
                    st.warning("Please select columns for Pairplot.")
            
            else:
                st.warning(f"Insufficient columns for **{plot_type}**. Ensure you have the necessary numeric/categorical variables.")
                
        except Exception as e:
            st.error(f"An error occurred during plotting: {e}")


    else:
        st.info("Please upload a dataset first in the **Data Intake** section.")


# 4. Missing Data Handling
elif menu_selection == "🩹 Data Cleaning":
    st.title("🩹 Missing Data Handling & Cleaning")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        missing_info = df.isnull().sum().reset_index()
        missing_info.columns = ['Column', 'Missing Count']
        missing_info['Missing Percentage (%)'] = (missing_info['Missing Count'] / len(df)) * 100
        missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
            by='Missing Count', ascending=False
        ).set_index('Column')

        col1, col2 = st.columns([2, 3])

        with col1:
            st.markdown("### Missing Value Summary")
            if missing_info.empty:
                st.success("🎉 **No missing values** found in the dataset.")
            else:
                st.warning(f"⚠️ **{len(missing_info)}** column(s) have missing values.")
                st.dataframe(missing_info)
        
        with col2:
            st.markdown("### Missing Data Visualization (Heatmap)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap='magma', ax=ax)
            ax.set_title('Missing Data Heatmap')
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 🛠️ Interactive Cleaning Options")
        
        action = st.selectbox(
            "Select an action to apply to the dataset:",
            ["Select an option...", "Drop Rows with ANY Missing Value", "Impute Numeric Columns with Mean"]
        )
        
        if action != "Select an option...":
            if st.button(f"Confirm and Apply: {action}", type="primary"):
                # Input Validation - Confirmation
                df_temp = st.session_state['df'].copy()
                
                if action == "Drop Rows with ANY Missing Value":
                    rows_before = len(df_temp)
                    df_temp.dropna(inplace=True)
                    rows_after = len(df_temp)
                    st.success(f"✅ Dropped **{rows_before - rows_after}** rows. Dataset updated.")
                    
                elif action == "Impute Numeric Columns with Mean":
                    numeric_cols = get_column_types(df_temp)[0]
                    for col in numeric_cols:
                        if df_temp[col].isnull().any():
                            mean_val = df_temp[col].mean()
                            df_temp[col].fillna(mean_val, inplace=True)
                    st.success("✅ Missing values in **numeric columns** filled with the column's mean. Dataset updated.")

                st.session_state['df'] = df_temp.copy()
                st.dataframe(st.session_state['df'].head())
                
        st.markdown("---")
        if st.button("🔄 Reset Data to Original Upload"):
            if st.session_state['original_df'] is not None:
                st.session_state['df'] = st.session_state['original_df'].copy()
                st.success("Dataset successfully **reset** to its original uploaded state.")
            else:
                st.warning("Original dataset backup not found.")
                    
    else:
        st.info("Please upload a dataset first in the **Data Intake** section.")

# 5. Download Report
elif menu_selection == "📥 Export Report":
    st.title("📥 Export Data Analysis Report")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        st.markdown("### Final Descriptive Report")
        stats_df = generate_descriptive_stats(df)
        st.dataframe(stats_df)

        @st.cache_data
        def convert_df_to_csv(df):
            # Cache the conversion
            return df.to_csv().encode('utf-8')

        csv_report = convert_df_to_csv(stats_df)

        st.download_button(
            label="💾 Download Descriptive Statistics as CSV",
            data=csv_report,
            file_name='professional_analysis_report.csv',
            mime='text/csv',
            type="primary"
        )
        st.markdown("---")
        
        # Option to download the currently cleaned/modified dataset
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="⬇️ Download Current (Cleaned) Dataset as CSV",
            data=csv_data,
            file_name='cleaned_dataset.csv',
            mime='text/csv',
        )

    else:

        st.info("Please upload a dataset and perform analysis to generate reports.")
