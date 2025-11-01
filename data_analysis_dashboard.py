import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Configuration ---
st.set_page_config(
    page_title="Data Analysis & Visualization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management for Data ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = None

# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads CSV file into a pandas DataFrame and stores it in session state."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("🚫 **Error:** The uploaded file is empty.")
                return None
            st.session_state['original_df'] = df.copy() # Store original for reset
            st.session_state['df'] = df
            return df
        except Exception as e:
            st.error(f"🚫 **Error reading CSV:** {e}")
            return None
    return None

def get_column_types(df):
    """Separates columns into numeric and categorical types."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols

def generate_descriptive_stats(df):
    """Generates descriptive statistics and includes NumPy calculations."""
    stats_df = df.describe().T
    
    # Calculate Mean, Median, Std using NumPy for demonstration
    num_cols = df.select_dtypes(include=np.number).columns
    
    # Create a DataFrame for NumPy-calculated stats
    np_stats = pd.DataFrame({
        'mean_np': [np.mean(df[col].dropna()) for col in num_cols],
        'median_np': [np.median(df[col].dropna()) for col in num_cols],
        'std_np': [np.std(df[col].dropna()) for col in num_cols]
    }, index=num_cols)
    
    # Merge with the standard describe output (optional, but shows NumPy use)
    stats_df = stats_df.join(np_stats, how='left')
    
    return stats_df

# --- Sidebar Navigation ---

st.sidebar.title("🧭 Navigation")
menu_selection = st.sidebar.radio(
    "Go to",
    ["📁 Dataset Upload", "📊 Data Summary", "📈 Visualization", "🩹 Missing Data Handling", "⬇️ Download Report", "ℹ️ About"]
)

# --- Main App Sections ---

# 1. Dataset Upload
if menu_selection == "📁 Dataset Upload":
    st.title("📁 Dataset Upload & Preview")
    
    uploaded_file = st.file_uploader(
        "Upload your CSV Dataset", 
        type=["csv"],
        help="Please upload a valid CSV file to begin the analysis."
    )
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("✅ Dataset loaded successfully!")
            st.markdown("### Data Preview (First 5 Rows)")
            st.dataframe(df.head())
            
            st.markdown("### Dataset Structure")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dimensions")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                
            with col2:
                st.subheader("Data Types")
                # Use st.table for better display of dtypes
                dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.dataframe(dtypes_df)

            # Check for column type warnings
            numeric_cols, categorical_cols = get_column_types(df)
            if not numeric_cols:
                st.warning("⚠️ **Warning:** No numeric columns found in the dataset.")
            if not categorical_cols:
                st.warning("⚠️ **Warning:** No categorical columns found in the dataset.")
    elif st.session_state['df'] is None:
        st.info("⬆️ Please upload a CSV file to start the analysis.")

# 2. Data Summary
elif menu_selection == "📊 Data Summary":
    st.title("📊 Data Summary")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        st.markdown("### Descriptive Statistics")
        stats_df = generate_descriptive_stats(df)
        st.dataframe(stats_df)
        
        st.markdown("### Column Data Types")
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtypes_df)

    else:
        st.info("Please upload a dataset first in the 'Dataset Upload' section.")

# 3. Data Visualization
elif menu_selection == "📈 Visualization":
    st.title("📈 Data Visualization")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        numeric_cols, categorical_cols = get_column_types(df)

        st.sidebar.header("🎨 Plot Options")
        plot_type = st.sidebar.selectbox(
            "Select Plot Type",
            ["Histogram", "Bar Chart", "Box Plot", "Heatmap (Correlation)", "Scatter Plot", "Pairplot"]
        )

        # --- Plotting Logic ---
        
        # 1. Histogram (Distribution of Numeric)
        if plot_type == "Histogram" and numeric_cols:
            st.subheader("Histogram: Distribution of a Numeric Variable")
            column = st.sidebar.selectbox("Select Numeric Column", numeric_cols)
            
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {column}')
            st.pyplot(fig)
        elif plot_type == "Histogram" and not numeric_cols:
             st.warning("Cannot generate Histogram. The dataset has no numeric columns.")

        # 2. Bar Chart (Frequency of Categorical)
        elif plot_type == "Bar Chart" and categorical_cols:
            st.subheader("Bar Chart: Frequency of a Categorical Variable")
            column = st.sidebar.selectbox("Select Categorical Column", categorical_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y=df[column], ax=ax, order = df[column].value_counts().index)
            ax.set_title(f'Frequency of {column}')
            ax.set_xlabel('Count')
            st.pyplot(fig)
        elif plot_type == "Bar Chart" and not categorical_cols:
             st.warning("Cannot generate Bar Chart. The dataset has no categorical columns.")
             
        # 3. Box Plot (Outliers)
        elif plot_type == "Box Plot" and numeric_cols:
            st.subheader("Box Plot: Outlier Detection")
            column = st.sidebar.selectbox("Select Numeric Column for Box Plot", numeric_cols)
            
            fig, ax = plt.subplots()
            sns.boxplot(y=df[column], ax=ax)
            ax.set_title(f'Box Plot of {column}')
            st.pyplot(fig)
        elif plot_type == "Box Plot" and not numeric_cols:
             st.warning("Cannot generate Box Plot. The dataset has no numeric columns.")

        # 4. Heatmap (Correlation)
        elif plot_type == "Heatmap (Correlation)" and numeric_cols and len(numeric_cols) > 1:
            st.subheader("Heatmap: Correlation Matrix of Numeric Variables")
            
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
        elif plot_type == "Heatmap (Correlation)" and (not numeric_cols or len(numeric_cols) < 2):
            st.warning("Cannot generate Correlation Heatmap. Need at least two numeric columns.")

        # 5. Scatter Plot (Relationship)
        elif plot_type == "Scatter Plot" and len(numeric_cols) >= 2:
            st.subheader("Scatter Plot: Relationship Between Two Variables")
            x_var = st.sidebar.selectbox("Select X-Axis Variable", numeric_cols)
            y_var = st.sidebar.selectbox("Select Y-Axis Variable", [c for c in numeric_cols if c != x_var])
            
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
            ax.set_title(f'Scatter Plot of {x_var} vs {y_var}')
            st.pyplot(fig)
        elif plot_type == "Scatter Plot" and len(numeric_cols) < 2:
             st.warning("Cannot generate Scatter Plot. Need at least two numeric columns.")
             
        # 6. Pairplot (Multi-variable relationships)
        elif plot_type == "Pairplot" and len(numeric_cols) >= 2:
            st.subheader("Pairplot: Multi-variable Relationships")
            max_cols = min(5, len(numeric_cols))
            selected_cols = st.sidebar.multiselect(
                "Select Columns (Max 5 Recommended)", 
                numeric_cols, 
                default=numeric_cols[:min(3, len(numeric_cols))]
            )

            if selected_cols:
                st.info("Generating Pairplot. This may take a moment for large datasets.")
                try:
                    # Limit sample size for performance on very large datasets
                    sample_df = df[selected_cols].sample(n=min(500, len(df)), random_state=42)
                    fig = sns.pairplot(sample_df.dropna())
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating Pairplot: {e}")
            else:
                st.warning("Please select at least one numeric column for the Pairplot.")
        elif plot_type == "Pairplot" and len(numeric_cols) < 2:
            st.warning("Cannot generate Pairplot. Need at least two numeric columns.")
        
    else:
        st.info("Please upload a dataset first in the 'Dataset Upload' section.")

# 4. Missing Data Handling
elif menu_selection == "🩹 Missing Data Handling":
    st.title("🩹 Missing Data Analysis and Cleaning")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        missing_info = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage (%)': missing_percentage
        })
        missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
            by='Missing Count', ascending=False
        )
        
        st.markdown("### Missing Value Summary")
        if missing_info.empty:
            st.success("🎉 **No missing values** found in the dataset.")
        else:
            st.warning(f"⚠️ **{len(missing_info)}** column(s) have missing values.")
            st.dataframe(missing_info)
            
            st.markdown("### Missing Data Visualization (Heatmap)")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Data Heatmap')
            st.pyplot(fig)
            
            st.markdown("### Interactive Cleaning Options")
            action = st.radio(
                "Select a missing data action:",
                ["Do Nothing", "Drop Missing Rows (dropna)", "Fill Missing Values with Mean (Imputation)"]
            )
            
            if action != "Do Nothing":
                confirm_button = st.button(f"Confirm and Apply: {action}")
                
                if confirm_button:
                    # Input Validation - Confirmation
                    st.session_state['df_before_action'] = st.session_state['df'].copy()
                    
                    if action == "Drop Missing Rows (dropna)":
                        rows_before = len(st.session_state['df'])
                        st.session_state['df'].dropna(inplace=True)
                        rows_after = len(st.session_state['df'])
                        st.success(f"✅ Dropped **{rows_before - rows_after}** rows with missing values.")
                        
                    elif action == "Fill Missing Values with Mean (Imputation)":
                        numeric_cols = get_column_types(st.session_state['df'])[0]
                        for col in numeric_cols:
                            mean_val = st.session_state['df'][col].mean()
                            st.session_state['df'][col].fillna(mean_val, inplace=True)
                        st.success("✅ Missing values in **numeric columns** filled with the column's mean.")

                    st.markdown("#### **Resulting Data Preview**")
                    st.dataframe(st.session_state['df'].head())

            st.markdown("---")
            if st.button("Reset Data to Original Upload"):
                if st.session_state['original_df'] is not None:
                    st.session_state['df'] = st.session_state['original_df'].copy()
                    st.success("🔄 Dataset has been **reset** to its original uploaded state.")
                else:
                    st.warning("Original dataset not found. Please re-upload.")
                    
    else:
        st.info("Please upload a dataset first in the 'Dataset Upload' section.")

# 5. Download Report
elif menu_selection == "⬇️ Download Report":
    st.title("⬇️ Download Descriptive Report")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        st.markdown("### Descriptive Statistics Report")
        stats_df = generate_descriptive_stats(df)
        st.dataframe(stats_df)

        @st.cache_data
        def convert_df_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv_report = convert_df_to_csv(stats_df)

        st.download_button(
            label="Download Descriptive Statistics as CSV",
            data=csv_report,
            file_name='descriptive_analysis_report.csv',
            mime='text/csv',
        )
    else:
        st.info("Please upload a dataset first in the 'Dataset Upload' section to generate a report.")

# 6. About Section
elif menu_selection == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    This interactive **Data Analysis and Visualization Dashboard** is a Python mini-project built using **Streamlit**. 
    It demonstrates a complete end-to-end data analytics workflow.
    
    ### 💻 Key Technologies Used:
    * **Streamlit:** For creating the interactive web application interface.
    * **Pandas:** For efficient data loading, cleaning, manipulation, and summarization.
    * **NumPy:** For performing core numerical computations (Mean, Median, Std Dev).
    * **Seaborn/Matplotlib:** For creating aesthetic and informative statistical visualizations.
    
    ### ✨ Core Features:
    * Upload any CSV file.
    * View dataset dimensions and data types.
    * Generate comprehensive descriptive statistics.
    * Explore data patterns with various plots (Histogram, Box Plot, Heatmap, etc.).
    * Identify and handle missing data through dropping rows or mean imputation.
    * Export the descriptive analysis report.
    """)
    st.markdown("---")
    st.markdown("Developed as a demonstration of core Data Science/Analytics tools in Python.")