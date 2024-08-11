import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Advanced EDA and CSV Modification App")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data with caching to avoid reloading on every rerun
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    
    # Display the dataset
    st.write("### Dataset Preview")
    st.write(df.head())

    # Data Filtration
    st.write("### Data Filtration")
    if st.checkbox("Apply filters"):
        filter_column = st.selectbox("Select column to filter", df.columns)
        
        if df[filter_column].dtype in [np.int64, np.float64]:
            min_value, max_value = st.slider("Select range", float(df[filter_column].min()), float(df[filter_column].max()), (float(df[filter_column].min()), float(df[filter_column].max())))
            df_filtered = df[(df[filter_column] >= min_value) & (df[filter_column] <= max_value)]
        else:
            unique_values = df[filter_column].unique()
            selected_values = st.multiselect("Select values to filter", unique_values, default=unique_values)
            df_filtered = df[df[filter_column].isin(selected_values)]
        
        st.write("Filtered Data Preview")
        st.write(df_filtered.head())
    else:
        df_filtered = df

    # Handle Categorical Data
    st.write("### Handle Categorical Data")
    if st.checkbox("Encode Categorical Data"):
        cat_column = st.selectbox("Select categorical column to encode", df_filtered.select_dtypes(include=['object']).columns)
        encoding_type = st.selectbox("Select encoding type", ["Label Encoding", "One-Hot Encoding"])
        if encoding_type == "Label Encoding":
            le = LabelEncoder()
            df_filtered[cat_column] = le.fit_transform(df_filtered[cat_column])
        elif encoding_type == "One-Hot Encoding":
            df_filtered = pd.get_dummies(df_filtered, columns=[cat_column])
        st.write(f"Encoded Data Preview ({encoding_type})")
        st.write(df_filtered.head())

    # Feature Engineering
    st.write("### Feature Engineering")
    if st.checkbox("Create a New Feature"):
        st.write("Create a new feature by applying arithmetic operations between columns.")
        col1, col2 = st.columns(2)
        with col1:
            feature_column_1 = st.selectbox("Select first column", df_filtered.select_dtypes(include=[np.number]).columns)
        with col2:
            feature_column_2 = st.selectbox("Select second column", df_filtered.select_dtypes(include=[np.number]).columns)
        
        operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])
        new_feature_name = st.text_input("Enter name for new feature")
        
        if st.button("Create Feature"):
            if operation == "Add":
                df_filtered[new_feature_name] = df_filtered[feature_column_1] + df_filtered[feature_column_2]
            elif operation == "Subtract":
                df_filtered[new_feature_name] = df_filtered[feature_column_1] - df_filtered[feature_column_2]
            elif operation == "Multiply":
                df_filtered[new_feature_name] = df_filtered[feature_column_1] * df_filtered[feature_column_2]
            elif operation == "Divide":
                df_filtered[new_feature_name] = df_filtered[feature_column_1] / df_filtered[feature_column_2]
            st.write("New Feature Created")
            st.write(df_filtered.head())

    # Scaling Columns
    st.write("### Scale Numeric Data")
    if st.checkbox("Scale columns"):
        scale_column = st.selectbox("Select column to scale", df_filtered.select_dtypes(include=[np.number]).columns)
        scaler_type = st.selectbox("Select scaler", ["Min-Max Scaler", "Standard Scaler"])
        if scaler_type == "Min-Max Scaler":
            scaler = MinMaxScaler()
        elif scaler_type == "Standard Scaler":
            scaler = StandardScaler()
        df_filtered[scale_column] = scaler.fit_transform(df_filtered[[scale_column]])
        st.write(f"Scaled Data Preview ({scaler_type})")
        st.write(df_filtered.head())

    # Drop Columns
    st.write("### Modify Dataset")
    if st.checkbox("Drop a column"):
        drop_column = st.multiselect("Select columns to drop", df_filtered.columns)
        if drop_column:
            df_filtered = df_filtered.drop(columns=drop_column)
            st.write("Updated Data Preview")
            st.write(df_filtered.head())

    # Visualization
    st.write("### Visualization")
    plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot", "Scatter plot", "Correlation Heatmap", "Line plot", "Bar plot", "Pair Plot", "Pie Chart"])

    # Generate plots directly
    if plot_type == "Histogram":
        selected_column = st.selectbox("Select column for histogram", df_filtered.select_dtypes(include=[np.number]).columns)
        bins = st.slider("Number of bins", 10, 50, 30)
        color = st.color_picker("Select color", "#4CAF50")
        st.write(f"Histogram for {selected_column}")
        fig = px.histogram(df_filtered, x=selected_column, nbins=bins, color_discrete_sequence=[color])
        st.plotly_chart(fig)
        
    elif plot_type == "Boxplot":
        selected_column = st.selectbox("Select column for boxplot", df_filtered.select_dtypes(include=[np.number]).columns)
        st.write(f"Boxplot for {selected_column}")
        fig = px.box(df_filtered, y=selected_column)
        st.plotly_chart(fig)
    
    elif plot_type == "Scatter plot":
        x_column = st.selectbox("Select X column", df_filtered.select_dtypes(include=[np.number]).columns)
        y_column = st.selectbox("Select Y column", df_filtered.select_dtypes(include=[np.number]).columns)
        color = st.color_picker("Select color", "#4CAF50")
        st.write(f"Scatter plot between {x_column} and {y_column}")
        fig = px.scatter(df_filtered, x=x_column, y=y_column, color_discrete_sequence=[color])
        st.plotly_chart(fig)
    
    elif plot_type == "Correlation Heatmap":
        st.write("Correlation Heatmap")
        corr_matrix = df_filtered.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig)
    
    elif plot_type == "Line plot":
        x_column = st.selectbox("Select X column for line plot", df_filtered.select_dtypes(include=[np.number]).columns)
        y_column = st.selectbox("Select Y column for line plot", df_filtered.select_dtypes(include=[np.number]).columns)
        color = st.color_picker("Select color", "#4CAF50")
        st.write(f"Line plot between {x_column} and {y_column}")
        fig = px.line(df_filtered, x=x_column, y=y_column, color_discrete_sequence=[color])
        st.plotly_chart(fig)
    
    elif plot_type == "Bar plot":
        x_column = st.selectbox("Select X column for bar plot", df_filtered.columns)
        y_column = st.selectbox("Select Y column for bar plot", df_filtered.select_dtypes(include=[np.number]).columns)
        color = st.color_picker("Select color", "#4CAF50")
        st.write(f"Bar plot between {x_column} and {y_column}")
        fig = px.bar(df_filtered, x=x_column, y=y_column, color_discrete_sequence=[color])
        st.plotly_chart(fig)

    elif plot_type == "Pair Plot":
        st.write("Pair Plot")
        selected_columns = st.multiselect("Select columns for pair plot", df_filtered.select_dtypes(include=[np.number]).columns)
        if len(selected_columns) > 1:
            sns.pairplot(df_filtered[selected_columns])
            st.pyplot()
        else:
            st.write("Please select more than one column for pair plot.")

    elif plot_type == "Pie Chart":
        selected_column = st.selectbox("Select column for pie chart", df_filtered.columns)
        color = st.color_picker("Select color", "#4CAF50")
        st.write(f"Pie chart for {selected_column}")
        fig = px.pie(df_filtered, names=selected_column, color_discrete_sequence=[color])
        st.plotly_chart(fig)

    # Data Export
    st.write("### Export Data")
    if st.checkbox("Export filtered/processed data"):
        export_filename = st.text_input("Enter file name for export", "modified_data.csv")
        st.write("Click the button below to export the data")
        st.download_button(label="Export Data as CSV", data=df_filtered.to_csv().encode('utf-8'), file_name=export_filename, mime='text/csv')
