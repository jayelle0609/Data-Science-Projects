
# Summary Statistics for DataFrame
def summarize_data(df):
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSummary statistics:\n", df.describe())
    print("\nFirst few rows:\n", df.head())

    summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Unique Values': df.nunique(),
        'Mean': df.mean(),
        'Std Dev': df.std(),
        'Min': df.min(),
        'Max': df.max()
    })
    return summary



# Normalize a column
def normalize_column(df, col_name):
    if col_name in df.columns:
        df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df   



# Remove missing values (not advised)
def remove_missing_values(df):
    df = df.dropna()
    return df



# Remove duplicates
def remove_duplicates(df):
    df = df.drop_duplicates()
    return df



# Convert categorical columns to numerical (ORDINAL, has order, label encoding to preserve order)
def convert_categorical_to_numerical(df, col_name):
    """
    Converts a specified categorical column to numerical using ordinal encoding.
    """
    if col_name in df.columns:
        df[col_name] = pd.Categorical(df[col_name]).codes
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df

# Convert categorical columns to numerical (Nominal, no order, ONE-HOT to avoid implicit ordering)
def convert_categorical_to_one_hot(df, col_name):
    """
    Converts a specified categorical column to numerical using one-hot encoding.
    """
    if col_name in df.columns:
        df = pd.get_dummies(df, columns=[col_name], drop_first=True)
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df

# Encode categorical columns using Label Encoding
def encode_categorical(df, col_name):
    """
    Encodes a specified categorical column using label encoding.
    """
    if col_name in df.columns:
        le = LabelEncoder()
        df[col_name] = le.fit_transform(df[col_name])
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df



# Impute missing values with mean or median or mode of column, for specific columns
def impute_missing_values(df, col_name, strategy='mean'):   
    """
    Imputes missing values in a specified column using the given strategy.
    """
    if col_name in df.columns:
        if strategy == 'mean':
            df[col_name].fillna(df[col_name].mean(), inplace=True)
        elif strategy == 'median':
            df[col_name].fillna(df[col_name].median(), inplace=True)
        elif strategy == 'mode':
            df[col_name].fillna(df[col_name].mode()[0], inplace=True)
        else:
            print("Invalid strategy. Use 'mean', 'median', or 'mode'.")
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df



# Convert data types
def convert_data_types(df, *col_names, new_type):
    """
    Converts one or more specified columns to a new data type.
    """
    for col in col_names:
        if col in df.columns:
            try:
                df[col] = df[col].astype(new_type)
            except ValueError as e:
                print(f"Error converting column '{col}' to {new_type}: {e}")
        else:
            print(f"Column '{col}' does not exist in the DataFrame.")
    return df




# Scaling numerical columns
def scale_numerical_columns(df, cols, scaler='standard'):
    """
    Scales specified numerical columns in the DataFrame using the given scaler.
    """
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        print("Invalid scaler. Use 'standard' or 'minmax'.")
        return df

    for col in cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
        else:
            print(f"Column '{col}' does not exist in the DataFrame.")
    return df



# Convert date columns to datetime format
def convert_to_datetime(df, col_name):
    if col_name in df.columns:
        try:
            df[col_name] = pd.to_datetime(df[col_name])
        except Exception as e:
            print(f"Error converting column '{col_name}' to datetime: {e}")
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df



# Import necessary libraries
def import_libraries():
    """
    Imports commonly used libraries for data analysis.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    return pd, np, plt, sns, train_test_split, StandardScaler, MinMaxScaler, LabelEncoder, SimpleImputer

# Calculate correlation matrix
def calculate_correlation(df):
    """
    Calculates the correlation matrix of the DataFrame.
    """
    corr_matrix = df.corr()
    print("Correlation Matrix:\n", corr_matrix)
    return corr_matrix

# Visualize correlation matrix
def visualize_correlation(corr_matrix):
    """
    Visualizes the correlation matrix using a heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix Heatmap")
    plt.show()



# Plot histogram of a column
def plot_histogram(df, col_name):
    """
    Plots a histogram of the specified column in the DataFrame.
    """
    if col_name in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col_name], bins=30, kde=True)
        plt.title(f"Histogram of {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Frequency")
        plt.show()
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")



# Plot scatter plot between two columns
def plot_scatter(df, col_x, col_y):
    """
    Plots a scatter plot between two specified columns in the DataFrame.
    """
    if col_x in df.columns and col_y in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=col_x, y=col_y)
        plt.title(f"Scatter Plot between {col_x} and {col_y}")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()
    else:
        print(f"Columns '{col_x}' or '{col_y}' do not exist in the DataFrame.")



# Plot box plot of multiple columns
def plot_boxplot(df, cols):
    """
    Plots a box plot for multiple specified columns in the DataFrame.
    """
    if all(col in df.columns for col in cols):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[cols])
        plt.title("Box Plot of Selected Columns")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"One or more columns {cols} do not exist in the DataFrame.") 



# Machine Learning cross validation, train-test split, and model evaluation
def ml_cross_validation(df, target_col, model, test_size=0.2, random_state=42):
    """
    Performs train-test split, cross-validation, and model evaluation.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    
    return X_train, X_test, y_train, y_test

# Perform feature selection using correlation
def feature_selection_by_correlation(df, target_col, threshold=0.5):
    """
    Selects features based on correlation with the target column.
    """
    if target_col in df.columns:
        corr = df.corr()[target_col].abs()
        selected_features = corr[corr > threshold].index.tolist()
        selected_features.remove(target_col)  # Remove target column from features
        print(f"Selected features based on correlation with {target_col}: {selected_features}")
        return selected_features
    else:
        print(f"Target column '{target_col}' does not exist in the DataFrame.")
        return []
    


# Statistics tests  
def perform_statistical_tests(df, col1, col2):
    """
    Performs statistical tests (t-test and ANOVA) on two specified columns in the DataFrame.
    """
    from scipy import stats

    if col1 in df.columns and col2 in df.columns:
        # T-test
        t_stat, p_value_t = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
        print(f"T-test between {col1} and {col2}: t-statistic = {t_stat}, p-value = {p_value_t}")

        # ANOVA
        f_stat, p_value_f = stats.f_oneway(df[col1].dropna(), df[col2].dropna())
        print(f"ANOVA between {col1} and {col2}: F-statistic = {f_stat}, p-value = {p_value_f}")
    else:
        print(f"One or both columns '{col1}' and '{col2}' do not exist in the DataFrame.")



# Load dataset
def load_dataset(file_path):
    """
    Loads a dataset from the specified file path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
# Save DataFrame to CSV
def save_to_csv(df, file_path):
    """
    Saves the DataFrame to a CSV file at the specified file path.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")