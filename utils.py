"""
ml_super_utils.py
-----------------
All-in-one machine learning utility and pipeline builder.

Includes:
✅ Outlier & anomaly detection (Z-score, IQR, Isolation Forest, DBSCAN, One-Class SVM)
✅ Auto numeric/categorical detection
✅ ColumnTransformer preprocessing
✅ RandomForest pipeline
✅ GridSearchCV & RandomizedSearchCV
✅ Model evaluation, visualization, save/load
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint
import joblib


# ======================================================
# 1️⃣ OUTLIER DETECTION
# ======================================================

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR rule."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"✅ Removed outliers using IQR from columns: {columns}")
    return df


def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score."""
    for col in columns:
        z = np.abs(stats.zscore(df[col]))
        df = df[z < threshold]
    print(f"✅ Removed outliers using Z-score (|Z| < {threshold}) from: {columns}")
    return df


def clean_data_all(df, columns, z_thresh=3, use_iqr=True):
    """Combined outlier cleaning using IQR + Z-score."""
    print("🔧 Starting combined outlier cleaning...")
    if use_iqr:
        df = remove_outliers_iqr(df, columns)
    df = remove_outliers_zscore(df, columns, threshold=z_thresh)
    print(f"✅ Combined cleaning complete. Remaining shape: {df.shape}")
    return df


# ======================================================
# 2️⃣ ANOMALY DETECTION
# ======================================================

def detect_anomalies_isolation_forest(df, columns, contamination=0.05):
    """Detect anomalies using Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_iforest'] = model.fit_predict(df[columns])
    print(f"✅ Isolation Forest applied (contamination={contamination})")
    return df


def detect_anomalies_dbscan(df, columns, eps=0.5, min_samples=5):
    """Detect anomalies using DBSCAN."""
    X = df[columns].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    df['anomaly_dbscan'] = db.labels_
    print(f"✅ DBSCAN applied (eps={eps}, min_samples={min_samples})")
    return df


def detect_anomalies_oneclass_svm(df, columns, nu=0.05, kernel='rbf'):
    """Detect anomalies using One-Class SVM."""
    model = OneClassSVM(kernel=kernel, nu=nu, gamma='auto')
    df['anomaly_ocsvm'] = model.fit_predict(df[columns])
    print(f"✅ One-Class SVM applied (kernel={kernel}, nu={nu})")
    return df


# ======================================================
# 3️⃣ VISUALIZATION
# ======================================================

def plot_outliers(df, column, title="Outlier Visualization"):
    """Plot boxplot and histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(df[column])
    axes[0].set_title(f"{title} - Boxplot")
    axes[1].hist(df[column], bins=30, color='gray')
    axes[1].set_title(f"{title} - Histogram")
    plt.tight_layout()
    plt.show()


# ======================================================
# 4️⃣ AUTO COLUMN DETECTION
# ======================================================

def detect_column_types(df, target):
    """Automatically detect numeric and categorical columns."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("🔍 Numeric columns:", num_cols)
    print("🔍 Categorical columns:", cat_cols)
    return num_cols, cat_cols


# ======================================================
# 5️⃣ PIPELINE CONSTRUCTION
# ======================================================

def build_pipeline(num_cols, cat_cols):
    """Build pipeline with ColumnTransformer and RandomForest."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    print("✅ ML pipeline created with ColumnTransformer + RandomForest")
    return pipeline


# ======================================================
# 6️⃣ TRAIN, EVALUATE, TUNE
# ======================================================

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate pipeline."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n✅ Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return pipeline


def tune_hyperparameters(pipeline, X_train, y_train, method='random'):
    """GridSearchCV or RandomizedSearchCV for RandomForest."""
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    param_dist = {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': randint(3, 30),
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    }

    if method == 'grid':
        print("🔍 Running GridSearchCV...")
        search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    else:
        print("🎲 Running RandomizedSearchCV...")
        search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                    n_iter=10, cv=5, n_jobs=-1, random_state=42,
                                    scoring='accuracy', verbose=1)

    search.fit(X_train, y_train)
    print("🏆 Best Parameters:", search.best_params_)
    print("✅ Best Score:", search.best_score_)
    return search.best_estimator_


# ======================================================
# 7️⃣ SAVE / LOAD
# ======================================================

def save_pipeline(pipeline, filename="model_pipeline.joblib"):
    """Save pipeline."""
    joblib.dump(pipeline, filename)
    print(f"💾 Pipeline saved as {filename}")


def load_pipeline(filename="model_pipeline.joblib"):
    """Load saved pipeline."""
    print(f"📂 Loading pipeline from {filename}...")
    return joblib.load(filename)


# ======================================================
# 8️⃣ DEMO RUN
# ======================================================

if __name__ == "__main__":
    # Example dataset
    df = pd.DataFrame({
        'age': [25, 32, 47, 51, 62, 23, 44, 36, 52, 46],
        'income': [50000, 60000, 80000, 72000, 90000, 40000, 82000, 61000, 95000, 76000],
        'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F'],
        'city': ['New York', 'Paris', 'Paris', 'London', 'London', 'Tokyo', 'Tokyo', 'Paris', 'London', 'New York'],
        'purchased': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    })

    target = 'purchased'
    X, y = df.drop(columns=[target]), df[target]

    # Clean data
    df = clean_data_all(df, ['age', 'income'])
    df = detect_anomalies_isolation_forest(df, ['age', 'income'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Detect columns
    num_cols, cat_cols = detect_column_types(df, target)

    # Build pipeline
    pipeline = build_pipeline(num_cols, cat_cols)

    # Train & evaluate
    pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    # Tune
    best_pipeline = tune_hyperparameters(pipeline, X_train, y_train, method='random')

    # Save + reload
    save_pipeline(best_pipeline)
    loaded_pipeline = load_pipeline()

    # Predict new data
    new_data = pd.DataFrame({
        'age': [29, 55],
        'income': [55000, 88000],
        'gender': ['F', 'M'],
        'city': ['Paris', 'London']
    })

    preds = loaded_pipeline.predict(new_data)
    print("\n🔮 Predictions for new input:")
    print(preds)