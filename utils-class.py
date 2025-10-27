"""
ml_super_utils_class.py
-----------------------
All-in-one class-based ML utilities and pipeline builder.

Usage (basic):
    ml = MLSuperUtils(df, target='purchased')
    ml.clean_data(columns=['age','income'])
    ml.detect_anomalies_iforest(columns=['age','income'])
    ml.build_pipeline()
    ml.train()
    ml.tune(method='random')  # or 'grid'
    ml.save_pipeline("model.joblib")
    preds = ml.predict(new_df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib


class MLSuperUtils:
    def __init__(self, df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize with dataframe and target column name.
        Keeps an internal copy of the dataframe to avoid accidental external mutation.
        """
        self.df = df.copy()
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
        self.target = target
        self.random_state = random_state
        self.test_size = test_size

        # Placeholders
        self.num_cols = None
        self.cat_cols = None
        self.pipeline = None
        self.best_pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # --------------------------
    # Column detection & helpers
    # --------------------------
    def detect_column_types(self):
        """Auto-detect numeric and categorical feature columns (excludes target)."""
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.target in num_cols:
            num_cols.remove(self.target)
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols, self.cat_cols = num_cols, cat_cols
        print("🔍 Numeric columns:", self.num_cols)
        print("🔍 Categorical columns:", self.cat_cols)
        return self.num_cols, self.cat_cols

    # --------------------------
    # Outlier removal
    # --------------------------
    def remove_outliers_iqr(self, columns: list):
        """Remove univariate outliers using IQR for provided numeric columns."""
        for col in columns:
            if col not in self.df.columns:
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        print(f"✅ Removed IQR outliers from: {columns}")
        return self.df

    def remove_outliers_zscore(self, columns: list, threshold: float = 3.0):
        """Remove univariate outliers using Z-score (|Z| < threshold)."""
        for col in columns:
            if col not in self.df.columns:
                continue
            z = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
            self.df = self.df[z < threshold]
        print(f"✅ Removed Z-score outliers (|Z| < {threshold}) from: {columns}")
        return self.df

    def clean_data(self, columns: list, use_iqr: bool = True, z_thresh: float = 3.0):
        """
        Combined cleaning pipeline.
        columns: list of numeric columns to clean (IQR/Z-score applied per column).
        """
        print("🔧 Starting cleaning pipeline...")
        if use_iqr:
            self.remove_outliers_iqr(columns)
        self.remove_outliers_zscore(columns, threshold=z_thresh)
        print("✅ Cleaning complete. Data shape:", self.df.shape)
        return self.df

    # --------------------------
    # Anomaly detection
    # --------------------------
    def detect_anomalies_iforest(self, columns: list, contamination: float = 0.05):
        """Add column 'anomaly_iforest' to df (1 normal, -1 anomaly)."""
        model = IsolationForest(contamination=contamination, random_state=self.random_state)
        self.df['anomaly_iforest'] = model.fit_predict(self.df[columns])
        print(f"✅ IsolationForest anomalies added (contamination={contamination})")
        return self.df

    def detect_anomalies_dbscan(self, columns: list, eps: float = 0.5, min_samples: int = 5):
        """Add column 'anomaly_dbscan' to df (-1 = anomaly)."""
        X = self.df[columns].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        self.df['anomaly_dbscan'] = db.labels_
        print(f"✅ DBSCAN applied (eps={eps}, min_samples={min_samples})")
        return self.df

    def detect_anomalies_ocsvm(self, columns: list, nu: float = 0.05, kernel: str = 'rbf'):
        """Add column 'anomaly_ocsvm' to df (1 normal, -1 anomaly)."""
        model = OneClassSVM(kernel=kernel, nu=nu, gamma='auto')
        self.df['anomaly_ocsvm'] = model.fit_predict(self.df[columns])
        print(f"✅ One-Class SVM applied (nu={nu}, kernel={kernel})")
        return self.df

    # --------------------------
    # Visualization
    # --------------------------
    def plot_outliers(self, column: str, figsize=(10, 4)):
        """Plot boxplot and histogram for one column."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found.")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].boxplot(self.df[column].dropna())
        axes[0].set_title(f"{column} - Boxplot")
        axes[1].hist(self.df[column].dropna(), bins=30)
        axes[1].set_title(f"{column} - Histogram")
        plt.tight_layout()
        plt.show()

    # --------------------------
    # Pipeline building
    # --------------------------
    def build_pipeline(self, model=None, numeric_transformer=None, categorical_transformer=None):
        """
        Build a ColumnTransformer pipeline and attach a model (default RandomForestClassifier).
        If detect_column_types hasn't been called, it is run automatically.
        """
        if self.num_cols is None or self.cat_cols is None:
            self.detect_column_types()

        if numeric_transformer is None:
            numeric_transformer = Pipeline([('scaler', StandardScaler())])
        if categorical_transformer is None:
            categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ], remainder='drop'
        )

        if model is None:
            model = RandomForestClassifier(random_state=self.random_state)

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        print("✅ Pipeline built (preprocessor + model).")
        return self.pipeline

    # --------------------------
    # Train / Evaluate / Split
    # --------------------------
    def prepare_split(self, stratify: bool = True):
        """Create train/test split from current self.df."""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        strat = y if stratify and y.nunique() > 1 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=strat
        )
        print("🔀 Train/test split prepared:", self.X_train.shape, self.X_test.shape)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """Train the pipeline on the prepared train set. Call prepare_split() first (or it will run automatically)."""
        if self.pipeline is None:
            self.build_pipeline()
        if self.X_train is None:
            self.prepare_split()
        self.pipeline.fit(self.X_train, self.y_train)
        print("✅ Pipeline trained.")
        return self.pipeline

    def evaluate(self, pipeline: Pipeline = None):
        """Evaluate pipeline on test set and print accuracy + classification report."""
        if pipeline is None:
            pipeline = self.pipeline
        if pipeline is None:
            raise ValueError("No pipeline available. Build and train a pipeline first.")
        if self.X_test is None:
            self.prepare_split()
        preds = pipeline.predict(self.X_test)
        print("✅ Evaluation results:")
        print("Accuracy:", accuracy_score(self.y_test, preds))
        print(classification_report(self.y_test, preds))
        return preds

    # --------------------------
    # Hyperparameter search
    # --------------------------
    def tune(self, method: str = 'random', n_iter: int = 10, cv: int = 5, scoring: str = 'accuracy'):
        """
        Tune hyperparameters for RandomForest inside the pipeline.
        method: 'grid' or 'random'
        Returns the best pipeline (fitted).
        """
        if self.pipeline is None:
            self.build_pipeline()

        # parameter grids target the step named 'model' in pipeline
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

        if self.X_train is None:
            self.prepare_split()

        if method == 'grid':
            print("🔍 Running GridSearchCV...")
            search = GridSearchCV(self.pipeline, param_grid, cv=cv, n_jobs=-1, scoring=scoring, verbose=1)
        else:
            print("🎲 Running RandomizedSearchCV...")
            search = RandomizedSearchCV(self.pipeline, param_distributions=param_dist,
                                        n_iter=n_iter, cv=cv, n_jobs=-1, random_state=self.random_state,
                                        scoring=scoring, verbose=1)
        search.fit(self.X_train, self.y_train)
        self.best_pipeline = search.best_estimator_
        print("🏆 Best params:", search.best_params_)
        print("✅ Best CV score:", search.best_score_)
        return self.best_pipeline

    # --------------------------
    # Save / Load / Predict
    # --------------------------
    def save_pipeline(self, filename: str = "model_pipeline.joblib"):
        """Save current best_pipeline (or pipeline) to disk via joblib."""
        pl = self.best_pipeline if self.best_pipeline is not None else self.pipeline
        if pl is None:
            raise ValueError("No pipeline available to save.")
        joblib.dump(pl, filename)
        print(f"💾 Pipeline saved to {filename}")

    def load_pipeline(self, filename: str = "model_pipeline.joblib"):
        """Load pipeline from disk and set as best_pipeline."""
        self.best_pipeline = joblib.load(filename)
        print(f"📂 Pipeline loaded from {filename}")
        return self.best_pipeline

    def predict(self, X_new: pd.DataFrame, pipeline: Pipeline = None):
        """Predict on new data using chosen pipeline (best_pipeline if available)."""
        pl = pipeline if pipeline is not None else (self.best_pipeline if self.best_pipeline is not None else self.pipeline)
        if pl is None:
            raise ValueError("No pipeline available. Build or load a pipeline first.")
        return pl.predict(X_new)

    # --------------------------
    # Utility: cross-validation quick check
    # --------------------------
    def cross_val_score(self, cv: int = 5):
        """Return cross-validation scores for the current pipeline on full dataset."""
        if self.pipeline is None:
            self.build_pipeline()
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        print(f"CV scores: {scores}\nMean: {scores.mean():.4f}")
        return scores


# --------------------------
# Demo usage
# --------------------------
if __name__ == "__main__":
    # tiny example data
    data = {
        'age': [25, 32, 47, 51, 62, 23, 44, 36, 52, 46],
        'income': [50000, 60000, 80000, 72000, 90000, 40000, 82000, 61000, 95000, 76000],
        'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F'],
        'city': ['New York', 'Paris', 'Paris', 'London', 'London', 'Tokyo', 'Tokyo', 'Paris', 'London', 'New York'],
        'purchased': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    ml = MLSuperUtils(df, target='purchased')
    ml.detect_column_types()                    # auto detect columns
    ml.clean_data(columns=['age', 'income'])    # IQR + Zscore cleaning
    ml.detect_anomalies_iforest(columns=['age', 'income'])
    ml.build_pipeline()                         # build pipeline with default RF
    ml.prepare_split()
    ml.train()
    ml.evaluate()
    best = ml.tune(method='random', n_iter=10)  # Randomized or method='grid'
    ml.evaluate(best)
    ml.save_pipeline("demo_pipeline.joblib")
    loaded = ml.load_pipeline("demo_pipeline.joblib")

    new_data = pd.DataFrame({
        'age': [29, 55],
        'income': [57000, 88000],
        'gender': ['F', 'M'],
        'city': ['Paris', 'London']
    })
    print("Predictions:", ml.predict(new_data))