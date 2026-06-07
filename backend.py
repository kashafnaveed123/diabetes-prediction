# ============================================================
# backend.py  —  THE BRAIN OF THE DIABETES PREDICTION PROJECT 🧠
# ============================================================
#
# 📌 HOW TO RUN THIS FILE:
#     python backend.py
#
# 📌 WHAT THIS FILE DOES (like Colab cells, but in functions):
#
#   SECTION 1 → Import libraries
#   SECTION 2 → Download dataset from Kaggle
#   SECTION 3 → Load CSV into a DataFrame
#   SECTION 4 → Explore the data  (like df.head(), df.info())
#   SECTION 5 → Clean & preprocess the data
#   SECTION 6 → Split into train & test sets
#   SECTION 7 → Train the Random Forest model
#   SECTION 8 → Evaluate (accuracy, confusion matrix, report)
#   SECTION 9 → Plot functions (charts for the frontend)
#   SECTION 10→ Self-test  (runs when you do: python backend.py)
#
# ============================================================


# ============================================================
# SECTION 1 — Import Libraries
# ============================================================
# Same imports you used in your Colab notebooks!
# ============================================================

import os                          # For working with files & folders
import glob                        # For searching files with a pattern
import json
from pathlib import Path

import joblib
import kagglehub                   # To download datasets from Kaggle
from dotenv import load_dotenv

import numpy  as np                # Numbers & arrays
import pandas as pd                # Data tables (DataFrames)

# Matplotlib — draw charts
import matplotlib
matplotlib.use("Agg")              # Needed for Streamlit (no pop-up windows)
import matplotlib.pyplot as plt

import seaborn as sns              # Prettier charts built on top of matplotlib

# Scikit-learn — all the ML tools
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree            import plot_tree

import warnings
warnings.filterwarnings("ignore")  # Hide noisy warnings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"


# ============================================================
# SECTION 2 — Download the Dataset from Kaggle
# ============================================================
#
# In Colab you pasted:
#   import kagglehub
#   path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
#
# We do the SAME THING here, inside a function.
# kagglehub automatically reads your ~/.kaggle/kaggle.json key.
# The file is downloaded ONCE and cached — next run is instant!
#
# ============================================================

def download_dataset() -> str:
    """
    Download the diabetes dataset from Kaggle using kagglehub.

    Returns:
        csv_path (str) — full path to the diabetes_prediction_dataset.csv file
    """

    ensure_kaggle_credentials()

    print(" Downloading dataset from Kaggle...")
    print("   (First time takes ~5 seconds · After that it uses a local cache)")

    # This is EXACTLY the code from the Kaggle dataset page!
    folder_path = kagglehub.dataset_download(
        "iammustafatz/diabetes-prediction-dataset"
    )

    print(f"✅ Dataset folder: {folder_path}")

    # Inside the folder there is a CSV file — let's find it automatically
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"❌ No CSV found in {folder_path}. "
            "Try deleting the cache folder and re-running."
        )

    csv_path = csv_files[0]        # Use the first CSV found
    print(f"📄 CSV file: {csv_path}")
    return csv_path


def ensure_kaggle_credentials() -> None:
    """
    Make Kaggle credentials available for kagglehub.

    Reads credentials from the project's .env file and also writes a
    ~/.kaggle/kaggle.json file if it does not already exist.
    """

    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")

    if not username or not api_key:
        raise EnvironmentError(
            
            "Missing Kaggle credentials. Add KAGGLE_USERNAME and KAGGLE_KEY to .env."
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    if not kaggle_json_path.exists():
        kaggle_json_path.write_text(
            json.dumps({"username": username, "key": api_key}),
            encoding="utf-8",
        )
        os.chmod(kaggle_json_path, 0o600)


# ============================================================
# SECTION 3 — Load the CSV into a Pandas DataFrame
# ============================================================
#
# In Colab:
#   df = pd.read_csv("diabetes_prediction_dataset.csv")
#   df.head()
#
# ============================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Read the CSV file into a pandas DataFrame.

    Parameters:
        csv_path (str) — path returned by download_dataset()

    Returns:
        df (DataFrame) — the raw, unmodified dataset
    """

    df = pd.read_csv(csv_path)
    print(f"\n📊 Data loaded!  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ============================================================
# SECTION 4 — Explore the Data
# ============================================================
#
# In Colab:
#   print(df.dtypes)
#   print(df.describe())
#   print(df.isnull().sum())
#   print(df['diabetes'].value_counts())
#
# ============================================================

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the dataset for display in the frontend.

    Returns a dict with:
        shape           — (rows, columns) tuple
        dtypes          — column names and their data types
        null_counts     — missing values per column
        describe        — statistical summary (mean, std, min, max …)
        class_counts    — how many 0s and 1s in 'diabetes' column
        class_balance   — % diabetic vs non-diabetic
    """

    diabetic     = int(df["diabetes"].sum())
    non_diabetic = int(len(df) - diabetic)
    total        = len(df)

    summary = {
        "shape"        : df.shape,
        "dtypes"       : df.dtypes.to_dict(),
        "null_counts"  : df.isnull().sum().to_dict(),
        "describe"     : df.describe().round(2),
        "class_counts" : {"Diabetic": diabetic, "Non-Diabetic": non_diabetic},
        "class_balance": {
            "Diabetic %"    : round(diabetic     / total * 100, 1),
            "Non-Diabetic %": round(non_diabetic / total * 100, 1),
        },
        "total_rows"   : total,
        "total_cols"   : df.shape[1],
    }
    return summary


# ============================================================
# SECTION 5 — Clean & Preprocess the Data
# ============================================================
#
# Raw data has text columns (gender, smoking_history).
# Machine Learning models need NUMBERS — so we convert text → numbers.
#
# In Colab this is called "Encoding Categorical Variables":
#   from sklearn.preprocessing import LabelEncoder
#   le = LabelEncoder()
#   df['gender'] = le.fit_transform(df['gender'])
#
# We also remove the rare "Other" gender value (only a few rows).
#
# ============================================================

# These are the column names the model will use as input features.
# We store them here so both backend and frontend can access them.
FEATURE_COLUMNS = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "hypertension",
    "heart_disease",
    "gender_encoded",
    "smoking_encoded",
]

TARGET_COLUMN = "diabetes"      # What we are predicting (0 = no, 1 = yes)

# Human-readable labels for the gender and smoking columns
GENDER_MAPPING  = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAPPING = {
    "never"         : 0,
    "No Info"       : 1,
    "current"       : 2,
    "former"        : 3,
    "ever"          : 4,
    "not current"   : 5,
}


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the raw DataFrame for machine learning.

    Steps:
        1. Drop duplicate rows
        2. Remove rows where gender = "Other" (very rare, causes noise)
        3. Encode 'gender' as a number   (Female=0, Male=1)
        4. Encode 'smoking_history' as a number
        5. Select only the columns the model needs

    Returns:
        df_clean (DataFrame) — ready for train/test split
    """

    df_clean = df.copy()            # Never modify the original!

    # ── Step 1: Drop exact duplicate rows ────────────────────
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped  = before - len(df_clean)
    if dropped > 0:
        print(f"   🗑️  Removed {dropped} duplicate rows")

    # ── Step 2: Remove "Other" gender rows ───────────────────
    df_clean = df_clean[df_clean["gender"] != "Other"]

    # ── Step 3: Encode gender (text → number) ────────────────
    df_clean["gender_encoded"] = df_clean["gender"].map(GENDER_MAPPING)

    # ── Step 4: Encode smoking history (text → number) ───────
    df_clean["smoking_encoded"] = df_clean["smoking_history"].map(SMOKING_MAPPING)

    # ── Step 5: Drop rows where encoding produced NaN ─────────
    df_clean = df_clean.dropna(
        subset=["gender_encoded", "smoking_encoded"]
    )

    print(f"   ✅ After cleaning: {len(df_clean)} rows remain")

    return df_clean


# ============================================================
# SECTION 6 — Split into Train & Test Sets
# ============================================================
#
# In Colab:
#   X = df[feature_columns]
#   y = df['diabetes']
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# ============================================================

def split_data(df_clean: pd.DataFrame, test_size_pct: int, random_state: int):
    """
    Split the cleaned data into training and test sets.

    Parameters:
        df_clean      — preprocessed DataFrame (from preprocess_data)
        test_size_pct — e.g. 20 means 20% for testing
        random_state  — integer seed (same seed = same split every time)

    Returns:
        X_train, X_test  — input features (the questions)
        y_train, y_test  — target labels  (the answers)
    """

    X = df_clean[FEATURE_COLUMNS]   # Input columns  (what the model sees)
    y = df_clean[TARGET_COLUMN]     # Output column  (what it predicts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size_pct / 100,   # 20 → 0.20
        random_state = random_state,
        stratify     = y,     # Keep same 0/1 ratio in both splits
    )

    print(f"\n✂️  Split:  {len(X_train)} training rows  |  {len(X_test)} test rows")
    return X_train, X_test, y_train, y_test


# ============================================================
# SECTION 7 — Train the Random Forest Model
# ============================================================
#
# In Colab:
#   from sklearn.ensemble import RandomForestClassifier
#   model = RandomForestClassifier(n_estimators=100, max_depth=10)
#   model.fit(X_train, y_train)
#
# ============================================================

def train_model(
    X_train,
    y_train,
    n_estimators    : int = 100,
    max_depth       : int = 10,
    min_samples_split: int = 2,
    random_state    : int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.

    Parameters:
        X_train, y_train  — training split
        n_estimators      — number of trees to build
        max_depth         — max depth of each tree
        min_samples_split — min samples needed to split a node
        random_state      — seed for reproducibility

    Returns:
        model — trained RandomForestClassifier (ready to predict!)
    """

    print(f"\n🌲 Training Random Forest  "
          f"({n_estimators} trees, max depth {max_depth})...")

    model = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        random_state      = random_state,
        n_jobs            = -1,     # Use all CPU cores → faster!
    )

    model.fit(X_train, y_train)    # ← LEARNING happens here!

    print("✅ Training complete!")
    return model


def save_model(model, model_path: Path = MODEL_PATH) -> Path:
    """Save a trained model so the app can reuse it later."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    return model_path


def load_saved_model(model_path: Path = MODEL_PATH):
    """Load an already-trained model from disk."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run training first."
        )

    return joblib.load(model_path)


def train_and_save_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    random_state: int = 42,
):
    """Run the full training pipeline once and save the model."""

    csv_path = download_dataset()
    df = load_data(csv_path)
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        test_size_pct=20,
        random_state=random_state,
    )

    model = train_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    save_model(model)
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    return model, results


# ============================================================
# SECTION 8 — Evaluate the Model
# ============================================================
#
# In Colab:
#   y_pred = model.predict(X_test)
#   print(accuracy_score(y_test, y_pred))
#   print(confusion_matrix(y_test, y_pred))
#   print(classification_report(y_test, y_pred))
#
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """
    Evaluate a trained model and collect all metrics.

    Returns a dict with everything the frontend needs to display results.
    """

    # ── Predictions ──────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_prob       = model.predict_proba(X_test)[:, 1]   # Probability of class 1

    # ── Accuracy ─────────────────────────────────────────────
    test_acc  = accuracy_score(y_test,  y_pred)       * 100
    train_acc = accuracy_score(y_train, y_pred_train) * 100
    gap       = train_acc - test_acc

    # ── ROC-AUC score (a better metric for imbalanced datasets) ─
    auc_score = roc_auc_score(y_test, y_prob) * 100

    # ── Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    # True Negatives, False Positives, False Negatives, True Positives
    tn, fp, fn, tp = cm.ravel()

    # ── Classification Report as DataFrame ───────────────────
    report_dict = classification_report(
        y_test, y_pred,
        target_names = ["No Diabetes (0)", "Diabetes (1)"],
        output_dict  = True,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    # ── ROC Curve data ────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # ── Sample predictions (first 20 rows) ───────────────────
    labels = ["✅ No Diabetes", "🚨 Diabetes"]
    sample_df = pd.DataFrame({
        "Actual"       : [labels[i] for i in list(y_test)[:20]],
        "Predicted"    : [labels[i] for i in list(y_pred)[:20]],
        "Confidence %" : [round(p * 100, 1) for p in list(y_prob)[:20]],
        "Correct?"     : [
            "✅ Yes" if a == p else "❌ No"
            for a, p in zip(list(y_test)[:20], list(y_pred)[:20])
        ],
    })

    return {
        # Accuracy numbers
        "test_accuracy"  : round(test_acc,  2),
        "train_accuracy" : round(train_acc, 2),
        "overfit_gap"    : round(gap,       2),
        "auc_score"      : round(auc_score, 2),

        # Confusion matrix breakdown
        "conf_matrix"    : cm,
        "true_negatives" : int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives" : int(tp),

        # Full report and sample table
        "report_df"      : report_df,
        "sample_df"      : sample_df,

        # ROC curve raw data
        "roc_fpr"        : fpr,
        "roc_tpr"        : tpr,
    }


# ============================================================
# SECTION 9 — Chart / Plot Functions
# ============================================================
# Each function draws ONE chart and returns a matplotlib Figure.
# The frontend (app.py) calls these and displays them with st.pyplot()
# ============================================================

# ── 9a: Class Distribution ──────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    """Bar chart — how many diabetic vs non-diabetic patients."""

    counts = df["diabetes"].value_counts().sort_index()
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{val:,}",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    plt.tight_layout()
    return fig


# ── 9b: Correlation Heatmap ──────────────────────────────────
def plot_correlation_heatmap(df_clean: pd.DataFrame):
    """Heatmap showing how strongly each feature correlates with diabetes."""

    num_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    corr     = df_clean[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot    = True,
        fmt      = ".2f",
        cmap     = "RdYlGn",
        center   = 0,
        ax       = ax,
        linewidths = 0.5,
        linecolor  = "white",
    )
    ax.set_title("Feature Correlation Heatmap\n"
                 "(Closer to 1 = strong positive link with diabetes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── 9c: Confusion Matrix ─────────────────────────────────────
def plot_confusion_matrix(cm):
    """Heatmap of the confusion matrix — right vs wrong predictions."""

    labels = ["No Diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = labels,
        yticklabels = labels,
        ax          = ax,
        linewidths  = 1,
        linecolor   = "white",
        annot_kws   = {"size": 16, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label",    fontsize=11)
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


# ── 9d: Feature Importance ───────────────────────────────────
def plot_feature_importance(model):
    """Horizontal bar chart of feature importances from the forest."""

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    feat_sorted = [FEATURE_COLUMNS[i] for i in indices]
    imp_sorted  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        feat_sorted[::-1],
        imp_sorted[::-1],
        color     = "#3498db",
        edgecolor = "white",
        linewidth = 0.8,
    )
    ax.set_title("Feature Importance — Which input matters most?",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for bar, val in zip(bars, imp_sorted[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    top_feature = feat_sorted[0]
    top_score   = round(float(imp_sorted[0]), 4)
    return fig, top_feature, top_score


# ── 9e: ROC Curve ────────────────────────────────────────────
def plot_roc_curve(fpr, tpr, auc_score: float):
    """
    ROC (Receiver Operating Characteristic) Curve.
    The closer to the top-left corner, the better!
    AUC = Area Under Curve  (100% = perfect, 50% = random guessing)
    """

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"Random Forest (AUC = {auc_score:.1f}%)")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5,
            linestyle="--", label="Random Guess (AUC = 50%)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Model Discrimination Ability",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    plt.tight_layout()
    return fig


# ── 9f: Age vs BMI coloured by Diabetes ──────────────────────
def plot_age_bmi_scatter(df_clean: pd.DataFrame, sample_n: int = 2000):
    """
    Scatter plot of Age vs BMI, coloured by whether the patient has diabetes.
    We sample to keep it fast.
    """

    df_sample = df_clean.sample(n=min(sample_n, len(df_clean)), random_state=42)
    colors     = df_sample["diabetes"].map({0: "#2ecc71", 1: "#e74c3c"})

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df_sample["age"], df_sample["bmi"],
        c=colors, alpha=0.5, edgecolors="none", s=18,
    )
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("BMI", fontsize=11)
    ax.set_title("Age vs BMI  (🟢 No Diabetes  |  🔴 Diabetes)",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    from matplotlib.patches import Patch
    legend = [Patch(color="#2ecc71", label="No Diabetes"),
              Patch(color="#e74c3c", label="Diabetes")]
    ax.legend(handles=legend)
    plt.tight_layout()
    return fig


# ── 9g: One Decision Tree ────────────────────────────────────
def plot_single_tree(model):
    """Show the first tree in the forest (depth capped at 3)."""

    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(
        model.estimators_[0],
        feature_names = FEATURE_COLUMNS,
        class_names   = ["No Diabetes", "Diabetes"],
        filled        = True,
        rounded       = True,
        max_depth     = 3,
        ax            = ax,
        fontsize      = 9,
    )
    ax.set_title("Decision Tree #1  (depth capped at 3 for readability)",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


# ============================================================
# SECTION 10 — Self-Test  (runs ONLY when you do: python backend.py)
# ============================================================
# This is like running ALL your Colab cells one by one.
# Great for testing that everything works before running the UI!
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("  🧪  backend.py  —  Full Pipeline Self-Test")
    print("=" * 60)

    # ── Step 1: Download ────────────────────────────────────
    print("\n[STEP 1]  Download dataset from Kaggle")
    csv_path = download_dataset()

    # ── Step 2: Load ────────────────────────────────────────
    print("\n[STEP 2]  Load CSV into DataFrame")
    df = load_data(csv_path)
    print(df.head())

    # ── Step 3: Summary ─────────────────────────────────────
    print("\n[STEP 3]  Data Summary")
    summary = get_data_summary(df)
    print(f"   Rows        : {summary['total_rows']:,}")
    print(f"   Columns     : {summary['total_cols']}")
    print(f"   Diabetic    : {summary['class_counts']['Diabetic']:,}  "
          f"({summary['class_balance']['Diabetic %']}%)")
    print(f"   Non-Diabetic: {summary['class_counts']['Non-Diabetic']:,}  "
          f"({summary['class_balance']['Non-Diabetic %']}%)")
    print(f"   Missing vals: {sum(summary['null_counts'].values())}")

    # ── Step 4: Preprocess ──────────────────────────────────
    print("\n[STEP 4]  Preprocess (encode categories, remove duplicates)")
    df_clean = preprocess_data(df)

    # ── Step 5: Split ───────────────────────────────────────
    print("\n[STEP 5]  Train/Test Split  (80% train, 20% test)")
    X_train, X_test, y_train, y_test = split_data(df_clean, 20, 42)

    # ── Step 6: Train ───────────────────────────────────────
    print("\n[STEP 6]  Train Random Forest")
    model = train_model(X_train, y_train, n_estimators=100, max_depth=10)
    save_model(model)

    # ── Step 7: Evaluate ────────────────────────────────────
    print("\n[STEP 7]  Evaluate")
    results = evaluate_model(model, X_train, X_test, y_train, y_test)

    print(f"\n   Test  Accuracy : {results['test_accuracy']}%")
    print(f"   Train Accuracy : {results['train_accuracy']}%")
    print(f"   Overfit Gap    : {results['overfit_gap']}%")
    print(f"   ROC-AUC Score  : {results['auc_score']}%")
    print(f"\n   Confusion Matrix:")
    print(f"   True Negatives  (Correctly said NO) : {results['true_negatives']:,}")
    print(f"   False Positives (Said YES, was NO)  : {results['false_positives']:,}")
    print(f"   False Negatives (Said NO,  was YES) : {results['false_negatives']:,}")
    print(f"   True Positives  (Correctly said YES): {results['true_positives']:,}")

    # ── Step 8: Feature Importance ──────────────────────────
    print("\n[STEP 8]  Feature Importance")
    _, top_feat, top_score = plot_feature_importance(model)
    print(f"   Most important feature: '{top_feat}'  (score = {top_score})")

    print()
    print("=" * 60)
    print("  ✅  ALL STEPS PASSED — backend.py works correctly!")
    print("  👉  Now run:  streamlit run app.py")
    print("=" * 60)
    print()
