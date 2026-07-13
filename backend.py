# ============================================================
# backend.py  —  THE BRAIN OF THE DIABETES PREDICTION PROJECT 🧠
# ============================================================
#
# 📌 HOW TO RUN THIS FILE:
#     python backend.py
#
# 📌 WHAT THIS FILE DOES (just like Colab cells, but as functions):
#
#   STEP 1 → Import the libraries we need
#   STEP 2 → Download the dataset from Kaggle
#   STEP 3 → Load the CSV file into a table (DataFrame)
#   STEP 4 → Look at the data (summary, counts, etc.)
#   STEP 5 → Clean the data & turn text into numbers
#   STEP 6 → Split the data into "train" and "test" sets
#   STEP 7 → Train the neural network model
#   STEP 8 → Check how good the model is (accuracy, etc.)
#   STEP 9 → Draw charts to visualize everything
#   STEP 10 → Self-test: run everything when this file is executed
#
# Nothing fancy here — every function does ONE simple job,
# so it's easy to read from top to bottom. 🙂
# ============================================================


# ------------------------------------------------------------
# STEP 1 — Import the libraries we need
# ------------------------------------------------------------
import os                  # for working with files & folders
import glob                # for finding files that match a pattern
import json
from pathlib import Path

import joblib               # for saving/loading our trained model
import kagglehub            # for downloading the dataset from Kaggle
from dotenv import load_dotenv

import numpy as np          # numbers & arrays
import pandas as pd         # data tables (DataFrames)

import matplotlib
matplotlib.use("Agg")       # so charts don't try to pop up a window
import matplotlib.pyplot as plt
import seaborn as sns       # nicer-looking charts

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")   # hide noisy warning messages

load_dotenv()   # reads secret keys (like our Kaggle login) from a .env file

# Where we will save our trained model on disk
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"

# ---- Settings for our neural network (feel free to tweak these!) ----
HIDDEN_LAYERS = (64, 32)   # 1st hidden layer = 64 neurons, 2nd = 32 neurons
ALPHA = 0.0005             # L2 regularization (helps prevent overfitting)
LEARNING_RATE = 0.001      # how big each learning "step" is
MAX_ITER = 120             # number of training epochs (passes over the data)

# The columns (features) our model will look at to make a prediction
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

TARGET_COLUMN = "diabetes"   # what we are trying to predict (0 = no, 1 = yes)

# Mappings to turn text columns into numbers
GENDER_MAPPING = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAPPING = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5,
}


# ------------------------------------------------------------
# STEP 2 — Download the dataset from Kaggle
# ------------------------------------------------------------
def ensure_kaggle_credentials():
    """
    Make sure our Kaggle username & key are available.
    They should be stored in a .env file like this:

        KAGGLE_USERNAME=your_username
        KAGGLE_KEY=your_key
    """
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")

    if not username or not api_key:
        raise EnvironmentError(
            "Missing Kaggle credentials. Add KAGGLE_USERNAME and KAGGLE_KEY to your .env file."
        )

    # kagglehub needs these as environment variables
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key

    # kagglehub also likes to find a kaggle.json file — so we create one
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    if not kaggle_json_path.exists():
        kaggle_json_path.write_text(json.dumps({"username": username, "key": api_key}))
        os.chmod(kaggle_json_path, 0o600)


def download_dataset():
    """
    Download the diabetes dataset from Kaggle.
    Returns the full path to the CSV file.
    """
    ensure_kaggle_credentials()

    print("⬇️  Downloading dataset from Kaggle...")
    folder_path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
    print(f"✅ Dataset folder: {folder_path}")

    # Find the CSV file inside the downloaded folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {folder_path}")

    csv_path = csv_files[0]
    print(f"📄 CSV file: {csv_path}")
    return csv_path


# ------------------------------------------------------------
# STEP 3 — Load the CSV into a DataFrame
# ------------------------------------------------------------
def load_data(csv_path):
    """Read the CSV file into a pandas table (DataFrame)."""
    df = pd.read_csv(csv_path)
    print(f"📊 Data loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ------------------------------------------------------------
# STEP 4 — Explore the data
# ------------------------------------------------------------
def get_data_summary(df):
    """
    Build a simple dictionary of useful facts about the dataset,
    so the frontend (app.py) can display them easily.
    """
    diabetic = int(df["diabetes"].sum())
    non_diabetic = int(len(df) - diabetic)
    total = len(df)

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "describe": df.describe().round(2),
        "class_counts": {"Diabetic": diabetic, "Non-Diabetic": non_diabetic},
        "class_balance": {
            "Diabetic %": round(diabetic / total * 100, 1),
            "Non-Diabetic %": round(non_diabetic / total * 100, 1),
        },
        "total_rows": total,
        "total_cols": df.shape[1],
    }


# ------------------------------------------------------------
# STEP 5 — Clean & preprocess the data
# ------------------------------------------------------------
def preprocess_data(df):
    """
    Get the raw data ready for machine learning:
      1. Remove duplicate rows
      2. Drop the rare "Other" gender rows
      3. Turn gender & smoking history text into numbers
    """
    df_clean = df.copy()   # never touch the original data!

    # 1. Remove duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"🗑️  Removed {before - len(df_clean)} duplicate rows")

    # 2. Drop rare "Other" gender rows (too few to learn from)
    df_clean = df_clean[df_clean["gender"] != "Other"]

    # 3. Encode text columns as numbers
    df_clean["gender_encoded"] = df_clean["gender"].map(GENDER_MAPPING)
    df_clean["smoking_encoded"] = df_clean["smoking_history"].map(SMOKING_MAPPING)

    # Drop any rows where encoding failed (unexpected text values)
    df_clean = df_clean.dropna(subset=["gender_encoded", "smoking_encoded"])

    print(f"✅ After cleaning: {len(df_clean)} rows remain")
    return df_clean


# ------------------------------------------------------------
# STEP 6 — Split into train & test sets
# ------------------------------------------------------------
def split_data(df_clean, test_size_pct=20, random_state=42):
    """
    Split the data into a training set (to teach the model)
    and a test set (to check how well it learned).
    """
    X = df_clean[FEATURE_COLUMNS]   # the "questions" (inputs)
    y = df_clean[TARGET_COLUMN]     # the "answers" (what we predict)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size_pct / 100,
        random_state=random_state,
        stratify=y,   # keep the same 0/1 ratio in both sets
    )

    print(f"✂️  Split: {len(X_train)} training rows | {len(X_test)} test rows")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# STEP 7 — Train the neural network
# ------------------------------------------------------------
def train_model(X_train, y_train):
    """
    Train a simple feed-forward neural network to predict diabetes.
    We use a Pipeline so the data is automatically scaled before training.
    """
    print(f"🧠 Training neural network (layers={HIDDEN_LAYERS}, epochs={MAX_ITER})...")

    model = Pipeline([
        ("scaler", StandardScaler()),          # scales numbers so the network learns better
        ("classifier", MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=ALPHA,
            learning_rate_init=LEARNING_RATE,
            max_iter=MAX_ITER,
            early_stopping=True,     # stop early if it's no longer improving
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)   # ← this is where the actual learning happens!

    print("✅ Training complete!")
    return model


def save_model(model, model_path=MODEL_PATH):
    """Save the trained model to disk so we can reuse it later."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    return model_path


def load_saved_model(model_path=MODEL_PATH):
    """Load a model that was already trained and saved."""
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at {model_path}. Train one first!")
    return joblib.load(model_path)


def train_and_save_model():
    """Run the whole pipeline from start to finish, and save the result."""
    csv_path = download_dataset()
    df = load_data(csv_path)
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    model = train_model(X_train, y_train)
    save_model(model)

    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    return model, results


# ------------------------------------------------------------
# STEP 8 — Evaluate the model
# ------------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Test the model and collect all the numbers/tables the frontend
    needs in order to show how good the model is.
    """
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of "has diabetes"

    test_acc = accuracy_score(y_test, y_pred) * 100
    train_acc = accuracy_score(y_train, y_pred_train) * 100
    auc_score = roc_auc_score(y_test, y_prob) * 100

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_df = pd.DataFrame(
        classification_report(
            y_test, y_pred,
            target_names=["No Diabetes (0)", "Diabetes (1)"],
            output_dict=True,
        )
    ).transpose().round(3)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # A small table showing the first 20 predictions
    labels = ["✅ No Diabetes", "🚨 Diabetes"]
    sample_df = pd.DataFrame({
        "Actual": [labels[i] for i in list(y_test)[:20]],
        "Predicted": [labels[i] for i in list(y_pred)[:20]],
        "Confidence %": [round(p * 100, 1) for p in list(y_prob)[:20]],
        "Correct?": [
            "✅ Yes" if a == p else "❌ No"
            for a, p in zip(list(y_test)[:20], list(y_pred)[:20])
        ],
    })

    return {
        "test_accuracy": round(test_acc, 2),
        "train_accuracy": round(train_acc, 2),
        "overfit_gap": round(train_acc - test_acc, 2),
        "auc_score": round(auc_score, 2),
        "conf_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "report_df": report_df,
        "sample_df": sample_df,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
    }


# ------------------------------------------------------------
# STEP 9 — Charts (each function draws ONE chart)
# ------------------------------------------------------------
def plot_class_distribution(df):
    """Bar chart: how many diabetic vs non-diabetic patients are in the data."""
    counts = df["diabetes"].value_counts().sort_index()
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=colors)
    ax.set_title("Class Distribution", fontweight="bold")
    ax.set_ylabel("Number of Patients")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}",
                ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df_clean):
    """Heatmap: how strongly each feature relates to diabetes."""
    corr = df_clean[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    """Heatmap: shows correct vs incorrect predictions."""
    labels = ["No Diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    return fig


def plot_feature_importance(model):
    """
    Horizontal bar chart: which input features matter most to the model.
    For a neural network, we estimate this from the first layer's weights.
    """
    classifier = model.named_steps["classifier"]
    first_layer_weights = np.abs(classifier.coefs_[0])
    importances = np.linalg.norm(first_layer_weights, axis=1)
    importances = importances / importances.sum()   # turn into percentages

    order = np.argsort(importances)[::-1]
    feat_sorted = [FEATURE_COLUMNS[i] for i in order]
    imp_sorted = importances[order] 

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feat_sorted[::-1], imp_sorted[::-1], color="#3498db")
    ax.set_title("Feature Importance — Which input matters most?", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    top_feature = feat_sorted[0]
    top_score = round(float(imp_sorted[0]), 4)
    return fig, top_feature, top_score


def plot_roc_curve(fpr, tpr, auc_score):
    """
    ROC Curve: shows how well the model tells the two classes apart.
    Closer to the top-left corner = better. AUC 100% = perfect, 50% = random guessing.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"Neural Network (AUC = {auc_score:.1f}%)")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Guess (AUC = 50%)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_age_bmi_scatter(df_clean, sample_n=2000):
    """Scatter plot: Age vs BMI, colored by whether the patient has diabetes."""
    df_sample = df_clean.sample(n=min(sample_n, len(df_clean)), random_state=42)
    colors = df_sample["diabetes"].map({0: "#2ecc71", 1: "#e74c3c"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_sample["age"], df_sample["bmi"], c=colors, alpha=0.5, s=18)
    ax.set_xlabel("Age")
    ax.set_ylabel("BMI")
    ax.set_title("Age vs BMI  (🟢 No Diabetes | 🔴 Diabetes)", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_network_architecture(model):
    """A simple picture of the neural network's layers (input → hidden → output)."""
    classifier = model.named_steps["classifier"]
    hidden_layers = classifier.hidden_layer_sizes
    if isinstance(hidden_layers, int):
        hidden_layers = (hidden_layers,)

    layer_sizes = [len(FEATURE_COLUMNS), *hidden_layers, 1]
    layer_titles = ["Input", *[f"Hidden {i+1}" for i in range(len(hidden_layers))], "Output"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title("Feed-Forward Neural Network Architecture", fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, len(layer_sizes) + 1)
    ax.set_ylim(0, 10)

    y_positions = np.linspace(2, 8, 5)
    for layer_idx, (size, title) in enumerate(zip(layer_sizes, layer_titles), start=1):
        ax.text(layer_idx, 9.1, f"{title}\n{size} units", ha="center", fontweight="bold")
        node_count = min(size, len(y_positions))
        for y in np.linspace(2, 8, node_count):
            ax.add_patch(plt.Circle((layer_idx, y), 0.18, color="#3498db"))

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# STEP 10 — Self-test (runs only when you do: python backend.py)
# ------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  🧪  backend.py — Full Pipeline Self-Test")
    print("=" * 60)

    print("\n[1/7] Downloading & loading data...")
    csv_path = download_dataset()
    df = load_data(csv_path)

    print("\n[2/7] Data summary...")
    summary = get_data_summary(df)
    print(f"   Rows: {summary['total_rows']:,} | Columns: {summary['total_cols']}")
    print(f"   Diabetic: {summary['class_counts']['Diabetic']:,} "
          f"({summary['class_balance']['Diabetic %']}%)")

    print("\n[3/7] Cleaning & preprocessing...")
    df_clean = preprocess_data(df)

    print("\n[4/7] Splitting into train/test...")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("\n[5/7] Training the model...")
    model = train_model(X_train, y_train)
    save_model(model)

    print("\n[6/7] Evaluating the model...")
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"   Test Accuracy : {results['test_accuracy']}%")
    print(f"   Train Accuracy: {results['train_accuracy']}%")
    print(f"   ROC-AUC Score : {results['auc_score']}%")

    print("\n[7/7] Feature importance...")
    _, top_feature, top_score = plot_feature_importance(model)
    print(f"   Most important feature: '{top_feature}' (score = {top_score})")

    print("\n" + "=" * 60)
    print("  ✅ ALL STEPS PASSED — backend.py works correctly!")
    print("  👉 Now run: streamlit run app.py")
    print("=" * 60 + "\n")