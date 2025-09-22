# eda_telco.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

# ===============================
# 1. Load Data
# ===============================
DEFAULT_DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DEFAULT_DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ===============================
# 2. Plots Folder
# ===============================
PLOTS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_PATH), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# 3. Helper to save Matplotlib plots
# ===============================
def save_plot(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# ===============================
# 4. Data Tables as PNGs
# ===============================

def save_table(df_summary, title, filename):
    """Render dataframe summary as a table and save as PNG"""
    fig, ax = plt.subplots(figsize=(12, df_summary.shape[0] * 0.4 + 1))
    ax.axis('off')
    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        rowLabels=df_summary.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    save_plot(fig, filename)

def save_summary_tables(df):
    numeric_summary = df.describe().T
    categorical_summary = df.describe(include=["object"]).T
    save_table(numeric_summary, "Numerical Summary", "numeric_summary.png")
    save_table(categorical_summary, "Categorical Summary", "categorical_summary.png")
    print("Saved summary tables as PNGs.")

# ===============================
# 5. EDA Charts (all saved as PNG)
# ===============================

def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", ax=ax)
    ax.set_title("Churn Distribution")
    save_plot(fig, "churn_distribution.png")

def plot_gender_churn(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="gender", hue="Churn", palette="Set1", ax=ax)
    ax.set_title("Gender vs Churn")
    save_plot(fig, "gender_vs_churn.png")

def plot_tenure_hist(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df["tenure"], bins=30, kde=True, color="steelblue", ax=ax)
    ax.set_title("Customer Tenure Distribution")
    save_plot(fig, "tenure_distribution.png")

def plot_monthlycharges(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, ax=ax)
    ax.set_title("Monthly Charges Distribution by Churn")
    save_plot(fig, "monthlycharges_by_churn.png")

def plot_contract_churn(df):
    fig, ax = plt.subplots(figsize=(7,4))
    sns.countplot(data=df, x="Contract", hue="Churn", palette="Set3", ax=ax)
    ax.set_title("Contract Type vs Churn")
    save_plot(fig, "contract_vs_churn.png")

def plot_paymentmethod(df):
    fig, ax = plt.subplots(figsize=(9,4))
    sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette="muted", ax=ax)
    ax.set_title("Payment Method vs Churn")
    save_plot(fig, "paymentmethod_vs_churn.png")

def plot_internetservice(df):
    fig, ax = plt.subplots(figsize=(7,4))
    sns.countplot(data=df, x="InternetService", hue="Churn", palette="coolwarm", ax=ax)
    ax.set_title("Internet Service vs Churn")
    save_plot(fig, "internetservice_vs_churn.png")

def plot_tenure_charges(df):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="Churn", alpha=0.7, ax=ax)
    ax.set_title("Tenure vs Monthly Charges")
    save_plot(fig, "tenure_vs_monthlycharges.png")

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")

# ===============================
# 6. Run All
# ===============================
if __name__ == "__main__":
    # Save summary tables as PNG
    save_summary_tables(df)

    # Save all plots as PNG
    plot_churn_distribution(df)
    plot_gender_churn(df)
    plot_tenure_hist(df)
    plot_monthlycharges(df)
    plot_contract_churn(df)
    plot_paymentmethod(df)
    plot_internetservice(df)
    plot_tenure_charges(df)
    plot_heatmap(df)

    print(f"EDA complete. All plots & tables saved to {PLOTS_DIR}")