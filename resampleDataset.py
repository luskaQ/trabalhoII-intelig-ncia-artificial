from sklearn.utils import resample
import pandas as pd

df = pd.read_csv(
    "dataset/enhanced_student_habits_performance_dataset.csv"
)

df_majoritaria = df[df["dropout_risk"] == "No"]
df_minoritaria = df[df["dropout_risk"] == "Yes"]

n_amostras = len(df_minoritaria)
df_majoritaria_downsampled = resample(df_majoritaria, replace=False, n_samples=n_amostras, random_state=42)

df_balanceado = pd.concat([
    df_minoritaria,
    df_majoritaria_downsampled # type: ignore
]) # type: ignore

df_balanceado.to_csv(
    "dataset/balanced_dataset.csv",
    index=False
)

ids_utilizados = set(df_balanceado["student_id"])

df_nao_usado = df[
    ~df["student_id"].isin(ids_utilizados)
]

df_nao_usado.to_csv(
    "dataset/nao_usado_no_resample.csv",
    index=False
)

print("Original:")
print(df["dropout_risk"].value_counts())

print("\nBalanceado:")
print(df_balanceado["dropout_risk"].value_counts())

print("\nNão utilizado:")
print(df_nao_usado["dropout_risk"].value_counts())