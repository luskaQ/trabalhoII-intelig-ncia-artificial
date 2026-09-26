import pandas as pd
import numpy as np

def gerar_dataset_formatado(dataset):
    df = pd.read_csv(dataset)

    df = df.drop("student_id", axis=1)

    alvo = df["dropout_risk"]
    features = df.drop(columns=["dropout_risk"])

    X_df = pd.get_dummies(features, dtype=int)
    X = X_df.to_numpy()

    Y = (alvo == "Yes").astype(int).to_numpy()

    print("classe no:", np.sum(Y == 0))
    print("classe yes:", np.sum(Y == 1))

    return X, Y



#gerar_dataset_formatado()