import pandas as pd
import numpy as np

def gerar_dataset_formatado():
    df = pd.read_csv("dataset/enhanced_student_habits_performance_dataset.csv")
    alvo = df['dropout_risk']
    features = df.drop(columns=['dropout_risk'])
    X_df = pd.get_dummies(features, dtype=int)
    X = X_df.values 


    Y_numerico, categorias_y = pd.factorize(alvo)
    Y = Y_numerico

    print(X_df.head())
    
    print(Y_numerico)
  
    return



gerar_dataset_formatado()