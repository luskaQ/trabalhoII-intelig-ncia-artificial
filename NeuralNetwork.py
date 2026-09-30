import pandas as pd
import numpy as np
from sklearn.utils import resample


class NeuralNetwork:
    def __init__(self, num_epocas, taxa_treino, X, Y,num_camadas_ocultas = 1, num_neuronios_camada_oculta = 16, paciencia = 10) -> None:
        self.num_camadas_ocultas = num_camadas_ocultas
        self.num_neuronios_camada_oculta = num_neuronios_camada_oculta
        self.num_epocas = num_epocas
        self.taxa_treino = taxa_treino
        self.X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) #estabilizar dados entre 0 e 1
        self.Y = Y.reshape(-1, 1) # faz virar uma matriz coluna (m, 1) ao invez de um vetor 1D
        self.cria_camadas()
        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        self.paciencia = paciencia
        pass
    
    def reLU(self, x):
        return np.maximum(0, x)
    
    def normalizar(self, X):
        intervalo = self.x_max - self.x_min
        intervalo[intervalo == 0] = 1

        return (X - self.x_min) / intervalo
    
    def reLU_derivada(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        expX = np.exp(x - np.max(x, axis=1, keepdims=True))
        return expX / np.sum(expX, axis = 1, keepdims = True)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def cria_camadas(self):
        self.quantidade_dados = self.X.shape[0]
        self.quantidade_features = self.X.shape[1]
        self.quantidade_classes_saida = len(np.unique(self.Y))
        self.W = {} #cada posicao do dicionario guarda uma matriz de pesos
        self.b = {}
        np.random.seed(42)
        self.W[1] = np.random.randn(self.quantidade_features, self.num_neuronios_camada_oculta) * np.sqrt(2 / self.quantidade_features)
        self.b[1] = np.zeros((1, self.num_neuronios_camada_oculta))
        for i in range(2, self.num_camadas_ocultas + 1):
            self.W[i] = np.random.randn(self.num_neuronios_camada_oculta, self.num_neuronios_camada_oculta) * np.sqrt(2 / self.num_neuronios_camada_oculta)
            self.b[i] = np.zeros((1, self.num_neuronios_camada_oculta))
        self.W_saida = np.random.randn(self.num_neuronios_camada_oculta, 1) * 0.01
        self.b_saida = np.zeros((1, 1))
            
    def treinar(self):
        best_acc = 0
        early_stopping = 0
        for epoca in range(self.num_epocas):
            tx_mutavel = self.taxa_treino
            #feed forward
            Z = {}
            A = {0: self.X} # meu A[0] é a propria entrada
            
            for i in range(1, self.num_camadas_ocultas + 1):
                Z[i] = np.dot(A[i-1], self.W[i]) + self.b[i] #Resultado atual é o resultado da camda anterior vezes os pesoas atuais + bias
                A[i] = self.reLU(Z[i])
                
            Z_saida = np.dot(A[self.num_camadas_ocultas], self.W_saida) + self.b_saida
            A_saida = self.sigmoid(Z_saida)
            dZ_saida = A_saida - self.Y
            
            #backpropagation

            peso_classe_1 = 5.0 

            erro = A_saida - self.Y

            pesos = np.where(self.Y == 1, peso_classe_1, 1.0)

            dZ_saida = erro * pesos

                
            dW_saida = (1 / self.quantidade_dados) * np.dot(A[self.num_camadas_ocultas].T, dZ_saida)
            db_saida = (1 / self.quantidade_dados) * np.sum(dZ_saida, axis=0, keepdims=True)
            
            dW = {}
            db = {}
            
            dZ_prox = dZ_saida
            W_prox = self.W_saida
            
            for i in range(self.num_camadas_ocultas, 0, -1):
                dZ = np.dot(dZ_prox, W_prox.T) * self.reLU_derivada(Z[i])
                dW[i] = (1 / self.quantidade_dados) * np.dot(A[i-1].T, dZ)
                db[i] = (1 / self.quantidade_dados) * np.sum(dZ, axis=0, keepdims=True)
                
                dZ_prox = dZ
                W_prox = self.W[i]
            self.W_saida -= tx_mutavel * dW_saida
            self.b_saida -= tx_mutavel * db_saida
            
            for i in range(1, self.num_camadas_ocultas + 1):
                self.W[i] -= tx_mutavel * dW[i]
                self.b[i] -= tx_mutavel * db[i]
            if epoca % 100 == 0:
                loss = -np.mean(self.Y * np.log(A_saida + 1e-8) + (1 - self.Y) * np.log(1 - A_saida + 1e-8)) #cross entropy  binaria
                
                previsoes = (A_saida > 0.5).astype(int)
                acuracia = np.mean(previsoes == self.Y)
                if(acuracia > best_acc):
                    best_acc = acuracia
                    early_stopping = 0
                else:
                    early_stopping += 1
                print(f"Época {epoca:4d} | Perda (Loss): {loss:.4f} | Acurácia: {acuracia * 100:.2f}%")
            
            if(early_stopping == self.paciencia):
                print("Early stopping na epoca {epoca}")
                break
                
    def prever(self, X_previsao):
        Z = {}
        A = {0: X_previsao}
                    
        for i in range(1, self.num_camadas_ocultas + 1):
                        Z[i] = np.dot(A[i-1], self.W[i]) + self.b[i] #Resultado atual é o resultado da camda anterior vezes os pesoas atuais + bias
                        A[i] = self.reLU(Z[i])
                        
        Z_saida = np.dot(A[self.num_camadas_ocultas], self.W_saida) + self.b_saida
        A_saida = self.sigmoid(Z_saida)
        
        return A_saida
        
                
import DataSet

X, Y = DataSet.gerar_dataset_formatado("dataset/balanced_dataset.csv")
df = np.column_stack((X, Y))
df = pd.DataFrame(df)
df.rename(columns={df.columns[-1]: 'dropout_risk'}, inplace=True)

train_set = df.sample(frac=0.8, random_state=42)
test_set = df.drop(train_set.index)

X_train = np.array(train_set.drop(columns=['dropout_risk']))
Y_train = np.array(train_set['dropout_risk'])

X_test = np.array(test_set.drop(columns=['dropout_risk']))
Y_test = np.array(test_set['dropout_risk'])

dfNo = train_set[train_set['dropout_risk'] == 0]
dfYes = train_set[train_set['dropout_risk'] == 1]
print("classe no - train_set: ", len(dfNo))
print("classe yes - train_set: ", len(dfYes))

dfNo = test_set[test_set['dropout_risk'] == 0]
dfYes = test_set[test_set['dropout_risk'] == 1]
print("classe no - test_set ", len(dfNo))
print("classe yes - test_set: ", len(dfYes))




rede = NeuralNetwork(
    num_epocas=100000, 
    taxa_treino=0.01, 
    X=X_train, 
    Y=Y_train, 
    num_camadas_ocultas=3, 
    num_neuronios_camada_oculta=16,
    paciencia=10
)
X_test = rede.normalizar(X_test)

rede.treinar() # type: ignore

tp = fp = tn = fn = 0

for i in range(X_test.shape[0]):
    pred = rede.prever(X_test[i, :])
    if pred < 0.5 and Y_test[i] == 0:
        tn += 1
    elif(pred < 0.5 and Y_test[i] == 1):
        fn += 1
    elif(pred >= 0.5 and Y_test[i] == 1):
        tp += 1
    elif(pred >= 0.5 and Y_test[i] == 0):
        fp += 1
acuracia = (tn+tp)/(tp+tn+fp+fn+1e-7)
recall = tp / (tp+fn+1e-7)
precision = tp / (tp+fp+1e-7)
f1 = (2 * precision * recall) / (precision + recall +1e-7)
tnr = tn/(tn+fp+1e-7)
print(tp, fp, tn, fn)
print("accuracy: ", acuracia)
print("recall: ", recall)
print("precision: ", precision)
print("f1: ", f1)
print("tnr: ", tnr)
quantidade_acertos = tn + tp
print("total: ", tn+tp+fp+fn)
print("acertos: ", quantidade_acertos)


new_x, new_y = DataSet.gerar_dataset_formatado("dataset/nao_usado_no_resample.csv")
new_x = rede.normalizar(new_x)
print("classe no - new_y ", np.sum(new_y == 0))
print("classe yes - new_y: ", np.sum(new_y == 1))

tp = fp = tn = fn = 0

for i in range(new_x.shape[0]):

    pred = rede.prever(new_x[i, :])

    if pred < 0.5 and new_y[i] == 0:
        tn += 1

    elif pred < 0.5 and new_y[i] == 1:
        fn += 1

    elif pred >= 0.5 and new_y[i] == 1:
        tp += 1

    elif pred >= 0.5 and new_y[i] == 0:
        fp += 1

print(tp, fp, tn, fn)
acuracia = (tn + tp) / (tp + tn + fp + fn + 1e-7)

recall = tp / (tp + fn + 1e-7)

precision = tp / (tp + fp + 1e-7)

f1 = (2 * precision * recall) / (precision + recall + 1e-7)

tnr = tn / (tn + fp + 1e-7)


print("accuracy:", acuracia)
print("recall:", recall)
print("precision:", precision)
print("f1:", f1)
print("tnr:", tnr)