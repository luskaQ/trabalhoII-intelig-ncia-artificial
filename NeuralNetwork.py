import pandas as pd
import numpy as np


class NeuralNetwork:
    def __init__(self, num_epocas, taxa_treino, X, Y,num_camadas_ocultas = 1, num_neuronios_camada_oculta = 16) -> None:
        self.num_camadas_ocultas = num_camadas_ocultas
        self.num_neuronios_camada_oculta = num_neuronios_camada_oculta
        self.num_epocas = num_epocas
        self.taxa_treino = taxa_treino
        self.X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) #estabilizar dados entre 0 e 1
        self.Y = Y.reshape(-1, 1) # faz virar uma matriz coluna (m, 1) ao invez de um vetor 1D
        self.cria_camadas()
        pass
    
    def reLU(self, x):
        return np.maximum(0, x)
    
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
        self.W[1] = np.random.randn(self.quantidade_features, self.num_neuronios_camada_oculta) * 0.01
        self.b[1] = np.zeros((1, self.num_neuronios_camada_oculta))
        for i in range(2, self.num_camadas_ocultas + 1):
            self.W[i] = np.random.randn(self.num_neuronios_camada_oculta, self.num_neuronios_camada_oculta) * 0.01
            self.b[i] = np.zeros((1, self.num_neuronios_camada_oculta))
        self.W_saida = np.random.randn(self.num_neuronios_camada_oculta, 1) * 0.01
        self.b_saida = np.zeros((1, 1))
            
    def treinar(self):
        for epoca in range(self.num_epocas):
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
            dZ_saida = A_saida - self.Y
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
            self.W_saida -= self.taxa_treino * dW_saida
            self.b_saida -= self.taxa_treino * db_saida
            
            for i in range(1, self.num_camadas_ocultas + 1):
                self.W[i] -= self.taxa_treino * dW[i]
                self.b[i] -= self.taxa_treino * db[i]
            if epoca % 100 == 0:
                loss = -np.mean(self.Y * np.log(A_saida + 1e-8) + (1 - self.Y) * np.log(1 - A_saida + 1e-8)) #cross entropy  binaria
                
                previsoes = (A_saida > 0.5).astype(int)
                acuracia = np.mean(previsoes == self.Y)
                print(f"Época {epoca:4d} | Perda (Loss): {loss:.4f} | Acurácia: {acuracia * 100:.2f}%")
                
                
import DataSet

X, Y = DataSet.gerar_dataset_formatado()
                
rede = NeuralNetwork(
    num_epocas=1000, 
    taxa_treino=0.1, 
    X=X, 
    Y=Y, 
    num_camadas_ocultas=2, 
    num_neuronios_camada_oculta=32
)

rede.treinar()