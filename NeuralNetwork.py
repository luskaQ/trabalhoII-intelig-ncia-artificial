import pandas as pd
import numpy as np


class NeuralNetwork:
    def __init__(self) -> None:
        pass
    
    def reLU(self, x):
        return max(0, x)
    
    def reLU_derivada(self, x):
        if x > 0:
            return 1.0
        return (0.0)
    
    