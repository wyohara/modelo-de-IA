import numpy as np

from src.modelo.tensor.rope.tensor import Tensor
from src.modelo.tensor.rope.tensor_foward import TensorFoward
from src.modelo.tensor.rope.tensor_backward import TensorBackward
from src.modelo.tensor.rope.ferramentas import Ferramentas

class Embeding():
    def __init__(self):
        super().__init__()
        
        #valores iniciais
        self.__tensor = Tensor(num_head=4)

        # Listas para armazenar variáveis intermediárias de cada cabeça
        self.head_out_list = []

    def foward(self, valor_entrada:np):
        return TensorFoward().foward(tensor=self.__tensor, valor_entrada=valor_entrada)
    
    def backward(self, valor_entrada:np, valor_saida:np):
        return TensorBackward().backward(self.__tensor, valor_entrada, valor_saida)

    def gerar_loss(self, valor_saida, x2_norm):
        loss = np.mean((x2_norm - valor_saida) ** 2)
        print(f"Loss inicial (num_heads={self.__tensor.num_heads}): {loss:.6f}")


def embeding_teste():
    X = np.array([[1., 0., 1.],  [0., 1., 0.],  [1., 1., 1.]])
    Y =  np.array([
            [0.5, 0.2, 0.3],
            [0.1, 0.9, 0.0],
            [0.8, 0.7, 0.6]
        ])

    embeding = Embeding()
    saida_inicial = embeding.foward(X)
    embeding.gerar_loss(saida_inicial, Y)
    for i in range(100):
        saida_final = embeding.backward(X, Y)
        embeding.gerar_loss(saida_final, Y)