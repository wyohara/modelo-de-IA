import numpy as np

from src.modelo.tensor.original.ferramentas import Ferramentas


class Tensor(Ferramentas):
    def __init__(self, tam_token=3, dim_model=3, dim_head=3, num_head=1, learning_rate=0.1):

        self.tam_token = tam_token
        #dimensão do modelo de entrada
        self.dim_model = dim_model
        # dimensão por cabeça (igual a d_model por simplicidade)
        if int(dim_model/num_head) ==0:
            dim_head =1
        else:
            dim_head= int(dim_model/num_head)
        self.dim_head = dim_head
        self.num_heads = num_head
        np.random.seed(42)

        # Inicialização dos pesos
        #representa a dimensão do que o token deseja
        self.W_Query = np.random.randn(self.num_heads, self.dim_model, self.dim_head) * 0.1
        #representa a dimensão de que valores o token possui
        self.W_Key = np.random.randn(self.num_heads, self.dim_model, self.dim_head) * 0.1
        #camada que mostra representa o valor que o token possui quando é alta atenção nele
        self.W_Value = np.random.randn(self.num_heads, self.dim_model, self.dim_head) * 0.1
        #camada de normalização de saída
        self.W_Output = np.random.randn(self.num_heads * self.dim_head, self.dim_model) * 0.1

        # Feed-forward
        self.W_ff1 = np.random.randn(self.dim_model, 10) * 0.1 #camada de pesos
        self.b_ff1 = np.random.randn(10) * 0.1 #camada de bias
        self.W_ff2 = np.random.randn(10, self.dim_model) * 0.1 #camada de
        self.b_ff2 = np.random.randn(self.dim_model) * 0.1

        self.learning_rate = learning_rate

    def zerar_pesos(self,tensor_base):
        # Inicializar gradientes com 0 na mesmas dimensões
        self.W_Query = np.zeros_like(tensor_base.W_Query)
        self.W_Key = np.zeros_like(tensor_base.W_Key)
        self.W_Value = np.zeros_like(tensor_base.W_Value)
        self.W_Output = np.zeros_like(tensor_base.W_Output)
        self.W_ff1 = np.zeros_like(tensor_base.W_ff1)
        self.b_ff1 = np.zeros_like(tensor_base.b_ff1)
        self.W_ff2 = np.zeros_like(tensor_base.W_ff2)
        self.b_ff2 = np.zeros_like(tensor_base.b_ff2)

        