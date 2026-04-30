import numpy as np


class Tensor:
    def __init__(self, dim_model:int, heads:int, dim_k:int=0, dim_v:int=0, teste=True):
        '''
        Classe principal que opera o tensor
        Params:
            dim_model: dimensão do modelo, ou quantos parâmetros de entrada ele aceita
            heads: numero de cabeças de atenção
            dim_k, dim_v: dimensões usadas na cabeça. Se 0 usa divisao inteira de dim_model // heads
        '''
        self.dim_model = dim_model
        self.heads = heads
        self.dim_K = dim_k if dim_k is not 0 else dim_model // heads
        self.dim_V = dim_v if dim_v is not 0 else dim_model // heads

        if teste:          
            np.random.seed(42)

        '''
        Definindo os pesos iniciais de Q, K, V e o peso final O após a concatenação obedecendo:
        MultiHead(Q,K,V ) = Concat(head1,...,headh)W^o
        '''
    
        self.W_K = np.random.randn(self.heads, self.dim_model, self.dim_K) * 0.1
        self.W_Q = np.random.randn(self.heads, self.dim_model, self.dim_K) * 0.1
        self.W_V = np.random.randn(self.heads, self.dim_model, self.dim_V) * 0.1

        # Projeção final após concatenação
        self.W_O = np.random.randn(self.heads * self.dim_V, self.dim_model)
    
