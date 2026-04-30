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
        self.dim_K = dim_k if dim_k != 0 else dim_model // heads
        self.dim_V = dim_v if dim_v != 0 else dim_model // heads

        if teste: #definindo uma seed fixa para o numpy gerar sempre o mesmo valor no random
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
    
if __name__ == "__main__":
    cabecas = 4
    dim_model = 64

    #criando o tensor
    tensor = Tensor(dim_model, cabecas)

    tensor_shape = tensor.W_Q.shape
    print("======================================\n\tCódigo do Tensor_1")
    print("======================================")
    print(f'Formato do tensor: {tensor.W_Q.shape}:')
    print(f'\t- {tensor.W_Q.shape[0]} Cabeças')
    print(f'\t- {tensor.W_Q.shape[1]} dimensões de valores')
    print(f'\t- {tensor.W_Q.shape[2]} dimensões de cabeça')
    