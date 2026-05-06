import numpy as np


from src.modelo.transformer.modulos.ferramentas import relu, normalizador_camada

class CamadaFeedFoward:
    def __init__(self, dim_model:int, dim_ff=None, teste=True):
        '''
        Modelo da rede neural feed foward simplificada para o tensor.
        Params:
            dim_model: tamanho da dimensão do modelo
            dim_ff: é a dimensão do feed foward normalmente 4*dim_model
        '''
        #definindo uma seed fixa para o numpy gerar sempre o mesmo valor e tornar teste repetível
        if teste:
            np.random.seed(42)
        # dimensões do modelo usado  
        self.dim_model = dim_model       
        # Fator gamma do layernorm, cada dimensão tem seu gamma e layernorm
        self.gamma = np.array([1]*dim_model, dtype=np.float16) 
        # Fator beta do layernorm, cada dimensão tem seu beta e layernorm
        self.beta = np.array([0]*dim_model, dtype=np.float16) 
        
        # Valor segundo o paper "Attention Is All You Need"
        # dim_ff é 4*dim_model
        if dim_ff:
            self.dim_ff = dim_ff
        else:
            self.dim_ff = 4 * dim_model
        
        #gerando os pesos e bias do feed foward camada 1
        self.W1 = (np.random.randn(dim_model, self.dim_ff) * 0.1).astype(np.float16)
        self.b1 = (np.zeros(self.dim_ff)).astype(np.float16)
        
        # gerando os pesos e bias do feed foward camada 2
        # Perceba que no segundo peso é invertido dim_ff e dim_model
        # O que irá gerar uma matriz quadrada dim_model x dim_model
        self.W2 = (np.random.randn(self.dim_ff, dim_model) * 0.1).astype(np.float16)
        self.b2 = (np.zeros(dim_model)).astype(np.float16)

    def forward(self, saida_camada_atencao:np.array)->np.array:
        '''
        Método que aplica uma rede neural feed foward a saída da atenção.
        Permite inserir aleatoriedade a projeção linear original
            saida_camada_atencao = valor que saiu da atencao (batch, seq_len, dim_model)
        '''
        # Aplicando o primeiro feed foward
        self.ff1 = saida_camada_atencao @ self.W1 + self.b1

        # Realizando a ativação com ReLU por ser mais simples
        self.ff1 = relu(self.ff1)

        # Aplicando o segundo feed foward
        self.ff2 = (self.ff1 @ self.W2) + self.b2
        self.ff2 = relu(self.ff2)
        return self.ff2


    def add_norm_layer(self, ff:np.array, saida_atencao:np.array):        
        '''
        Camada que aplica o add e norm layer no transformer
        '''
        ff_soma = ff + saida_atencao
        x, x_norm, media, dp, x_scoreZ, gamma = normalizador_camada(ff_soma, 
                                                                    gamma=self.gamma, 
                                                                    beta=self.beta)
        return x_norm