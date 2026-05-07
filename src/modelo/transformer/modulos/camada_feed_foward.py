import numpy as np


from src.modelo.transformer.modulos.ferramentas import normalizador_camada, normalizador_camada_backward

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
        self.w1 = (np.random.randn(dim_model, self.dim_ff) * 0.1).astype(np.float16)
        self.b1 = (np.zeros(self.dim_ff)).astype(np.float16)
        
        # gerando os pesos e bias do feed foward camada 2
        # Perceba que no segundo peso é invertido dim_ff e dim_model
        # O que irá gerar uma matriz quadrada dim_model x dim_model
        self.w2 = (np.random.randn(self.dim_ff, dim_model) * 0.1).astype(np.float16)
        self.b2 = (np.zeros(dim_model)).astype(np.float16)

    def forward(self, saida_camada_atencao:np.array)->np.array:
        '''
        Método que aplica uma rede neural feed foward a saída da atenção.
        Permite inserir aleatoriedade a projeção linear original
            saida_camada_atencao = valor que saiu da atencao (batch, seq_len, dim_model)
        '''
        # Aplicando o primeiro feed foward
        self.ff1 = saida_camada_atencao @ self.w1 + self.b1

        # Realizando a ativação com ReLU por ser mais simples
        self.ff1 = self.__relu(self.ff1)

        # Aplicando o segundo feed foward
        self.ff2 = (self.ff1 @ self.w2) + self.b2
        self.ff2 = self.__relu(self.ff2)
        return self.ff2

    def backward(self, d_ff2:np.array):
        '''
        Método que realiza a retropropagação (backward) para corrigir os pesos.
        Params:
                ff2_out: resultado resultado corrigido da camada add_norm_backward 
                    hape (batch, seq_len, dim_model).
        Returns:
            dx : Gradiente em relação à entrada `saida_att` (propagado para a atenção).
                Shape (batch, seq_len, dim_model).
            dW1 : Gradiente para a matriz de pesos da primeira camada (W1).
                Shape (dim_model, dim_ff).
            db1 : Gradiente para o bias da primeira camada (b1). Shape (dim_ff,).
            dW2 : Gradiente para a matriz de pesos da segunda camada (W2).
                Shape (dim_ff, dim_model).
            db2 : Gradiente para o bias da segunda camada (b2). Shape (dim_model,).
            dgamma : Gradiente para o parâmetro gamma da LayerNorm. Shape (dim_model,).
            dbeta : Gradiente para o parâmetro beta da LayerNorm. Shape (dim_model,).
        '''
        ff_out = d_ff2.copy()
        d_x_residual = d_ff2.copy()

        # ----- Backward da segunda camada linear (W2, b2) -----
        # Recupera a soma sobre os eixos:
        #   b (batch) e s (seq_len)
        #   mantem i (dimensão da camada oculta) e j (dimensão de d_saida).
        # Resultado é uma matriz (dim_ff, dim_model).
        dW2 = np.einsum('b s i, b s j -> i j', self.ff1, ff_out)  
        db2 = np.sum(ff_out, axis=(0, 1))

        # Gradiente em relação à saída da ativação (hidden)
        dh = ff_out @ self.w2.T 

        # -----  Backward da ReLU -----
        # Derivada da ReLU: 1 onde ff1 > 0, senão 0
        d_relu = (self.ff1 > 0).astype(float)
        dh_ff1 = dh * d_relu

        # ----- Backward da primeira camada linear (W1, b1) -----
        dW1 = np.einsum('b s i, b s j -> i j', d_ff2, dh_ff1)
        db1 = np.sum(dh_ff1, axis=(0, 1))
        # Gradiente em relação à entrada (saida_atencao)
        dx_ff = dh_ff1 @ self.w1.T 
        self.ff_soma = d_x_residual+ dx_ff
        return self.ff_soma, dW1, db1, dW2, db2
    
    def camada_add_norm(self, ff:np.array, entrada_camada:np.array):        
        '''
        Camada que aplica o add e norm layer no transformer
        '''
        self.ff_soma = ff + entrada_camada
        x, x_norm, media, dp, x_scoreZ, gamma = normalizador_camada(self.ff_soma, 
                                                                    gamma=self.gamma, 
                                                                    beta=self.beta)
        return x_norm
    
    def camada_add_norm_backward(self, dy:np.array):
        '''
        Reversão da camada add_norm 
            Params:
                dy: rótulos para comparação após usar o gradiente de perda mse
            Returns:
                dff2: correção do feedfoward antes da normalização e add_layer
                dgamma: Correção do fator gammma do feedfoward
                dbeta: Correção do fator beta do feedfoward
        '''
        d_ff2, dgamma, dbeta = normalizador_camada_backward(self.ff_soma, dy, self.gamma)
        return d_ff2, dgamma, dbeta
    
    def corrigir_pesos(self, dW1:np.array, db1:np.array, dW2:np.array, db2:np.array, dgamma:np.array, dbeta:np.array, taxa_aprendizado=0.1):
        '''
        Método que corrige os pesos da camada do feedfoward
            dgamma : Gradiente da perda em relação ao parâmetro gamma.
                Deve ter o mesmo formato que self.gamma (dim_model,).
            dbeta : Gradiente da perda em relação ao parâmetro beta.
                Deve ter o mesmo formato que self.beta (dim_model,).
        '''
        self.w1 -= taxa_aprendizado * dW1
        self.w2 -= taxa_aprendizado * dW2
        self.b1 -= taxa_aprendizado * db1
        self.b2 -= taxa_aprendizado * db2
        
        self.gamma = self.gamma - taxa_aprendizado * dgamma
        self.beta  = self.beta  - taxa_aprendizado * dbeta

    def __relu(self,x):
        return np.maximum(0, x)
    