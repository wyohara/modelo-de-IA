import numpy as np

from src.modelo.tensor.modelo_personalizado.modulos.ferramentas_tensor import FerramentasTensor

class FeedFoward:
    def __init__(self, dim_model:int, dim_ff=None, teste=True):
        '''
        Modelo da rede neural feed foward simplificada para o tensor.
        Params:
            dim_model: dimensão original do modelo do tensor - quantos parametros são aceitos
            dim_ff: é a dimensão do feed foward normalmente 4*dim_model
        '''
        #definindo uma seed fixa para o numpy gerar sempre o mesmo valor e tornar teste repetível
        if teste:
            np.random.seed(42)

        self.__ferramentas = FerramentasTensor()
        self.dim_model = dim_model # dimensões do modelo usado        
        self.gamma = np.array([1]*dim_model) # Fator gamma do layernorm, cada dimensão tem seu gamma e layernorm
        self.beta = np.array([0]*dim_model) # Fator beta do layernorm, cada dimensão tem seu beta e layernorm

        # Valor segundo o paper "Attention Is All You Need"
        # dim_ff é 4*dim_model
        dim_ff =dim_ff if dim_ff else 4 * dim_model
        
        #gerando os pesos e bias do feed foward camada 1
        self.W1 = np.random.randn(dim_model, dim_ff) * 0.1
        self.b1 = np.zeros(dim_ff)
        
        #gerando os pesos e bias do feed foward camada 2
        # Perceba que no segundo peso é invertido dim_ff e dim_model
        # O que irá gerar uma matriz quadrada dim_model x dim_model
        self.W2 = np.random.randn(dim_ff, dim_model) * 0.1
        self.b2 = np.zeros(dim_model)
    
    def forward_completo(self, saida_atencao)->np.array:
        '''
        Método que aplica uma rede neural feed foward a saída da atenção.
        Permite inserir aleatoriedade a projeção linear original
            saida_atencao = valor que saiu da atencao (batch, seq_len, dim_model)
        '''
        # Aplicando o primeiro feed foward
        self.ff1 = saida_atencao @ self.W1 + self.b1

        # Realizando a ativação com ReLU por ser mais simples
        self.camada_oculta = self.relu(self.ff1)

        # Aplicando o segundo feed foward
        self.ff2 = (self.camada_oculta @ self.W2) + self.b2
        self.ff2_soma = self.ff2 + saida_atencao
        return self.__ferramentas.layer_norm(self.ff2_soma, 
                                                   gamma=self.gamma, 
                                                   beta=self.beta)

    
    def foward(self, saida_atencao)->np.ndarray:
        self.forward_completo(saida_atencao)
        ff_soma, ff_out, ff_mean, ff_std, ff_out_norm, gamma = self.forward_completo(saida_atencao)
        return ff_out

    def gerar_gradiente_perda_mse(self, resultado, rotulos_teste):
        """
        Gera o gradiente de perda d_saida em relação ao rotulo.
        Usada onde a saída é um valor contínuo e erros com distribuição 
            gaussiana (maximiza a verossimilhança sob normalidade).
        Penaliza grandes erros quadraticamente. É mais sensível a outliers.
        Use MSE quando seu objetivo é prever um número real (regressão).
        Params:
            rotulos_teste: resultado esperado pela rede neural
            resultado: resultado da rede neural feed foward normalizada
        """
        #recupera os parametros da ultima ativação
        batch_size, seq_len, dim_model = self.ff2.shape
        # Recupera o número total de tokens
        N = batch_size * seq_len  
        #Aplica o MSE
        d_saida = 2 * (resultado - rotulos_teste) / (N * dim_model)
        return d_saida
    
    def gerar_gradiente_perda_cross_entropy(self, resultado, rotulos_teste):
        '''
        Usada em classificação
            a saída é uma distribuição de probabilidades sobre classes discretas.
        Maximiza a log‑verossimilhança de uma distribuição multinomial.
        Não é adequada para valores contínuos (exige probabilidades).
        Penaliza principalmente previsões confiantes erradas (log de probabilidade muito negativa).
        Params:
            rotulos_teste: resultado esperado pela rede neural
            resultado: resultado da rede neural feed foward normalizada
        '''
        #recupera os parametros da ultima ativação
        batch_size, seq_len, dim_model = self.ff2.shape

        logits_reshape = resultado.reshape(-1, dim_model)
        max_logits = np.max(logits_reshape, axis=1, keepdims=True)
        e_x = np.exp(logits_reshape - max_logits)
        probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        probs = probs.reshape(resultado.shape)

        # Converter rotulos_teste para one‑hot se necessário
        if rotulos_teste.shape != resultado.shape:
            y_one_hot = np.zeros_like(resultado)
            y_one_hot[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], rotulos_teste] = 1
        else:
            y_one_hot = rotulos_teste
        d_saida = probs - y_one_hot
        return d_saida

    def gerar_backward(self, gradiente_perda, media_ff, std_ff, out_norm_ff, 
                       gamma, saida_att, soma):
        '''
        Método que realiza a retropropagação (backward) para corrigir os pesos.
        Params:
            gradiente_perda: Gradiente de perda calculado pelo MSE ou cross entropy
                Shape (batch, seq_len, dim_model).
            
            Fornecido por FerramentasTensor.layer_norm_backward
                media_ff: O valor da média do feedfoward
                    Shape (batch, seq_len, dim_model).
                std_ff: O desvio padrão do feed foward (shape broadcastável).                
                out_norm_ff: Valor corrigido do feed foward antes de aplicar beta e gamma
                gamma: valor do gamma usado
                saida_att: resultado da camada de atenção usado no feedfoward
                    hape (batch, seq_len, dim_model).
                soma: Entrada original da LayerNorm (saída do FF + entrada residual).
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
        # ----- Backward da segunda camada linear (W2, b2) -----
        #realiza o retropropagação da normalização, achando os valores corrigidos
        d_ff2, dgamma, dbeta = self.__ferramentas.layer_norm_backward(gradiente_perda, 
                                                                       soma, media_ff, 
                                                                       std_ff, out_norm_ff, 
                                                                       gamma)
        ff_out = d_ff2.copy()
        d_x_residual = d_ff2.copy()

        # ----- Backward da segunda camada linear (W2, b2) -----
        # Recupera a soma sobre os eixos:
        #   b (batch) e s (seq_len)
        #   mantem i (dimensão da camada oculta) e j (dimensão de d_saida).
        # Resultado é uma matriz (dim_ff, dim_model).
        dW2 = np.einsum('b s i, b s j -> i j', self.camada_oculta, ff_out)  
        db2 = np.sum(ff_out, axis=(0, 1))

        # Gradiente em relação à saída da ativação (hidden)
        dh = ff_out @ self.W2.T 

        # -----  Backward da ReLU -----
        # Derivada da ReLU: 1 onde ff1 > 0, senão 0
        d_relu = (self.ff1 > 0).astype(float)
        dh_ff1 = dh * d_relu

        # ----- Backward da primeira camada linear (W1, b1) -----
        dW1 = np.einsum('b s i, b s j -> i j', saida_att, dh_ff1)
        db1 = np.sum(dh_ff1, axis=(0, 1))
        # Gradiente em relação à entrada (saida_atencao)
        dx_ff = dh_ff1 @ self.W1.T 
        dx = d_x_residual+ dx_ff
        return dx, dW1, db1, dW2, db2, dgamma, dbeta  
    
    def corrigir_pesos(self, dW1, db1, dW2, db2, dgamma, dbeta, taxa_aprendizado=0.001):
        '''
        Método que corrige os pesos da camada do feedfoward
            dgamma : Gradiente da perda em relação ao parâmetro gamma.
                Deve ter o mesmo formato que self.gamma (dim_model,).
            dbeta : Gradiente da perda em relação ao parâmetro beta.
                Deve ter o mesmo formato que self.beta (dim_model,).
        '''
        self.W1 -= taxa_aprendizado * dW1
        self.W2 -= taxa_aprendizado * dW2
        self.b1 -= taxa_aprendizado * db1
        self.b2 -= taxa_aprendizado * db2
        
        self.gamma = self.gamma - taxa_aprendizado * dgamma
        self.beta  = self.beta  - taxa_aprendizado * dbeta
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def swish(x, beta=1.0):
        return x * (1 / (1 + np.exp(-beta * x)))