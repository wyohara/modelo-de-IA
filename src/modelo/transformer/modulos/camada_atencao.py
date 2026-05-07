import numpy as np
from src.modelo.transformer.modulos.ferramentas import normalizador_camada
from src.modelo.transformer.modulos.ferramentas import normalizador_camada, normalizador_camada_backward


class CamadaAtencao:
    def __init__(self, dim_model:int, num_heads:int, dim_k:int=0, dim_v:int=0, teste=True):
        '''
        Camada principal de ativacação da atenção
        Params:
            dim_model: dimensão do modelo, ou quantos parâmetros de entrada ele aceita
            num_heads: numero de cabeças de atenção
            dim_k, dim_v: dimensões usadas na cabeça. Se 0 usa divisao inteira de dim_model // heads
        '''
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_k = dim_k if dim_k != 0 else dim_model // num_heads
        self.dim_v = dim_v if dim_v != 0 else dim_model // num_heads

        self.gamma = np.array([1]*dim_model, dtype=np.float16) # Fator gamma do layernorm, cada dimensão tem seu gamma e layernorm
        self.beta = np.array([0]*dim_model, dtype=np.float16) # Fator beta do layernorm, cada dimensão tem seu beta e layernorm

        if teste:
            # definindo uma seed fixa para o numpy gerar sempre o mesmo valor no random
            np.random.seed(42)
        self.teste = teste

        # iniciando os valores da camada de atenção com numpy        
        self.W_K = (np.random.randn(self.num_heads, self.dim_model, self.dim_k) * 0.1).astype(np.float16)
        self.W_Q = (np.random.randn(self.num_heads, self.dim_model, self.dim_k) * 0.1).astype(np.float16)
        self.W_V = (np.random.randn(self.num_heads, self.dim_model, self.dim_v) * 0.1).astype(np.float16)

        # Projeção final após concatenação
        self.W_O = (np.random.randn(self.num_heads * self.dim_k, self.dim_model)).astype(np.float16)  
        self.b_O = (np.random.randn(self.dim_model)).astype(np.float16)  
    
    def foward(self, embedding_Q:np.array, embedding_K:np.array, embedding_V:np.array)->np.array:
        '''
        Método que calcula a atenção completa a partir dos vetores de embedding
        '''
        att_list = []
        self.Q_list, self.K_list, self.V_list, self.attn_list = [], [], [], []
        for h in range(self.num_heads): #propagando ao longo de cada cabeça
            # Projetar Q, K, V com as matrizes da cabeça i           
            Q_i = embedding_Q @ self.W_Q[h]   # (batch, seq_len, d_k)
            K_i = embedding_K @ self.W_K[h]   # (batch, seq_len, d_k)
            V_i = embedding_V @ self.W_V[h]   # (batch, seq_len, d_v)
            
            #Calcula a atenção para cada cabeça
            atencao_saida, attn_i = self.__calcular_atencao(Q_i, K_i, V_i)
            att_list.append(atencao_saida)

            #salvando os dados em uma lista para backward
            self.Q_list.append(Q_i)
            self.K_list.append(K_i)
            self.V_list.append(V_i)
            self.attn_list.append(attn_i)

        #concatena a saída de acordo com o útimo eixo, no caso as cabecas
        # Ao concatenar retorna ao shape original no caso (cabecas, dim_model, dim_K)
        self.head_concat = np.concatenate(att_list, axis=-1)
        
        #Calcula a saída multiplicando pelo peso de saída
        self.x_soma_residual = self.head_concat @ self.W_O
        self.x_soma_residual += self.b_O

        return self.x_soma_residual

    def backward(self, d_att:np.array):
        """
        Backward da MHA.

        Parâmetros
        ----------
        d_att : Atenção corrigida da normalização ou diretamente da feed foward
            Shape (batch, seq_len, dim_model).

        Retorna
        -------
        dx_entrada : Gradiente em relação à entrada `embeding_Q` (propagado para camadas anteriores).
            Shape (batch, seq_len, dim_model).
        dW_Q_list, dW_K_list, dW_V_list : Gradientes para as matrizes de cada cabeça cada um 
            shape (dim_model, d_k)
        dW_O : Gradiente para a matriz de saída W_O 
            shape (num_heads*d_k, dim_model)
        db_O : Gradiente para o bias da projeção de saída
            shape (dim_model,)
        """
        # 2. Backward da residual: soma = x + embeding_Q
        d_x = d_att.copy()                # gradiente para saida_concatenada @ W_O
        d_emb_residual = d_att.copy()     # parte do gradiente para embeding_Q via residual

        # 3. Backward da projeção de saída (W_O)
        dW_O = np.einsum('b s i, b s j -> i j', self.head_concat, d_x)  
        db_O = np.sum(d_x, axis=(0, 1))
        d_concat = d_x @ self.W_O.T

        # 4. Dividir gradiente entre as cabeças
        d_heads = np.split(d_concat, self.num_heads, axis=-1)

        # 5. Processar cada cabeça: backward da atenção e das projeções Q,K,V
        dW_Q_list, dW_K_list, dW_V_list = [], [], []
        d_emb_total = np.zeros_like(self.x_soma_residual)   # acumula gradiente vindo das cabeças

        for h in range(self.num_heads):
            # Backward da atenção (uma cabeça)
            dQ_h, dK_h, dV_h = self.__atencao_backward(
                d_heads[h],
                self.Q_list[h],
                self.K_list[h],
                self.V_list[h],
                self.attn_list[h]
            )

            # Gradientes para as matrizes de projeção (W_Q[h], etc.)
            dW_Q_h = np.einsum('bsi, bsj -> ij', self.x_soma_residual, dQ_h)   # (dim_model, d_k)
            dW_K_h = np.einsum('bsi, bsj -> ij', self.x_soma_residual, dK_h)
            dW_V_h = np.einsum('bsi, bsj -> ij', self.x_soma_residual, dV_h)

            dW_Q_list.append(dW_Q_h)
            dW_K_list.append(dW_K_h)
            dW_V_list.append(dW_V_h)

            # Gradiente em relação à entrada (embeding_Q) via essa cabeça
            d_emb_total += dQ_h @ self.W_Q[h].T + dK_h @ self.W_K[h].T + dV_h @ self.W_V[h].T

        # 6. Gradiente final para a entrada: contribuição residual + contribuição das cabeças
        dx_entrada = d_emb_residual + d_emb_total

        return (dx_entrada,
                dW_Q_list, dW_K_list, dW_V_list,
                dW_O, db_O)
    
    def camada_add_norm(self, att:np.array, embedding_Q:np.array)->np.array:
        '''
        Camada que aplica o add e norm layer no transformer
        '''
        # somando o embedding_Q ao resultado para preservar o gradiente
        self.x_soma_residual = att + embedding_Q

        x, x_norm, media, dp, x_scoreZ, gamma = normalizador_camada(self.x_soma_residual, 
                                                                    gamma=self.gamma, 
                                                                    beta=self.beta)
        return x_norm
    
    def camada_add_norm_backward(self, dy:np.array):
        '''
        Reversão da camada add_norm 
            Params:
                att_norm resultado da camada normalizada da saída da atenção
                dy: rótulos para comparação após usar o gradiente de perda mse
            Returns:
                d_att: correção do feedfoward antes da normalização e add_layer
                dgamma: Correção do fator gammma do feedfoward
                dbeta: Correção do fator beta do feedfoward
        '''
        d_soma, dgamma, dbeta = normalizador_camada_backward(self.x_soma_residual, dy, self.gamma)
        return d_soma, dgamma, dbeta
    
    def __calcular_atencao(self, w_q:list, w_k:list, w_v:list)->np.array:
        """
        Aplica a formula da atenção escalar por meio de produto escalar.
        Q, K, V: tensores de dimensões (batch, seq_len, d_k)
        Retorna: saída (batch, seq_len, d_v) e pesos de atenção (batch, seq_len, seq_len)
        """
        #garantindo que dim_k>0
        if self.dim_k==0:
            dim_k = 1e-5
        else:
            dim_k = self.dim_k

        # primeira parte da formula softmax(Q*K^T)
        # A transposição ocorre somente na dim_model e dim_k.
        # Isso resulta em uma matriz quadrada
        score_QK = (w_q @ w_k.transpose(0,2,1))/ np.sqrt(dim_k)
        atencao_normalizada = self.__softmax(score_QK, axis=-1)
        saida = atencao_normalizada @ w_v
        return saida, atencao_normalizada
    
    def __atencao_backward(self, dout, Q, K, V, attn):
        """
        Backward da atenção escalonada (single head).

        Parâmetros
        ----------
        dout : np.ndarray
            Gradiente em relação à saída da atenção (out) – shape (batch, seq_len, d_v).
        Q, K, V : np.ndarray
            Matrizes de consulta, chave e valor (shape batch, seq_len, d_k para Q/K; d_v para V).
        attn : np.ndarray
            Pesos de atenção (softmax) calculados no forward – shape (batch, seq_len, seq_len).

        Retorna
        -------
        dQ, dK, dV : np.ndarray
            Gradientes em relação a Q, K, V (mesmas shapes das entradas).
        """
        d_k = Q.shape[-1]
        # dV = attn^T @ dout
        dV = np.einsum('b i j, b j k -> b i k', attn, dout)
        # d_attn = dout @ V^T
        d_attn = dout @ V.transpose(0, 2, 1)
        # Derivada do softmax (estável)
        d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
        d_scores = d_scores / np.sqrt(d_k)
        # Gradientes para Q e K
        dQ = d_scores @ K
        dK = d_scores.transpose(0, 2, 1) @ Q
        return dQ, dK, dV
    
    def __softmax(self, x, axis=-1)->np.array:
        """Aplica o softmax para normalizar os valores em valores entre 0 e 1."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    

    