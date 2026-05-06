import numpy as np
from src.modelo.transformer.modulos.ferramentas import normalizador_camada


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
    
    def gerar_atencao(self, embedding_Q:np.array, embedding_K:np.array, embedding_V:np.array)->np.array:
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
        self.x = self.head_concat @ self.W_O

        return self.x

    def add_norm_layer(self, att:np.array, embedding_Q:np.array)->np.array:
        '''
        Camada que aplica o add e norm layer no transformer
        '''
        # somando o embedding_Q ao resultado para preservar o gradiente
        self.x = att + embedding_Q

        x, x_norm, media, dp, x_scoreZ, gamma = normalizador_camada(self.x, 
                                                                    gamma=self.gamma, 
                                                                    beta=self.beta)
        return x_norm
    
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
    
    def __softmax(self, x, axis=-1)->np.array:
        """Aplica o softmax para normalizar os valores em valores entre 0 e 1."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)