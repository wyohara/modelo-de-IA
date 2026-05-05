import numpy as np
from src.modelo.tensor.modelo_personalizado.modulos.ferramentas_tensor import FerramentasTensor


class AtencaoMulticabeca:
    def __init__(self, seq_len:int, cabecas:int, dim_k:int=0, dim_v:int=0, teste=True):
        '''
        Classe principal que opera o tensor
        Params:
            dim_model: dimensão do modelo, ou quantos parâmetros de entrada ele aceita
            heads: numero de cabeças de atenção
            dim_k, dim_v: dimensões usadas na cabeça. Se 0 usa divisao inteira de dim_model // heads
        '''
        self.seq_len = seq_len
        self.num_heads = cabecas
        self.dim_K = dim_k if dim_k != 0 else seq_len // cabecas
        self.dim_V = dim_v if dim_v != 0 else seq_len // cabecas

        self.gamma = np.array([1]*seq_len) # Fator gamma do layernorm, cada dimensão tem seu gamma e layernorm
        self.beta = np.array([0]*seq_len) # Fator beta do layernorm, cada dimensão tem seu beta e layernorm
        self.__ferramentas = FerramentasTensor()

        if teste: #definindo uma seed fixa para o numpy gerar sempre o mesmo valor no random
            np.random.seed(42)
        self.__iniciar_valores()        
    
    def __iniciar_valores(self):
        '''
        Definindo os pesos iniciais de Q, K, V e o peso final O após a concatenação obedecendo:
        MultiHead(Q,K,V ) = Concat(head1,...,headh)W^o
        '''    
        self.W_K = np.random.randn(self.num_heads, self.seq_len, self.dim_K) * 0.1
        self.W_Q = np.random.randn(self.num_heads, self.seq_len, self.dim_K) * 0.1
        self.W_V = np.random.randn(self.num_heads, self.seq_len, self.dim_V) * 0.1

        # Projeção final após concatenação
        self.W_O = np.random.randn(self.num_heads * self.dim_K, self.seq_len)
    
    def gerar_atencao_completa(self, embeding_Q, embeding_K, embeding_V):
        '''
        Método que calcula a atenção a partir das matrizes QKV do multihead attention
        '''
        att_list = []
        self.Q_list, self.K_list, self.V_list, self.attn_list = [], [], [], []
        for h in range(self.num_heads): #propagando ao longo de cada cabeça
            # Projetar Q, K, V com as matrizes da cabeça i                        
            Q_i = embeding_Q @ self.W_Q[h]   # (batch, seq_len, d_k)
            K_i = embeding_K @ self.W_K[h]   # (batch, seq_len, d_k)
            V_i = embeding_V @ self.W_V[h]   # (batch, seq_len, d_v)
            
            #Calcula a atenção para cada cabeça
            atencao_saida, attn_i = self.__ferramentas.calcular_atencao(Q_i, K_i, V_i)
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
        x = self.head_concat @ self.W_O
        self.x_soma = x + embeding_Q

        att_soma, att_out, mean, std, out_norm, gamma = self.__ferramentas.layer_norm(self.x_soma)
        return (att_soma, att_out, mean, std, out_norm, gamma)
    
    def gerar_atencao(self, embeding_Q, embeding_K, embeding_V):
        att_soma, att_out, mean, std, out_norm, gamma = self.gerar_atencao_completa(embeding_Q, embeding_K, embeding_V)
        return att_out
    
    def gerar_backward(self, ff_soma, out, mean, std, out_norm, gamma):
        """
        Backward da MHA.

        Parâmetros
        ----------
        gradiente_perda : np.ndarray
            Gradiente da perda em relação à **saída** da MHA (após layer norm).
            Shape (batch, seq_len, dim_model).

        Retorna
        -------
        dx_entrada : np.ndarray
            Gradiente em relação à entrada `embeding_Q` (propagado para camadas anteriores).
            Shape (batch, seq_len, dim_model).
        dW_Q_list, dW_K_list, dW_V_list : list of np.ndarray
            Gradientes para as matrizes de cada cabeça (cada um shape (dim_model, d_k)).
        dW_O : np.ndarray
            Gradiente para a matriz de saída W_O (shape (num_heads*d_k, dim_model)).
        db_O : np.ndarray
            Gradiente para o bias da projeção de saída (shape (dim_model,)).
        dgamma_ln, dbeta_ln : np.ndarray
            Gradientes dos parâmetros gamma e beta da layer norm (se existirem).
        """
        # 1. Backward da layer norm (pós‑residual)
        d_soma, dgamma_ln, dbeta_ln = self.__ferramentas.layer_norm_backward(
            ff_soma,
            out,           # entrada original da layer norm
            mean,
            std,
            out_norm,
            gamma
        )

        # 2. Backward da residual: soma = x + embeding_Q
        d_x = d_soma.copy()                # gradiente para saida_concatenada @ W_O
        d_emb_residual = d_soma.copy()     # parte do gradiente para embeding_Q via residual

        # 3. Backward da projeção de saída (W_O)
        dW_O = np.einsum('b s i, b s j -> i j', self.head_concat, d_x)  
        db_O = np.sum(d_x, axis=(0, 1))
        d_concat = d_x @ self.W_O.T

        # 4. Dividir gradiente entre as cabeças
        d_heads = np.split(d_concat, self.num_heads, axis=-1)

        # 5. Processar cada cabeça: backward da atenção e das projeções Q,K,V
        dW_Q_list, dW_K_list, dW_V_list = [], [], []
        d_emb_total = np.zeros_like(self.x_soma)   # acumula gradiente vindo das cabeças

        for h in range(self.num_heads):
            # Backward da atenção (uma cabeça)
            dQ_h, dK_h, dV_h = self.__attention_backward(
                d_heads[h],
                self.Q_list[h],
                self.K_list[h],
                self.V_list[h],
                self.attn_list[h]
            )

            # Gradientes para as matrizes de projeção (W_Q[h], etc.)
            dW_Q_h = np.einsum('bsi, bsj -> ij', self.x_soma, dQ_h)   # (dim_model, d_k)
            dW_K_h = np.einsum('bsi, bsj -> ij', self.x_soma, dK_h)
            dW_V_h = np.einsum('bsi, bsj -> ij', self.x_soma, dV_h)

            dW_Q_list.append(dW_Q_h)
            dW_K_list.append(dW_K_h)
            dW_V_list.append(dW_V_h)

            # Gradiente em relação à entrada (embeding_Q) via essa cabeça
            d_emb_total += dQ_h @ self.W_Q[h].T + dK_h @ self.W_K[h].T + dV_h @ self.W_V[h].T

        # 6. Gradiente final para a entrada: contribuição residual + contribuição das cabeças
        dx_entrada = d_emb_residual + d_emb_total

        return (dx_entrada,
                dW_Q_list, dW_K_list, dW_V_list,
                dW_O, db_O,
                dgamma_ln, dbeta_ln)
    
    def __attention_backward(self, dout, Q, K, V, attn):
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
    
    def corrigir_pesos_mha(self, dW_Q_list, dW_K_list, dW_V_list, dW_O, db_O, 
                       dgamma_ln, dbeta_ln, taxa_aprendizado=0.001):
        # Atualiza as matrizes das cabeças
        for h in range(self.num_heads):
            self.W_Q[h] -= taxa_aprendizado * dW_Q_list[h]
            self.W_K[h] -= taxa_aprendizado * dW_K_list[h]
            self.W_V[h] -= taxa_aprendizado * dW_V_list[h]
        # Atualiza a projeção de saída
        self.W_O -= taxa_aprendizado * dW_O
        # Atualiza parâmetros da layer norm (se existirem)
        self.gamma -= taxa_aprendizado * dgamma_ln
        self.beta  -= taxa_aprendizado * dbeta_ln