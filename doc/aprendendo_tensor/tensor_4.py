import numpy as np


def softmax(x, axis=-1):
    """Aplica o softmax para normalizar os valores em valores entre 0 e 1."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def calcular_atencao(Q, K, V):
    """
    Aplica a formula da atenção escalar por meio de produto escalar.
    Q, K, V: tensores de dimensões (..., seq_len, d_k)
    Retorna: saída (..., seq_len, d_v) e pesos de atenção (..., seq_len, seq_len)
    """
    dim_k = Q.shape[-1] # pega a dimensão da cabeça
    #primeira parte da formula softmax(Q*K^T)
    # A transposição ocorre somente na dim_model e dim_k.
    # Isso resulta em uma matriz quadrada
    score_QK = (Q @ K.transpose(0,2,1))/ np.sqrt(dim_k)
    atencao_normalizada = softmax(score_QK, axis=-1)
    saida = atencao_normalizada @ V
    return saida, atencao_normalizada

class Tensor:
    def __init__(self, dim_model:int, cabecas:int, dim_k:int=0, dim_v:int=0, teste=True):
        '''
        Classe principal que opera o tensor
        Params:
            dim_model: dimensão do modelo, ou quantos parâmetros de entrada ele aceita
            heads: numero de cabeças de atenção
            dim_k, dim_v: dimensões usadas na cabeça. Se 0 usa divisao inteira de dim_model // heads
        '''
        self.dim_model = dim_model
        self.cabecas = cabecas
        self.dim_K = dim_k if dim_k != 0 else dim_model // cabecas
        self.dim_V = dim_v if dim_v != 0 else dim_model // cabecas

        if teste: #definindo uma seed fixa para o numpy gerar sempre o mesmo valor no random
            np.random.seed(42)

        '''
        Definindo os pesos iniciais de Q, K, V e o peso final O após a concatenação obedecendo:
        MultiHead(Q,K,V ) = Concat(head1,...,headh)W^o
        '''
    
        self.W_K = np.random.randn(self.cabecas, self.dim_model, self.dim_K) * 0.1
        self.W_Q = np.random.randn(self.cabecas, self.dim_model, self.dim_K) * 0.1
        self.W_V = np.random.randn(self.cabecas, self.dim_model, self.dim_V) * 0.1

        # Projeção final após concatenação
        self.W_O = np.random.randn(self.cabecas * self.dim_V, self.dim_model)
    
    def propagar(self, embeding_Q, embeding_K, embeding_V):
        '''
        Após o texto de entrada ser transformado em tokens e gerado um embeding,
        ele é propagado no tensor para gerar os resultados.
        Propagar é usar um embeding para multiplicar os vetores do tensor 
        e assim gerar os resultados.
        Params:
            embeding_Q: valor da dimensão Q do embeding
            embeding_K: valor da dimensão K do embeding
            embeding_V: valor da dimensão K do embeding         
        '''
        
        saida_cabecas = []
        for h in range(self.cabecas): #propagando ao longo de cada cabeça
            # Projetar Q, K, V com as matrizes da cabeça i            
            Q_i = embeding_Q @ self.W_Q[h]   # (batch, seq_len, d_k)
            K_i = embeding_K @ self.W_K[h]   # (batch, seq_len, d_k)
            V_i = embeding_V @ self.W_V[h]   # (batch, seq_len, d_v)
            #Calcula a atenção para cada cabeça
            atencao_saida, _ = calcular_atencao(Q_i, K_i, V_i)
            saida_cabecas.append(atencao_saida)

        #concatena a saída de acordo com o útimo eixo, no caso as cabecas
        # Ao concatenar retorna ao shape original no caso (cabecas, dim_model, dim_K)
        saida_concatenada = np.concatenate(saida_cabecas, axis=-1)        
        #Calcula a saída multiplicando pelo peso de saída
        saida = saida_concatenada @ self.W_O
        return saida

    
if __name__ == "__main__":
    cabecas = 4
    dim_model = 64  #número de dimensões do modelo
    tam_batch = 2   # número de frases processadas em paralelo ou blocos de dados
    compr_seq = 10     # número de tokens por frase ou bloco de dados (comprimento da sequência)

    # --------------------------------------------------------
    # Simulando uma camada de embedding aprendida (opcional)
    # --------------------------------------------------------
    # Aqui mostramos como seria uma camada de embedding que transforma índices de tokens

    tam_vocab = 1000 #supondo que o vocabulario tenha 1000 tokens
    matriz_embedding = np.random.randn(tam_vocab, dim_model) * 0.1

    # Frase transformada em tokens, usa duas listas pois o batch é 2
    frase_em_tokens = np.array([[5, 23, 78, 12, 45, 99, 200, 3, 6, 0],
                                [5, 23, 78, 12, 45, 99, 200, 3, 6, 0]])
    
    if frase_em_tokens.shape[0] == 2:
        # uma consulta simples para achar os tokens
        # embedding (2,10,64)
        frase_embedding = matriz_embedding[frase_em_tokens]

        tensor = Tensor(dim_model, cabecas)
        saida_frase = tensor.propagar(frase_embedding, frase_embedding, frase_embedding)

        print("======================================\n\tCódigo do Tensor_4")
        print("======================================")
        print(f'Tamanho da matriz de embedding: {matriz_embedding.shape}')
        print(f'Tamanho da frase para embedding: {frase_embedding.shape}')
        print(f'Formato da frase embeding: {frase_em_tokens.shape}')
        print(f'\t- Saída  {saida_frase.shape} ')
        
        print("======================================\n\tGerando logits e tokens de saída")
        print("======================================")

        #gerando o logit a partir da saida do tensor e da matriz de embedding
        logits = saida_frase @ matriz_embedding.T

        #usando softmax para gerar a probabilidade (saida, embedding)
        probabilidades_tokens = softmax(logits)

        #Escolhendo o token com maior probabilidade
        tokens_ids = np.argmax(probabilidades_tokens, axis=-1)
        
        print(f'matriz embedding transposta no formato {matriz_embedding.T.shape}')
        print(f'\t- Saída do tensor: {saida_frase.shape} ')
        print(f'logit criado no formato {logits.shape}')
        print(f'probabilidades dos tokens criado no formato {probabilidades_tokens.shape}')
        print(f'Id do token com maior probabilidade é no formato {tokens_ids.shape}')
        
        # Para percorrer e exibir todos os tokens de todas as posições (cuidado: pode ser grande):
        for b in range(tokens_ids.shape[0]):  
            for s in range(tokens_ids.shape[1]):  # seq_len
                token_id = tokens_ids[b, s]
                print(f'batch {b}, posição {s}: id do token = {token_id}')