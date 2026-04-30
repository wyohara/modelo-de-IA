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
    score_QK = Q @ K.transpose(0,2,1)   # (..., seq_q, seq_k)
    atencao_normalizada = softmax(score_QK, axis=-1/ np.sqrt(dim_k))
    saida = atencao_normalizada @ V
    return saida, atencao_normalizada

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
        self.cabecas = heads
        self.dim_K = dim_k if dim_k != 0 else dim_model // heads
        self.dim_V = dim_v if dim_v != 0 else dim_model // heads

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
    dim_model = 64

    #criando o tensor
    tensor = Tensor(dim_model, cabecas)

    tensor_shape = tensor.W_Q.shape
    print("======================================\n\tCódigo do Tensor_2")
    print("======================================")
    print(f'Formato do tensor: {tensor.W_Q.shape}:')
    print(f'\t- {tensor.W_Q.shape[0]} Cabeças')
    print(f'\t- {tensor.W_Q.shape[1]} dimensões de valores')
    print(f'\t- {tensor.W_Q.shape[2]} dimensões de cabeça')
    

    print("======================================")    
    #simulando o embeding
    tam_batch = 4 #batch são o total de blocos de textos processados paralelamente
    tam_bloco = 64
    dim_model = tensor.dim_model
    embeding_Q = np.random.randn(tam_batch, tam_bloco, dim_model)
    embeding_K = np.random.randn(tam_batch, tam_bloco, dim_model)
    embeding_V = np.random.randn(tam_batch, tam_bloco, dim_model)    

    saida = tensor.propagar(embeding_Q,embeding_K,embeding_V)
    print("\n✅ Multi-Head Attention executada com sucesso!")    
    print("Embeding de entrada no formato: ", embeding_Q.shape)
    print("Formato do tensor na concatenação: ", tensor.W_K[0].shape)
    print("Cada cabeça trabalha em subespaços de dimensão dim_k =", tensor.dim_K)
    print("Concatenação das", cabecas, "cabeças produz", cabecas * tensor.dim_V, "dimensões")
    print("Projeção final reduz para", dim_model, "dimensões")
    print("Resultado final igual a ", saida.shape, "dimensões")