import numpy as np

from src.modelo.tensor.original.tensor import Tensor
from src.modelo.tensor.original.ferramentas import Ferramentas

class TensorFoward():
    def __init__(self):
        self.__ferramentas = Ferramentas()

        self.iniciar_dados_intermediários_vazios()
        

    def foward(self, tensor:Tensor, valor_entrada:np):
        '''Foward que salva os dados intermediários para poder realizar o treinamento'''
        self.iniciar_dados_intermediários_vazios()
        head_out_list=[]
        # [etapa 1]calculando os valores de saida das camadas Q K V do tensor
        for i in range(tensor.num_heads):
            Q = valor_entrada @ tensor.W_Query[i]
            K = valor_entrada @ tensor.W_Key[i]
            V = valor_entrada @ tensor.W_Value[i]
            scores = Q @ K.T / np.sqrt(tensor.dim_head)
            attn_weights = self.__ferramentas.softmax(scores, axis=-1)
            head_out = attn_weights @ V
            head_out_list.append(head_out)

            self.Query_list.append(Q)
            self.Key_list.append(K)
            self.Value_list.append(V)
            self.scores_list.append(scores)
            self.attn_weights_list.append(attn_weights)

        #[Etapa 2] Concatenação das cabeças ao longo da última dimensão (axis=-1)
        self.concat_heads = np.concatenate(head_out_list, axis=-1)
        #[Etapa 3] Projeção de saída da atenção usando W_Output
        projecao_out = self.concat_heads @ tensor.W_Output

        #[Etapa 4] calculando a saída (entrada + projeção) e normalizando
        self.X1 = valor_entrada + projecao_out
        self.X1_norm = self.__ferramentas.layer_norm(self.X1)

        #[Etapa 5] calculando o Feed-forward somando o peso e bias de normalização
        self.feed_foward_linear = self.X1_norm @ tensor.W_ff1 + tensor.b_ff1
        # removendo valor negativo pois não tem valor no gradiente usando o Relu
        self.feed_foward_hidden = self.__ferramentas.relu(self.feed_foward_linear)
        #aplicando o bias e peso da segunda camada
        self.ff_out = self.feed_foward_hidden @ tensor.W_ff2 + tensor.b_ff2

        #[Etapa 6] Residual e layer norm final
        self.X2 = self.X1_norm + self.ff_out
        #valor normalizado com a última camada do output
        self.X2_norm = self.__ferramentas.layer_norm(self.X2)         
        return self.X2_norm
    
    def iniciar_dados_intermediários_vazios(self):
        # Listas para armazenar variáveis intermediárias de cada cabeça
        self.Query_list = []
        self.Key_list = []
        self.Value_list = []
        self.scores_list = []
        self.attn_weights_list=[]     