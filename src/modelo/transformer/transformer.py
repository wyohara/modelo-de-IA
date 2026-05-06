import numpy as np

from src.modelo.transformer.modulos.camada_atencao import CamadaAtencao
from src.modelo.transformer.modulos.camada_feed_foward import CamadaFeedFoward
from src.modelo.transformer.modulos.transformer_utils import TransformerUtils


class Transformer:
    def __init__(self, dim_model=512, num_heads=8):
        self.camada_atencao = CamadaAtencao(dim_model, num_heads)
        self.camada_feed_Foward = CamadaFeedFoward(dim_model)
        self.utils = TransformerUtils()

    def aplicar_tensor_padrao(self, embedding_Q:np.array, embedding_K:np.array, embedding_V:np.array):

        #aplicando a atenção
        att = self.camada_atencao.gerar_atencao(embedding_Q, embedding_K, embedding_V)
        saida_norm_att = self.camada_atencao.add_norm_layer(att, embedding_Q)

        #aplicando o feedfoward
        ff = self.camada_feed_Foward.forward(saida_norm_att)
        saida_norm_ff = self.camada_feed_Foward.add_norm_layer(ff, saida_norm_att)

        return saida_norm_ff