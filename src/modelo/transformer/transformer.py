import numpy as np

from src.modelo.transformer.modulos.camada_atencao import CamadaAtencao
from src.modelo.transformer.modulos.camada_feed_foward import CamadaFeedFoward
from src.modelo.transformer.modulos.transformer_utils import TransformerUtils

from src.modelo.transformer.modulos.ferramentas import gradiente_perda_mse


class Transformer:
    def __init__(self, dim_model=512, num_heads=8, teste=False):
        self.camada_atencao = CamadaAtencao(dim_model, num_heads, teste=teste)
        self.camada_feed_Foward = CamadaFeedFoward(dim_model, teste=teste)
        self.utils = TransformerUtils()

    def aplicar_tensor_padrao(self, embedding_Q:np.array, embedding_K:np.array, embedding_V:np.array):

        #aplicando a atenção
        att = self.camada_atencao.foward(embedding_Q, embedding_K, embedding_V)
        saida_norm_att = self.camada_atencao.camada_add_norm(att, embedding_Q)

        #aplicando o feedfoward
        ff = self.camada_feed_Foward.forward(saida_norm_att)
        saida_norm_ff = self.camada_feed_Foward.camada_add_norm(ff, saida_norm_att)

        return saida_norm_ff
    
    def aplicar_backward_padrao(self, x:np.array, y:np.array):
        dy = gradiente_perda_mse(x, y)

        # Gerando a correção do add e norm do feedfoward
        d_x, dgamma, dbeta  = self.camada_feed_Foward.camada_add_norm_backward(dy)
        self.camada_feed_Foward.gamma -= dgamma
        self.camada_feed_Foward.beta -= dbeta

        # Gerando a correção do feedfoward
        dx, dW1, db1, dW2, db2 = self.camada_feed_Foward.backward(d_x)
        self.camada_feed_Foward.w1 -= dW1
        self.camada_feed_Foward.b1 -= db1
        self.camada_feed_Foward.w2 -= dW2
        self.camada_feed_Foward.b2 -= db2

        # Gerando a correção do add e norm da atenção
        datt, dgamma, dbeta= self.camada_atencao.camada_add_norm_backward(dx)
        self.camada_atencao.gamma -= dgamma
        self.camada_atencao.beta -= dbeta

        # Gerando a correção da atenção
        dx_entrada, dW_Q_list, dW_K_list, dW_V_list, dW_O, db_O = self.camada_atencao.backward(datt)
        self.camada_atencao.W_Q -= dW_Q_list
        self.camada_atencao.W_K -= dW_K_list
        self.camada_atencao.W_V -= dW_V_list        
        self.camada_atencao.W_O -= dW_O 
        self.camada_atencao.b_O -= db_O

