import numpy as np

from src.modelo.tensor.rope.tensor import Tensor
from src.modelo.tensor.rope.tensor_foward import TensorFoward
from src.modelo.tensor.rope.ferramentas import Ferramentas


class TensorBackward():
    def __init__(self):
        self.__ferramentas = Ferramentas()
    
    def backward(self, tensor_inicial:Tensor, valor_entrada, rotulos):
        tensor_zeros = Tensor()
        tensor_zeros.zerar_pesos(tensor_inicial)
        
        foward = TensorFoward()
        foward.foward(tensor_inicial, valor_entrada)
        
        # Gradiente da perda (MSE)
        gradiente_perda = 2 * (foward.X2_norm - rotulos) / foward.X2_norm.size
        
        # Backward das camadas finais (LayerNorm, FFN, residuals) - igual ao original
        d_x2 = self.__ferramentas.layer_norm_backward(gradiente_perda, foward.X2)
        dX1_norm = d_x2.copy()
        dff_out = d_x2.copy()
        
        backward_W_ff2 = foward.feed_foward_hidden.T @ dff_out
        backward_b_ff2 = np.sum(dff_out, axis=0)
        dff_hidden = dff_out @ tensor_inicial.W_ff2.T
        dff_hidden[foward.feed_foward_linear <= 0] = 0
        
        backward_W_ff1 = foward.X1_norm.T @ dff_hidden
        backward_b_ff1 = np.sum(dff_hidden, axis=0)
        dX1_norm_from_ff = dff_hidden @ tensor_inicial.W_ff1.T
        
        dX1_norm_total = dX1_norm + dX1_norm_from_ff
        dX1 = self.__ferramentas.layer_norm_backward(dX1_norm_total, foward.X1)
        
        dX_from_resid = dX1.copy()
        dmha_out = dX1.copy()
        
        backward_W_Output = foward.concat_heads.T @ dmha_out
        dconcat_heads = dmha_out @ tensor_inicial.W_Output.T
        
        dX_from_attn = np.zeros_like(valor_entrada)
        
        # Recupera as tabelas RoPE usadas no forward
        cos = foward.cos
        sin = foward.sin
        
        for i in range(tensor_inicial.num_heads):
            start = i * tensor_inicial.dim_head
            end = (i + 1) * tensor_inicial.dim_head
            dhead_out = dconcat_heads[:, start:end]
            
            # Dados da cabeça (Q_rot, K_rot, V)
            Q_rot = foward.Query_list[i]
            K_rot = foward.Key_list[i]
            V = foward.Value_list[i]
            attn_weights = foward.attn_weights_list[i]
            scores = foward.scores_list[i]
            
            # Backward da atenção (igual ao original, usando Q_rot e K_rot)
            dV = attn_weights.T @ dhead_out
            dattn = dhead_out @ V.T
            
            dscores = np.zeros_like(scores)
            for row in range(scores.shape[0]):
                y = attn_weights[row]
                dL_dy = dattn[row]
                dL_dx = y * (dL_dy - np.sum(dL_dy * y))
                dscores[row] = dL_dx
            
            dS = dscores / np.sqrt(tensor_inicial.dim_head)
            dQ_rot = dS @ K_rot
            dK_rot = dS.T @ Q_rot
            
            # ========== NOVO: Retropropagação através do RoPE ==========
            # Como a rotação é ortogonal, o gradiente é rotacionado com sinal oposto.
            if cos is not None and sin is not None and cos.shape[1] > 0:
                dQ_orig = self.__ferramentas.apply_rotary_emb(dQ_rot, cos, -sin)
                dK_orig = self.__ferramentas.apply_rotary_emb(dK_rot, cos, -sin)
            else:
                # Se dim_head == 1 ou tabelas vazias, RoPE é identidade
                dQ_orig = dQ_rot
                dK_orig = dK_rot
            # ===========================================================
            
            # Gradientes para os pesos lineares (usando os gradientes originais)
            tensor_zeros.W_Query[i] = valor_entrada.T @ dQ_orig
            tensor_zeros.W_Key[i]   = valor_entrada.T @ dK_orig
            tensor_zeros.W_Value[i] = valor_entrada.T @ dV
            
            # Contribuição para o gradiente da entrada (X)
            dX_from_attn += (dQ_orig @ tensor_inicial.W_Query[i].T +
                             dK_orig @ tensor_inicial.W_Key[i].T +
                             dV @ tensor_inicial.W_Value[i].T)
        
        # Atualização dos pesos (SGD) - igual ao original
        lr = tensor_inicial.learning_rate
        tensor_inicial.W_Query -= lr * tensor_zeros.W_Query
        tensor_inicial.W_Key   -= lr * tensor_zeros.W_Key
        tensor_inicial.W_Value -= lr * tensor_zeros.W_Value
        tensor_inicial.W_Output -= lr * backward_W_Output
        tensor_inicial.W_ff1    -= lr * backward_W_ff1
        tensor_inicial.b_ff1    -= lr * backward_b_ff1
        tensor_inicial.W_ff2    -= lr * backward_W_ff2
        tensor_inicial.b_ff2    -= lr * backward_b_ff2
        
        # Retorna o forward atualizado (para a próxima iteração)
        return foward.foward(tensor_inicial, valor_entrada)
