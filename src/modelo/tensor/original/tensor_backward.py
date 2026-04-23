import numpy as np

from src.modelo.tensor.original.tensor import Tensor
from src.modelo.tensor.original.tensor_foward import TensorFoward
from src.modelo.tensor.original.ferramentas import Ferramentas


class TensorBackward():
    def __init__(self):
        self.__ferramentas = Ferramentas()
    
    def backward(self, tensor_inicial:Tensor, valor_entrada, rotulos):
        tensor_zeros = Tensor()
        tensor_zeros.zerar_pesos(tensor_inicial)
        
        foward = TensorFoward()
        foward.foward(tensor_inicial, valor_entrada)

        #[Etapa 0] Gradiente da perda em relação a X2_norm
        # Usa a fórmula para calcular o MSE em n pontos de dados é escrita como 1n∑i=1n(yi-yi^)2, em que y é o valor real e ŷ é o valor previsto.
        # Quadrar o erro significa que o valor resultante é sempre positivo: dessa forma, o MSE avalia apenas a magnitude do erro, não sua direção
        gradiente_perda = 2 * (foward.X2_norm - rotulos) / foward.X2_norm.size

        #[Etapa 6] Backward onde normaliza a saída do foward usando o gradiente de perda
        # Gera o novo X2 corrigido
        d_x2 = self.__ferramentas.layer_norm_backward(gradiente_perda, foward.X2)

        # Residual (X2 = X1_norm + ff_out)
        dX1_norm = d_x2.copy()
        dff_out = d_x2.copy()

        #[Etapa 5] Backward feed-forward
        backward_W_ff2 = foward.feed_foward_hidden.T @ dff_out
        backward_b_ff2 = np.sum(dff_out, axis=0)
        dff_hidden = dff_out @ tensor_inicial.W_ff2.T

        # Backward para corrigir o ff_hidden
        dff_hidden[foward.feed_foward_linear <= 0] = 0

        backward_W_ff1 = foward.X1_norm.T @ dff_hidden
        backward_b_ff1 = np.sum(dff_hidden, axis=0)
        dX1_norm_from_ff = dff_hidden @ tensor_inicial.W_ff1.T

        # Soma das contribuições para dX1_norm
        dX1_norm_total = dX1_norm + dX1_norm_from_ff

        # Backward primeira layer norm
        dX1 = self.__ferramentas.layer_norm_backward(dX1_norm_total, foward.X1)

        # Residual (X1 = X + mha_out)
        dX_from_resid = dX1.copy()
        dmha_out = dX1.copy()

        # Backward projeção de saída da atenção
        backward_W_Output = foward.concat_heads.T @ dmha_out           # (num_heads*d_k, d_model)
        dconcat_heads = dmha_out @ tensor_inicial.W_Output.T            # (3, num_heads*d_k)

        # Backward através das cabeças
        dX_from_attn = np.zeros_like(valor_entrada)  # acumulador

        
        for i in range(tensor_inicial.num_heads):
            start = i * tensor_inicial.dim_head
            end = (i + 1) * tensor_inicial.dim_head
            dhead_out = dconcat_heads[:, start:end]   # (3,3) gradiente para a saída da cabeça i

            # Recuperar valores desta cabeça
            Q = foward.Query_list[i]
            K = foward.Key_list[i]
            V = foward.Value_list[i]
            attn_weights = foward.attn_weights_list[i]
            scores = foward.scores_list[i]

            # Backward da atenção para esta cabeça
            # 1) dV
            dV = attn_weights.T @ dhead_out

            # 2) dattn
            dattn = dhead_out @ V.T

            # 3) Backward softmax
            dscores = np.zeros_like(scores)
            for row in range(np.array(scores).shape[0]):
                y = attn_weights[row]
                dL_dy = dattn[row]
                dL_dx = y * (dL_dy - np.sum(dL_dy * y))
                dscores[row] = dL_dx

            # 4) Backward scores
            dS = dscores / np.sqrt(tensor_inicial.dim_head)
            dQ = dS @ K
            dK = dS.T @ Q

            # 5) Gradientes dos pesos
            tensor_zeros.W_Query[i] = valor_entrada.T @ dQ
            tensor_zeros.W_Key[i] = valor_entrada.T @ dK
            tensor_zeros.W_Value[i] = valor_entrada.T @ dV

            # Contribuição para o gradiente de X
            dX_from_attn += dQ @ tensor_inicial.W_Query[i].T + dK @ tensor_inicial.W_Key[i].T + dV @ tensor_inicial.W_Value[i].T

        # Gradiente total de X (não será usado para atualizar X, pois é entrada fixa)
        dX_total = dX_from_resid + dX_from_attn

        # ---------------------------
        # Atualização dos pesos (SGD)
        # ---------------------------
        learning_rate = tensor_inicial.learning_rate
        tensor_inicial.W_Query -= learning_rate * tensor_zeros.W_Query
        tensor_inicial.W_Key -= learning_rate * tensor_zeros.W_Key
        tensor_inicial.W_Value -= learning_rate * tensor_zeros.W_Value
        tensor_inicial.W_Output -= learning_rate * backward_W_Output
        tensor_inicial.W_ff1 -= learning_rate * backward_W_ff1
        tensor_inicial.b_ff1 -= learning_rate * backward_b_ff1
        tensor_inicial.W_ff2 -= learning_rate * backward_W_ff2
        tensor_inicial.b_ff2 -= learning_rate * backward_b_ff2

        return foward.foward(tensor_inicial, valor_entrada)