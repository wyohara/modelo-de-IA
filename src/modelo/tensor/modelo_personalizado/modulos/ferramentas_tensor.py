import numpy as np


class FerramentasTensor:

    def softmax(self, x, axis=-1):
        """Aplica o softmax para normalizar os valores em valores entre 0 e 1."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


    def calcular_atencao(self, w_q, w_k, w_v):
        """
        Aplica a formula da atenção escalar por meio de produto escalar.
        Q, K, V: tensores de dimensões (..., seq_len, d_k)
        Retorna: saída (..., seq_len, d_v) e pesos de atenção (..., seq_len, seq_len)
        """
        dim_k = w_q.shape[-1] # pega a dimensão da cabeça
        # primeira parte da formula softmax(Q*K^T)
        # A transposição ocorre somente na dim_model e dim_k.
        # Isso resulta em uma matriz quadrada
        score_QK = (w_q @ w_k.transpose(0,2,1))/ np.sqrt(dim_k)
        atencao_normalizada = self.softmax(score_QK, axis=-1)
        saida = atencao_normalizada @ w_v
        return saida, atencao_normalizada
    
    def layer_norm(self, ff2_soma, eps=1e-6, gamma=1, beta=0):
        """Retorna normalizado, média e desvio padrão.
            Params:
                gamma: fator de escala do vetor
                beta: fator de deslocamento do vetor
                eps: valor pequeno para evitar divisão por zero no std
                
            Returns:
                out: resultado normalizado com beta e gama
                mean: média dos valores
                std: desvio padrão dos valores
                out_norm: valor normalizado da saída
                gamma: valor original do fator de correção"""
        mean = ff2_soma.mean(axis=-1, keepdims=True)
        var = ((ff2_soma - mean) ** 2).mean(axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        out_norm = (ff2_soma - mean) / std
        out_corrigido = gamma * out_norm + beta
        return ff2_soma, out_corrigido, mean, std, out_norm, gamma

    def layer_norm_backward(self, dout, x, mean, std, x_norm, gamma):
        """
        Retropropagação ou backward do layerNorm com parâmetros gamma (escala) e beta (deslocamento).
        Calcula os gradientes da perda em relação à entrada x, aos parâmetros gamma e beta.
        A operação forward corresponde a:
            y = gamma * (x - mean) / std + beta
        onde mean e std são calculados ao longo da última dimensão de x.

        Params:
            dout:Gradiente da perda em relação à saída da layer norm (y).
                Shape (batch, seq_len, dim_model).
            x: Entrada original da layer norm (antes da normalização).
                Shape (batch, seq_len, dim_model).
            mean: Média de x ao longo da última dimensão (broadcastável).
                Shape (batch, seq_len, 1) ou algo que se alinhe.
            std: Desvio padrão (com eps) de x ao longo da última dimensão.
                Shape (batch, seq_len, 1).
            x_norm: Valor normalizado
                Shape (batch, seq_len, dim_model).
            gamma : Parâmetro de escala.

        Returns
            dx: Gradiente da perda em relação à entrada x.
                Shape (batch, seq_len, dim_model).
            dgamma: Gradiente da perda em relação a gamma.
                Shape igual a gamma (geralmente (dim_model,)).
            dbeta: Gradiente da perda em relação a beta.
                Shape igual a gamma (geralmente (dim_model,)).
        """
        N = x.shape[-1]
        dgamma = np.sum(dout * x_norm, axis=(0, 1))
        dbeta = np.sum(dout, axis=(0, 1))

        dnorm = dout * gamma
        dvar = -np.sum(dnorm * (x - mean) * 0.5 * std**(-3), axis=-1, keepdims=True)
        dmean = -np.sum(dnorm / std, axis=-1, keepdims=True) - 2 * dvar * (x - mean).mean(axis=-1, keepdims=True)
        dx = dnorm / std + dvar * 2 * (x - mean) / N + dmean / N
        return dx, dgamma, dbeta