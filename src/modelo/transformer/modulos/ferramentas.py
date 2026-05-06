import numpy as np

def normalizador_camada(x, eps=1e-6, gamma=1, beta=0):
    """
    Função de normalização da camada de atenção normalizado, média e desvio padrão.
    Ele calcula o score z, que calcula quantos desvios padrão está da média
        Params:
            x: Tensor que será normalizado
            gamma: fator de escala do vetor
            beta: fator de deslocamento do vetor
            eps: valor pequeno para evitar divisão por zero no std
            
        Returns:
            x: Tensor utilizado na normalização
            x_corrigido: Tensor normalizado com a correção  beta e gamma
            media: média dos valores do Tensor
            dp: desvio padrão do Tensor em relação a média
            x_scoreZ: Tensor normalizado com o score z
            gamma: valor original do fator de correção gamma
    """
    media = x.mean(axis=-1, keepdims=True)
    # A variancia obtida a partir da média
    variancia = ((x - media) ** 2).mean(axis=-1, keepdims=True)
    dp = np.sqrt(variancia + eps)
    x_scoreZ = (x - media) / dp
    x_corrigido = gamma * x_scoreZ + beta
    return x, x_corrigido, media, dp, x_scoreZ, gamma


def normalizador_camada_backward(dy, x, gamma, eps=1e-5):
    '''
    Backward do normalizador de camada
        dy: gradiente de perda em relação à saída
            shape (batch, seq_len, d_model)
        x: entrada original
            shape (batch, seq_len, d_model)
        gamma: Fator de correção gamma usado
    '''
    N = x.shape[-1]
    media = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    x_scoreZ = (x - media) / std

    dbeta = np.sum(dy, axis=(0, 1))
    dgamma = np.sum(dy * x_scoreZ, axis=(0, 1))

    dx_hat = dy * gamma          # shape broadcast
    # derivada em relação a x
    dx = (1. / N / std) * (N * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - x_scoreZ * np.sum(dx_hat * x_scoreZ, axis=-1, keepdims=True))
    return dx, dgamma, dbeta

def relu(x):
    return np.maximum(0, x)