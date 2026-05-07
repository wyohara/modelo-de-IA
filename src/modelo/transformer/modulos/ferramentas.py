import numpy as np

def normalizador_camada(x, gamma:np.array, beta:np.array, eps=1e-6):
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


def normalizador_camada_backward(x_corrigido:np.array, dy:np.array, gamma:np.array, eps=1e-6):
    '''
    Backward do normalizador de camada
        dy: gradiente de perda em relação à saída
            shape (batch, seq_len, d_model)
        x: entrada original
            shape (batch, seq_len, d_model)
        gamma: Fator de correção gamma usado
    '''
    N = x_corrigido.shape[-1]
    media = np.mean(x_corrigido, axis=-1, keepdims=True)
    var = np.var(x_corrigido, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    x_scoreZ = (x_corrigido - media) / std
    dbeta = np.sum(dy, axis=(0, 1))
    dgamma = np.sum(dy * x_scoreZ, axis=(0, 1))

    dx_hat = dy * gamma          # shape broadcast
    # derivada em relação a x
    dx = (1. / N / std) * (N * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - x_scoreZ * np.sum(dx_hat * x_scoreZ, axis=-1, keepdims=True))
    return dx, dgamma, dbeta


def gradiente_perda_mse(resultado:np.array, rotulos_teste:np.array):
    """
    Gera o gradiente de perda d_saida em relação ao rotulo.
    Usada onde a saída é um valor contínuo e erros com distribuição 
        gaussiana (maximiza a verossimilhança sob normalidade).
    Penaliza grandes erros quadraticamente. É mais sensível a outliers.
    Use MSE quando seu objetivo é prever um número real (regressão).
    Params:
        rotulos_teste: resultado esperado pela rede neural
        resultado: resultado da rede neural feed foward normalizada
    """
    #recupera os parametros da ultima ativação
    batch, seq_len, dim_model = resultado.shape
    # Recupera o número total de tokens
    N = batch * seq_len  
    #Aplica o MSE
    gradiente_perda = 2 * (resultado - rotulos_teste) / (N * dim_model)
    return gradiente_perda


def softmax(x:np.array, axis=-1):
    """Aplica o softmax para normalizar os valores em valores entre 0 e 1."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def gradiente_perda_cross_entropy(resultado:np.array, rotulos_teste:np.array):
        '''
        Usada em classificação
            a saída é uma distribuição de probabilidades sobre classes discretas.
        Maximiza a log‑verossimilhança de uma distribuição multinomial.
        Não é adequada para valores contínuos (exige probabilidades).
        Penaliza principalmente previsões confiantes erradas (log de probabilidade muito negativa).
        Params:
            rotulos_teste: resultado esperado pela rede neural
            resultado: resultado da rede neural feed foward normalizada
        '''
        #recupera os parametros da ultima ativação
        batch_size, seq_len, dim_model = resultado.shape

        logits_reshape = resultado.reshape(-1, dim_model)
        max_logits = np.max(logits_reshape, axis=1, keepdims=True)        
        probs = softmax(logits_reshape - max_logits).reshape(resultado.shape)

        # Converter rotulos_teste para one‑hot se necessário
        if rotulos_teste.shape != resultado.shape:
            y_one_hot = np.zeros_like(resultado)
            y_one_hot[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], rotulos_teste] = 1
        else:
            y_one_hot = rotulos_teste
        
        gradiente_perda = probs - y_one_hot
        return gradiente_perda
