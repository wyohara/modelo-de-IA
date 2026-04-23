import numpy as np


class Ferramentas():

    @staticmethod
    def layer_norm( x:np, eps=1e-5):
        """Normalização de camada simples: subtrai média e divide pelo desvio padrão."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    @staticmethod
    def softmax(x, axis=-1):
        """Softmax estável."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def layer_norm_backward(dout, x:np, eps=1e-5):
        """Backward simplificado da layer norm."""
        mu = x.mean(axis=-1, keepdims=True) #média da ultima dimensão
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True) # calcula a variância
        desvio_padrao = np.sqrt(var + eps)
        dnorm = dout / desvio_padrao
        dmean = -dnorm.mean(axis=-1, keepdims=True)
        dvar = - (dout * ((x - mu) / desvio_padrao)).mean(axis=-1, keepdims=True) / desvio_padrao
        dx = dnorm + dmean + dvar * (x - mu) * 2 / x.shape[-1]
        return dx