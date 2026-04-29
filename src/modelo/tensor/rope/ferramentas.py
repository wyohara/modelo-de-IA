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
    
    @staticmethod
    def precompute_rope_cos_sin(seq_len, dim_head, base=10000.0):
        """
        Pré-computa as matrizes de cosseno e seno para RoPE.
        Retorna:
            cos: (seq_len, dim_head//2) ou (seq_len, dim_head) - valores de cosseno por par
            sin: (seq_len, dim_head//2) - valores de seno por par
        """
        # Garante que dim_head seja par para aplicação por pares; se ímpar, ignora o último
        half_dim = dim_head // 2
        if half_dim == 0:
            # dim_head = 1, não há pares; retorna arrays vazios (RoPE nulo)
            return np.zeros((seq_len, 0)), np.zeros((seq_len, 0))
        
        # Theta_i = base^(-2i/dim_head)
        theta = 1.0 / (base ** (np.arange(0, half_dim) / half_dim))   # (half_dim,)
        
        # posições: (seq_len, 1)
        positions = np.arange(seq_len).reshape(-1, 1)
        
        # ângulos: pos * theta  -> (seq_len, half_dim)
        angles = positions * theta
        
        cos = np.cos(angles)
        sin = np.sin(angles)
        return cos, sin
    
    @staticmethod
    def apply_rotary_emb(x, cos, sin):
        """
        Aplica a rotação RoPE ao tensor x (seq_len, dim_head).
        Se dim_head for ímpar, a última dimensão permanece inalterada.
        """
        d = x.shape[-1]
        half = d // 2
        if half == 0:
            return x.copy()
        
        # Separa pares (even, odd)
        x_even = x[..., 0::2]   # (seq_len, half)
        x_odd  = x[..., 1::2]   # (seq_len, half)
        
        # Trunca cos/sin para o número correto de pares
        cos = cos[:, :half]
        sin = sin[:, :half]
        
        # Rotação: (x_even, x_odd) -> (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        
        # Reconstitui o tensor na ordem original: par, ímpar, par, ímpar, ...
        y = np.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd
        return y