from src.modelo.tensor.modelo_personalizado.modulos.atencao_multicabeca import AtencaoMulticabeca
from src.modelo.tensor.modelo_personalizado.modulos.feed_foward import FeedFoward

import numpy as np


class Tensor:
    def __init__(self, seq_len:int, num_heads:int, dim_k:int=0, dim_v:int=0, teste=True):
        self.mha = AtencaoMulticabeca(seq_len=seq_len, cabecas=num_heads, 
                                      dim_k=dim_k, dim_v=dim_v, teste=teste)
        self.feedfoward = FeedFoward(dim_model=seq_len, teste=teste)

    
    def converter_lista_para_tensor_shape(self, lista_tokens:list)->np.ndarray:           
        num_zeros = self.mha.seq_len - len(lista_tokens)
        tokens_np = np.array(lista_tokens) 
        #caso seja uma lista simples     
        if len(tokens_np.shape) == 1: 
            tokens_np = tokens_np.reshape(1, 1, len(tokens_np))
        else:
            tokens_np = tokens_np.reshape(tokens_np.shape[0], 1, tokens_np.shape[1])
            print('reshape parcial ', tokens_np.shape)
        arr_padded = np.pad(tokens_np, ((0, 0), (0, 0), (0, self.mha.seq_len-tokens_np.shape[2])), constant_values=0)
        return arr_padded
    
    @staticmethod
    def similaridade(x:np.ndarray, y:np.ndarray):
        dot = np.sum(x * y, axis=-1)          # (2, 10)

        # 2. Normas
        normA = np.linalg.norm(x, axis=-1)    # (2, 10)
        normB = np.linalg.norm(y, axis=-1)    # (2, 10)

        # 3. Cosseno da similaridade
        epsilon = 1e-8
        cos_sim = dot / (normA * normB + epsilon)

        print("Formato da similaridade:", cos_sim)  # (2, 10)
        return cos_sim[0]

    @staticmethod
    def rotacionar_por_cosseno(v, u, cosseno_alvo):
        """
        Rotaciona cada vetor no lote (batch, seq_len, dim) para que seu cosseno com
        o respectivo vetor em u seja igual a `cosseno_alvo`.
        Preserva a norma de cada v.

        Parâmetros:
            v : np.ndarray, shape (batch, seq_len, dim)
            u : np.ndarray, shape (batch, seq_len, dim)
            cosseno_alvo : float

        Retorna:
            v_rot : np.ndarray, mesma forma de v
        """
        # Normaliza u ao longo da última dimensão
        u_norm = u / np.linalg.norm(u, axis=-1, keepdims=True)  # (batch, seq_len, dim)
        
        # Norma de v (mantém a última dimensão como 1 para broadcast)
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)      # (batch, seq_len, 1)
        
        # Cosseno atual: (v · u_norm) / ||v||
        # Produto escalar: soma sobre dim (axis=-1)
        cos_atual = np.sum(v * u_norm, axis=-1, keepdims=True) / v_norm   # (batch, seq_len, 1)
        theta_atual = np.arccos(np.clip(cos_atual, -1, 1))
        theta_alvo = np.arccos(np.clip(cosseno_alvo, -1, 1))
        
        # Componente paralela a u: (v_norm * cos_atual) * u_norm
        v_paralelo = (v_norm * cos_atual) * u_norm          # (batch, seq_len, dim)
        
        # Componente perpendicular: v - v_paralelo
        v_perp = v - v_paralelo
        norma_perp = np.linalg.norm(v_perp, axis=-1, keepdims=True)   # (batch, seq_len, 1)
        
        # Trata vetores que estão exatamente na direção de u (norma_perp ≈ 0)
        # Para esses, cria uma direção perpendicular arbitrária
        mask = (norma_perp < 1e-8).squeeze(-1)   # bool, shape (batch, seq_len)
        if np.any(mask):
            # Cria um vetor auxiliar ortogonal a u_norm
            # Um jeito simples para qualquer dim: começa com vetor de uns e subtrai a projeção
            aux = np.ones_like(v)                # (batch, seq_len, dim)
            # Projeta aux em u_norm
            proj_aux = np.sum(aux * u_norm, axis=-1, keepdims=True) * u_norm
            aux_perp = aux - proj_aux
            norma_aux = np.linalg.norm(aux_perp, axis=-1, keepdims=True)
            aux_perp = aux_perp / (norma_aux + 1e-8)
            # Substitui v_perp e norma_perp para os índices onde mask é True
            v_perp = np.where(mask[..., None], aux_perp, v_perp)
            norma_perp = np.where(mask[..., None], 1.0, norma_perp)
        
        # Vetor unitário perpendicular
        v_perp_unit = v_perp / (norma_perp + 1e-8)
        
        # Novo vetor: mantém a mesma norma, mas com ângulo theta_alvo em relação a u
        v_rot = v_norm * (np.cos(theta_alvo) * u_norm + np.sin(theta_alvo) * v_perp_unit)
        return v_rot