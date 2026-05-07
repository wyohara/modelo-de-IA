import numpy as np

from src.modelo.transformer.modulos.ferramentas import gradiente_perda_mse, gradiente_perda_cross_entropy


class TransformerUtils:
    @staticmethod
    def similaridade(x:np.array, y:np.array):
        dot = np.sum(x * y, axis=-1)          # (2, 10)

        # 2. Normas
        normA = np.linalg.norm(x, axis=-1)    # (2, 10)
        normB = np.linalg.norm(y, axis=-1)    # (2, 10)

        # 3. Cosseno da similaridade
        epsilon = 1e-8
        cos_sim = dot / (normA * normB + epsilon)
        return cos_sim[0]
    
    @staticmethod
    def perda_mse(x:np.array, y:np.array):
        return gradiente_perda_mse(x, y)
    
    @staticmethod
    def perda_mse(x:np.array, y:np.array):
        return gradiente_perda_cross_entropy(x, y)