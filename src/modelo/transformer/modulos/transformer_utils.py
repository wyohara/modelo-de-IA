import numpy as np


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

        print("Formato da similaridade:", cos_sim)  # (2, 10)
        return cos_sim[0]