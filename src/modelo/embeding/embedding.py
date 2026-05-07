from src.modelo.transformer.transformer import Transformer

class Embedding:
    def __init__(self):
        self.dataset_emebedding = []
        self.transformer = Transformer(dim_model=50,num_heads=5)
        
    def carregar_dataset(self):
        with open ('src/media/dataset/embedding/lx_wordSim_353.txt', 'r', encoding='utf-8') as f:
            texto = f.read()
            texto = texto.replace("\n",'\t')
            t =texto.strip().split("\t")
            for i in range(0,len(t)-1,4):
                if i >4 :
                    t[i-1] = float(t[i-1].strip())
                self.dataset_emebedding.append(t[i-4:i])
    
    
