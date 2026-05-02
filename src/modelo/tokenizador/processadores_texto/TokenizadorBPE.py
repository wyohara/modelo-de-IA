from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

class TokenizadorBPE:
    def __init__(self, max_tam_token=20):
        '''
        Classe que aplica o BPE (Byte Pair Encode) e salva as quantidadades em um csv
        Params:
            max_tam_token: Tamanho máximo do token gerado pelo BPE, por padrão 20 (20 letras)
        '''
        self.__dataset = Path('src/media/dataset/')        
        self.__arquivo_lista_bpe = Path('src/media/dados_processados/lista_bpe.csv')
        self.max_tam_token = max_tam_token
    
    def processar_textos(self):      
        '''
        Método principal que carrega os datasets e para cada arquivo de texto aplica o bpe
        '''
        lista_arquivos = [f for f in self.__dataset.iterdir() if f.is_file()]

        #percorre a lista de arquivos e processa apenas os arquivos txt
        for arquivo in lista_arquivos:
            if arquivo.name.split('.')[-1] == 'txt':
                print(f">>> processando {arquivo.name}")
                self.__aplicar_bpe(arquivo)

    def __aplicar_bpe(self, arquivo:Path):
        '''
        Método principal que realiza o BPE:
            - carrega o texto e separa com split()
            - remove espaços em branco se houver
            - percorre com dois iteradores cada palavra cortando o texto e somando a quantidade
            Params:
                arquivo: caminho do arquivo de texto a ser lido
        '''

        lista_bpe = defaultdict()
        
        # carregando o arquivo para processar
        with open(arquivo,'r', encoding='utf-8') as arq:
            texto = arq.read().strip()
            texto = re.findall(r'\S+|\s*', texto)

        # ordena por tamanho do maior para o menor arquivo para otimizar o loop
        lista_texto = sorted(texto, key=len, reverse=True)

        #percorre cada palavra do texto
        for palavra in lista_texto:
            for i in range(len(palavra)):
                for j in range(min(len(palavra), self.max_tam_token)):
                    # Faz um slice da palavra
                    if i<=j:
                        subpalavra = palavra[i:j+1]
                        try:
                            lista_bpe[subpalavra]['quantidade'] += 1
                        except KeyError:
                            lista_bpe[subpalavra]=defaultdict()
                            lista_bpe[subpalavra]['quantidade'] = 1

        #Após percorrer o texto salva os dados
        self.salvar_e_incrementar_bpe(lista_bpe)     

    def salvar_e_incrementar_bpe(self, lista_bpe:defaultdict):
        '''
        Método que salva e incrementa as quantidades de subpalavras que existirem
        Params:
            lista_bpe: defaultdict contendo os dados a serem salvos
        '''
        arquivo_csv = str(self.__arquivo_lista_bpe)
        # Converte o defaultdict/dict para DataFrame
        df_novo = pd.DataFrame([
            {'subpalavra': k, 'quantidade': v['quantidade']}
            for k, v in lista_bpe.items()
        ])

        try:
            # Tenta ler o arquivo existente
            df_existente = pd.read_csv(arquivo_csv, encoding='utf-8')
            
            # Concatena os dois DataFrames
            df_combinado = pd.concat([df_existente, df_novo], ignore_index=True)
            
            # Agrupa por 'subpalavra' e agrega:
            df_resultado = df_combinado.groupby('subpalavra', as_index=False)['quantidade'].sum()

            #ordena o resultado por quantidade do maior para o menor
            df_resultado = df_resultado.sort_values('quantidade', ascending=False)
            
            df_resultado.to_csv(arquivo_csv, index=False, encoding='utf-8')

        except FileNotFoundError:
            df_novo.to_csv(arquivo_csv, index=False, encoding='utf-8')
            print(f"Arquivo criado: {arquivo_csv}")