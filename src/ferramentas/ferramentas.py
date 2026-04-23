
import time
from functools import wraps

def texto_para_hex(texto:str):
    if type(texto)== str:
        return texto.encode('utf-8').hex()
    else:
        raise TypeError

def hex_para_texto(texto_hex):
    return bytes.fromhex(texto_hex.replace("#",'')).decode('utf-8',errors='surrogateescape')


def medir_tempo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        print(f"Função '{func.__name__}' executou em {fim - inicio:.4f} s.\n")
        return resultado
    return wrapper