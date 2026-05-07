import pandas as pd

with open ('src/media/dataset/embedding/lx_wordSim_353.txt', 'r', encoding='utf-8') as f:
    texto = f.read()
    texto = texto.replace("\n",'\t')
    t =texto.strip().split("\t")
    nt = []
    for i in range(0,len(t)-1,4):
        if i >4 :
            t[i-1] = float(t[i-1].strip())
        nt.append(t[i-4:i])

for i in nt:
    print(i)
