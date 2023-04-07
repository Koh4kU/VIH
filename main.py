import os
import string
i, j=0, 0
files=[[],[],[],[]]
percentaje=[]
for directory in os.listdir("./resources/dataset"):
    for file in os.listdir("./resources/dataset/"+directory):
        files[i].append(file)
        j+=1
    percentaje.append(j)
    i+=1
diccionario=""
with open("./resources/diccionario.txt", encoding="utf-8") as dic:
    diccionario=dic.read()
replace_from="áéíóúÁÉÍÓÚ"
replace_to="aeiouAEIOU"
trans_table=diccionario.maketrans(replace_from, replace_to)
string_replace=diccionario.translate(trans_table)
string_final=string_replace.lower()
print(string_final)

