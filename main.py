import os
import string
i, j=0, 0
files=[[],[],[],[]]
percentaje=[]

diccionario=""
with open("./resources/diccionario.txt", encoding="utf-8") as dic:
    diccionario=dic.read()
replace_from="áéíóúÁÉÍÓÚ"
replace_to="aeiouAEIOU"
trans_table=diccionario.maketrans(replace_from, replace_to)
string_replace=diccionario.translate(trans_table)
string_final=string_replace.lower()

#function that gives a value of VIH if res >5 have VIH
result=[]
string_temp=""
res1=0
res2=0
string_list=string_final.split("\n")
i=0
for directory in os.listdir("./resources/dataset"):
    for file in os.listdir("./resources/dataset/"+directory):
        if file.endswith(".txt") or file.endswith(".json") or file.endswith(".csv"):
            with open("./resources/dataset/" + directory + "/" + file, encoding="utf-8") as f:
                string_temp = f.read()
            for z in string_list:
                if (z in string_temp):
                    res1 += 1
            if (res1 > 0):
                res2 += 1

            print(f"({file}, {res1}, {directory})")
            res1 = 0
    result.append(res2)
    res2 = 0
    percentaje.append(j)
print(result)





