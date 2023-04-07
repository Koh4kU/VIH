import os
import string
i, j=0, 0
files=[[],[],[],[]]
percentaje=[]
for directory in os.listdir("./resources/dataset"):
    j=0
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

#function that gives a value of VIH negative=no, 0=maybe, positive=yes
result=[]
string_temp=""
res1=0
res2=0
string_list=string_final.split("\n")
i=0
for directory in os.listdir("./resources/dataset"):
    for j in files[i]:
        if(j is ("*.txt" or "*.csv" or "*.json" or "*.xls*" or "*.word*")):
            with open("./resources/dataset/"+directory+"/"+j) as f:
                string_temp=f.read()
            for z in string_list:
                if(z in string_temp):
                    res1+=1
                else:
                    res1-=1
            if(res1>0):
                res2+=1
            res1=0
    result.append(res2)
    res2=0
    i+=1
print(result)




