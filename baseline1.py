import os
import string
i, j=0, 0
files=[[],[],[],[]]
percentaje=[]
total=[]
diccionario=""
with open("resources/diccionario1.txt", encoding="utf-8") as dic:
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
    directories=os.listdir("./resources/dataset/"+directory)
    total.append(len(directories))
    for file in directories:
        if file.endswith(".txt") or file.endswith(".json") or file.endswith(".csv"):

            with open("./resources/dataset/" + directory + "/" + file, encoding="utf-8") as f:
                string_temp = f.read().lower()
                trans_table2 = string_temp.maketrans(replace_from, replace_to)
                string_cases = string_temp.translate(trans_table)
            for z in string_list:
                if (z in string_cases and ("vih" in z or "sida" == z or "hiv" == z)):
                    res1+=10
                    #print(f"({file}, {res1}, {directory})")
                    #print("Matcheo-> " + z + " <-")
                elif z in string_cases:
                    res1 += 1
                    #print(f"({file}, {res1}, {directory})")
                    #print("Matcheo-> "+z+" <-")
            if (res1 >= 10):
                res2 += 1

            res1 = 0
    result.append(res2)
    res2 = 0
#print("Con VIH: "+ str(result))
#print("Total arhcivos: " +str(total))
media=0
for i in range(0,4):
    if i == 0:
        no_vih=100-((result[i] * 100) / total[i])
        percentaje.append("%.2f" % no_vih)
        media+=no_vih
    else:
        percentaje.append("%.2f" % ((result[i] * 100) / total[i]))
        media+=((result[i] * 100) / total[i])
media=media/4
print(f"""Porcentaje correcto: {percentaje} \tMedia: {"%.2f" % media}""")
print(f"""Falsos positivos: {"%.2f" % (100-float(percentaje[0]))}""")
print(f"""Falsos negativos: {"%.2f" % (100-((float)(percentaje[1])+(float)(percentaje[2])+(float)(percentaje[3]))/3)}""")






