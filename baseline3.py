import nltk
import os
#nltk.download()
def baseline3():
    with open("./resources/sintomas_definitoriosF", encoding="Utf-8") as f:
        definitorios=f.read()
    with open("./resources/sintomas_indicatoriosF", encoding="Utf-8") as f:
        indicatorios=f.read()
    with open("./resources/sintomas_infeccionAgudaF", encoding="Utf-8") as f:
        infeccionAguda=f.read()
    with open("./resources/enfermedades_sexualesF", encoding="Utf-8") as f:
        enfSexuales=f.read()
    with open("./resources/sociodemograficosF", encoding="Utf-8") as f:
        sociodemografico=f.read()


    #Definitorios
    formatted_definitorios=formatDicc(definitorios)
    list_definitoriosTuple=formatList(formatted_definitorios)
    #print(list_definitoriosTokenized)

    formatted_indicatorios = formatDicc(indicatorios)
    list_indicatoriosTuple = formatList(formatted_indicatorios)
    #print(list_indicatoriosTokenized)

    formatted_infAguda = formatDicc(infeccionAguda)
    list_infAgudaTuple = formatList(formatted_infAguda)
    #print(list_infAgudaTokenized)

    formatted_enfSexual = formatDicc(enfSexuales)
    list_enfSexualTuple = formatList(formatted_enfSexual)

    list_socioDemograficoTuple=formatList(sociodemografico)
    #print(list_socioDemograficoTokenized)

    total = 0
    total_list=[]
    result=[]
    for directory in os.listdir("./resources/dataset"):
        directories = os.listdir("./resources/dataset/" + directory)
        total_list.append(len(directories))
        for file in directories:
            if file.endswith(".txt") or file.endswith(".json") or file.endswith(".csv"):
                with open("./resources/dataset/" + directory + "/" + file, encoding="utf-8") as f:
                    string_temp = f.read().lower()
                string_formatted=formatDicc(string_temp)

                #Todo el tocho de código para comparar los diccionarios con el txt
                temp=0
                temp_string=""
                es_vih=False
                for tuple in list_definitoriosTuple:
                    temp_string=listToString(tuple[0])
                    #print(temp_string)
                    #print(tuple[0])

                    if temp_string in string_formatted:
                        es_vih=True
                        break
                if es_vih:
                    total+=1
                else:
                    temp = 0
                    prob = 0
                    #print(f'''Dir-> {directory}\t File-> {file}\tProb->{prob}\n\n''')
                    prob = getProb(list_indicatoriosTuple, string_formatted)
                    #print(f'''ProbInd->{prob}\n\n''')
                    #print(f'''ProbAguda->{prob}\n\n''')
                    prob += getProb(list_enfSexualTuple, string_formatted)
                    #print(f'''ProbSex->{prob}\n\n''')

                    # Solo comprobar las infecciones si hay enfermedades anteriores
                    if prob != 0:
                        prob += getProb(list_infAgudaTuple, string_formatted)

                    #Sociodemográficos, diferente algoritmo
                    range_temp=""
                    range1=0
                    range2=0
                    for tuple in list_socioDemograficoTuple:

                        if "50" in tuple[0]:
                            for i in range(50, 100):
                                if str(i) + " años" in string_formatted:
                                    prob += float(tuple[1])
                        elif "50" not in tuple[0]:
                            range_temp = tuple[0]
                            range1 = (int)(range_temp[0:2])
                            range2 = (int)(range_temp[3:5])
                            for i in range(range1, range2):
                                if str(i) + " años" in string_formatted:
                                    prob += float(tuple[1])
                                    break
                    if prob > 7:
                        total += 1
                string_formatted=""
                #print(f"""Total {file}-> {total}""")
        result.append(total)
        total = 0
    media = 0
    percentaje=[]
    true_negatives = (total_list[0] - result[0])
    true_positives = sum(result[1:])
    accuracy = (true_positives + true_negatives) / sum(total_list)

    false_negatives = sum(total_list[1:]) - sum(result[1:])
    false_positive = result[0]

    precision=true_positives/(true_positives+false_positive)
    recall=true_positives/(true_positives+false_negatives)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(
        f"""True positive-> {true_positives / sum(total_list)}\tTrue negative-> {true_negatives / sum(total_list)}\tFalse negatives -> {false_negatives / sum(total_list)}\t False positive->{false_positive / sum(total_list)}""")
    print(f"""Accuracy -> {accuracy}\tPrecision-> {precision}\tRecall-> {recall}\tF1 score-> {f1_score}""")


def formatDicc(s):
    replace_from = "áéíóúÁÉÍÓÚ(),:;"
    replace_to = "aeiouAEIOU     "
    trans_table = s.maketrans(replace_from, replace_to)
    string_replace = s.translate(trans_table)
    string_final = string_replace.lower()
    #string_final=s.lower()
    return string_final
def formatList(s):
    tuple_list = []
    list=[]
    list = s.split("\n")
    tuple_list = [(list[i], list[i + 1]) for i in range(0, len(list) - 1, 2)]
    return tuple_list

def getProb(list, string_formatted):
    temp=0
    temp_string=""
    prob=0
    for tuple in list:
        temp_string=listToString(tuple[0])
        if temp_string in string_formatted:
            #print(temp_string)
            #print(string_stemmed)
            if(temp_string == 'fatigmalestasteni' or temp_string == 'cefale'
                        or temp_string == 'linfadenopatiperiferadenopati' or temp_string == 'faringitis'
                        or temp_string == 'altergastrointestinal', 'diarre' or temp_string == 'mononucleosis'
                        or temp_string == 'sindrommononuclefiebradenopatimialgi') and (
                        "linfopenia" + " <500") in string_formatted:
                prob+=1

            elif ("leucopenia" in string_formatted
                        and "trombopenia" in string_formatted):
                prob+=1
            prob += (float)(tuple[1])

    return prob

def listToString(list):
    string=""
    for i in list:
        string+=i
    return string

if __name__== "__main__":
    baseline3()

