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

    stop_words = nltk.corpus.stopwords.words("spanish")
    spanish_stemmer = nltk.stem.SnowballStemmer("spanish")

    #Definitorios
    formatted_definitorios=formatDicc(definitorios)
    list_definitoriosTuple=formatList(formatted_definitorios)
    list_definitoriosTokenized=tokenize(list_definitoriosTuple, stop_words, spanish_stemmer)
    #print(list_definitoriosTokenized)

    formatted_indicatorios = formatDicc(indicatorios)
    list_indicatoriosTuple = formatList(formatted_indicatorios)
    list_indicatoriosTokenized = tokenize(list_indicatoriosTuple, stop_words, spanish_stemmer)
    #print(list_indicatoriosTokenized)

    formatted_infAguda = formatDicc(infeccionAguda)
    list_infAgudaTuple = formatList(formatted_infAguda)
    list_infAgudaTokenized = tokenize(list_infAgudaTuple, stop_words, spanish_stemmer)
    #print(list_infAgudaTokenized)

    formatted_enfSexual = formatDicc(enfSexuales)
    list_enfSexualTuple = formatList(formatted_enfSexual)
    list_enfSexualTokenized = tokenize(list_enfSexualTuple, stop_words, spanish_stemmer)

    list_socioDemograficoTuple=formatList(sociodemografico)
    list_socioDemograficoTokenized=tokenize(list_socioDemograficoTuple, stop_words, spanish_stemmer)
    #print(list_socioDemograficoTokenized)

    string_stemmed=""
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
                tokenized=nltk.tokenize.word_tokenize(string_formatted)
                #print(tokenized)
                for tokens in tokenized:
                    if tokens not in stop_words:
                        string_stemmed+=spanish_stemmer.stem(tokens)+" "
                #print(list_stemmed)
                #print(string_stemmed)
                #Todo el tocho de código para comparar los diccionarios con el txt
                temp=0
                temp_string=""
                es_vih=False
                for tuple in list_definitoriosTokenized:
                    temp_string=listToString(tuple[0])
                    #print(temp_string)
                    #print(tuple[0])

                    if temp_string in string_stemmed:
                        es_vih=True
                        break
                if es_vih:
                    total+=1
                else:
                    temp = 0
                    prob = 0
                    #print(f'''Dir-> {directory}\t File-> {file}\tProb->{prob}\n\n''')
                    prob = getProb(list_indicatoriosTokenized, string_stemmed, spanish_stemmer)
                    #print(f'''ProbInd->{prob}\n\n''')
                    prob += getProb(list_infAgudaTokenized, string_stemmed, spanish_stemmer)
                    #print(f'''ProbAguda->{prob}\n\n''')
                    prob += getProb(list_enfSexualTokenized, string_stemmed, spanish_stemmer)
                    #print(f'''ProbSex->{prob}\n\n''')
                    #Sociodemográficos, diferente algoritmo
                    range_temp=""
                    range1=0
                    range2=0
                    for tuple in list_socioDemograficoTuple:

                        if "50" in tuple[0]:
                            for i in range(50,100):
                                if str(i)+" años" in string_formatted:
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
                string_stemmed=""
                #print(f"""Total {file}-> {total}""")
        result.append(total)
        total = 0
    media = 0
    percentaje=[]
    for i in range(0, 4):
        if i == 0:
            no_vih = 100 - ((result[i] * 100) / total_list[i])
            percentaje.append("%.2f" % no_vih)
            media += no_vih
        else:
            percentaje.append("%.2f" % ((result[i] * 100) / total_list[i]))
            media += ((result[i] * 100) / total_list[i])
    media = media / 4
    print(f"""Porcentaje correcto: {percentaje} \tMedia: {"%.2f" % media}""")
    print(f"""Falsos positivos: {"%.2f" % (100 - float(percentaje[0]))}""")
    print(f"""Falsos negativos: {"%.2f" % (100 - ((float)(percentaje[1]) + (float)(percentaje[2]) + (float)(percentaje[3])) / 3)}""")


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
def tokenize(t, stop_words, spanish_stemmer):
    list_tokenized=[]
    list_stemmed=[]
    for i in t:
        tokenized=nltk.tokenize.word_tokenize(i[0])
        for j in tokenized:
            if j not in stop_words:
                list_stemmed.append(spanish_stemmer.stem(j))
                #print(f"""Stemm-> {j}""")
        #print(list_stemmed)
        list_tokenized.append((list_stemmed, i[1]))
        list_stemmed=[]
    return list_tokenized

def getProb(list, string_stemmed, spanish_stemmer):
    temp=0
    temp_string=""
    prob=0
    for tuple in list:
        temp_string=listToString(tuple[0])
        if temp_string in string_stemmed:
            #print(temp_string)
            #print(string_stemmed)
            if(temp_string == 'fatigmalestasteni' or temp_string == 'cefale'
                        or temp_string == 'linfadenopatiperiferadenopati' or temp_string == 'faringitis'
                        or temp_string == 'altergastrointestinaldiarre' or temp_string == 'mononucleosis'
                        or temp_string == 'sindrommononuclefiebradenopatimialgi') and (
                        spanish_stemmer.stem("linfopenia")) + " <500" in string_stemmed:
                prob+=1

            elif (spanish_stemmer.stem("leucopenia") in string_stemmed
                        and spanish_stemmer.stem("trombopenia") in string_stemmed):
                prob+=1
            prob += (float)(tuple[1])

    return prob

def listToString(list):
    string=""
    for i in list:
        string+=i+" "
    return string

if __name__== "__main__":
    baseline3()

