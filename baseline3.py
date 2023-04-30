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
    print(list_definitoriosTokenized)

    list_stemmed=[]
    for directory in os.listdir("./resources/dataset"):
        directories = os.listdir("./resources/dataset/" + directory)
        for file in directories:
            if file.endswith(".txt") or file.endswith(".json") or file.endswith(".csv"):
                with open("./resources/dataset/" + directory + "/" + file, encoding="utf-8") as f:
                    string_temp = f.read().lower()
                string_formatted=formatDicc(string_temp)
                tokenized=nltk.tokenize.word_tokenize(string_formatted)
                #print(tokenized)
                for tokens in tokenized:
                    if tokens in stop_words:
                        tokenized.remove(tokens)
                    else:
                        list_stemmed.append(spanish_stemmer.stem(tokens))
                print(list_stemmed)
                #Todo el tocho de código para comparar los diccionarios con el txt
                list_stemmed=[]

def formatDicc(s):
    replace_from = "áéíóúÁÉÍÓÚ"
    replace_to = "aeiouAEIOU"
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
            if j in stop_words:
                #print(f"""Token-> {j}""")
                tokenized.remove(j)
            else:
                list_stemmed.append(spanish_stemmer.stem(j))
                #print(f"""Stemm-> {j}""")
        #print(list_stemmed)
        list_tokenized.append((list_stemmed, i[1]))
        list_stemmed=[]
    return list_tokenized


if __name__== "__main__":
    baseline3()

