

import pdfplumber
with pdfplumber.open('resources/Glossary-Spanish-HIVinfo.pdf') as pdf:
    text = pdf.pages[6:206]
    bold_text=""
    for i in range(len(text)):
        bold_text =text[i].filter(lambda obj: obj["object_type"] == "char" and "Bold" in obj["fontname"])
        final_text=bold_text.extract_text().split("\n")
        for j in final_text:
            if "VEA:" not in j and "SINÓNIMO(S):" not in j and j not in "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ" and not j.isnumeric():
                with open("resources/diccionario.txt", "a", encoding="utf-8") as f:
                    f.write(j)
                    f.write("\n")








