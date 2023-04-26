with open("values.txt") as f:
    s=f.read()
replace=s.replace(",", ".")
print(replace)
with open("enfermedades_sexuales", encoding="Utf-8") as vih:
    s=vih.read()

split=s.split("\n")
split2=replace.split("\n")
final=""
for i in range(len(split)):
    final = final + split[i] + "\n" + split2[i] + "\n"


with open("enfermedades_sexualesF", "w", encoding="Utf-8") as write:
    write.write(final)