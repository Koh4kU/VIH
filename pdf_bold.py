import PyPDF2

read_pdf = PyPDF2.PdfReader("resources/Glossary-Spanish-HIVinfo.pdf")

page=read_pdf.pages[5]
print(page)
item=page["/PieceInfo"]['/InDesign']['/PageItemUIDToLocationDataMap']
print(page.extract_text())
for i in range(len(item)):
    print(item["/"+str(i)])

'''
page=read_pdf.pages[4:6]
items=[]
for i in page:
    items.append(i["/PieceInfo"]['/InDesign']['/PageItemUIDToLocationDataMap'])

print(items)


for i in range(len(items)):
    for j in range(len(items[i])):
        print(items[i][j]["/"+str(j)])
'''



