import pandas as pd
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2




with open('D:\ProjetoIntegradoPython\censo_2000_2010_2022.CSV',mode='r') as arq:
    leitor = csv.reader(arq, delimiter=';')
    linhas = 0
    linhas2 = 0
    popTotal2000 = 0
    popTotal2010 = 0
    popTotal2022 = 0
    qtdeMunicipios = 0

    listaMunicipio = []
    listaCenso2000 = []
    listaCenso2010 = []
    listaCenso2022 = []
 
    listaTaxaCrescPop0010 = []
    listaTaxaCrescPop1022 = []

    listaPorcRepEstado2000 = []
    listaPorcRepEstado2010 = []
    listaPorcRepEstado2022 = []


    for coluna in leitor:
    
    
    
        # if linhas == 0:
            # f'Colunas: - f usado para formatar a string
            # print(f'Colunas: {" ".join(coluna)}')
            # linhas += 1
        # else:
            # coluna[0] = está na primeira coluna da linha
            # print(f'\tElemento {coluna[0]} é o {coluna[1]}, de simbolo {coluna[2]}')
            # linhas += 1 - incremento para a proxima linha
            # linhas += 1
        # print(f'Lidas {linhas} linhas')
        
        
       # print(f'{coluna[0]} {coluna[1]} {coluna[2]} {coluna[3]}')
        municipio = coluna[0]
        censo2000 = coluna[1]
        censo2010 = coluna[2]
        censo2022 = coluna[3]

        listaMunicipio.append(municipio)
        #calculo populacao para cada censo
        popTotal2000 = popTotal2000 + int(censo2000)
        popTotal2010 = popTotal2010 + int(censo2010)
        popTotal2022 = popTotal2022 + int(censo2022)
        qtdeMunicipios = qtdeMunicipios + 1

        linhas += 1

   #  print(linhas)

    #print(popTotal2000) 



with open('D:\ProjetoIntegradoPython\censo_2000_2010_2022.CSV',mode='r') as arq:
    leitor2 = csv.reader(arq, delimiter=';')

    taxaCresc0010 = 0
    taxaCresc1022 = 0

    calcMediaCresc0010 = 0
    calcMediaCresc1022 = 0

    for coluna2 in leitor2:

        # print(f'{coluna[0]} {coluna[1]} {coluna[2]} {coluna[3]}')
        municipio = coluna2[0]
        censo2000 = coluna2[1]
        censo2010 = coluna2[2]
        censo2022 = coluna2[3]


        calcTaxaCresc0010 = (float(censo2010) * 100) / float(censo2000) - 100
        taxaCresc0010 = str(calcTaxaCresc0010)[:6]
        #Calculo Media Grafico 2000 2010
        calcMediaCresc0010 = calcMediaCresc0010 + float(taxaCresc0010)

        calctaxaCresc1022 = (float(censo2022) * 100) / float(censo2010) - 100
        taxaCresc1022 = str(calctaxaCresc1022)[:6]
        #Calculo Media Grafico 2010 2020
        calcMediaCresc1022 = calcMediaCresc1022 + float(taxaCresc1022)

        calcPorcPop2000 = ((int(censo2000) * 100) / int(popTotal2000))
        PorcentagemPop2000 = str(calcPorcPop2000)[:6]

        calcPorcPop2010 = (int(censo2010) * 100) / int(popTotal2010)
        PorcentagemPop2010 = str(calcPorcPop2010)[:6]

        calcPorcPop2022 = (int(censo2022) * 100) / int(popTotal2022)
        PorcentagemPop2022 = str(calcPorcPop2022)[:6]

        listaCenso2000.append(censo2000)
        listaCenso2010.append(censo2010)
        listaCenso2022.append(censo2022)

        listaPorcRepEstado2000.append(PorcentagemPop2000)
        listaPorcRepEstado2010.append(PorcentagemPop2010)
        listaPorcRepEstado2022.append(PorcentagemPop2022)

        listaTaxaCrescPop0010.append(taxaCresc0010)
        listaTaxaCrescPop1022.append(taxaCresc1022)


tabelaCenso = pd.DataFrame(
    data = zip(listaMunicipio,listaCenso2000,listaCenso2010,listaCenso2022,listaPorcRepEstado2000,listaPorcRepEstado2010,listaPorcRepEstado2022,listaTaxaCrescPop0010,listaTaxaCrescPop1022), 
    columns=["Municipio","Censo 2000","Censo 2010","Censo 2022","% Rep População comparando com o total do Estado 2000","% Rep População comparando com o total do Estado 2010","% Rep População comparando com o total do Estado 2022","Taxa Crescimento entre 2000 - 2010","Taxa Crescimento entre 2010 - 2022"]
    )

tabelaCenso.sort_values(by='Censo 2000')

tabelaCenso.to_excel("TabelaCensoConsolidado.xlsx")


#Calculo Media Crescimento entre 2000 2010

print(calcMediaCresc0010)
print(linhas)
mediaCresc0010 = calcMediaCresc0010 / linhas

x = np.arange(11)

refPorCresc  = [float(listaTaxaCrescPop0010[0]),float(listaTaxaCrescPop0010[1]),float(listaTaxaCrescPop0010[2]),float(listaTaxaCrescPop0010[3]),float(listaTaxaCrescPop0010[4]),float(listaTaxaCrescPop0010[5]),float(listaTaxaCrescPop0010[6]),float(listaTaxaCrescPop0010[7]),float(listaTaxaCrescPop0010[8]),float(listaTaxaCrescPop0010[9]),float(listaTaxaCrescPop0010[10])]
refMedia0010 = [mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010]

width = 0.2
plt.bar(x-0.2, refPorCresc, width, color='cyan')
plt.bar(x, refMedia0010, width, color='orange')
plt.xticks(x,[str(listaMunicipio[0]), str(listaMunicipio[1]), str(listaMunicipio[2]), str(listaMunicipio[3]), str(listaMunicipio[4]), str(listaMunicipio[5]), str(listaMunicipio[6]), str(listaMunicipio[7]), str(listaMunicipio[8]), str(listaMunicipio[9]), str(listaMunicipio[10])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2000-2010")

barra_labelMun = plt.bar(x-0.2, refPorCresc, width)
barra_labelMedia = plt.bar(x, refMedia0010, width)
plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.show()

#Calculo Media Crescimento entre 2010 2022

print(calcMediaCresc1022)
print(linhas)
mediaCresc1022 = calcMediaCresc0010 / linhas



x = np.arange(11)

refPorCresc1022  = [float(listaTaxaCrescPop1022[0]),float(listaTaxaCrescPop1022[1]),float(listaTaxaCrescPop1022[2]),float(listaTaxaCrescPop1022[3]),float(listaTaxaCrescPop1022[4]),float(listaTaxaCrescPop1022[5]),float(listaTaxaCrescPop1022[6]),float(listaTaxaCrescPop1022[7]),float(listaTaxaCrescPop1022[8]),float(listaTaxaCrescPop1022[9]),float(listaTaxaCrescPop1022[10])]
refMedia1020 = [mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022]

width = 0.2
plt2.bar(x-0.2, refPorCresc1022, width, color='cyan')
plt2.bar(x, refMedia0010, width, color='orange')
plt2.xticks(x,[str(listaMunicipio[0]), str(listaMunicipio[1]), str(listaMunicipio[2]), str(listaMunicipio[3]), str(listaMunicipio[4]), str(listaMunicipio[5]), str(listaMunicipio[6]), str(listaMunicipio[7]), str(listaMunicipio[8]), str(listaMunicipio[9]), str(listaMunicipio[10])])
plt2.xlabel("Municípios")
plt2.ylabel("Taxa crescimento 2010-2022")

barra_labelMun = plt.bar(x-0.2, refPorCresc1022, width)
barra_labelMedia = plt.bar(x, refMedia1020, width)
plt2.bar_label(barra_labelMun)
plt2.bar_label(barra_labelMedia)
plt2.show()

print("Planilha gerado com sucesso!!")