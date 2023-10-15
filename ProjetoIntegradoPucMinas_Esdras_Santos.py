
# Inicializacao
#numpy - usado para processamento numérico
#pandas - usado para manipulação de bases de dados
#pyplot - usado para visualização de dados
#seaborn - usado para visualização de dados
 

import pandas as pd
import numpy as np

import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns


#processamento dos dados e particionamento em base de teste (train_test_split)
from sklearn.model_selection import train_test_split

# processamento de alguns dados (preprocessing)
from sklearn import preprocessing
# analise de transformação de dados
from sklearn.linear_model import LinearRegression

import csv

np.set_printoptions(threshold=None, precision=2)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.precision',2)




# ========== Funcao para Classificar o Porte do Municipios ==========
# Pequeno: até 100 mil habitantes
# Media - de 100 mil até 500 mil habitantes
# Grande - de 500 mil até 3 milhões habitante
# Metropole - acima de 3 milhões habitante

def ClassificacaoPorteMunicipio(PopulacaoRef):

    if (PopulacaoRef <= 100000):
        ClassifMunicipio = "Pequeno"
    elif (PopulacaoRef >= 100001) and (PopulacaoRef <= 500000):
        ClassifMunicipio = "Medio"
    elif (PopulacaoRef >= 500001) and (PopulacaoRef <= 3000000):
        ClassifMunicipio = "Grande"
    else:
        ClassifMunicipio = "Metropole"
    
    return ClassifMunicipio

# ====================================================================



with open('D:\ProjetoIntegradoPython\censo_2000_2010_2022_atualizado.CSV',mode='r') as arq:
    leitor = csv.reader(arq, delimiter=';')

    
    #variaveis Globais

    popTotal2000 = 0
    popTotal2010 = 0
    popTotal2022 = 0
    popTotalEstimada2032 = 0
    popTotalEstimada2042 = 0
    qtdeMunicipios = 0


    listaMunicipio = []
    listaCenso2000 = []
    listaCenso2010 = []
    listaCenso2022 = []
    listaCresc2032 = []
    listaCresc2042 = []
    
    listaClassifMunicipio2000 = []
    listaClassifMunicipio2010 = []
    listaClassifMunicipio2022 = []
    listaClassifMunicipio2032 = []
    listaClassifMunicipio2042 = []


    listaTaxaCrescPop0010 = []
    listaTaxaCrescPop1022 = []
    listaTaxaCrescPop2232 = []
    listaTaxaCrescPop3242 = []
    
    listaPorcRepEstado2000 = []
    listaPorcRepEstado2010 = []
    listaPorcRepEstado2022 = []
    listaPorcRepEstimadoEstado2032 = []
    listaPorcRepEstimadoEstado2042 = []

    taxaCresc2232 = 0
    taxaCresc3242 = 0
    calcMediaCresc2232 = 0
    calcMediaCresc3242 = 0

    listapopTotalEstimada2032 = []
    listapopTotalEstimada2042 = []

    for coluna in leitor:
          
        
       # print(f'{coluna[0]} {coluna[1]} {coluna[2]} {coluna[3]}')
        municipio = coluna[0]
        censo2000 = coluna[1]
        censo2010 = coluna[2]
        censo2022 = coluna[3]

# =================================================previsao da populacao ===============================================
        # Calculo para estimativa da populacao em 2032 e 2042

         # Dados fornecidos
        anos = np.array([2000, 2010, 2012]).reshape(-1, 1)  # reshape para tornar 'anos' uma matriz bidimensional
        populacoes = np.array([censo2000, censo2010, censo2022])

        # Criar um modelo de regressão linear
        modelo = LinearRegression()

        
        # Treinar o modelo
        modelo.fit(anos, populacoes)

        # Fazer previsões para os anos fornecidos
        anos_para_prever = np.array([2032, 2042]).reshape(-1, 1)  # Anos para os quais você deseja prever a população
        previsoes = modelo.predict(anos_para_prever)
        
        listaPrev = []
        listaPrev = str(previsoes).replace("[ ","").replace("[","").replace(" ]","").replace("]","").split()

        prev2032_str = listaPrev[0]
        #prev2032 = prev2032_str.split('.')[0]
        prev2032 = previsoes[0]

        prev2042_str = listaPrev[1]
        #prev2042 = prev2042_str.split('.')[0]
        prev2042 = previsoes[1]


        listaCresc2032.append(prev2032)
        listaCresc2042.append(prev2042)
        

        #Classificacao do Municipios
        #2000
        TipoMunicipio2000 = ClassificacaoPorteMunicipio(int(censo2000))
        listaClassifMunicipio2000.append(TipoMunicipio2000)

        #2010
        TipoMunicipio2010 = ClassificacaoPorteMunicipio(int(censo2010))
        listaClassifMunicipio2010.append(TipoMunicipio2010)

        #2022
        TipoMunicipio2022 = ClassificacaoPorteMunicipio(int(censo2022))
        listaClassifMunicipio2022.append(TipoMunicipio2022)

        #2032
        TipoMunicipio2032 = ClassificacaoPorteMunicipio(int(prev2032))
        listaClassifMunicipio2032.append(TipoMunicipio2032)
 
        #2042
        TipoMunicipio2042 = ClassificacaoPorteMunicipio(int(prev2042))
        listaClassifMunicipio2042.append(TipoMunicipio2042)

        listaMunicipio.append(municipio)


        #calculo populacao para cada censo
        popTotal2000 = popTotal2000 + int(censo2000)
        popTotal2010 = popTotal2010 + int(censo2010)
        popTotal2022 = popTotal2022 + int(censo2022)
        popTotalEstimada2032 = popTotalEstimada2032 + int(prev2032)
        popTotalEstimada2042 = popTotalEstimada2042 + int(prev2042)

        qtdeMunicipios = qtdeMunicipios + 1


        listaCenso2000.append(censo2000)
        listaCenso2010.append(censo2010)
        listaCenso2022.append(censo2022)
        listapopTotalEstimada2032.append(popTotalEstimada2032)
        listapopTotalEstimada2042.append(popTotalEstimada2042)

    listaPopulacaoTotalEstado = []
    listaPopulacaoTotalEstado.append(popTotal2000)        
    listaPopulacaoTotalEstado.append(popTotal2010)     
    listaPopulacaoTotalEstado.append(popTotal2022)     
    listaPopulacaoTotalEstado.append(popTotalEstimada2032)     
    listaPopulacaoTotalEstado.append(popTotalEstimada2042)     
# ================================================= fim previsao da populacao ===============================================

with open('D:\ProjetoIntegradoPython\censo_2000_2010_2022_atualizado.CSV',mode='r') as arq:
    leitor = csv.reader(arq, delimiter=';')

    taxaCresc0010 = 0
    taxaCresc1022 = 0


    calcMediaCresc0010 = 0
    calcMediaCresc1022 = 0


    for coluna2 in leitor:

        # print(f'{coluna[0]} {coluna[1]} {coluna[2]} {coluna[3]}')
        municipio = coluna2[0]
        censo2000 = coluna2[1]
        censo2010 = coluna2[2]
        censo2022 = coluna2[3]

        # Calculo da taxa de crescimento da populacao entre 2000 e 2010
        calcTaxaCresc0010 = (float(censo2010) * 100) / float(censo2000) - 100
        taxaCresc0010 = str(calcTaxaCresc0010)[:6]

        # Calculo da taxa de crescimento da populacao entre 2010 e 2022
        calctaxaCresc1022 = (float(censo2022) * 100) / float(censo2010) - 100
        taxaCresc1022 = str(calctaxaCresc1022)[:6]

 
         #Calculo Media Crescimento p/ Grafico 2000 2010
        calcMediaCresc0010 = calcMediaCresc0010 + float(taxaCresc0010)

        #Calculo Media Crescimento p/ Grafico 2010 2022
        calcMediaCresc1022 = calcMediaCresc1022 + float(taxaCresc1022)


        #Porcetagem da Populacao: % da populacao do municipio comparado com o Estado
        calcPorcPop2000 = ((int(censo2000) * 100) / int(popTotal2000))
        PorcentagemPop2000 = str(calcPorcPop2000)[:6]

        calcPorcPop2010 = (int(censo2010) * 100) / int(popTotal2010)
        PorcentagemPop2010 = str(calcPorcPop2010)[:6]

        calcPorcPop2022 = (int(censo2022) * 100) / int(popTotal2022)
        PorcentagemPop2022 = str(calcPorcPop2022)[:6]
        

        listaPorcRepEstado2000.append(PorcentagemPop2000)
        listaPorcRepEstado2010.append(PorcentagemPop2010)
        listaPorcRepEstado2022.append(PorcentagemPop2022)

        listaTaxaCrescPop0010.append(taxaCresc0010)
        listaTaxaCrescPop1022.append(taxaCresc1022)


# Calculo da % da população e taxa de crescimento para os anos previstos 2032 e 2032


for i in range(len(listaMunicipio)):
    
    #Porcetagem da Populacao: % da populacao do municipio comparado com o Estado
    
    # ====> 2032 <====
    calcPorcPop2032 = (int(listaCresc2032[i]) * 100) / int(popTotalEstimada2032)
    PorcentagemPop2032 = str(calcPorcPop2032)[:6]

    # Calculo da taxa de crescimento da populacao entre 2022 e 2032
    calctaxaCresc2232 = (float(listaCresc2032[i]) * 100) / float(listaCenso2022[i]) - 100
    taxaCresc2232 = str(calctaxaCresc2232)[:6]

    #Calculo Media Crescimento p/ Grafico  2022 2032
    calcMediaCresc2232 = calcMediaCresc2232 + float(taxaCresc2232)

   
    listaPorcRepEstimadoEstado2032.append(PorcentagemPop2032) 
    listaTaxaCrescPop2232.append(taxaCresc2232) 

    # ====> 2042 <====
    calcPorcPop2042 = (int(listaCresc2042[i]) * 100) / int(popTotalEstimada2042)
    PorcentagemPop2042 = str(calcPorcPop2042)[:6]

    # Calculo da taxa de crescimento da populacao entre 2032 e 2042
    calctaxaCresc3242 = (float(listaCresc2042[i]) * 100) / float(listaCresc2032[i]) - 100
    taxaCresc3242 = str(calctaxaCresc3242)[:6]

    #Calculo Media Crescimento p/ Grafico  2032 2042
    calcMediaCresc3242 = calcMediaCresc3242 + float(taxaCresc3242)

   
    listaPorcRepEstimadoEstado2042.append(PorcentagemPop2042) 
    listaTaxaCrescPop3242.append(taxaCresc3242) 



# Consolidacao de todos os dados gerados em um arquivo na extensão Excel - xlsx
# Os dados estao ordenados Decrescente baseado na populacao do Censo 2000

data = {
        'Municipio' :  listaMunicipio,

        'População em 2000' :  listaCenso2000,
        'Classificação do Porte do Municipio 2000' :  listaClassifMunicipio2000,
        '% Pop. comparado com o Estado 2000' :  listaPorcRepEstado2000,

        'População em 2010' :  listaCenso2010,
        'Classificação do Porte do Municipio 2010' :  listaClassifMunicipio2010,
        '% Pop. comparado com o Estado 2010' :  listaPorcRepEstado2010,

        'População em 2022' :  listaCenso2022,
        'Classificação do Porte do Municipio 2022' :  listaClassifMunicipio2022,
        '% Pop. comparado com o Estado 2022' :  listaPorcRepEstado2022,

        'Estimativa da População em 2032' :  listaCresc2032,
        'Classificação do Porte do Municipio 2032' :  listaClassifMunicipio2032,
        '% Pop. comparado com o Estado 2032' : listaPorcRepEstimadoEstado2032,

        'Estimativa da População em 2042' :  listaCresc2042,
        'Classificação do Porte do Municipio 2042' :  listaClassifMunicipio2042,
        '% Pop. comparado com o Estado 2042' : listaPorcRepEstimadoEstado2042,

        'Taxa Crescimento entre 2000 - 2010' :  listaTaxaCrescPop0010,
        'Taxa Crescimento entre 2010 - 2022' :  listaTaxaCrescPop1022,
        'Taxa Crescimento entre 2022 - 2032' :  listaTaxaCrescPop2232,
        'Taxa Crescimento entre 2032 - 2042' :  listaTaxaCrescPop3242
        }

tabelaCenso = pd.DataFrame(data)

tabelaCenso = tabelaCenso.sort_values(by=['% Pop. comparado com o Estado 2000'], ascending=False)

tabelaCenso.to_excel("D:\ProjetoIntegradoPython\TabelaDosCensosConsolidados.xlsx",index=False)

tabelaCenso.to_csv("D:\ProjetoIntegradoPython\TabelaDosCensosConsolidadosPosAnalise.csv",index=False,encoding='utf-8')

listaTaxaCrescPop0010_float = [float(item) for item in listaTaxaCrescPop0010]
listaTaxaCrescPop1022_float = [float(item) for item in listaTaxaCrescPop1022]
listaTaxaCrescPop2232_float = [float(item) for item in listaTaxaCrescPop2232]
listaTaxaCrescPop3242_float = [float(item) for item in listaTaxaCrescPop3242]

medianaTaxaCrescPop0010 = np.median(listaTaxaCrescPop0010_float)
medianaTaxaCrescPop1022 = np.median(listaTaxaCrescPop1022_float)
medianaTaxaCrescPop2232 = np.median(listaTaxaCrescPop2232_float)
medianaTaxaCrescPop3242 = np.median(listaTaxaCrescPop3242_float)

listaTaxaCrescEstado = []
listaTaxaCrescEstado.append(medianaTaxaCrescPop0010)
listaTaxaCrescEstado.append(medianaTaxaCrescPop1022)
listaTaxaCrescEstado.append(medianaTaxaCrescPop2232)
listaTaxaCrescEstado.append(medianaTaxaCrescPop3242)

# =========================================>>> Grafico taxa de crescimento populacional entre os anos de 2000 - 2010 <<<=========================================

leituraArquivo = pd.read_csv("D:\ProjetoIntegradoPython\TabelaDosCensosConsolidadosPosAnalise.csv")

from pandas.core.api import isnull

#Utiliza-se a mediana pois podem existir os Outliers nesta coleção de cados

# Supondo que 'Classificação do Porte do Municipio 2000' é a coluna de classificação e
# 'Taxa Crescimento entre 2000 - 2010' é a coluna de taxa de crescimento


# ========== municipios classificados como Pequenos 2000 2010 ====================================================================================================

mediaCresc0010 = np.median([el for el in leituraArquivo["Taxa Crescimento entre 2000 - 2010"] if (np.isnan(el) == False)])

medianaMuniPequeno0010 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Pequeno','Taxa Crescimento entre 2000 - 2010'].astype('float')
calculoMedianaPequena0010 = np.median(medianaMuniPequeno0010)

dadosMunicipiosPequeno0010 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Pequeno','Taxa Crescimento entre 2000 - 2010']
nomesMunicipiosPequeno0010 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Pequeno','Municipio']

dados10MunicipiosPequeno0010 = dadosMunicipiosPequeno0010[:12].values
nomes10MunicipiosPequeno0010 = nomesMunicipiosPequeno0010[:12].values

x = np.arange(8)

refPorCrescPequeno0010  = [float(dados10MunicipiosPequeno0010[0]),float(dados10MunicipiosPequeno0010[1]),float(dados10MunicipiosPequeno0010[2]),float(dados10MunicipiosPequeno0010[3]),float(dados10MunicipiosPequeno0010[4]),float(dados10MunicipiosPequeno0010[5]),float(dados10MunicipiosPequeno0010[6]),float(dados10MunicipiosPequeno0010[7])]
refMedia0010 = [mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010]
refMedia0010MunPeq0010 = [calculoMedianaPequena0010,calculoMedianaPequena0010,calculoMedianaPequena0010,calculoMedianaPequena0010,calculoMedianaPequena0010,calculoMedianaPequena0010,calculoMedianaPequena0010,calculoMedianaPequena0010]

width = 0.2

plt.bar(x-0.2, refPorCrescPequeno0010, width, color='#6495ED',label='Taxa Muninípio')
plt.bar(x, refMedia0010, width, color='orange',label='Taxa Média do Estado')
plt.bar(x-0.4, refMedia0010MunPeq0010, width, color='green',label='Taxa Média do Estado para Mun.  Pequenos')

plt.xticks(x,[str(nomes10MunicipiosPequeno0010[0]), str(nomes10MunicipiosPequeno0010[1]), str(nomes10MunicipiosPequeno0010[2]), str(nomes10MunicipiosPequeno0010[3]), str(nomes10MunicipiosPequeno0010[4]), str(nomes10MunicipiosPequeno0010[5]), str(nomes10MunicipiosPequeno0010[6]), str(nomes10MunicipiosPequeno0010[7])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2000-2010 dos Municipios Classificados como Pequeno")

barra_labelMun = plt.bar(x-0.2, refPorCrescPequeno0010, width)
barra_labelMedia = plt.bar(x, refMedia0010, width)
barra_labelMediaMunPeq = plt.bar(x-0.4, refMedia0010MunPeq0010, width)

plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.bar_label(barra_labelMediaMunPeq)

plt.legend()
plt.show()


# ========== FIM **** municipios classificados como Pequenos 2000 2010 ====================================================================================================



# ========== municipios classificados como Medio 2000 2010 ====================================================================================================

mediaCresc0010 = np.median([el for el in leituraArquivo["Taxa Crescimento entre 2000 - 2010"] if (np.isnan(el) == False)])

medianaMuniMedio0010 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Pequeno','Taxa Crescimento entre 2000 - 2010'].astype('float')
calculoMedianaMedio0010 = np.median(medianaMuniMedio0010)

dadosMunicipiosMedio0010 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Medio','Taxa Crescimento entre 2000 - 2010']
nomesMunicipiosMedio0010 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Medio','Municipio']

dados10MunicipiosMedio0010 = dadosMunicipiosMedio0010[:12].values
nomes10MunicipiosMedio0010 = nomesMunicipiosMedio0010[:12].values

x = np.arange(8)

refPorCrescMedio0010  = [float(dados10MunicipiosMedio0010[0]),float(dados10MunicipiosMedio0010[1]),float(dados10MunicipiosMedio0010[2]),float(dados10MunicipiosMedio0010[3]),float(dados10MunicipiosMedio0010[4]),float(dados10MunicipiosMedio0010[5]),float(dados10MunicipiosMedio0010[6]),float(dados10MunicipiosMedio0010[7])]
refMedia0010 = [mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010]
refMedia0010MunMedio0010 = [calculoMedianaMedio0010,calculoMedianaMedio0010,calculoMedianaMedio0010,calculoMedianaMedio0010,calculoMedianaMedio0010,calculoMedianaMedio0010,calculoMedianaMedio0010,calculoMedianaMedio0010]


width = 0.2

plt.bar(x-0.2, refPorCrescMedio0010, width, color='#6495ED',label='Taxa Muninípio')
plt.bar(x, refMedia0010, width, color='orange',label='Taxa Média do Estado')
plt.bar(x-0.4, refMedia0010MunMedio0010, width, color='green',label='Taxa Média do Estado para Mun. Medio')


plt.xticks(x,[str(nomes10MunicipiosMedio0010[0]), str(nomes10MunicipiosMedio0010[1]), str(nomes10MunicipiosMedio0010[2]), str(nomes10MunicipiosMedio0010[3]), str(nomes10MunicipiosMedio0010[4]), str(nomes10MunicipiosMedio0010[5]), str(nomes10MunicipiosMedio0010[6]), str(nomes10MunicipiosMedio0010[7])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2000-2010 dos Municipios Classificados como Medio")

barra_labelMun = plt.bar(x-0.2, refPorCrescMedio0010, width)
barra_labelMedia = plt.bar(x, refMedia0010, width)
barra_labelMediaMunPeq = plt.bar(x-0.4, refMedia0010MunMedio0010, width)

plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.bar_label(barra_labelMediaMunPeq)

plt.legend()
plt.show()


# ========== FIM **** municipios classificados como Medio 2000 2010 ====================================================================================================



# ========== municipios classificados como Grande ou Metropole 2000 2010 ====================================================================================================

mediaCresc0010 = np.median([el for el in leituraArquivo["Taxa Crescimento entre 2000 - 2010"] if (np.isnan(el) == False)])

medianaMuniGranMetrop0010 = tabelaCenso.loc[(tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Grande') | (tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Metropole'),'Taxa Crescimento entre 2000 - 2010'].astype('float')
calculoMedianaGranMetrop0010 = np.median(medianaMuniGranMetrop0010)

dadosMunicipiosGrandeMetropole0010 = tabelaCenso.loc[(tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Grande') | (tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Metropole'),'Taxa Crescimento entre 2000 - 2010']

nomesMunicipiosGrandeMetropole0010 = tabelaCenso.loc[(tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Grande') | (tabelaCenso['Classificação do Porte do Municipio 2000'] == 'Metropole'),'Municipio']

dados10MunicipiosGrandeMetropole0010 = dadosMunicipiosGrandeMetropole0010[:12].values
nomes10MunicipiosGrandeMetropole0010 = nomesMunicipiosGrandeMetropole0010[:12].values

x = np.arange(8)

refPorCrescGrandeMetropole0010  = [float(dados10MunicipiosGrandeMetropole0010[0]),float(dados10MunicipiosGrandeMetropole0010[1]),float(dados10MunicipiosGrandeMetropole0010[2]),float(dados10MunicipiosGrandeMetropole0010[3]),float(dados10MunicipiosGrandeMetropole0010[4]),float(dados10MunicipiosGrandeMetropole0010[5]),float(dados10MunicipiosGrandeMetropole0010[6]),float(dados10MunicipiosGrandeMetropole0010[7])]
refMedia0010 = [mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010,mediaCresc0010]
refMedia0010MunGranMetrop0010 = [calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010,calculoMedianaGranMetrop0010]


width = 0.2

plt.bar(x-0.2, refPorCrescGrandeMetropole0010, width, color='#6495ED',label='Taxa Muninípio')
plt.bar(x, refMedia0010, width, color='orange',label='Taxa Média do Estado')
plt.bar(x-0.4, refMedia0010MunGranMetrop0010, width, color='green',label='Taxa Média do Estado para Mun. Grande e Metrópole')


plt.xticks(x,[str(nomes10MunicipiosGrandeMetropole0010[0]), str(nomes10MunicipiosGrandeMetropole0010[1]), str(nomes10MunicipiosGrandeMetropole0010[2]), str(nomes10MunicipiosGrandeMetropole0010[3]), str(nomes10MunicipiosGrandeMetropole0010[4]), str(nomes10MunicipiosGrandeMetropole0010[5]), str(nomes10MunicipiosGrandeMetropole0010[6]), str(nomes10MunicipiosGrandeMetropole0010[7])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2000-2010 dos Municipios Classificados como Grande ou Metropole")

barra_labelMun = plt.bar(x-0.2, refPorCrescGrandeMetropole0010, width)
barra_labelMedia = plt.bar(x, refMedia0010, width)
barra_labelMediaMunPeq = plt.bar(x-0.4, refMedia0010MunGranMetrop0010, width)

plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.bar_label(barra_labelMediaMunPeq)

plt.legend()
plt.show()


# ========== FIM **** municipios classificados como Grande ou Metropole 2000 2010 ====================================================================================================



# =========================================>>> Grafico taxa de crescimento populacional entre os anos de 2010 - 2022 <<<=========================================


# ========== municipios classificados como Pequenos 2010 2022 ====================================================================================================

mediaCresc1022 = np.median([el for el in leituraArquivo["Taxa Crescimento entre 2010 - 2022"] if (np.isnan(el) == False)])

medianaMuniPequeno1022 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Pequeno','Taxa Crescimento entre 2010 - 2022'].astype('float')
calculoMedianaPequena1022 = np.median(medianaMuniPequeno1022)

dadosMunicipiosPequeno1022 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Pequeno','Taxa Crescimento entre 2010 - 2022']
nomesMunicipiosPequeno1022 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Pequeno','Municipio']

dados10MunicipiosPequeno1022 = dadosMunicipiosPequeno1022[:12].values
nomes10MunicipiosPequeno1022 = nomesMunicipiosPequeno1022[:12].values

x = np.arange(8)

refPorCrescPequeno1022  = [float(dados10MunicipiosPequeno1022[0]),float(dados10MunicipiosPequeno1022[1]),float(dados10MunicipiosPequeno1022[2]),float(dados10MunicipiosPequeno1022[3]),float(dados10MunicipiosPequeno1022[4]),float(dados10MunicipiosPequeno1022[5]),float(dados10MunicipiosPequeno1022[6]),float(dados10MunicipiosPequeno1022[7])]
refMedia1022 = [mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022]
refMedia0010MunPeq1022 = [calculoMedianaPequena1022,calculoMedianaPequena1022,calculoMedianaPequena1022,calculoMedianaPequena1022,calculoMedianaPequena1022,calculoMedianaPequena1022,calculoMedianaPequena1022,calculoMedianaPequena1022]

width = 0.2

plt.bar(x-0.2, refPorCrescPequeno1022, width, color='#6495ED',label='Taxa Muninípio')
plt.bar(x, refMedia1022, width, color='orange',label='Taxa Média do Estado')
plt.bar(x-0.4, refMedia0010MunPeq1022, width, color='green',label='Taxa Média do Estado para Mun. Pequenos')

plt.xticks(x,[str(nomes10MunicipiosPequeno1022[0]), str(nomes10MunicipiosPequeno1022[1]), str(nomes10MunicipiosPequeno1022[2]), str(nomes10MunicipiosPequeno1022[3]), str(nomes10MunicipiosPequeno1022[4]), str(nomes10MunicipiosPequeno1022[5]), str(nomes10MunicipiosPequeno1022[6]), str(nomes10MunicipiosPequeno1022[7])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2010-2022 dos Municipios Classificados como Pequeno")

barra_labelMun = plt.bar(x-0.2, refPorCrescPequeno1022, width)
barra_labelMedia = plt.bar(x, refMedia1022, width)
barra_labelMediaMunPeq = plt.bar(x-0.4, refMedia0010MunPeq1022, width)

plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.bar_label(barra_labelMediaMunPeq)

plt.legend()
plt.show()


# ========== FIM **** municipios classificados como Pequenos 2000 2010 ====================================================================================================


# ========== municipios classificados como Medio 2010 2022 ====================================================================================================

mediaCresc1022 = np.median([el for el in leituraArquivo["Taxa Crescimento entre 2010 - 2022"] if (np.isnan(el) == False)])

medianaMuniMedio1022 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Pequeno','Taxa Crescimento entre 2010 - 2022'].astype('float')
calculoMedianaMedio1022 = np.median(medianaMuniMedio1022)

dadosMunicipiosMedio1022 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Medio','Taxa Crescimento entre 2010 - 2022']
nomesMunicipiosMedio1022 = tabelaCenso.loc[tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Medio','Municipio']

dados10MunicipiosMedio1022 = dadosMunicipiosMedio1022[:12].values
nomes10MunicipiosMedio1022 = nomesMunicipiosMedio1022[:12].values

x = np.arange(8)

refPorCrescMedio1022  = [float(dados10MunicipiosMedio1022[0]),float(dados10MunicipiosMedio1022[1]),float(dados10MunicipiosMedio1022[2]),float(dados10MunicipiosMedio1022[3]),float(dados10MunicipiosMedio1022[4]),float(dados10MunicipiosMedio1022[5]),float(dados10MunicipiosMedio1022[6]),float(dados10MunicipiosMedio1022[7])]
refMedia1022 = [mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022]
refMedia1022MunMedio1022 = [calculoMedianaMedio1022,calculoMedianaMedio1022,calculoMedianaMedio1022,calculoMedianaMedio1022,calculoMedianaMedio1022,calculoMedianaMedio1022,calculoMedianaMedio1022,calculoMedianaMedio1022]


width = 0.2

plt.bar(x-0.2, refPorCrescMedio1022, width, color='#6495ED',label='Taxa Muninípio')
plt.bar(x, refMedia1022, width, color='orange',label='Taxa Média do Estado')
plt.bar(x-0.4, refMedia1022MunMedio1022, width, color='green',label='Taxa Média do Estado para Mun. Medio')


plt.xticks(x,[str(nomes10MunicipiosMedio1022[0]), str(nomes10MunicipiosMedio1022[1]), str(nomes10MunicipiosMedio1022[2]), str(nomes10MunicipiosMedio1022[3]), str(nomes10MunicipiosMedio1022[4]), str(nomes10MunicipiosMedio1022[5]), str(nomes10MunicipiosMedio1022[6]), str(nomes10MunicipiosMedio1022[7])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2010-2022 dos Municipios Classificados como Medio")

barra_labelMun = plt.bar(x-0.2, refPorCrescMedio1022, width)
barra_labelMedia = plt.bar(x, refMedia1022, width)
barra_labelMediaMunPeq = plt.bar(x-0.4, refMedia1022MunMedio1022, width)

plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.bar_label(barra_labelMediaMunPeq)

plt.legend()
plt.show()


# ========== FIM **** municipios classificados como Medio 2000 2010 ====================================================================================================


# ========== municipios classificados como Grande ou Metropole 2000 2010 ====================================================================================================

mediaCresc1022 = np.median([el for el in leituraArquivo["Taxa Crescimento entre 2010 - 2022"] if (np.isnan(el) == False)])

medianaMuniGranMetrop1022 = tabelaCenso.loc[(tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Grande') | (tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Metropole'),'Taxa Crescimento entre 2010 - 2022'].astype('float')
calculoMedianaGranMetrop1022 = np.median(medianaMuniGranMetrop1022)

dadosMunicipiosGrandeMetropole1022 = tabelaCenso.loc[(tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Grande') | (tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Metropole'),'Taxa Crescimento entre 2010 - 2022']

nomesMunicipiosGrandeMetropole1022 = tabelaCenso.loc[(tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Grande') | (tabelaCenso['Classificação do Porte do Municipio 2010'] == 'Metropole'),'Municipio']

dados10MunicipiosGrandeMetropole1022 = dadosMunicipiosGrandeMetropole1022[:12].values
nomes10MunicipiosGrandeMetropole1022 = nomesMunicipiosGrandeMetropole1022[:12].values

x = np.arange(8)

refPorCrescGrandeMetropole1022  = [float(dados10MunicipiosGrandeMetropole1022[0]),float(dados10MunicipiosGrandeMetropole1022[1]),float(dados10MunicipiosGrandeMetropole1022[2]),float(dados10MunicipiosGrandeMetropole1022[3]),float(dados10MunicipiosGrandeMetropole1022[4]),float(dados10MunicipiosGrandeMetropole1022[5]),float(dados10MunicipiosGrandeMetropole1022[6]),float(dados10MunicipiosGrandeMetropole1022[7])]
refMedia1022 = [mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022,mediaCresc1022]
refMedia1022MunGranMetrop1022 = [calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022,calculoMedianaGranMetrop1022]


width = 0.2

plt.bar(x-0.2, refPorCrescGrandeMetropole1022, width, color='#6495ED',label='Taxa Muninípio')
plt.bar(x, refMedia1022, width, color='orange',label='Taxa Média do Estado')
plt.bar(x-0.4, refMedia1022MunGranMetrop1022, width, color='green',label='Taxa Média do Estado para Mun. Grande e Metrópole')


plt.xticks(x,[str(nomes10MunicipiosGrandeMetropole1022[0]), str(nomes10MunicipiosGrandeMetropole1022[1]), str(nomes10MunicipiosGrandeMetropole1022[2]), str(nomes10MunicipiosGrandeMetropole1022[3]), str(nomes10MunicipiosGrandeMetropole1022[4]), str(nomes10MunicipiosGrandeMetropole1022[5]), str(nomes10MunicipiosGrandeMetropole1022[6]), str(nomes10MunicipiosGrandeMetropole1022[7])])
plt.xlabel("Municípios")
plt.ylabel("Taxa crescimento 2010-2022 dos Municipios Classificados como Grande ou Metropole")

barra_labelMun = plt.bar(x-0.2, refPorCrescGrandeMetropole1022, width)
barra_labelMedia = plt.bar(x, refMedia1022, width)
barra_labelMediaMunPeq = plt.bar(x-0.4, refMedia1022MunGranMetrop1022, width)

plt.bar_label(barra_labelMun)
plt.bar_label(barra_labelMedia)
plt.bar_label(barra_labelMediaMunPeq)

plt.legend()
plt.show()


# ========== FIM **** municipios classificados como Grande ou Metropole 2000 2010 ====================================================================================================




# ========== Crescimento do Estados utilizando dados de 2000 2010 2022 e os estimados para 2032 e 2032 ====================================================================================================


x = np.arange(4)


width = 0.2

plt.bar(x, listaTaxaCrescEstado, width)

plt.xticks(x,["Taxa Cresc.2010","Taxa Cresc. 2022","Taxa Cresc. 2032","Taxa Cresc. 2042"])

plt.ylabel("Taxa Crescimento e Estimativa Populacional do Estado entre 2010 a 2042")

barra_labelMedia = plt.bar(x, listaTaxaCrescEstado, width)


plt.bar_label(barra_labelMedia)

plt.legend()
plt.show()


print("Dados gerados com sucesso!!!")


