import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
pd.options.mode.chained_assignment = None

#-------------------------------------------
#Funções extras

def colocaAlvoComeco(df_inp, nome_alvo):
    df = df_inp.copy()
    cols = list(df.columns)
    ind = cols.index(nome_alvo)
    if(ind != 0):
        cols_new = [cols[ind]] + cols[:ind] + cols[ind+1:] 
        df = df[cols_new]
    return df

#-------------------------------------------
#Analise exploratória dos dados

#Analise de uma variavel contínua ou discreta ordenável (qualitativa ordinal ou quantitativa discreta)
#OBS: No caso de qualitativa ordinal tem que converter para inteiro antes, de acordo com a ordem
def analiseVariableContinuousOrDiscreteSorted(df_inp, df_test_inp, col, is_continuous):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    if(df[col].dtype != np.float64 and df[col].dtype != np.int64):
        df[col] = pd.to_numeric(df[col], downcast = "float")
    if(df_test[col].dtype != np.float64 and df_test[col].dtype != np.int64):
        df_test[col] = pd.to_numeric(df_test[col], downcast = "float")
    
    print('*******************' + col + '*******************')
    df_aux = pd.concat([df[col], df_test.rename(columns = {col: col + '_test'})[col + '_test']], axis = 1)
    print(df_aux[[col, col + '_test']].describe().transpose())
    if(is_continuous):
        sns.distplot(df[col], label = 'Treino', kde = True)
        sns.distplot(df_test[col], label = 'Teste', kde = True)
    else:
        sns.distplot(df[col], label = 'Treino', kde = False, bins = np.arange(min(df[col]), max(df[col]) + 2))
        sns.distplot(df_test[col], label = 'Teste', kde = False, bins = np.arange(min(df_test[col]), max(df_test[col]) + 2))
    plt.legend()
    plt.show()
    
    print('------' + 'Nulos' + '------')
    df2 = df[df[col].isnull() == True][col]
    df_test2 = df_test[df_test[col].isnull() == True][col]
    print('Treino: ' + str(len(df2)) + ' / ' + str(round(len(df2)/len(df), 4)))
    print('Teste: ' + str(len(df_test2)) + ' / ' + str(round(len(df_test2)/len(df_test), 4)))
    print(' ')
    
    print('------' + 'Quantidade de valores distintos' + '------')
    df2 = df[df[col].isnull() == False][col]
    df_test2 = df_test[df_test[col].isnull() == False][col]
    values = list(dict.fromkeys(df2))
    values_test = list(dict.fromkeys(df_test2)) 
    print('Treino: ' + str(len(values)) + ' / ' + str(round(len(values)/len(df2), 4)))
    print('Teste: ' + str(len(values_test)) + ' / ' + str(round(len(values_test)/len(df_test2), 4)))
    print(' ')
    
#Analise de uma variavel discreta e não ordenável (qualitativa nominal)
def analiseVariableDiscreteUnsorted(df, df_test, col):
    print('*******************' + col + '*******************')
    df2 = df[df[col].isnull() == False]
    df_test2 = df_test[df_test[col].isnull() == False]
    print('Não-Nulos Treino: ' + str(len(df2[col])))
    print('Não-Nulos Teste: ' +str(len(df_test2[col])))
    df_counts = df[col].value_counts(normalize = True).rename('percentage').reset_index()
    df_counts2 = df_test2[col].value_counts(normalize = True).rename('percentage').reset_index()
    fig, ax =plt.subplots(1,2)
    sns.barplot(x = 'index', y = 'percentage', data = df_counts, ax = ax[0]);
    sns.barplot(x = 'index', y = 'percentage', data = df_counts2, ax = ax[1]);
    #df_aux = pd.concat([df2[col], df_test2.rename(columns = {col: col + '_test'})[col + '_test']], axis = 1)
    #sns.countplot(x = col + '_test',  data = df_aux, ax = ax[0], order = df_aux[col + '_test'].value_counts().index);
    #sns.countplot(x = col + '_test', data = df_aux, ax = ax[1], order = df_aux[col + '_test'].value_counts().index);
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()
    
    print('------' + 'Nulos' + '------')
    df2 = df[df[col].isnull() == True][col]
    df_test2 = df_test[df_test[col].isnull() == True][col]
    print('Treino: ' + str(len(df2)) + ' / ' + str(round(len(df2)/len(df), 4)))
    print('Teste: ' + str(len(df_test2)) + ' / ' + str(round(len(df_test2)/len(df_test), 4)))
    print(' ')
    
    print('------' + 'Quantidade de valores distintos' + '------')
    df2 = df[df[col].isnull() == False][col]
    df_test2 = df_test[df_test[col].isnull() == False][col]
    values = list(dict.fromkeys(df2))
    values_test = list(dict.fromkeys(df_test2)) 
    print('Treino: ' + str(len(values)) + ' / ' + str(round(len(values)/len(df2), 4)))
    print('Teste: ' + str(len(values_test)) + ' / ' + str(round(len(values_test)/len(df_test2), 4)))
    print(' ')

#Printa quais são as possíveis variáveis que podem sem nulas e quais podem ser nulas ao mesmo tempo
def printNullRows(df):
    tudo = []
    for i in range(0, len(df)):
        row = df.iloc[i,:]
        row_aux = row[row.isnull()]
        if(len(row_aux) > 0):
            lista = list(row_aux.index)
            tudo.append(str(lista))
    values = list(dict.fromkeys(tudo))
    print(values)
    print('-------')

#Printa a quantidade de nulos e de valores distintos nas variveis no treino e no teste
#(É o máximo que dá pra fazer no começo para uma valiável quanlitativa nominal com muitos valores distintos)    
def distinctValues(df, df_test, col, flag_print):
    df_aux = df[df[col].isnull() == False]
    df_aux2 = df_test[df_test[col].isnull() == False]
    values = list(dict.fromkeys(df_aux[col]))
    values2 = list(dict.fromkeys(df_aux2[col]))
    print(col + ':')
    print('Quantidade: Total / Nulos / Distintos')
    print('-Treino: ' + str(len(df[col])) + ' / ' +  str(len(df[col]) - len(df_aux)) + ' / ' + str(len(values)))
    print('-Teste: ' + str(len(df_test[col])) + ' / ' +  str(len(df_test[col]) - len(df_aux2)) + ' / ' + str(len(values2)))
    if(flag_print):
        print('Valores Distintos no Treino:')
        print(values)
        print('Valores Distintos no Teste:')
        print(values2)
    print('-------')

#Printa as linhas que tem nulo
def createDFsWithNulls(df):
    tudo = []
    inds = []
    for i in range(0, len(df)):
        row = df.iloc[i,:]
        row_aux = row[row.isnull()]
        if(len(row_aux) > 0):
            lista = list(row_aux.index)
            tudo.append(str(lista))
            inds.append(i)
    values = list(dict.fromkeys(tudo))
    dft = pd.DataFrame(columns = ['ind', 'type_null'])
    dft['ind'] = inds
    dft['type_null'] = tudo
    dfs_null = []
    for v in values:
        dft_temp = dft[dft['type_null'] == v]
        inds_temp = list(dft_temp['ind'])
        df_temp = df.loc[inds_temp, :].copy()
        dfs_null.append(df_temp)
    return dfs_null
    
#Cria tabela com numero de ocorrencias de cada valor da variavel, e as probabilidades condicionais
def createDFOcorrProbValues(df, col, nome_alvo):
    df = df[df[col].isnull() == False]
    values = list(dict.fromkeys(df[col]))      
    
    values_alvo = list(dict.fromkeys(df[nome_alvo]))
    dfr = pd.DataFrame(columns = ['value_alvo', 'lista_prob'])
    dfr['value_alvo'] = values_alvo
    dfr['lista_prob'] = [[] for i in range(0, len(dfr))]
    
    ind = []
    lista_ocorr = []
    for i in range(0, len(values)):
        ind.append(i)
        df_aux = df[df[col] == values[i]]
        lista_ocorr.append(len(df_aux))
        for j in range(0, len(dfr)):
            df_aux2 = df_aux[df_aux[nome_alvo] == values_alvo[j]]
            if(len(df_aux) > 0):
                prob_aux = len(df_aux2)/len(df_aux)
                dfr['lista_prob'][j].append(prob_aux)
            else:
                dfr['lista_prob'][j].append(0)
    
    dfv = pd.DataFrame(columns = ['value', 'ocorr'])
    dfv['value'] = values
    dfv['ocorr'] = lista_ocorr
    dfv['ocorr_frac'] = np.array(lista_ocorr)/len(df)
    for i in range(0, len(dfr)):
        dfv[nome_alvo + '_' + str(dfr['value_alvo'][i])] = dfr['lista_prob'][i]
    
    dfv = dfv.sort_values(by = 'ocorr', ascending = False).reset_index(drop = True)
    return dfv

#-------------------------------------------
#Analise de ganho de informação com variaveis discretas e não ordenáveis

#Calcula a entropia (mede quanto está bagunçada a informação da coluna alvo no dataframe)
#considerando que o alvo tem valores discretos (e não ordenáveis)
def calcEntropia(df, col_alvo):
    entropia = 0
    if(len(df) > 0):
        values_alvo = list(dict.fromkeys(df[col_alvo]))
        for v in values_alvo:
            df_aux = df[df[col_alvo] == v]
            prob = len(df_aux)/len(df)
            if(prob != 0):
                entropia = entropia + (-1)*prob*np.log2(prob)
    else:
        entropia = -np.inf
    return entropia

#Ganho de informação (com base na entropia) de variaveis discretas
def calcGainInforDiscreteUnsorted(df, col, nome_alvo):
    df = df[df[col].isnull() == False]
    values = list(dict.fromkeys(df[col]))
    entropia_inicial = calcEntropia(df, nome_alvo)
    ind = []
    entropia = 0
    for i in range(0, len(values)):
        ind.append(i)
        df_aux = df[df[col] == values[i]]
        entropia = entropia + (len(df_aux)/len(df))*calcEntropia(df_aux, nome_alvo)
    delta_entropia = entropia - entropia_inicial
    gain_info = (-1)*delta_entropia
    return gain_info

#-------------------------------------------
#Aproxima a variavel por uma discreta e não ordenável (já usa o fato de ser ordenável na aproximação)
#Inspirado no modelo MDL: https://repositorio.ufscar.br/bitstream/handle/ufscar/9322/DissDMBS.pdf?sequence=1&isAllowed=y

def applySplitIndex(df, col, lista_splits):
    df_temp = df.copy()
    df_temp[col + '_aux1'] = [0 for i in range(0, len(df_temp))]
    df_temp[col + '_aux2'] = [0 for i in range(0, len(df_temp))]
    for split in lista_splits:
        if(len(split) > 1):
            df_aux = df.loc[split,:]
            df_temp[col][split] = [df_aux[col].mean() for i in range(0, len(split))]
            df_temp[col + '_aux1'][split] = [min(df_aux[col]) for i in range(0, len(split))]
            df_temp[col + '_aux2'][split] = [max(df_aux[col]) for i in range(0, len(split))]
        else:
            ind = split[0]
            df_temp[col][ind] = df[col][ind]
            df_temp[col + '_aux1'][split] = [df[col][ind] for i in range(0, len(split))]
            df_temp[col + '_aux2'][split] = [df[col][ind] for i in range(0, len(split))]
    df_temp = df_temp.sort_values(by = col, ascending = True)
    return df_temp

#Condição para aceitar uma divisão pelo método MDL
def condicaoMDL(gain, df, lista_splits, nome_alvo):
    df1 = df.loc[lista_splits[0],:]
    df2 = df.loc[lista_splits[1],:]
    entropia = calcEntropia(df, nome_alvo)
    entropia1 = calcEntropia(df1, nome_alvo)
    entropia2 = calcEntropia(df2, nome_alvo)
    l = len(list(dict.fromkeys(df[nome_alvo])))
    l1 = len(list(dict.fromkeys(df1[nome_alvo])))
    l2 = len(list(dict.fromkeys(df2[nome_alvo])))
    Delta = np.log2(np.power(3, l) - 2) - (l*entropia - l1*entropia1 - l2*entropia2)
    n = len(df)
    gain_ref = (np.log2(n - 1)/n) + (Delta/n)
    if(gain > gain_ref):
        return True
    else:
        return False  

#Encontra o melhor split binário (usado para encontrar a melhor discretização de variáveis continuas)
def findBestSplit(df_inp, col, nome_alvo, sub_lista, bins):
    df = df_inp.loc[sub_lista,:].copy()
    lista_gain = []
    inds = []
    for i in range(1, len(sub_lista)):
        lista_splits = [sub_lista[:i], sub_lista[i:]]
        if(df[col][sub_lista[i-1]] != df[col][sub_lista[i]]):
            df_temp = applySplitIndex(df, col, lista_splits)
            gain = calcGainInforDiscreteUnsorted(df_temp, col, nome_alvo)
            lista_gain.append(gain)
            inds.append(i)
    if(len(lista_gain) > 0):
        maxi_gain = max(lista_gain)
        aux = lista_gain.index(maxi_gain)
        indice = inds[aux]
        lista_splits = [sub_lista[:indice], sub_lista[indice:]]
        if(bins < 1 and condicaoMDL(maxi_gain, df, lista_splits, nome_alvo) == False):
            lista_splits = [sub_lista]
    else:
        lista_splits = [sub_lista]
    return lista_splits

#Converte da melhor forma uma variavel continua (ou discreta ordenável) para discreta 
def bestConvertContinuousToDiscrete(df, col, nome_alvo, bins):
    df = df[df[col].isnull() == False]
    
    if(df[col].dtype != np.float64 and df[col].dtype != np.int64):
        df[col] = pd.to_numeric(df[col], downcast = "float")
    
    if(bins < 1):
        len_values_ini = len(list(dict.fromkeys(df[col])))
    else:
        len_values_ini = bins
    
    df = df.sort_values(by = col, ascending = True)
    indices = list(df.index)
    lista_splits = [indices]
    last_gain = 0
    
    while(len(lista_splits) < len_values_ini):
        lista_gain_aux = []
        lista_splits_aux = []
        for i in range(0, len(lista_splits)):
            sub_lista = lista_splits[i]
            if(len(sub_lista) > 1):
                sub_lista_split = findBestSplit(df, col, nome_alvo, sub_lista, bins)
                lista_splits_temp = lista_splits.copy() 
                if(len(sub_lista) > 1):
                    lista_splits_temp.remove(sub_lista)
                    lista_splits_temp.extend(sub_lista_split)
                    df_temp = applySplitIndex(df, col, lista_splits_temp)
                    gain = calcGainInforDiscreteUnsorted(df_temp, col, nome_alvo)
                    lista_splits_aux.append(lista_splits_temp)
                    lista_gain_aux.append(gain)
        if(len(lista_gain_aux) > 0):
            max_gain = max(lista_gain_aux)
            if(max_gain > last_gain):
                ind = lista_gain_aux.index(max_gain)
                lista_splits = lista_splits_aux[ind].copy()
                last_gain = max_gain
            else:
                break
        else:
            break
        
    df_temp = applySplitIndex(df, col, lista_splits)
    return df_temp

#-------------------------------------------
#Analise plot de prob das variaveis e ganhos de informação

#Plot das probabilidade condicionais discretas nao ordenaveis
def plotConditionalProb(df, col, nome_alvo, **kwargs):
    flag_sorted = kwargs.get('flag_sorted', False)
    flag_countinuous = kwargs.get('flag_countinuous', False)

    df = df[df[col].isnull() == False]
    
    values = list(dict.fromkeys(df[col]))
    if(flag_sorted):
        if(df[col].dtype == np.float64 or df[col].dtype == np.int64):
            values.sort()
    
    values_alvo = list(dict.fromkeys(df[nome_alvo]))
    dfr = pd.DataFrame(columns = ['value_alvo', 'lista_prob'])
    dfr['value_alvo'] = values_alvo
    dfr['lista_prob'] = [[] for i in range(0, len(dfr))]
    
    ind = []
    for i in range(0, len(values)):
        ind.append(i)
        df_aux = df[df[col] == values[i]]
        for j in range(0, len(dfr)):
            df_aux2 = df_aux[df_aux[nome_alvo] == values_alvo[j]]
            if(len(df_aux) > 0):
                prob_aux = len(df_aux2)/len(df_aux)
                dfr['lista_prob'][j].append(prob_aux)
            else:
                dfr['lista_prob'][j].append(0)    
    if(flag_sorted):
        ind = values
    
    fig, ax = plt.subplots()
    for j in range(0, len(dfr)): 
        if(flag_sorted):
            line, = ax.plot(ind, dfr['lista_prob'][j], 'o', label = nome_alvo + '_' + str(dfr['value_alvo'][j]))
            if(flag_countinuous == False):
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            colunas = list(df.columns)
            if(colunas.index(col + '_aux1') != -1):
                ind1 = list(dict.fromkeys(df[col  + '_aux1']))
                ind2 = list(dict.fromkeys(df[col  + '_aux2']))
                lista_probs = dfr['lista_prob'][j]
                for k in range(0, len(ind1)):
                    if(ind1[k] != ind2[k]):
                        ax.plot([ind1[k], ind2[k]], [lista_probs[k], lista_probs[k]], color = line.get_color())
                        if(k < len(ind1)-1 and ind1[k+1] != ind2[k+1]):
                            ax.plot([ind2[k], ind2[k]], [lista_probs[k], lista_probs[k+1]], color = line.get_color())
            else:
                ax.plot(ind, dfr['lista_prob'][j], color = line.get_color(), label = nome_alvo + '_' + str(dfr['value_alvo'][j]))
        else:
            plt.scatter(ind, dfr['lista_prob'][j], label = nome_alvo + '_' + str(dfr['value_alvo'][j]))
            plt.xticks(range(len(values)), values)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])    
    ax.legend(loc = 'center left', ncol = 1, frameon = True, bbox_to_anchor=(1, 0.5))
    plt.show()
 
    gain_info = calcGainInforDiscreteUnsorted(df, col, nome_alvo)   
    gain_info_per_bit = calcGainInforDiscreteUnsorted(df, col, nome_alvo) / np.log2(len(values))    
    print('Ganho de Informação: ' + str(round(gain_info, 4)))
    print('Ganho de Informação por Bit: ' + str(round(gain_info_per_bit, 4)))

#Plot das probabilidade condicionais de variaveis continuas ou ordenaveis
def plotConditionalProbContinuousOrDiscreteSorted(df_inp, col, nome_alvo, bins):
    df = df_inp.copy()
    df_temp = bestConvertContinuousToDiscrete(df, col, nome_alvo, bins)
    plotConditionalProb(df_temp, col, nome_alvo, flag_sorted = True)
    df_temp = df_temp.drop(col  + '_aux1', axis = 1)
    df_temp = df_temp.drop(col  + '_aux2', axis = 1)

#-------------------------------------------