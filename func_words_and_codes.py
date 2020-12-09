import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import display_progress_ipython as display
import itertools
import func_aux_analise as faa
pd.options.mode.chained_assignment = None

#----------------------------------------------------
#Funções auxiliares de tratamento

#Invete uma string
def reverse_slicing(s):
    if(len(s) > 0):
        return s[::-1]
    else:
        return s

#Pega só os caracteres que são letras de uma string
def splitString_alpha(str): 
    alpha = "" 
    num = "" 
    special = "" 
    for i in range(len(str)): 
        if (str[i].isdigit()): 
            num = num+ str[i] 
        elif((str[i] >= 'A' and str[i] <= 'Z') or
             (str[i] >= 'a' and str[i] <= 'z')): 
            alpha += str[i] 
        else: 
            special += str[i] 
    #return alpha, num, special
    return alpha

#Pega só os caracteres que são numeros de uma string (retorna '0' se não tiver número)
def splitString_num(str): 
    alpha = "" 
    num = "" 
    special = "" 
    for i in range(len(str)): 
        if (str[i].isdigit()): 
            num = num+ str[i] 
        elif((str[i] >= 'A' and str[i] <= 'Z') or
             (str[i] >= 'a' and str[i] <= 'z')): 
            alpha += str[i] 
        else: 
            special += str[i] 
    #return alpha, num, special
    if(num == ''):
        num = 'S' #Adicionado pra avisar cabines sem número
    return num

#Retorna uma posição da string ou retorna 0 se a posição não existir
def get_pos_string(s, pos): 
    try:
        return s[pos]
    except:
        return '-1' #Adicionado para avisar quando já acabou um código de digitos

#----------------------------------------------------
#Funções para fazer o tratamento de cada coluna específica

#Separa os nomes na lista de palavras que o compõe
def split_words_name(df_inp):
    df = df_inp[df_inp['Name'].isnull() == False].copy()
    df_null = df_inp[df_inp['Name'].isnull() == True].copy()
    
    #Remove os elementos desnecessários dos nomes (separadores e marcadores)
    df['Name'] = df['Name'].str.replace(',', '')
    df['Name'] = df['Name'].str.replace('.', '')
    df['Name'] = df['Name'].str.replace('(', '')
    df['Name'] = df['Name'].str.replace(')', '')
    df['Name'] = df['Name'].str.replace('"', '')
    df['Name_words'] = df['Name'].str.split(' ') #split as strings dos nomes pelos espaços
    
    df = pd.concat([df, df_null], ignore_index = False).sort_index()
    df = df.drop('Name', axis = 1) 
    return df

#Separa o prefixo do número do ticket e separa o prefixo nas palavras principais que o compõe
def split_pref_num_ticket(df_inp):
    df = df_inp[df_inp['Ticket'].isnull() == False].copy()
    df_null = df_inp[df_inp['Ticket'].isnull() == True].copy()
    
    #Separa o ticket no número e no seu prefixo (se possuir)
    df['Ticket_rev'] = [reverse_slicing(s) for s in df['Ticket']]
    lista = []
    for s in df['Ticket_rev']:
        if(s.find(' ') != -1):
            lista.append(s[:s.find(' ')])
        else:
            if(s.isnumeric()):
                lista.append(s)
            else:
                lista.append('')
    df['Ticket_rev'] = lista
    df['Ticket_num'] = [reverse_slicing(s) for s in df['Ticket_rev']]
    df['Ticket_pref'] = [df['Ticket'][i].replace(df['Ticket_num'][i], '') for i in range(0, len(df))]
    lista = []
    for s in df['Ticket_pref']:
        if(s == ''):
            lista.append('SEM_PREF')
        else:
            lista.append(s.replace(' ', ''))
    df['Ticket_pref'] = lista
    #Remove os elementos desnecessários (separadores e marcadores)
    df['Ticket_pref'] = df['Ticket_pref'].str.replace('.', '')
    df['Ticket_pref'] = df['Ticket_pref'].str.replace('/', ' ')
    df['Ticket_pref'] = df['Ticket_pref'].str.split(' ') #split as strings dos nomes pelos espaços
    
    df = pd.concat([df, df_null], ignore_index = False).sort_index() 
    df = df.drop('Ticket_rev', axis = 1)
    df = df.drop('Ticket', axis = 1)    
    return df

#Separa em uma contagem de quantas cabines existem, e lista de prefixos delas e a lista de números delas
def split_pref_num_and_count_cabins(df_inp):
    df = df_inp[df_inp['Cabin'].isnull() == False].copy()
    df_null = df_inp[df_inp['Cabin'].isnull() == True].copy()
    
    df['list_cabins'] = df['Cabin'].str.split(' ') #split as strings dos nomes pelos espaços
    df['Cabin_quant'] = [len(c) for c in df['list_cabins']]
    
    lista_pref = []
    lista_num = []
    for lista in df['list_cabins']:
        alphas = [splitString_alpha(s) for s in lista]
        alphas = list(dict.fromkeys(alphas))
        nums = [splitString_num(s) for s in lista]
        nums = list(dict.fromkeys(nums))
        lista_pref.append(alphas)
        lista_num.append(nums)
    df['Cabin_pref'] = lista_pref
    df['Cabin_num'] = lista_num
    df = df.drop('list_cabins', axis = 1)
    
    df = pd.concat([df, df_null], ignore_index = False).sort_index()
    df = df.drop('Cabin', axis = 1)
    return df

#----------------------------------------------------
#Funções criação de lista de ocorrência

#Criação de lista de ocorrência de palavras
def create_df_ocorr_words(df_inp, col, verbose):
    df = df_inp[df_inp[col].isnull() == False].copy()
    ind = list(df.index)
    
    #Pega todas as palavras distintas (pode ser uma palavra por linha ou uma lista de palavras por linha)
    all_words = []
    for v in df[col]:
        if(type(v) == str):
            all_words.append(v)
        else:
            all_words.extend(v)
    all_words = list(dict.fromkeys(all_words))
        
    #Para cada palavra, faz uma listas dos indices no dataframe que tem essa palavra
    dfw = pd.DataFrame(columns = ['key', 'lista', 'num_ocorr'])
    dfw['key'] = all_words
    dh, prog_ant = [], 0
    for i in range(0, len(dfw)):
        if(verbose):
            dh, prog_ant = display.update_progress(i/len(dfw), 1, dh, prog_ant)
        lista_aux = []
        for j in range(0, len(df)):
            v = df[col][ind[j]]
            if(type(v) == str):
                if (dfw['key'][i] == v):
                    lista_aux.append(ind[j])
            else:
                if any(dfw['key'][i] == s for s in v):
                    lista_aux.append(ind[j])
        dfw['lista'][i] = lista_aux
    for i in range(0, len(dfw)):
        dfw['num_ocorr'][i] = len(dfw['lista'][i])
    dfw = dfw.sort_values(by = 'num_ocorr', ascending = False).reset_index(drop = True) #Ordena por número de ocorrencias nos nomes
    
    return dfw

#Criação de lista de ocorrência de digitos em uma dada posição
def create_df_ocorr_digit(df_inp, col, pos_dig, verbose):
    df = df_inp[df_inp[col].isnull() == False].copy()
    ind = list(df.index)

    #Inverte os digitos (fica mais fácil pegar o digito da posição de interesse)
    lista_aux = []
    for v in df[col]:
        if(type(v) == str):
            lista_aux.append(reverse_slicing(v))
        else:
            lista_aux.append([reverse_slicing(s) for s in v])
    df[col] = lista_aux
    
    #Cria uma coluna com o digito de interesse
    lista = []
    for v in df[col]:
        if(type(v) == str):
            aux = get_pos_string(v, pos_dig)
        else:
            aux = [get_pos_string(s, pos_dig) for s in v]
        lista.append(aux)
    df[col] = lista
    
    dfw = create_df_ocorr_words(df, col, verbose)
    dfw['pos_dig'] = [pos_dig for i in range(0, len(dfw))]
    cols = list(dfw.columns)
    cols = [cols[len(cols)-1]] + cols[:len(cols)-1]
    dfw = dfw[cols]
    return dfw

#Criação de lista de ocorrência de todos os digitos
def create_df_ocorr_digits(df_inp, col, verbose):
    df = df_inp[df_inp[col].isnull() == False].copy()
    ind = list(df.index)

    lista_dfw = []
    pos_dig = 0
    dfw = create_df_ocorr_digit(df, col, pos_dig, verbose)
    if(len(dfw) > 1):
        lista_dfw.append(dfw)
        
    while(len(dfw) > 1):
        pos_dig = pos_dig + 1
        dfw = create_df_ocorr_digit(df, col, pos_dig, verbose)
        if(len(dfw) > 1):
            lista_dfw.append(dfw)

    dfw = pd.concat(lista_dfw).reset_index(drop = True)
    dfw = dfw.sort_values(by = 'num_ocorr', ascending = False).reset_index(drop = True)
    return dfw

#----------------------------------------------------
#Funções de analise e filtro de lista de ocorrência

#Calcula o ganho de informação (por entropia)
#Considerando que os separadores tem valores binários
def calc_infor_df_occor(df_inp, dfw_inp, col, col_alvo):
    dfw = dfw_inp.copy()
    df = df_inp[df_inp[col].isnull() == False].copy()
    
    entropia_inicial = faa.calc_entropia(df, col_alvo)
    
    lista_entropia = []
    for i in range(0, len(dfw)):
        indices = dfw['lista'][i]
        if(len(df) > 0):
            df_aux1 = df.loc[indices,:]
            df_aux2 = df.drop(df_aux1.index)
            entropia1 = faa.calc_entropia(df_aux1, col_alvo)
            entropia2 = faa.calc_entropia(df_aux2, col_alvo)
            entropia = (len(df_aux1)/len(df))*entropia1 + (len(df_aux2)/len(df))*entropia2
        else:
            entropia = -np.inf
        lista_entropia.append(entropia)
        
    dfw['gain_info'] = (-1)*(np.array(lista_entropia) - entropia_inicial)
    dfw['gain_info/codigo'] = dfw['gain_info']/2 #2 pq as variaveis são todas binárias
    dfw = dfw.sort_values(by = 'gain_info', ascending = False).reset_index(drop = True) #Ordena pelo ganho de informação
    return dfw

#----------------------------------------------------
#Funções de busca de lista de ocorrências dado um conjunto de variaveis

#Procura lista de ocorrência de palavras
def find_ocorr_words(df_inp, dfw_inp, col, verbose):
    df = df_inp[df_inp[col].isnull() == False].copy()
    ind = list(df.index)
    
    #Para cada palavra, faz uma listas dos indices no dataframe que tem essa palavra
    dfw = pd.DataFrame(columns = ['key', 'lista', 'num_ocorr'])
    dfw['key'] = dfw_inp['key'].copy()
    dh, prog_ant = [], 0
    for i in range(0, len(dfw)):
        if(verbose):
            dh, prog_ant = display.update_progress(i/len(dfw), 1, dh, prog_ant)
        lista_aux = []
        for j in range(0, len(df)):
            v = df[col][ind[j]]
            if(type(v) == str):
                if (dfw['key'][i] == v):
                    lista_aux.append(ind[j])
            else:
                if any(dfw['key'][i] == s for s in v):
                    lista_aux.append(ind[j])
        dfw['lista'][i] = lista_aux
    for i in range(0, len(dfw)):
        dfw['num_ocorr'][i] = len(dfw['lista'][i])
    dfw = dfw.sort_values(by = 'num_ocorr', ascending = False).reset_index(drop = True) #Ordena por número de ocorrencias nos nomes
    
    return dfw

#Procura lista de ocorrência de todos os digitos
def find_ocorr_digits(df_inp, dfw_inp, col, verbose):
    df = df_inp[df_inp[col].isnull() == False].copy()
    ind = list(df.index)

    #Inverte os digitos (fica mais fácil pegar o digito da posição de interesse)
    lista_aux = []
    for v in df[col]:
        if(type(v) == str):
            lista_aux.append(reverse_slicing(v))
        else:
            lista_aux.append([reverse_slicing(s) for s in v])
    df[col] = lista_aux
    
    #Para cada palavra, faz uma listas dos indices no dataframe que tem essa palavra
    dfw = pd.DataFrame(columns = ['pos_dig', 'key', 'lista', 'num_ocorr'])
    dfw['pos_dig'] = dfw_inp['pos_dig'].copy()
    dfw['key'] = dfw_inp['key'].copy()
    dh, prog_ant = [], 0
    for i in range(0, len(dfw)):
        if(verbose):
            dh, prog_ant = display.update_progress(i/len(dfw), 1, dh, prog_ant)
        lista_aux = []
        for j in range(0, len(df)):
            v = df[col][ind[j]]
            if(type(v) == str):
                if (dfw['key'][i] == get_pos_string(v, dfw['pos_dig'][i])):
                    lista_aux.append(ind[j])
            else:
                if any(dfw['key'][i] == get_pos_string(s, dfw['pos_dig'][i]) for s in v):
                    lista_aux.append(ind[j])
        dfw['lista'][i] = lista_aux
    for i in range(0, len(dfw)):
        dfw['num_ocorr'][i] = len(dfw['lista'][i])
    dfw = dfw.sort_values(by = 'num_ocorr', ascending = False).reset_index(drop = True) #Ordena por número de ocorrencias nos nomes
    
    return dfw

#----------------------------------------------------
#Funções de criação de colunas com as novas variaveis

def create_new_columns(df_inp, dfw, col):
    df = df_inp.copy()
    colunas_dfw = list(dfw.columns)
    if(colunas_dfw[0] == 'key'):
        for i in range(0, len(dfw)):
            label = col + '_' + str(dfw['key'][i])
            indices = list(df_inp[df_inp[col].isnull() == False].index)
            df[label] = [np.nan for i in range(0, len(df))]
            df[label][indices] = [0 for i in range(0, len(indices))]
            df[label][dfw['lista'][i]] = [1 for i in range(0, len(dfw['lista'][i]))] 
    elif(colunas_dfw[0] == 'pos_dig'):
        for i in range(0, len(dfw)):
            label = col + '_d' + str(dfw['pos_dig'][i]) + '_' + str(dfw['key'][i])
            indices = list(df_inp[df_inp[col].isnull() == False].index)
            df[label] = [np.nan for i in range(0, len(df))]
            df[label][indices] = [0 for i in range(0, len(indices))]
            df[label][dfw['lista'][i]] = [1 for i in range(0, len(dfw['lista'][i]))]
    df = df.drop(col, axis = 1)
    return df
