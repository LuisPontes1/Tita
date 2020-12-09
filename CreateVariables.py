import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from itertools import groupby
pd.options.mode.chained_assignment = None
from MDLP import MDLP_Discretizer #https://github.com/navicto/Discretization-MDLPC
from sklearn.decomposition import PCA
from sklearn import preprocessing

#-------------------------------------------
#Funções auxiliares

#Invete uma string
def reverseSlicing(s):
    if(len(s) > 0):
        return s[::-1]
    else:
        return s

#Pega só os caracteres que são letras de uma string
def splitStringAlpha(str): 
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
def splitStringNumber(str): 
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

#Remove variaveis de PCA
def removeVarsPCA(X):
    colunas = list(X.columns)
    colunas_remover = []
    for col in colunas:
        if(col.find('var_pca') != -1):
            colunas_remover.append(col)
    X_limpo = X.drop(colunas_remover, axis = 1).copy()
    return X_limpo

#-------------------------------------------
#Cria variáveis que extraem informações que estão "escondidas" -> o modelo não tem "inteligência" pra descobrir

#Simplifica como o sexo é escrito
def simplificaStringSexo(df_inp):
    df = df_inp.copy()
    if(df_inp['Sex'][0] != 'M' and df_inp['Sex'][0] != 'F'):
        df['Sex'] = np.where(df['Sex'] == 'male', 'M', np.where(df['Sex'] == 'female', 'F', np.nan))
    return df

#Separa os nomes na lista de palavras que o compõe
def quebraNome(df_inp):
    df = df_inp[df_inp['Name'].isnull() == False].copy()
    df_null = df_inp[df_inp['Name'].isnull() == True].copy()
    
    df['Name'] = df['Name'].str.split(',') #split as strings dos nomes pela virgula
    df['Last_Name'] = [v[0] for v in df['Name']]
    df['Name'] = [v[1] for v in df['Name']]
    df['Name'] = df['Name'].str.lstrip()
    
    df['Name'] = df['Name'].str.split('.')
    df['Honorific'] = [v[0] for v in df['Name']]
    df['Name'] = [v[1] for v in df['Name']]
    df['Name'] = df['Name'].str.lstrip()
    
    df['Name'] = df['Name'].str.split('(')
    df['Name_Parentesis'] = [v[1] if len(v) > 1 else 'NAO_TEM ' for v in df['Name']]
    df['Name_Parentesis'] = [v[:len(v)-1] for v in df['Name_Parentesis']]
    df['Name_Parentesis'] = [v[:len(v)-1] if v[len(v)-1] == ')' else v for v in df['Name_Parentesis']]
    df['Name_Parentesis'] = ['"' + v if v.find('"') != -1 and v[0] != '"' else v for v in df['Name_Parentesis']]
    df['Name'] = [v[0] for v in df['Name']]
    df['Name'] = df['Name'].str.rstrip()
    
    df['Name'] = df['Name'].str.split('"')
    df['Nickname'] = [v[1] if len(v) > 1 else 'NAO_TEM' for v in df['Name']]
    df['Nickname'] = [v if v != '' else 'NAO_TEM' for v in df['Nickname']]
    df['Name'] = [v[0] for v in df['Name']]
    df['Name'] = df['Name'].str.rstrip()
    
    df['First_Name'] = df['Name']
    
    df = pd.concat([df, df_null], ignore_index = False, sort = False).sort_index()
    df = df.drop('Name', axis = 1)
    return df
    
#Separa o prefixo do número do ticket e separa o prefixo nas palavras principais que o compõe
def quebraTicket(df_inp):
    df = df_inp[df_inp['Ticket'].isnull() == False].copy()
    df_null = df_inp[df_inp['Ticket'].isnull() == True].copy()
    
    #Separa o ticket no número e no seu prefixo (se possuir)
    df['Ticket_rev'] = [reverseSlicing(s) for s in df['Ticket']]
    
    lista = []
    for s in df['Ticket_rev']:
        if(s.find(' ') != -1):
            lista.append(s[:s.find(' ')])
        else:
            if(s.isnumeric()):
                lista.append(s)
            else:
                lista.append('0')
    df['Ticket_rev'] = lista
    
    df['Ticket_num'] = [reverseSlicing(s) for s in df['Ticket_rev']]
    
    df['Ticket_pref'] = [df['Ticket'][i].replace(df['Ticket_num'][i], '') for i in range(0, len(df))]
    df['Ticket_pref'] = [s.replace(' ', '') if s != '' else 'SEM_PREF'  for s in df['Ticket_pref']]
    
    df['Ticket_pref'] = df['Ticket_pref'].str.replace('.', '')
    '''
    #Remove os elementos desnecessários (separadores e marcadores)
    df['Ticket_pref'] = df['Ticket_pref'].str.replace('/', ' ')
    df['Ticket_pref'] = df['Ticket_pref'].str.split(' ') #split as strings dos nomes pelos espaços
    '''
    
    df['Ticket_num'] = pd.to_numeric(df['Ticket_num'])
    df = pd.concat([df, df_null], ignore_index = False, sort = False).sort_index() 
    df = df.drop('Ticket_rev', axis = 1)
    df = df.drop('Ticket', axis = 1)    
    return df

#Separa em uma contagem de quantas cabines existem, e lista de prefixos delas e a lista de números delas
def quebraCabins(df_inp):
    df = df_inp[df_inp['Cabin'].isnull() == False].copy()
    df_null = df_inp[df_inp['Cabin'].isnull() == True].copy()
    
    df['list_cabins'] = df['Cabin'].str.split(' ') #split as strings dos nomes pelos espaços
    df['Cabin_quant'] = [len(c) for c in df['list_cabins']]
    
    lista_pref = []
    lista_num = []
    for lista in df['list_cabins']:
        alphas = [splitStringAlpha(s) for s in lista]
        #alphas = list(dict.fromkeys(alphas))
        nums = [splitStringNumber(s) for s in lista]
        nums = list(dict.fromkeys(nums))
        lista_pref.append(alphas)
        lista_num.append(nums)
    df['Cabin_pref'] = lista_pref
    df['Cabin_num'] = lista_num
    df = df.drop('list_cabins', axis = 1)
    
    #Pega o prefixo mais comum, quando tem mais de uma cabine
    lista_pref = []
    for v in df['Cabin_pref']:
        alphas_disc = list(dict.fromkeys(v))
        count = []
        for w in alphas_disc:
            count.append(np.sum([1 if c == w else 0 for c in alphas_disc]))
            dft = pd.DataFrame(columns = ['alpha', 'count'])
        dft['alpha'] = alphas_disc
        dft['count'] = count
        dft = dft.sort_values(by = 'count', ascending = False).reset_index(drop = True)
        lista_pref.append(dft['alpha'][0])
    df['Cabin_pref'] = lista_pref

    #Pega a média dos números das cabines que a pessoa estava
    lista_num = []
    for v in df['Cabin_num']:
        lista_aux = []
        for c in v:
            if(c != 'S'):
                lista_aux.append(float(c))
        if(len(lista_aux) > 0):
            lista_num.append(np.mean(lista_aux))
        else:
            lista_num.append(0)
    df['Cabin_num'] = lista_num

    df = pd.concat([df, df_null], ignore_index = False, sort = False).sort_index()
    df = df.drop('Cabin', axis = 1)
    return df

#Cria as variaveis de idade máxima e mínima das pessoas com o mesmo Ticket
def createMaxMinAge(df_inp, df_test_inp):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    
    tam_df = len(df)
    df_aux = pd.concat([df, df_test], ignore_index = False, sort = False).reset_index(drop = True)
    
    lista_aux1 = []
    lista_aux2 = []
    for i in range(0, len(df_aux)):
        df_temp = df_aux[(df_aux['Ticket_pref'] == df_aux['Ticket_pref'][i]) & (df_aux['Ticket_num'] == df_aux['Ticket_num'][i])]
        lista_aux1.append(max(df_temp['Age']))
        lista_aux2.append(min(df_temp['Age']))

    df['Age_max'] = lista_aux1[:tam_df]
    df_test['Age_max'] = lista_aux1[tam_df:]
    
    df['Age_min'] = lista_aux2[:tam_df]
    df_test['Age_min'] = lista_aux2[tam_df:]
    return df, df_test

#--------------------------------------------
#Cria variaveis que são potências e combinações lineares das existentes -> pois os modelos são Algebra Linear

#função que cria a combinação das variaveis de entrada até ordem k
def combinationsColumns(X, k):
    X_new = X.copy()
    for i in range(2, k+1):
        cc = list(combinations_with_replacement(X.columns, r = i))
        X_aux = pd.DataFrame(columns = cc)
        for c in cc:
            X_aux[c] = X[c[0]]
            if(i == 2):
                X_aux[c] = X_aux[c]*X_new[c[1]]
            else:
                X_aux[c] = X_aux[c]*X_new[c[1:]]
        X_new = pd.concat([X_new, X_aux], axis = 1, sort = False)
    return X_new  

#função que cria a combinação das variaveis de entrada até ordem k considerando termos de série de laurent também
def combinationsColumnsLaurent(X, k):
    cols_ini = list(X.columns)
    X_tot = X.copy()
    X_tot = X_tot.join((1/X_tot).add_prefix('1/'))
    X_inv = X_tot.drop(cols_ini, axis = 1).copy()
    cols_inv = list(X_inv.columns)
    
    #Para evitar de fazer a conta (1/x)*x = 1 ou (1/x)(1/x)*x = x
    vars = [[cols_ini[0]], [cols_inv[0]]]
    for i in range(1, len(cols_ini)):
        vars_temp = vars.copy()
        for j in range(0,len(vars_temp)):
            v_new1 = vars_temp[j].copy()
            v_new2 = vars_temp[j].copy()
            v_new1.append(cols_ini[i])
            v_new2.append(cols_inv[i])
            vars.pop(0)
            vars.append(v_new1)
            vars.append(v_new2)
    #Pega todas as colunas que vamos ter que calcular, removendo as repetições
    X_new = X_tot.copy()
    ccs = []
    for i in range(0,len(vars)):
        for j in range(2, k+1):
            cc = list(combinations_with_replacement(vars[i], r = j))
            ccs.extend(cc)
    ccs = list(dict.fromkeys(ccs))
    #Ordena por ordem crescente de termos
    tam_ccs = []
    for c in ccs:
        tam_ccs.append(len(c))
    df_ccs = pd.DataFrame(columns = ['ccs', 'tam'])
    df_ccs['ccs'] = ccs
    df_ccs['tam'] = tam_ccs
    df_ccs = df_ccs.sort_values(by = 'tam', ascending = True).reset_index(drop = True)
    #Calculas as novas colunas e adiciona do dataset
    X_aux = pd.DataFrame(columns = ccs)
    for c in list(df_ccs['ccs']):
        X_aux[c] = X_tot[c[0]]
        if(len(c) == 2):
            X_aux[c] = X_aux[c]*X_new[c[1]]
        else:
            X_aux[c] = X_aux[c]*X_new[c[1:]]
        X_new[c] = X_aux[c]
    return X_new

#Cria o dataframe com os coeficientes do termo
def createDfTermo(name):
    df_term = pd.DataFrame(columns = ['key', 'expo'])
    if(type(name) == str):
        if(name[0] == '1'):
            df_term.loc[len(df_term)] = [name[2:], -1]
        elif(name != 'cte'):
            df_term.loc[len(df_term)] = [name, 1]
        elif(name == 'cte'):
            df_term.loc[len(df_term)] = ['cte', 0]  
    elif(type(name) != str):
        hist = groupby(name)
        for key, group in hist:
            expo = len(list(group))
            if(key[0] == '1'):
                 df_term.loc[len(df_term)] = [key[2:], -1*expo]
            else:
                df_term.loc[len(df_term)] = [key, expo]
    return df_term

#Cria um nome fácil para colocar na variavel
def createName(df_term):
    new_name = ''
    for i in range(0, len(df_term)):
        key = df_term['key'][i]
        expo = df_term['expo'][i]
        if(expo != 0):
            if(expo != 1):
                new_name = new_name + '(' + key + '^' + str(expo) + ')'
            else:
                new_name = new_name + '(' + key + ')'
    if(len(df_term) == 1 and new_name != ''):
         new_name = new_name[1:len(new_name)-1]
    if(new_name == ''):
        new_name = 'cte'
    return new_name

#Expande as variaveis de entrada em suas combinações e separa a entrada da saída
def expandVariables(df_inp, df_test_inp, colunas, laurent, order):
    df = df_inp.copy()
    X = df[colunas].copy()
    if(laurent):
        for c in colunas:
            min_value1 = min(np.abs(df_inp[df_inp[c] != 0][c]))
            min_value2 = min(np.abs(df_test_inp[df_test_inp[c] != 0][c]))
            min_value = min(min_value1, min_value2)
            X[c] = X[c].replace(0, 0.5*min_value) #Tira os zeros e aproxima por um número "pequeno"
        X = combinationsColumnsLaurent(X, order)
    else:
        X = combinationsColumns(X, order)

    colunas_novas = X.columns
    for col in colunas_novas:
        df_term = createDfTermo(col)
        X = X.rename(columns = {col: createName(df_term)})
    
    df = df.drop(colunas, axis = 1)
    df = pd.concat([df, X], axis = 1, sort = False)
    return df

#-------------------------------------------
#Cria variaveis por categorização de variaveis contínuas ou discretas e ordenáveis

#Categoriza pelo critério MDL
def createCategozation(df_inp, df_test_inp, lista_categorizar, nome_alvo):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    for col in lista_categorizar:
        numeric_features = np.arange(df[[col]].shape[1])
        discretizer = MDLP_Discretizer(features = numeric_features)
        discretizer.fit(df[[col]].values, df[[nome_alvo]].values)
        X_discretized = discretizer.transform(df[[col]].values)
        X_test_discretized = discretizer.transform(df_test[[col]].values)
        values = [v[0] for v in X_discretized]
        values = list(dict.fromkeys(values))
        print('----------------------------')
        print(col)
        if(len(values) > 1):
            df[col + '_MDL'] = X_discretized
            df_test[col + '_MDL'] = X_test_discretized
            #print(str(discretizer._cuts[0]))
            print(str(discretizer._bin_descriptions[0]))
        else:
            print('Não tem uma boa discretização')
        
    return df, df_test

def createPCAVariables(df_inp, df_test_inp, lista_colunas, num_comp):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    len_start = len(df)
    pca = PCA(n_components = num_comp)
    df_aux = df[lista_colunas]
    df_test_aux = df_test[lista_colunas]
    df_aux = removeVarsPCA(df_aux)
    df_test_aux = removeVarsPCA(df_test_aux)
    df_aux = pd.concat([df_aux, df_test_aux], ignore_index = True, sort = False).reset_index(drop = True)
    pca.fit(df_aux)
    pca_fitted = pca.fit_transform(df_aux)
    X_pca = pd.DataFrame(pca_fitted, columns = ['var_pca_' + str(i) for i in range(0, num_comp)])
    df = pd.concat([df, X_pca.loc[df_aux.index[:len_start],:]], axis = 1, ignore_index = False, sort = False)
    df_test = pd.concat([df_test, X_pca.loc[df_aux.index[len_start:],:].reset_index(drop = True)], axis = 1, ignore_index = False, sort = False)
    df = df.loc[:,~df.columns.duplicated()]
    df_test = df_test.loc[:,~df_test.columns.duplicated()]
    return df, df_test
    
#-------------------------------------------
#Cria variáveis por contagem de valores

#Cria a variavel contando quantas linhas tem o mesmo valor da variavel da linha atual
#ou um conjuntos de variveis tem mesmo valor
def createCountingEqualValues(df_inp, df_test_inp, col, **kwargs):
    col_name = kwargs.get('col_name', '')
    
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    
    tam_df = len(df)
    df_aux = pd.concat([df, df_test], ignore_index = False, sort = False).reset_index(drop = True)
    
    lista_aux = []
    inds = df_aux.index
    for i in range(0, len(df_aux)):
        if(type(col) == str):
            if(df_aux.loc[inds[i], [col]].isnull().sum() == 0):
                df_temp = df_aux[df_aux[col] == df_aux[col][i]]
                lista_aux.append(len(df_temp))
            else:
                lista_aux.append(np.nan)
        else:
            if(df_aux.loc[inds[i], col].isnull().sum() == 0):
                df_temp = df_aux.copy()
                for c in col:
                    if(len(df_temp) > 0):
                        df_temp = df_temp[df_temp[c] == df_temp[c][i]]
                lista_aux.append(len(df_temp))
            else:
                lista_aux.append(np.nan)
                
    if(col_name == ''):        
        if(type(col) == str):
            nome_coluna = 'Equal_' + col
        else:
            nome_coluna = 'Equal_'
            for c in col:
                nome_coluna = nome_coluna + c + '_'
            nome_coluna = nome_coluna[:len(nome_coluna)-1]
    else:
        nome_coluna = col_name
    
    df[nome_coluna] = lista_aux[:tam_df]
    df_test[nome_coluna] = lista_aux[tam_df:]
    return df, df_test

#Repete o processo de contagem para uma lista de colunas de interesse
def createCountingEqual(df_inp, df_test_inp, lista_col):
    df = df_inp.copy()
    df_test = df_test_inp.copy()    
    for c in lista_col:
        df, df_test = createCountingEqualValues(df, df_test, c)
    return df, df_test

#-------------------------------------------
#Funções temporárias (dar uma olhada melhor!!)

#Categoriza variavel Honorific (separei na mão e no bom senso, tem jeito melhor?)
def categorizaHonorific(df_inp):
    df = df_inp.copy()
    indices0 = df[df['Honorific'].isin(['Mr', 'Miss', 'Mrs', 'Sir', 'Ms', 'Mlle', 'Mme'])].index
    indices1 = df[df['Honorific'].isin(['Major', 'Capt', 'Col'])].index
    indices2 = df[df['Honorific'].isin(['Master', 'Dr', 'Rev'])].index
    indices3 = df[df['Honorific'].isin(['the Countess', 'Lady', 'Don', 'Jonkheer', 'Dona'])].index
    df['Honorific_Cat'] = df['Honorific']
    df['Honorific_Cat'][indices0] = '0'
    df['Honorific_Cat'][indices1] = '1'
    df['Honorific_Cat'][indices2] = '2'
    df['Honorific_Cat'][indices3] = '3'
    #df = df.drop('Honorific', axis = 1)
    return df

def createAuxiliarContinuousVariables(df_inp, df_test_inp, col):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    maxi = max(df[col].max(), df_test[col].max())
    len_tot = len(df) + len(df_test)
    mean = (len(df)/len_tot)*df[col].mean() + (len(df_test)/len_tot)*df_test[col].mean()
    ln_min = min(min(np.log(df[df[col] > 0][col])), min(np.log(df_test[df_test[col] > 0][col])))
    dfs = [df, df_test]
    for dfa in dfs:
        dfa[col + '/max'] = dfa[col]/maxi
        dfa[col + '/mean'] = dfa[col]/mean
        #dfa[col + '2'] = dfa[col]**2
        dfa['Sin(' + col + ')'] = np.sin(dfa[col])
        dfa['Cos(' + col + ')'] = np.cos(dfa[col])
        dfa['ln(' + col + ')'] =  np.where(dfa[col] > 0, np.log(dfa[col]), ln_min - 1)      
    return dfs[0], dfs[1]

def criarVariaveisMagicas(df_inp, df_test_inp):
    dfs = [df_inp.copy(), df_test_inp.copy()]
    for df in dfs:
        df['Age_min+Age_max'] = df['Age_min'] + df['Age_max']
        df['|(Age_min,Age_max)|'] = ((df['Age_min']**2) + (df['Age_max']**2))**(1/2)
        df['|(Age,Fare)|'] = ((df['Age']**2) + (df['Fare']**2))**(1/2)
    lista = ['Fare', 'Age', 'Age_min', 'Age_max', 'Age_min+Age_max', '|(Age_min,Age_max)|', '|(Age,Fare)|']     
    df = dfs[0]
    df_test = dfs[1]
    for v in lista:
        df, df_test = createAuxiliarContinuousVariables(df, df_test, v)
    
    return df, df_test