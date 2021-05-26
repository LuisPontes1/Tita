import pandas as pd
import numpy as np
from itertools import combinations_with_replacement

#--------------------------------------------
#Cria variaveis que são potências e combinações lineares das existentes -> pois os modelos são Algebra Linear

#Recebe X como uma matriz, k é a ordem máxima da expansão que queremos
def expansao_serie_taylor(X, k):
    Xshape = X.shape
    num_linhas = Xshape[0]
    num_colunas = Xshape[1]
    X_exp = np.empty(shape = [num_linhas, 0])
    colunas_exp = []
    num_novas_comb = num_colunas
    num_comb = 0
    for i in range(2, k+1):
        num_novas_comb = num_novas_comb*(num_colunas + i - 1)/i
        num_comb = num_comb + num_novas_comb
        combinacoes = list(combinations_with_replacement(np.arange(num_colunas), r = i))
        X_aux = np.empty(shape = [num_linhas, 0])
        for c in combinacoes:
            if(i == 2):
                X_aux = np.append(X_aux, 
                                  (X[:, c[0]] * X[:, c[1]]).reshape(num_linhas, 1), 
                                  axis = 1)
            else:
                X_aux = np.append(X_aux, 
                                  (X[:, c[0]] * X_temp[:, dict_colunas_temp[c[1:]]]).reshape(num_linhas, 1), 
                                  axis = 1)
        dict_colunas_temp = dict(zip(combinacoes, np.arange(num_comb, dtype = np.int64) ))
        X_temp = X_aux
        colunas_exp.extend(combinacoes)
        X_exp = np.append(X_exp, X_aux, axis = 1)
    return X_exp, colunas_exp

def combinationsColumns(X, k):
    X_new = X.copy()
    for i in range(2, k+1):
        cc = list(combinations_with_replacement(X.columns, r = i))
        X_aux = pd.DataFrame(columns = cc)
        for c in cc:
            if(i == 2):
                X_aux[c] = X[c[0]].values * X_new[c[1]].values
            else:
                X_aux[c] = X[c[0]].values * X_new[c[1:]].values
        X_new = pd.concat([X_new, X_aux], axis = 1, sort = False)
    return X_new 

def combinationsColumnsLaurent(X, k):
    df_cols = pd.DataFrame(X.columns.tolist(), columns = ['cols_ini'])
    df_cols['cols_inv'] = '1/' + df_cols['cols_ini']
    
    #Para evitar de fazer a conta (1/x)*x = 1 ou (1/x)(1/x)*x = x
    varis = [[]]
    for indice, row in df_cols.iterrows():
        vars_temp = varis.copy()
        for var in vars_temp:
            v_new1 = var.copy()
            v_new2 = var.copy()
            v_new1.append(row['cols_ini'])
            v_new2.append(row['cols_inv'])
            varis.pop(0)
            varis.append(v_new1)
            varis.append(v_new2)

    #Pega todas as colunas que vamos ter que calcular, removendo as repetições
    ccs = []
    for var in varis:
        for j in range(2, k+1):
            cc = list(combinations_with_replacement(var, r = j))
            ccs.extend(cc)
    ccs = list(dict.fromkeys(ccs))
    
    #Ordena por ordem crescente de termos
    df_ccs = pd.DataFrame()
    df_ccs['ccs'] = ccs
    df_ccs['tam'] = df_ccs['ccs'].apply(lambda c: len(c))
    df_ccs = df_ccs.sort_values(by = 'tam', ascending = True).reset_index(drop = True)
   
    #Calculas as novas colunas e adiciona do dataset
    X_tot = X.copy()
    X_tot = X_tot.join((1/X_tot).add_prefix('1/'))    
    for c in df_ccs['ccs'].tolist():
        if(len(c) == 2):
            X_tot[c] = X_tot[c[0]].values * X_tot[c[1]].values
        else:
            X_tot[c] = X_tot[c[0]].values * X_tot[c[1:]].values
    return X_tot

#Expande as variaveis de entrada em suas combinações
def expandVariables(df, colunas_transf, laurent, order):
    if(laurent):
        X = combinationsColumnsLaurent(df[colunas_transf], order)
    else:
        X = combinationsColumns(df[colunas_transf], order) 
    return pd.concat([df.drop(colunas_transf, axis = 1), X], axis = 1, sort = False)