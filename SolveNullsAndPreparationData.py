import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

#pip install datawig
#pip install tpot
from sklearn import metrics
import datawig
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tpot as tpt

from sklearn import preprocessing

#-------------------------------------------
#Funções básicas de tratamento de nulos

def completeWithMean(df_inp, df_test_inp, col, force_int):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    
    df_aux = pd.concat([df, df_test], ignore_index = False, sort = False).reset_index(drop = True)
    
    df2_notnull = df_aux[df_aux[col].isnull() == False].copy()
    
    #Avalia o "modelo"
    queda, acuracia_erro = avaliaMedia(df2_notnull, col, force_int, 0.5, 10)
    
    #Calcula as predições na base total como treino
    if(force_int):
        media = round(df2_notnull[col].mean())
    else:
        media = df2_notnull[col].mean()
    y_prev = [media for i in range(0, len(df2_notnull))]
    y = df2_notnull[col]
    
    printaResumoModelo(col, y, y_prev, queda, acuracia_erro)   
    
    #Faz o preenchimento dos nulos
    if(force_int):
        df[col] = df[col].fillna(round(df_aux[col].mean()))
        df_test[col] = df_test[col].fillna(round(df_aux[col].mean()))
    else:
        df[col] = df[col].fillna(df_aux[col].mean())
        df_test[col] = df_test[col].fillna(df_aux[col].mean())
    
    return df, df_test

def completeWithSpecificValue(df_inp, df_test_inp, col, specific_value):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    
    df[col] = df[col].fillna(specific_value)
    df_test[col] = df_test[col].fillna(specific_value)
    
    return df, df_test

def completeWithMostFrequent(df_inp, df_test_inp, col):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    
    df_aux = pd.concat([df, df_test], ignore_index = False, sort = False).reset_index(drop = True)
    df[col] = df[col].fillna(df_aux[col].value_counts().index[0])
    df_test[col] = df_test[col].fillna(df_aux[col].value_counts().index[0])
    
    return df, df_test

def completeWithModel(df_inp, df_test_inp, colunas_dropar, colunas_dummerizar, nome_alvo, col_imp, nome_model):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    
    #Dropa as colunas que não vamos usar no modelo
    df, df_test = dropColumns(df, df_test, colunas_dropar)
    
    #Dummeriza as colunas que queremos (ou são necessárias)
    df, df_test = dummerizaVariaveis(df, df_test, nome_alvo, colunas_dummerizar)
    
    #Dropa a colunas de alvo
    df, df_test = dropColumns(df, df_test, [nome_alvo])
    
    #Junta os dois datasets
    tam_df = len(df)
    df2 = pd.concat([df, df_test], ignore_index = False, sort = False).reset_index(drop = True)
    
    #Pega só os valoes sem nulo (para treinar)
    df2_notnull = df2[df2[col_imp].isnull() == False].copy()
    #Pega só os valoes nulos (para imputar)
    df2_null = df2[df2[col_imp].isnull() == True].copy()
    
    #Faz o modelo
    if(nome_model == 'datawig'):
        imputer = datawig.SimpleImputer(input_columns = list(df2_notnull.drop(col_imp, axis = 1).columns),
                                        output_column = col_imp, output_path = 'imputer_model')
    elif(nome_model == 'rf'):
        imputer = RandomForestRegressor()
    elif(nome_model == 'xg'):
        imputer = xgb.XGBRegressor()
    elif(nome_model == 'tpot'):
        imputer = tpt.TPOTRegressor(generations = 5, population_size = 40, cv = 5, random_state = 42, n_jobs = -1, verbosity = 2)
        fracao = 0.75
        df_train = df2_notnull.sample(frac = fracao, replace = False).copy()
        df_valid = df2_notnull.drop(df_train.index).copy()
        imputer.fit(df_train.drop(col_imp, axis = 1), df_train[col_imp])
        imputer = imputer.fitted_pipeline_.steps[-1][1] #Pega o melhor modelo do tpot
    
    queda, acuracia_erro = avaliaModelo(imputer, df2_notnull, col_imp, 0.5, 10, nome_model)
    
    #Fit do modelo pra valer
    if(nome_model == 'datawig'):
        imputer.fit(train_df = df2_notnull, num_epochs = 50)
    elif(nome_model == 'xg'):
        fracao = 0.5
        df_train2 = df2_notnull.sample(frac = fracao, replace = False).copy()
        df_valid2 = df2_notnull.drop(df_train2.index).copy().reset_index(drop = True)
        df_train2 = df_train2.reset_index(drop = True)
        eval_set = [(df_train2.drop(col_imp, axis = 1), df_train2[col_imp]), (df_valid2.drop(col_imp, axis = 1), df_valid2[col_imp])]
        imputer.fit(df_train2.drop(col_imp, axis = 1), df_train2[col_imp].values.ravel(), eval_set = eval_set, verbose = False)
    else:
        imputer.fit(df2_notnull.drop(col_imp, axis = 1), df2_notnull[col_imp])
    
    #Calcula as predições na base total como treino
    if(nome_model == 'datawig'):
        y_prev = imputer.predict(df2_notnull.drop(col_imp, axis = 1))[col_imp + '_imputed']
    else:
        y_prev = imputer.predict(df2_notnull.drop(col_imp, axis = 1))
    min_value = min(df2_notnull[col_imp])
    max_value = max(df2_notnull[col_imp])
    y_prev = np.array([v if v >= min_value else min_value for v in y_prev])
    y_prev = np.array([v if v <= max_value else max_value for v in y_prev])
    y = df2_notnull[col_imp]
    
    printaResumoModelo(col_imp, y, y_prev, queda, acuracia_erro)  
    
    #Imputa os valores nos nulos
    if(nome_model == 'datawig'):
        y_prev = imputer.predict(df2_null.drop(col_imp, axis = 1))[col_imp + '_imputed']
    else:
        y_prev = imputer.predict(df2_null.drop(col_imp, axis = 1))
    min_value = min(df2_notnull[col_imp])
    max_value = max(df2_notnull[col_imp])
    y_prev = np.array([v if v >= min_value else min_value for v in y_prev])
    y_prev = np.array([v if v <= max_value else max_value for v in y_prev])    
    df2_null[col_imp] = y_prev
    
    df3 = pd.concat([df2_notnull, df2_null], ignore_index = False, sort = False).sort_index()
    df_aux = df3.iloc[:tam_df,:].copy().reset_index(drop = True)
    df_test_aux = df3.iloc[tam_df:,:].copy().reset_index(drop = True)
    df_inp_final = df_inp.copy()
    df_test_inp_final = df_test_inp.copy()
    df_inp_final[col_imp] = df_aux[col_imp]
    df_test_inp_final[col_imp] = df_test_aux[col_imp]
    return df_inp_final, df_test_inp_final

#-------------------------------------------
#Funções auxiliares de tratamento para aplicar nos modelos

def dropColumns(df_inp, df_test_inp, cols):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    for col in cols:
        try:
            df = df.drop(col, axis = 1)
        except:
            #print(col + ' não existe no primeiro df')
            pass
        try:
            df_test = df_test.drop(col, axis = 1)
        except:
            #print(col + ' não existe no segundo df')
            pass
    return df, df_test

#Faz as variaveis qualitativas nominais ficarem binárias
def dummerizaVariaveis(df_inp, df_test_inp, nome_alvo, columns_inp):
    df = df_inp.copy()
    df = df.drop(nome_alvo, axis = 1)
    df_test = df_test_inp.copy()
    
    len_start = len(df)
    df_aux = pd.concat([df, df_test], ignore_index = True, sort = False).copy().reset_index(drop = True)
    #df_aux = df.append(df_test, sort = False).copy().reset_index(drop = True)
    if(columns_inp == []):
        #df_dummy = pd.get_dummies(df_aux)
        df_dummy = pd.get_dummies(df_aux, dtype = np.int64)
    else:
        df_dummy = pd.get_dummies(df_aux, columns = columns_inp)
    
    df = df_dummy.iloc[:len_start,:].copy().reset_index(drop = True)
    df[nome_alvo] = df_inp[nome_alvo]
    df_test = df_dummy.iloc[len_start:,:].copy().reset_index(drop = True)
    return df, df_test

#Quebra os dados em entrada e saída para os modelos
def splitXY(df_inp, nome_alvo):
    df = df_inp.copy()
    colunas = list(df.columns)
    colunas.remove(nome_alvo)
    X = df[colunas]
    y = df[[nome_alvo]]
    return X, y

def standarizaVariaveis(df_inp, df_test_inp, lista_colunas):
    df = df_inp.copy()
    df_test = df_test_inp.copy()
    len_start = len(df)
    df_aux = df[lista_colunas]
    df_test_aux = df_test[lista_colunas]
    df_aux = pd.concat([df_aux, df_test_aux], ignore_index = True, sort = False).reset_index(drop = True)    
    X_scaled = preprocessing.scale(df_aux)
    X_scaled = pd.DataFrame(X_scaled, columns = list(df_aux.columns))
    df[lista_colunas] = X_scaled.iloc[df_aux.index[:len_start], lista_colunas]
    df_test[lista_colunas] = X_scaled.loc[df_aux.index[len_start:], lista_colunas]
    return df, df_test

#-------------------------------------------
#Funções auxiliares de tratamento para aplicar nos modelos (ESPECIFICAS PARA O TITANIC)

#Remove variaveis de Cabine (Função para o problema específico do Titanic)
def removeVarsCabins(X):
    colunas = list(X.columns)
    colunas_remover = []
    for col in colunas:
        if(col.find('Cabin') != -1):
            colunas_remover.append(col)
    X_limpo = X.drop(colunas_remover, axis = 1).copy()
    return X_limpo

#-------------------------------------------
#Funções de avaliação de fit modificadas

def printaResumoModelo(col, y, y_prev, queda, acuracia_erro):
    #Ajusta a acuracia esperada com base na avaliação de modelo
    acuracia = metrics.r2_score(y, y_prev)
    acuracia_esperada = acuracia - queda
    print('----------------------')
    print(col)
    print('Acurácia Esperada (R^2): ' + str(round(acuracia_esperada*100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(acuracia_erro*100, 1)))
    erro_esperado = np.std(y - y_prev)*np.sqrt(1 - acuracia_esperada)
    print("Erro Esperado:", np.mean(erro_esperado))

def avaliaMedia(df, nome_alvo, force_int, fracao, num_loop):
    score_train = []
    score_valid = []
    quedas = []
    erros_medio = []
    
    for i in range(0, num_loop):
        df_train = df.sample(frac = fracao, replace = False).copy()
        df_valid = df.drop(df_train.index).copy()

        if(force_int):
            media = round(df_train[nome_alvo].mean())
        else:
            media = df_train[nome_alvo].mean()
        yt_prev = [media for i in range(0, len(df_train))]
        yv_prev = [media for i in range(0, len(df_valid))]
        
        score_train.append(metrics.r2_score(df_train[nome_alvo], yt_prev))
        score_valid.append(metrics.r2_score(df_valid[nome_alvo], yv_prev))
        queda = metrics.r2_score(df_train[nome_alvo], yt_prev) - metrics.r2_score(df_valid[nome_alvo], yv_prev)
        erro_medio = np.mean(np.abs(yv_prev - df_valid[nome_alvo]))
        quedas.append(queda)
        erros_medio.append(erro_medio)

    queda_erro = np.std(quedas)
    score_erro = max(np.std(score_train), np.std(score_valid))
    return max(np.mean(quedas), 0), np.sqrt(queda_erro**2 + score_erro**2)

def avaliaModelo(model, df, nome_alvo, fracao, num_loop, nome_model):
    score_train = []
    score_valid = []
    quedas = []
    erros_medio = []
    
    for i in range(0, num_loop):
        df_train = df.sample(frac = fracao, replace = False).copy()
        df_valid = df.drop(df_train.index).copy()

        if(nome_model == 'datawig'):
            model.fit(train_df = df_train, num_epochs = 50)
        elif(nome_model == 'xg'):
            fracao = 0.5
            df_train2 = df_train.sample(frac = fracao, replace = False).copy()
            df_valid2 = df_train.drop(df_train2.index).copy()
            eval_set = [(df_train2.drop(nome_alvo, axis = 1), df_train2[nome_alvo]), (df_valid2.drop(nome_alvo, axis = 1), df_valid2[nome_alvo])]
            model.fit(df_train2.drop(nome_alvo, axis = 1), df_train2[nome_alvo].values.ravel(), eval_set = eval_set, verbose = False)
        else:
            model.fit(df_train.drop(nome_alvo, axis = 1), df_train[nome_alvo])

        if(nome_model == 'datawig'):
            yt_prev = model.predict(df_train.drop(nome_alvo, axis = 1))[nome_alvo + '_imputed']
            yv_prev = model.predict(df_valid.drop(nome_alvo, axis = 1))[nome_alvo + '_imputed']
        else:
            yt_prev = model.predict(df_train.drop(nome_alvo, axis = 1))
            yv_prev = model.predict(df_valid.drop(nome_alvo, axis = 1))
        min_value = min(df_train[nome_alvo])
        max_value = max(df_train[nome_alvo])
        yt_prev = np.array([v if v >= min_value else min_value for v in yt_prev])
        yt_prev = np.array([v if v <= max_value else max_value for v in yt_prev])
        yv_prev = np.array([v if v >= min_value else min_value for v in yv_prev])
        yv_prev = np.array([v if v <= max_value else max_value for v in yv_prev])
        
        score_train.append(metrics.r2_score(df_train[nome_alvo], yt_prev))
        score_valid.append(metrics.r2_score(df_valid[nome_alvo], yv_prev))
        queda = metrics.r2_score(df_train[nome_alvo], yt_prev) - metrics.r2_score(df_valid[nome_alvo], yv_prev)
        erro_medio = np.mean(np.abs(yv_prev - df_valid[nome_alvo]))
        quedas.append(queda)
        erros_medio.append(erro_medio)
    '''
    print('----------------------')
    print(nome_alvo)
    print('Acurácia Treino (R^2): ' + str(round(np.mean(score_train) * 100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(np.std(score_train) * 100, 1)))
    print('Acurácia Validação (R^2): ' + str(round(np.mean(score_valid) * 100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(np.std(score_valid) * 100, 1)))
    print('Queda de Acurácia: ' + str(round(np.mean(quedas) * 100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(np.std(quedas) * 100, 1)))
    print("Erro médio:", np.mean(erros_medio))
    '''
    queda_erro = np.std(quedas)
    score_erro = max(np.std(score_train), np.std(score_valid))
    return max(np.mean(quedas), 0), np.sqrt(queda_erro**2 + score_erro**2)
