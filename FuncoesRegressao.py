import sys
import random
import pandas as pd
import numpy as np
import DisplayProgressoIPython as mydisplay #biblioteca minha
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from scipy.stats import ks_2samp
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import gc

#########################################

#Filtra dataframe apenas com colunas que possuem uma certa string no nome
def selectColumns(df, string):
    colunas = list(df.columns)
    colunas_manter = []
    for col in colunas:
        if(col.find(string) != -1):
            colunas_manter.append(col)
    df_filtrado = df[colunas_manter]
    return df_filtrado
    
###############################

def calculaMetricasModelos(df_train_values, df_test_values, df_importance_vars, num_vars = 10, plot = False):    
    #Calcula a importância média e o desvio padrão das importâncias entre os modelos e amostras
    df_vars = pd.DataFrame()
    if(df_importance_vars.empty == True):
        print('Não tem Importância de Variáveis')
    else:
        df_vars['media'] = df_importance_vars.mean(axis = 1)
        df_vars['desvio'] = df_importance_vars.std(axis = 1).fillna(0)
        df_vars_plot = df_vars.sort_values(by = ['media'], ascending = False)
        df_vars_plot.index = [str(ind) for ind in df_vars_plot.index]
        #Plota a importância
        if(plot == True):
            num_max = min(num_vars, len(df_vars))
            fig, axs = plt.subplots(1, 1)
            sns.barplot(x = df_vars_plot['media'][:num_max], y = df_vars_plot.index[:num_max], xerr = df_vars_plot['desvio'][:num_max])
            plt.xlabel('Score de Importância')
            plt.ylabel('Variáveis')
            plt.title("Importância das Variáveis")
            plt.show()
    
    df_params = pd.DataFrame()
    num_amostras = len(selectColumns(df_train_values, 'v').columns)
    
    #Plota as curvas R^2
    #if(plot == True):
    #    fig, axs = plt.subplots(1, 2, figsize = [12, 4])
    r2s_train = []
    r2s_test = []
    rmses_train = []
    rmses_test = []
    for i in range(0, num_amostras):
        y_train = df_train_values['v'+str(i+1)].dropna(axis = 0)
        y_train_value = selectColumns(df_train_values, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
        r2_train = np.sqrt(metrics.r2_score(y_train, y_train_value))
        r2s_train.append(r2_train)
        rmse_train = (np.sum((y_train - y_train_value)**2)/len(y_train))**(0.5)
        rmses_train.append(rmse_train)
        if(len(df_test_values) > 0):
            y_test = df_test_values['v'+str(i+1)].dropna(axis = 0)
            y_test_value = selectColumns(df_test_values, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
            r2_test = np.sqrt(metrics.r2_score(y_test, y_test_value))
            r2s_test.append(r2_test)
            rmse_test = (np.sum((y_test - y_test_value)**2)/len(y_test))**(0.5)
            rmses_test.append(rmse_test)
        else:
            r2s_test.append(np.nan)
            rmses_test.append(np.nan)
    
    df_params['R2_Treino'] = r2s_train
    df_params['R2_Teste'] = r2s_test
    df_params['RMSE_Treino'] = rmses_train
    df_params['RMSE_Teste'] = rmses_test
    
    df_params.loc['Media'] = df_params.mean()
    df_params.loc['Desvio'] = df_params.std()
    if(plot == True):
        display(df_params.loc[['Media', 'Desvio'], :])
    
    return df_vars, df_params
    
############################################################

def avaliaModelosEstacionarios(models_inp, df, colunas_id, nome_alvo, frac_teste = 0.5, train_cv = 0, amostras = 10, seed = None):
    #Faz uma cópia dos modelos
    models = []
    for m in models_inp:
        models.append(clone(m))
    
    if(seed == None):
        #Faz as coisas serem realmente aleatórias
        seedValue = random.randrange(sys.maxsize)
        random.seed(seedValue)
    else:
        random.seed(seed)
    
    #Faz as divisões em treino e teste (nesse caso é melhor ser divisão balanceada)
    X = df.drop(colunas_id, axis = 1)
    X = X.drop(nome_alvo, axis = 1)
    y = df[[nome_alvo]]
    
    if(frac_teste == 0):
        frac_teste = 0.5
        so_tem_treino = True
        if(amostras != 1):
            print('Só é possível uma amostra quando não há fração de teste')
            amostras = 1
    else:
        so_tem_treino = False
    splits = StratifiedShuffleSplit(n_splits = amostras, test_size = frac_teste)
    splits.get_n_splits(df, [0 for l in range(0, len(df))])
    
    lista_vars = list(X.columns) #Pega a lista das variaveis de entrada
    
    #Inicia as bases que vai guardar os dados do modelo
    df_models = pd.DataFrame(columns = ['Nome_Modelo', 'Objeto'])
    df_train_values = pd.DataFrame()
    df_test_values = pd.DataFrame()
    df_importance_vars = pd.DataFrame()
    dh, prog_ant, tempo_ini = [], 0, 0
    i = 0
    
    for train_index, test_index in splits.split(df, [0 for l in range(0, len(df))]):
        dh, prog_ant, tempo_ini = mydisplay.updateProgress(i/amostras, 2, dh, prog_ant, tempo_ini)
        i = i + 1
        
        #Aplica a divisão de treino e teste
        if(so_tem_treino == False):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]
            df_train = df.iloc[train_index, :]
        else:
            X_train, y_train = X.iloc[df.index, :], y.iloc[df.index, :]
            X_test = pd.DataFrame(columns = X_train.columns)
            y_test = pd.DataFrame(columns = y_train.columns)
            df_train = df.iloc[df.index, :]
        
        #Treina os modelos e salva a importância das variaveis e probabilidades preditas de cada um
        j = 0
        for m in models:
            if(train_cv == 0):
                j = j + 1
                model_temp = clone(m)
                model_temp.fit(X_train, y_train[nome_alvo])
                #Salva a importância das variáveis e predições
                try:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp.feature_importances_, index = lista_vars)
                except:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                df_train_values['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp]
                if(len(X_test) > 0):
                    df_test_values['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_test)
                del model_temp
                gc.collect()
            else:
                #Faz as divisões para treino com validação cruzada (esse é bom que seja aleatório mesmo)
                splits2 = ShuffleSplit(n_splits = train_cv, test_size = 0.5)
                splits2.get_n_splits(df_train, [0 for l in range(0, len(df_train))])
                for train_index1, train_index2 in splits2.split(df_train, [0 for l in range(0, len(df_train))]):
                    j = j + 1
                    X_train1, X_train2 = X_train.iloc[train_index1, :], X_train.iloc[train_index2, :]
                    y_train1, y_train2 = y_train.iloc[train_index1, :], y_train.iloc[train_index2, :]
                    
                    eval_set1 = [(X_train1, y_train1[nome_alvo]), (X_train2, y_train2[nome_alvo])]
                    model_temp1 = clone(m)
                    model_temp1.fit(X_train1, y_train1[nome_alvo], eval_set = eval_set1, verbose = 0)
                    #Salva a importância das variáveis e predições
                    try:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp1.feature_importances_, index = lista_vars)
                    except:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                    df_train_values['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp1]
                    if(len(X_test) > 0):
                        df_test_values['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_test)
                    del eval_set1, model_temp1
                    gc.collect()
                    
                    j = j + 1
                    eval_set2 = [(X_train2, y_train2[nome_alvo]), (X_train1, y_train1[nome_alvo])]
                    model_temp2 = clone(m)
                    model_temp2.fit(X_train2, y_train2[nome_alvo], eval_set = eval_set2, verbose = 0)
                    #Salva a importância das variáveis e predições
                    try:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp2.feature_importances_, index = lista_vars)
                    except:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                    df_train_values['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp2]
                    if(len(X_test) > 0):
                        df_test_values['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_test)
                    del eval_set2, model_temp2
                    del X_train1, X_train2, y_train1, y_train2
                    gc.collect()
                del splits2
                gc.collect()
        del X_train, X_test, df_train
        gc.collect()
        
        #Salva as respostas para poder comparar a predição com a resposta depois
        if(len(df_test_values) > 0):
            df_train_values['v'+str(i)] = y_train.reset_index(drop = True)
            df_test_values['v'+str(i)] = y_test.reset_index(drop = True)
        else:
            df_train_values['v'+str(i)] = y_.reset_index(drop = True)
            df_test_values['v'+str(i)] = y.reset_index(drop = True)
        del y_train, y_test
        gc.collect()
        
        #Salva os índices do split dos dados de treino e teste
        if(len(df_test_values) > 0):
            df_train_values['i'+str(i)] = train_index
            df_test_values['i'+str(i)] = test_index
        else:
            df_train_values['i'+str(i)] = df.index
            df_test_values['i'+str(i)] = pd.DataFrame(columns = df_train_values.columns)
    
    for m in models:
        del m
    del models, X, y, lista_vars, splits
    gc.collect()
    dh, prog_ant, tempo_ini = mydisplay.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    return df_models, df_train_values, df_test_values, df_importance_vars
    
########################################################

#OBS: train_cv = -1 força encontrar regras atemporais em algoritmos com cross validation interno
def avaliaModelosTemporais(models_inp, df, colunas_id, nome_alvo, train_cv = 0, frac_teste = 0.5, coluna_tempo = None,
                           testar_reversao = True):
    #Faz uma cópia dos modelos
    models = []
    for m in models_inp:
        models.append(clone(m))
    
    #Faz as divisões em treino e teste (nesse caso é melhor ser divisão balanceada)
    X = df.drop(colunas_id, axis = 1)
    X = X.drop(nome_alvo, axis = 1)
    y = df[[nome_alvo]]
    
    if(frac_teste == 0):
        frac_teste = 0.5
        so_tem_treino = True
    else:
        so_tem_treino = False
    
    lista_vars = list(X.columns) #Pega a lista das variaveis de entrada
    
    #Inicia as bases que vai guardar os dados do modelo
    df_models = pd.DataFrame(columns = ['Nome_Modelo', 'Objeto'])
    df_train_values = pd.DataFrame()
    df_test_values = pd.DataFrame()
    df_importance_vars = pd.DataFrame()
    
    df_loop = pd.DataFrame(columns = ['train_index', 'test_index'])
    #Calcula as divisões de treino e teste no loop
    if(so_tem_treino == False):
        #Separa a amostra fora do tempo no futuro como teste
        datas = list(df[coluna_tempo].value_counts().index)
        datas.sort()
        
        datas_train = datas[:int(len(datas)*(1-frac_teste))]
        datas_ft = datas[int(len(datas)*(1-frac_teste)):]
        train_index = df[df[coluna_tempo].isin(datas_train)].index
        test_index = df[df[coluna_tempo].isin(datas_ft)].index
        df_loop.loc[len(df_loop)] = [train_index, test_index]
        del datas_train, datas_ft, train_index, test_index
        gc.collect()
        
        #Separa a amostra fora do tempo no passado como teste
        if(testar_reversao == True):
            datas_train = datas[int(len(datas)*(frac_teste)):]
            datas_ft = datas[:int(len(datas)*(frac_teste))]
            train_index = df[df[coluna_tempo].isin(datas_train)].index
            test_index = df[df[coluna_tempo].isin(datas_ft)].index
            df_loop.loc[len(df_loop)] = [train_index, test_index]
            del datas_train, datas_ft, train_index, test_index
            gc.collect()
            
        del datas
        gc.collect()
    else:
        datas = list(df[coluna_tempo].value_counts().index)
        datas.sort()
        datas_train = datas[:int(len(datas)*(0.5))]
        datas_ft = datas[int(len(datas)*(0.5)):]
        train_index = df[df[coluna_tempo].isin(datas_train)].index
        test_index = df[df[coluna_tempo].isin(datas_ft)].index
        df_loop.loc[len(df_loop)] = [train_index, test_index]
        del datas_train, datas_ft, train_index, test_index, datas
        gc.collect()
    
    dh, prog_ant, tempo_ini = [], 0, 0
    i = 0
    for l in range(0, len(df_loop)):
        dh, prog_ant, tempo_ini = mydisplay.updateProgress(i/len(df_loop), 2, dh, prog_ant, tempo_ini)
        i = i + 1
        
        indice = df_loop.index[l]
        train_index = df_loop.loc[indice, 'train_index']
        test_index = df_loop.loc[indice, 'test_index']
        
        if(so_tem_treino == False):
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
            y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]
            df_train = df.loc[train_index, :]
        else:
            X_train, y_train = X.loc[df.index, :], y.loc[df.index, :]
            X_test = pd.DataFrame(columns = X_train.columns)
            y_test = pd.DataFrame(columns = y_train.columns)
            df_train = df.loc[df.index, :]
    
        df_train_values_temp = pd.DataFrame()
        df_test_values_temp = pd.DataFrame()
    
        #Treina os modelos e salva a importância das variaveis e probabilidades preditas de cada um
        j = 0
        for m in models:
            if(train_cv == 0):
                j = j + 1
                model_temp = clone(m)
                model_temp.fit(X_train, y_train[nome_alvo])
                #Salva a importância das variáveis e predições
                try:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp.feature_importances_, index = lista_vars)
                except:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                df_train_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp]
                if(len(X_test) > 0):
                    df_test_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_test)
                del model_temp
                gc.collect()
            elif(train_cv > 0):
                #Faz as divisões para treino com validação cruzada (esse é bom que seja aleatório mesmo)
                splits2 = ShuffleSplit(n_splits = train_cv, test_size = 0.5)
                splits2.get_n_splits(df_train, [0 for l in range(0, len(df_train))])
                for train_index1, train_index2 in splits2.split(df_train, [0 for l in range(0, len(df_train))]):
                    j = j + 1
                    X_train1, X_train2 = X_train.iloc[train_index1, :], X_train.iloc[train_index2, :]
                    y_train1, y_train2 = y_train.iloc[train_index1, :], y_train.iloc[train_index2, :]

                    eval_set1 = [(X_train1, y_train1[nome_alvo]), (X_train2, y_train2[nome_alvo])]
                    model_temp1 = clone(m)
                    model_temp1.fit(X_train1, y_train1[nome_alvo], eval_set = eval_set1, verbose = 0)
                    #Salva a importância das variáveis e predições
                    try:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp1.feature_importances_, index = lista_vars)
                    except:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                    df_train_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp1]
                    if(len(X_test) > 0):
                        df_test_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_test)
                    del eval_set1, model_temp1
                    gc.collect()

                    j = j + 1
                    eval_set2 = [(X_train2, y_train2[nome_alvo]), (X_train1, y_train1[nome_alvo])]
                    model_temp2 = clone(m)
                    model_temp2.fit(X_train2, y_train2[nome_alvo], eval_set = eval_set2, verbose = 0)
                    #Salva a importância das variáveis e predições
                    try:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp2.feature_importances_, index = lista_vars)
                    except:
                        df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                    df_train_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp2]
                    if(len(X_test) > 0):
                        df_test_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_test)
                    del eval_set2, model_temp2
                    del X_train1, X_train2, y_train1, y_train2
                    gc.collect()
                del splits2
                gc.collect()
            elif(train_cv == -1):
                df_train = df_train.sort_values(coluna_tempo, ascending = True)
                train_index1 = df_train.iloc[:int(len(df_train)*0.5), :].index
                train_index2 = df_train.iloc[int(len(df_train)*0.5):, :].index
                j = j + 1
                X_train1, X_train2 = X_train.loc[train_index1, :], X_train.loc[train_index2, :]
                y_train1, y_train2 = y_train.loc[train_index1, :], y_train.loc[train_index2, :]

                eval_set1 = [(X_train1, y_train1[nome_alvo]), (X_train2, y_train2[nome_alvo])]
                model_temp1 = clone(m)
                model_temp1.fit(X_train1, y_train1[nome_alvo], eval_set = eval_set1, verbose = 0)
                #Salva a importância das variáveis e predições
                df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp1.feature_importances_, index = lista_vars)
                df_train_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp1]
                if(len(X_test) > 0):
                    df_test_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_test)
                del eval_set1, model_temp1
                gc.collect()

                j = j + 1
                eval_set2 = [(X_train2, y_train2[nome_alvo]), (X_train1, y_train1[nome_alvo])]
                model_temp2 = clone(m)
                model_temp2.fit(X_train2, y_train2[nome_alvo], eval_set = eval_set2, verbose = 0)
                #Salva a importância das variáveis e predições
                df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp2.feature_importances_, index = lista_vars)
                df_train_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp2]
                if(len(X_test) > 0):
                    df_test_values_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_test)
                del eval_set2, model_temp2
                del X_train1, X_train2, y_train1, y_train2
                gc.collect()
                
        del X_train, X_test, df_train
        gc.collect()

        #Salva as respostas para poder comparar a predição com a resposta depois
        if(len(df_test_values_temp) > 0):
            df_train_values_temp['v'+str(i)] = y_train.reset_index(drop = True)
            df_test_values_temp['v'+str(i)] = y_test.reset_index(drop = True)
        else:
            df_train_values_temp['v'+str(i)] = y.reset_index(drop = True)
        del y_train, y_test
        gc.collect()

        #Salva os índices do split dos dados de treino e teste
        if(len(df_test_values_temp) > 0):
            df_train_values_temp['i'+str(i)] = train_index
            df_test_values_temp['i'+str(i)] = test_index
        else:
            df_train_values_temp['i'+str(i)] = df.index
            df_test_values_temp = pd.DataFrame(columns = df_train_values_temp.columns)
            
        df_train_values = pd.concat([df_train_values, df_train_values_temp], sort = False, axis = 1)
        df_test_values = pd.concat([df_test_values, df_test_values_temp], sort = False, axis = 1)
        del df_train_values_temp, df_test_values_temp
        gc.collect()
    
    for m in models:
        del m
    del models, X, y, lista_vars, df_loop
    gc.collect()
    dh, prog_ant, tempo_ini = mydisplay.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    return df_models, df_train_values, df_test_values, df_importance_vars