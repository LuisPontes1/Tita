import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import joblib
from sklearn.base import clone
import math
from scipy.stats import ks_2samp
#Minhas Bibliotecas
import DisplayProgressIPython as display
pd.options.mode.chained_assignment = None

#pip install scikit-multilearn
#http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection import iterative_train_test_split

#-------------------------------------------
#Funções de manipulação de lista de modelos

def iniciaListaModelos():
    df_modelos = pd.DataFrame(columns = ['nome', 'modelo'])
    df_res = pd.DataFrame(columns = ['nome', 'acuracia', 'erro', 'fator_qual', 'fator_erro'])
    return df_modelos, df_res

def insereModelo(df_modelos, numero_modelo, clf, **kwargs):
    nome = 'Tpot' + str(numero_modelo)
    df_res = kwargs.get('df_res', 0)  
    df_metr = kwargs.get('df_metr', 0)  
    if(len(df_modelos[df_modelos['nome'] == nome]) == 0):
        df_modelos.loc[len(df_modelos)] = [nome, clf]
        if(type(df_res) != int):
            df_res.loc[len(df_res)] = [nome, df_metr['valor']['Acurácia'], df_metr['erro']['Acurácia'], 
                                        df_metr['valor']['Fator_Qual'], df_metr['erro']['Fator_Qual']] 
    else:
        df_modelos.loc[df_modelos[df_modelos['nome'] == nome].index, :] = [nome, clf]
        if(type(df_res) != int):
            df_res.loc[df_res[df_res['nome'] == nome].index, :] = [nome, df_metr['valor']['Acurácia'], df_metr['erro']['Acurácia'], 
                                                                    df_metr['valor']['Fator_Qual'], df_metr['erro']['Fator_Qual']]         
        
def salvaModelos(n, df_modelos, df_res, pasta_models):
    for i in range(0, len(df_modelos)):
        try:
            joblib_file = "modelos" + str(pasta_models) + "/joblib_models" + str(n) + "_m" + str(i) + ".pkl"
            joblib.dump(df_modelos['modelo'][i], joblib_file)
        except:
            pass
    df_res.to_csv('modelos' + str(pasta_models) + '/df_res' + str(n) + '.csv', index = False)

def carregaModelos(n, pasta_models):
    df_modelos = pd.DataFrame(columns = ['nome', 'modelo'])
    tem_modelo = True
    i = 0
    while(tem_modelo):
        try:
            joblib_file = "modelos" + str(pasta_models) + "/joblib_models" + str(n) + "_m" + str(i) + ".pkl"
            joblib_model = joblib.load(joblib_file)
            insereModelo(df_modelos, i, joblib_model)
            i = i + 1
        except:
            tem_modelo = False
    df_res = pd.read_csv('modelos' + str(pasta_models) + '/df_res' + str(n) + '.csv')
    return df_modelos, df_res

def getMelhoresModelos(df_modelos, df_res):
    indice = df_res[df_res['acuracia'] == df_res['acuracia'].max()].index[0]
    acuracia_max = df_res['acuracia'][indice]
    erro_max = df_res['erro'][indice]
    df_sub = pd.DataFrame(columns = ['indice', 'fator_qual', 'fator_erro'])
    df_sub.loc[len(df_sub)] = [indice, df_res['fator_qual'][indice], df_res['fator_erro'][indice]]
    for i in range(0, len(df_res)):
        if(i != indice and df_res['acuracia'][i] + df_res['erro'][i] > acuracia_max - erro_max):
            df_sub.loc[len(df_sub)] = [i, df_res['fator_qual'][i], df_res['fator_erro'][i]]
    indice = df_sub[df_sub['fator_qual'] == df_sub['fator_qual'].min()].index[0]
    fator_min = df_sub['fator_qual'][indice]
    erro_min = df_sub['fator_erro'][indice]
    df_sub2 = pd.DataFrame(columns = ['indice'])
    for i in range(0, len(df_sub)):
        if(df_sub['fator_qual'][i] - df_sub['fator_erro'][i] < fator_min + erro_min):
            df_sub2.loc[len(df_sub2)] = [df_sub['indice'][i]]    
    best_models = df_modelos.iloc[list(df_sub2['indice']), :]
    return best_models

#-------------------------------------------
#Funções auxiliares de visualização do modelo

def calculaCurvaROCMedia(fpr_inp, tpr_inp, thr_inp):
    fpr = []
    tpr = []
    thr = []
    for i in range(0, len(thr_inp)):
        fpr.extend(fpr_inp[i])
        tpr.extend(tpr_inp[i])
        thr.extend(thr_inp[i])
    df = pd.DataFrame()
    df['thr'] = thr
    df['fpr'] = fpr
    df['tpr'] = tpr
    df = df.sort_values(by = 'fpr', ascending = True).reset_index(drop = True)
  
    fpr_max = df['fpr'].max()
    fpr_min = df['fpr'].min()
    num = 100
    div = (fpr_max - fpr_min)/num
    df_disc = pd.DataFrame(columns = ['decile', 'fpr', 'tpr'])   
    for i in range(0, num - 1):
        df_aux = df[(df['fpr'] >= fpr_min + i*div) & (df['fpr'] < fpr_min + (i+1)*div)]
        df_disc.loc[len(df_disc)] = [i, df_aux['fpr'].mean(), df_aux['tpr'].mean()]
    df_aux = df_aux = df[(df['fpr'] >= fpr_min + (num-1)*div)]
    df_disc.loc[len(df_disc)] = [num - 1, df_aux['fpr'].mean(), df_aux['tpr'].mean()]   
    df_disc = df_disc.sort_values(by = 'fpr', ascending = True).reset_index(drop = True)
    
    df = df.groupby('thr').mean()
    fpr_tudo = df['fpr']
    tpr_tudo = df['tpr']
    
    return fpr_tudo, tpr_tudo, df_disc['fpr'], df_disc['tpr']

def plotaCurvaROC(fpr_t_inp, tpr_t_inp, thr_t_inp, fpr_v_inp, tpr_v_inp, thr_v_inp):
    #Plota a curva roc
    fpr_t, tpr_t, fpr_tm, tpr_tm = calculaCurvaROCMedia(fpr_t_inp, tpr_t_inp, thr_t_inp)
    fpr_v, tpr_v, fpr_vm, tpr_vm = calculaCurvaROCMedia(fpr_v_inp, tpr_v_inp, thr_v_inp)
    fig, axs = plt.subplots(1, 2, figsize = [10, 4])
    axs[0].scatter(fpr_t, tpr_t, color = 'darkorange')
    axs[0].plot(fpr_tm, tpr_tm, color = 'black', lw = 2)
    axs[1].scatter(fpr_v, tpr_v, color = 'darkorange')
    axs[1].plot(fpr_vm, tpr_vm, color = 'black', lw = 2)
    for i in range(0, 2):
        axs[i].plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_ylabel('True Positive Rate')
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()

def plotaCurvaKS(curva_1_inp, curva_0_inp, curva_1v_inp, curva_0v_inp):
    #Plota a curva ks
    fig, axs = plt.subplots(1, 2, figsize = [10, 4])
    for i in range(0, len(curva_1_inp)):
        curva_1 = curva_1_inp[i] 
        curva_0 = curva_0_inp[i] 
        curva_1v = curva_1v_inp[i] 
        curva_0v = curva_0v_inp[i]
        if(i == 0):        
            axs[0].plot(curva_1['prob'], curva_1['curva'], color = 'blue', lw = 2, label = 'Curva 1')
            axs[0].plot(curva_0['prob'], curva_0['curva'], color = 'red',lw = 2, label = 'Curva 0')
            axs[1].plot(curva_1v['prob'], curva_1v['curva'], color = 'blue', lw = 2, label = 'Curva 1')
            axs[1].plot(curva_0v['prob'], curva_0v['curva'], color = 'red',lw = 2, label = 'Curva 0')
        else:
            axs[0].plot(curva_1['prob'], curva_1['curva'], color = 'blue', lw = 2)
            axs[0].plot(curva_0['prob'], curva_0['curva'], color = 'red',lw = 2)
            axs[1].plot(curva_1v['prob'], curva_1v['curva'], color = 'blue', lw = 2)
            axs[1].plot(curva_0v['prob'], curva_0v['curva'], color = 'red',lw = 2)
    for i in range(0, 2):
        axs[i].set_xlabel('Probabilidade de 1')
        axs[i].set_ylabel('Probabilidade Acumulada')
        axs[i].legend()
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()

def plotaMatrizConfusao(y, cf_matrix_train, cf_matrix_valid, num_loop, flag_normalize):    
    #plota a matriz de confusão
    cf_matrix_train = cf_matrix_train.astype('float')/num_loop
    cf_matrix_valid = cf_matrix_valid.astype('float')/num_loop
    if(flag_normalize):
        cf_matrix_train = cf_matrix_train.astype('float') / cf_matrix_train.sum(axis=1)[:, np.newaxis]
        cf_matrix_valid = cf_matrix_valid.astype('float') / cf_matrix_valid.sum(axis=1)[:, np.newaxis]
    fig, axs = plt.subplots(1, 2, sharey = 'row')
    labels = list(dict.fromkeys(y))
    labels.sort(reverse = False)
    if(flag_normalize):
        disp = metrics.ConfusionMatrixDisplay(cf_matrix_train, display_labels = labels)
        disp.plot(ax = axs[0], values_format = '.2g')
        disp.ax_.set_title('Treino')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp = metrics.ConfusionMatrixDisplay(cf_matrix_valid, display_labels = labels)
        disp.plot(ax = axs[1], values_format = '.2g')
        disp.ax_.set_title('Validação')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
    else:
        disp = metrics.ConfusionMatrixDisplay(cf_matrix_train, display_labels = labels)
        disp.plot(ax = axs[0], values_format = 'd')
        disp.ax_.set_title('Treino')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp = metrics.ConfusionMatrixDisplay(cf_matrix_valid, display_labels = labels)
        disp.plot(ax = axs[1], values_format = 'd')
        disp.ax_.set_title('Validação')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
    fig.text(0.4, 0.1, 'Predicted label', ha = 'left')
    plt.subplots_adjust(wspace = 0.40, hspace = 0.1)
    fig.colorbar(disp.im_, ax = axs)
    plt.show()

def printaResumo(nome_metrica, score_train, score_valid, quedas_score):
    print('----------------------')
    print(nome_metrica + ' Treino: ' + str(round(np.mean(score_train) * 100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(np.std(score_train) * 100, 1)))
    print(nome_metrica + ' Validação: ' + str(round(np.mean(score_valid) * 100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(np.std(score_valid) * 100, 1)))
    print('Queda de ' + nome_metrica + ': ' + str(round(np.mean(quedas_score) * 100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(np.std(quedas_score) * 100, 1)))

def printaImportancias(df_vars, num_max_ini):    
    num_max = min(num_max_ini, len(df_vars))
    if(df_vars.empty == False):
        fig, axs = plt.subplots(1, 1)
        sns.barplot(x = df_vars['mean'][:num_max], y = df_vars.index[:num_max], xerr = df_vars['error'][:num_max])
        plt.xlabel('Score de Importância')
        plt.ylabel('Variáveis')
        plt.title("Visualização da Importância das Variáveis")
        plt.show()
    else:
        print("Não tem feature_importances")

def printaImportanciasConjuntas(df_vars_nn, df_vars_n, num_max_ini):
    fig, axs = plt.subplots(1, 2, figsize = [12, 4])
    if(df_vars_nn.empty == False):
        num_max = min(num_max_ini, len(df_vars_nn))
        sns.barplot(ax = axs[0], x = df_vars_nn['mean'][:num_max], y = df_vars_nn.index[:num_max], xerr = df_vars_nn['error'][:num_max])
    else:
        print("Não tem feature_importances")
    if(df_vars_n.empty == False):
        num_max = min(num_max_ini, len(df_vars_n))
        sns.barplot(ax = axs[1], x = df_vars_n['mean'][:num_max], y = df_vars_n.index[:num_max], xerr = df_vars_n['error'][:num_max])
    else:
        print("Não tem feature_importances")
    plt.subplots_adjust(wspace=0.80, hspace=0.1)
    plt.show()

def plotaParametrosMeioMeio(df_parametros, cf_matrix_train, cf_matrix_valid, y, num_loop):
    print('****Avaliação Meio a Meio****')
    printaResumo('Acurácia', df_parametros['acuracia_train'], df_parametros['acuracia_valid'], df_parametros['acuracia_queda'])
    plotaMatrizConfusao(y, cf_matrix_train, cf_matrix_valid, num_loop, flag_normalize = True)
    printaResumo('ROC', df_parametros['roc_train'], df_parametros['roc_valid'], df_parametros['roc_queda'])
    plotaCurvaROC(df_parametros['fpr_train'], df_parametros['tpr_train'], df_parametros['thr_train'], df_parametros['fpr_valid'], df_parametros['tpr_valid'], df_parametros['thr_valid']) 
    printaResumo('KS', df_parametros['ks_train'], df_parametros['ks_valid'], df_parametros['ks_queda'])
    plotaCurvaKS(df_parametros['curva_1_train'], df_parametros['curva_0_train'], df_parametros['curva_1_valid'], df_parametros['curva_0_valid'])
    printaResumo('Melhor Prob Corte', df_parametros['melhor_thr_train'], df_parametros['melhor_thr_valid'], df_parametros['melhor_thr_queda'])
    printaResumo('Melhor Acurácia', df_parametros['melhor_acuracia_train'], df_parametros['melhor_acuracia_valid'], df_parametros['melhor_acuracia_queda'])
    print(' ')

def plotaEvolucaoDinamica(df_dinamico):
    #Plota os dados de evolução dinâmica
    print('****Avaliação por Quantidade de Treino****')
    fig, axs = plt.subplots(1, 3, figsize = [15, 4])
    axs[0].errorbar(df_dinamico['fracao'], df_dinamico['acuracia_train']*100, yerr = df_dinamico['acuracia_train_erro']*100, fmt='-o', label = 'Acurácia Treino', lw = 2, color = 'blue')
    axs[0].errorbar(df_dinamico['fracao'], df_dinamico['acuracia_valid']*100, yerr = df_dinamico['acuracia_valid_erro']*100, fmt='-o', label = 'Acurácia Validação', lw = 2, color = 'red')
    axs[1].errorbar(df_dinamico['fracao'], df_dinamico['roc_train']*100, yerr = df_dinamico['roc_train_erro']*100, fmt='-o', label = 'ROC Treino', lw = 2, color = 'blue')
    axs[1].errorbar(df_dinamico['fracao'], df_dinamico['roc_valid']*100, yerr = df_dinamico['roc_valid_erro']*100, fmt='-o', label = 'ROC Validação', lw = 2, color = 'red')
    axs[2].errorbar(df_dinamico['fracao'], df_dinamico['ks_train']*100, yerr = df_dinamico['ks_train_erro']*100, fmt='-o', label = 'KS Treino', lw = 2, color = 'blue')
    axs[2].errorbar(df_dinamico['fracao'], df_dinamico['ks_valid']*100, yerr = df_dinamico['ks_valid_erro']*100, fmt='-o', label = 'KS Validação', lw = 2, color = 'red')
    for i in range(0, 3):
        axs[i].set_xlabel('Fração de Treino')
        axs[i].legend() 
        axs[i].set_ylim([40, 110])
    axs[2].set_ylim([-10, 110])        
    axs[0].set_ylabel('Acurácia')
    axs[1].set_ylabel('ROC')  
    axs[2].set_ylabel('KS')      
    plt.subplots_adjust(wspace=0.50, hspace=0.1)
    plt.show()
    print('---------Evolução Ajustando Probabilidade de Corte---------')
    fig, axs = plt.subplots(1, 2, figsize = [10, 4])
    axs[0].errorbar(df_dinamico['fracao'], df_dinamico['melhor_thr_train']*100, yerr = df_dinamico['melhor_thr_train_erro']*100, fmt='-o', label = 'Melhor Prob Corte Treino', lw = 2, color = 'blue')
    axs[0].errorbar(df_dinamico['fracao'], df_dinamico['melhor_thr_valid']*100, yerr = df_dinamico['melhor_thr_valid_erro']*100, fmt='-o', label = 'Melhor Prob Corte Validação', lw = 2, color = 'red')
    axs[1].errorbar(df_dinamico['fracao'], df_dinamico['melhor_acuracia_train']*100, yerr = df_dinamico['melhor_acuracia_train_erro']*100, fmt='-o', label = 'Melhor Acurácia Treino', lw = 2, color = 'blue')
    axs[1].errorbar(df_dinamico['fracao'], df_dinamico['melhor_acuracia_valid']*100, yerr = df_dinamico['melhor_acuracia_valid_erro']*100, fmt='-o', label = 'Melhor Acurácia Validação', lw = 2, color = 'red')
    for i in range(0, 2):
        axs[i].set_xlabel('Fração de Treino')
        axs[i].legend() 
    axs[1].set_ylim([40, 110])
    axs[0].set_ylabel('Probabilidade de Corte')
    axs[1].set_ylabel('Melhor Acurácia')
    plt.subplots_adjust(wspace=0.50, hspace=0.1)
    plt.show()

def printaResumoEsperado(df_metricas):   
    print('---------Valores Esperados---------')   
    print('Acurácia: ' + str(round(df_metricas.loc['Acurácia', 'valor']*100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(df_metricas.loc['Acurácia', 'erro']*100, 1)))
    print('ROC: ' + str(round(df_metricas.loc['ROC', 'valor']*100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(df_metricas.loc['ROC', 'erro']*100, 1)))
    print('KS: ' + str(round(df_metricas.loc['KS', 'valor']*100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(df_metricas.loc['KS', 'erro']*100, 1)))
    print('Melhor Prob Corte: ' + str(round(df_metricas.loc['Melhor Prob Corte', 'valor']*100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(df_metricas.loc['Melhor Prob Corte', 'erro']*100, 1)))
    print('Melhor Acurácia: ' + str(round(df_metricas.loc['Melhor Acurácia', 'valor']*100, 1)) + ' ' + u"\u00B1" + ' ' + str(round(df_metricas.loc['Melhor Acurácia', 'erro']*100, 1)))
    print('Fator de Qualidade: ' + str(round(df_metricas.loc['Fator_Qual', 'valor'], 2)) + ' ' + u"\u00B1" + ' ' + str(round(df_metricas.loc['Fator_Qual', 'erro'], 2)))

#--------------------------------------------
#Funções Auxiliares de Inserção

def criaDataFrameParametros():
    df_parametros = pd.DataFrame(columns = ['acuracia_train', 'acuracia_valid', 'acuracia_queda',
                                            'roc_train', 'roc_valid', 'roc_queda', 'fpr_train', 'tpr_train', 'thr_train', 'fpr_valid', 'tpr_valid', 'thr_valid',
                                            'ks_train', 'ks_valid', 'ks_queda', 'curva_1_train', 'curva_0_train', 'curva_1_valid', 'curva_0_valid',
                                            'melhor_thr_train', 'melhor_thr_valid', 'melhor_thr_queda',
                                            'melhor_acuracia_train', 'melhor_acuracia_valid', 'melhor_acuracia_queda'])
    return df_parametros

def criaDataFrameDinamico():
    df_dinamico = pd.DataFrame(columns = ['fracao', 'acuracia_train', 'acuracia_valid', 'acuracia_queda', 'acuracia_train_erro', 'acuracia_valid_erro', 'acuracia_queda_erro',
                                          'roc_train', 'roc_valid', 'roc_queda', 'roc_train_erro', 'roc_valid_erro', 'roc_queda_erro',
                                          'ks_train', 'ks_valid', 'ks_queda', 'ks_train_erro', 'ks_valid_erro', 'ks_queda_erro',
                                          'melhor_thr_train', 'melhor_thr_valid', 'melhor_thr_queda', 
                                          'melhor_thr_train_erro', 'melhor_thr_valid_erro', 'melhor_thr_queda_erro',
                                          'melhor_acuracia_train', 'melhor_acuracia_valid', 'melhor_acuracia_queda', 
                                          'melhor_acuracia_train_erro', 'melhor_acuracia_valid_erro', 'melhor_acuracia_queda_erro'])
    return df_dinamico

def insereParametrosDataframe(yt, yt_prev, yt_prob, yv, yv_prev, yv_prob, df_parametros):
    #Insere os parâmetros de Acurácia
    #acuracia = metrics.accuracy_score(yt, yt_prev)
    #acuraciav = metrics.accuracy_score(yv, yv_prev)
    acuracia = metrics.balanced_accuracy_score(yt, yt_prev)
    acuraciav = metrics.balanced_accuracy_score(yv, yv_prev)
    acuracia_queda = acuracia - acuraciav
    #Calcula os parâmetros de ROC
    #fpr, tpr, _ = metrics.roc_curve(yt, yt_prev)
    #fprv, tprv, _ = metrics.roc_curve(yv, yv_prev)
    fpr, tpr, thr = metrics.roc_curve(yt, yt_prob)
    fprv, tprv, thrv = metrics.roc_curve(yv, yv_prob)
    roc_auc = metrics.auc(fpr, tpr)
    roc_aucv = metrics.auc(fprv, tprv)
    roc_queda = roc_auc - roc_aucv
    ks, curva_1, curva_0 = calculaKS(yt, yt_prob)
    ksv, curva_1v, curva_0v = calculaKS(yv, yv_prob)
    ks_queda = ks - ksv
    melhor_thr, melhor_acuracia = calculaMelhorProbCorte(fpr, tpr, thr)
    melhor_thrv, melhor_acuraciav = calculaMelhorProbCorte(fprv, tprv, thrv)
    melhor_thr_queda = melhor_thr - melhor_thrv
    melhor_acuracia_queda = melhor_acuracia - melhor_acuraciav
    
    df_parametros.loc[len(df_parametros)] = [acuracia, acuraciav, acuracia_queda, roc_auc, roc_aucv, roc_queda, fpr, tpr, thr, fprv, tprv, thrv, ks, ksv, ks_queda, 
                                             curva_1, curva_0, curva_1v, curva_0v, 
                                             melhor_thr, melhor_thrv, melhor_thr_queda, melhor_acuracia, melhor_acuraciav, melhor_acuracia_queda] 
    return df_parametros

def acrescentaMatrizConfusao(yt, yt_prev, yv, yv_prev, cf_matrix_train, cf_matrix_valid):
    #Calcula e soma a matriz de canfusão na matriz de confusão total
    cf_matrix_train = cf_matrix_train + metrics.confusion_matrix(yt, yt_prev)
    cf_matrix_valid = cf_matrix_valid + metrics.confusion_matrix(yv, yv_prev)
    return cf_matrix_train, cf_matrix_valid

def insereDadosDinamicos(df_dinamico, df_parametros, fracao):
    #Organiza os dados de evolução dinâmica
    df_dinamico.loc[len(df_dinamico)] = [fracao, df_parametros['acuracia_train'].mean(), df_parametros['acuracia_valid'].mean(), df_parametros['acuracia_queda'].mean(),
                                         df_parametros['acuracia_train'].std(), df_parametros['acuracia_valid'].std(), df_parametros['acuracia_queda'].std(),
                                         df_parametros['roc_train'].mean(), df_parametros['roc_valid'].mean(), df_parametros['roc_queda'].mean(),
                                         df_parametros['roc_train'].std(), df_parametros['roc_valid'].std(), df_parametros['roc_queda'].std(),
                                         df_parametros['ks_train'].mean(), df_parametros['ks_valid'].mean(), df_parametros['ks_queda'].mean(),
                                         df_parametros['ks_train'].std(), df_parametros['ks_valid'].std(), df_parametros['ks_queda'].std(),
                                         
                                         df_parametros['melhor_thr_train'].mean(), df_parametros['melhor_thr_valid'].mean(), df_parametros['melhor_thr_queda'].mean(),
                                         df_parametros['melhor_thr_train'].std(), df_parametros['melhor_thr_valid'].std(), df_parametros['melhor_thr_queda'].std(),
                                         df_parametros['melhor_acuracia_train'].mean(), df_parametros['melhor_acuracia_valid'].mean(), df_parametros['melhor_acuracia_queda'].mean(),
                                         df_parametros['melhor_acuracia_train'].std(), df_parametros['melhor_acuracia_valid'].std(), df_parametros['melhor_acuracia_queda'].std()] 
    return df_dinamico

#--------------------------------------------
#Funções Auxiliares de cálculos

def calculaListaFracoes(y, num_parts):
    values_count = y.value_counts()
    sample_min = min(values_count)
    frac_min = sample_min/len(y)
    sample_start = math.ceil(1/frac_min)
    step = ((len(y) - sample_start) - sample_start)/(num_parts + 1)
    list_frac = []
    for i in range(0, num_parts):
        list_frac.append(sample_start + (num_parts - i)*step)
    list_frac = [x/len(y) for x in list_frac]
    #Embaralha para estimar o tempo de execução melhor
    list_frac2 = list_frac.copy()
    list_frac2.sort()
    list_frac_final = []
    for i in range(0, len(list_frac)):
        list_frac_final.append(list_frac[i])
        list_frac_final.append(list_frac2[i])
    list_frac = list_frac_final[:int(len(list_frac_final)/2)]
    if(0.5 not in list_frac):
        list_frac.append(0.5)
    #Lembrar de deixar o 0.5 sempre primeiro!!
    ind = list_frac.index(0.5)
    list_frac = [list_frac[ind]] + list_frac[:ind] + list_frac[ind+1:] 
    return list_frac  

def fitaModeloAmostra(model, df_train, nome_alvo, df_vars, **kwargs):
    model_temp = clone(model)
    #Treina o modelo clonado
    cv = kwargs.get('cv', False) #O cv é para XGBoost com cross-validation
    count = kwargs.get('count', -2)      
    if(cv):
        df1 = df_train.sample(frac = 0.5, replace = False).copy()
        df2 = df_train.drop(df1.index).copy()
        eval_set = [(df1.drop(nome_alvo, axis = 1), df1[nome_alvo]), (df2.drop(nome_alvo, axis = 1), df2[nome_alvo])]
        model_temp.fit(df1.drop(nome_alvo, axis = 1), df1[nome_alvo].values.ravel(), 
                        early_stopping_rounds = 10, eval_metric = ["error", "logloss"], eval_set = eval_set, verbose = False)
    else:
        model_temp.fit(df_train.drop(nome_alvo, axis = 1), df_train[nome_alvo])
    
    #Soma as importâncias para termos uma importância acumulada de todas as amostras   
    try:
        df_vars[str(count)] = pd.Series(model_temp.feature_importances_, index = list(df_train.drop(nome_alvo, axis = 1).columns))
        count = count + 1
    except:
        pass
    #Retorna o modelo treinado com a amostra, a quebra feita no dataset e a importância acumulada das variaveis
    if(count > -1):    
        return model_temp, df_vars, count
    else:
        return model_temp, df_vars

def selectColumns(df_vars, char):
    colunas = list(df_vars.columns)
    colunas_manter = []
    for col in colunas:
        if(col.find(char) != -1):
            colunas_manter.append(col)
    df_vars_filtrado = df_vars[colunas_manter].copy()
    return df_vars_filtrado

def ponderaImportancias(df_vars, df_dinamico):
    score = df_dinamico['acuracia_valid'] - 0.5
    score = np.array([max(v, 0) for v in score])
    score_erro = df_dinamico['acuracia_valid_erro']
    if(df_vars.empty == False):
        for i in range(0, len(score)):
            df_vars['m' + str(i)] = score[i] * df_vars['m' + str(i)]
            df_vars['s' + str(i)] = ((score[i]*df_vars['s' + str(i)])**2 + (score_erro[i]*df_vars['m' + str(i)])**2)**0.5
        df_vars['mean'] = selectColumns(df_vars, 'm').sum(axis = 1)/np.sum(score)
        values_aux = (selectColumns(df_vars, 's')**2)/(np.sum(score)**2)
        df_vars['error'] = (selectColumns(df_vars, 'm').std(axis = 1).values**2 + values_aux.sum(axis = 1))**0.5
        df_vars = df_vars[['mean', 'error']]
        df_vars = df_vars.sort_values(by = ['mean'], ascending = False)
        #Calcula a importância acumulada
        df_vars['acum'] = df_vars['mean'].cumsum()
        df_vars['error^2'] = df_vars['error']**2
        df_vars['acum_error^2'] = df_vars['error^2'].cumsum()
        df_vars['acum_error^2'] = df_vars['acum_error^2']**0.5
        df_vars['acum_error'] = df_vars['acum_error^2']
        for i in range(0, len(df_vars)):
            df_vars['acum_error'][i] = df_vars['acum_error'][i]/(i+1)
        df_vars = df_vars[['mean', 'error', 'acum', 'acum_error']]
        #NORMALIZA NA FORÇA!!! (POR QUE ESTOU PRECISANDO FAZER ISSO??)
        #df_vars = df_vars/df_vars['acum'].max()
    return df_vars

def ponderaImportanciasSub(df_vars, df_vars_temp, num_loop, j):
    if(df_vars_temp.empty == False):
        df_vars['m' + str(j)] = selectColumns(df_vars_temp, 'm').mean(axis = 1)
        values_aux = (selectColumns(df_vars_temp, 's')**2)/(num_loop**2)
        df_vars['s' + str(j)] = (selectColumns(df_vars_temp, 'm').std(axis = 1).values**2 + values_aux.sum(axis = 1))**0.5
    return df_vars

def calculaMetricasEsperadas(y, y_prev, y_prob, df_dinamico):
    #acuracia = metrics.accuracy_score(y, y_prev)
    acuracia = metrics.balanced_accuracy_score(y, y_prev)
    fpr, tpr, thr = metrics.roc_curve(y, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ks, _, _ = calculaKS(y, y_prob)
    melhor_thr, melhor_acuracia = calculaMelhorProbCorte(fpr, tpr, thr)
    
    acuracia_esperada, acuracia_esperada_erro, acuracia_queda, acuracia_queda_erro = calculaMetricaEsperada('acuracia', acuracia, df_dinamico)
    roc_esperada, roc_esperada_erro, _, _ = calculaMetricaEsperada('roc', roc_auc, df_dinamico)
    ks_esperada, ks_esperada_erro, _, _ = calculaMetricaEsperada('ks', ks, df_dinamico)
    melhor_thr_esperada, melhor_thr_esperada_erro, _, _ = calculaMetricaEsperada('melhor_thr', melhor_thr, df_dinamico)
    melhor_acuracia_esperada, melhor_acuracia_esperada_erro, _, _ = calculaMetricaEsperada('melhor_acuracia', melhor_acuracia, df_dinamico)
    
    #Calcula o fator de qualidade (quanto mais próximo de zero, melhor)
    fator_qualidade, fator_erro = calculaFatorQualidade(acuracia_esperada, acuracia_esperada_erro, acuracia_queda, acuracia_queda_erro)
    
    df_metricas = pd.DataFrame(columns = ['valor', 'erro'])
    df_metricas.loc['Acurácia', :] = [acuracia_esperada, acuracia_esperada_erro]
    df_metricas.loc['ROC', :] = [roc_esperada, roc_esperada_erro]
    df_metricas.loc['KS', :] = [ks_esperada, ks_esperada_erro]
    df_metricas.loc['Melhor Prob Corte', :] = [melhor_thr_esperada, melhor_thr_esperada_erro]
    df_metricas.loc['Melhor Acurácia', :] = [melhor_acuracia_esperada, melhor_acuracia_esperada_erro]
    df_metricas.loc['Fator_Qual', :] = [fator_qualidade, fator_erro]
    return df_metricas

def calculaMetricaEsperada(col_ini, score, df_dinamico):
    col = col_ini + '_queda'
    col2 = col + '_erro'
    df = df_dinamico.copy()
    df[col + '_pond'] = df[col]*df['fracao']/df['fracao'].sum()
    df[col + '^2_pond'] = df[col]*df[col]*df['fracao']/df['fracao'].sum()
    df[col + '_erro_pond^2'] = (df[col2]**2)*(df['fracao']**2)/(df['fracao'].sum()**2)  
    score_queda = df[col + '_pond'].sum()
    score_queda_erro = (df[col + '_erro_pond^2'].sum() + df[col + '^2_pond'].sum() - score_queda**2)**0.5
    score_erro = df_dinamico[col_ini + '_train_erro'].max()
    score_esperada = score - score_queda
    score_esperada_erro = (score_erro**2 + score_queda_erro**2)**0.5
    return score_esperada, score_esperada_erro, score_queda, score_queda_erro

def calculaFatorQualidade(metrica, metrica_erro, queda, queda_erro):
    if(np.abs(metrica - 0.5) < metrica_erro):
        fator_qualidade = np.nan
        fator_erro = 0
    else:
        if(metrica - 0.5 > 0):
            ganho = metrica - 0.5
            fator_qualidade = queda/ganho
            fator_erro = ((queda_erro/ganho)**2 + (queda*metrica_erro/(ganho**2))**2)**0.5
        else:
            fator_qualidade = np.nan
            fator_erro = 0
    return fator_qualidade, fator_erro

def calculaKS(y, y_prob):
    df_ks = pd.DataFrame(columns = ['prob', 'alvo'])
    df_ks['prob'] = y_prob
    df_ks['alvo'] = y
    df_ks = df_ks.sort_values(by = 'prob', ascending = False).reset_index(drop = True)
    div = int(len(y)/10)
    df_ks_disc = pd.DataFrame(columns = ['decile', 'alvo_1', 'alvo_0'])
    tam_1 = len(df_ks[df_ks['alvo'] == 1])
    tam_0 = len(df_ks[df_ks['alvo'] == 0])    
    for i in range(0, 9):
        df_aux = df_ks.iloc[i*div:(i+1)*div,:]
        df_ks_disc.loc[len(df_ks_disc)] = [i+1, len(df_aux[df_aux['alvo'] == 1])/tam_1, len(df_aux[df_aux['alvo'] == 0])/tam_0]
    df_aux = df_ks.iloc[9*div:,:]
    df_ks_disc.loc[len(df_ks_disc)] = [10, len(df_aux[df_aux['alvo'] == 1])/tam_1, len(df_aux[df_aux['alvo'] == 0])/tam_0]
    df_ks_disc['diff'] = df_ks_disc['alvo' + '_1'] - df_ks_disc['alvo' + '_0']
    df_ks_disc['acum'] = df_ks_disc['diff'].cumsum()
    ks = df_ks_disc['acum'].max()
    ks = ks_2samp(df_ks[df_ks['alvo'] == 1]['prob'], df_ks[df_ks['alvo'] == 0]['prob'])[0] #Outro método
    
    df_ks = df_ks.sort_values(by = 'prob', ascending = True).reset_index(drop = True)
    curva_1 = df_ks[df_ks['alvo'] == 1].copy()
    curva_0 = df_ks[df_ks['alvo'] == 0].copy()
    curva_0['alvo'] = 1
    curva_1['curva'] = curva_1['alvo'].cumsum()/tam_1
    curva_0['curva'] = curva_0['alvo'].cumsum()/tam_0
    return ks, curva_1, curva_0

def calculaMelhorProbCorte(fpr, tpr, thr):
    df = pd.DataFrame()
    df['thr'] = thr
    df['fpr'] = fpr
    df['tpr'] = tpr 
    df['acuracia'] = (df['tpr'] + 1 - df['fpr'])/2
    melhor_acuracia = df['acuracia'].max()
    melhor_thr = df[df['acuracia'] == melhor_acuracia]['thr'].mean()
    return melhor_thr, melhor_acuracia

#--------------------------------------------
#Funções Auxiliares de split de dados
 
def splitBalanceado(df_ini, nome_alvo, fracao):
    #Quebra em treino e validação mantendo a proporção da resposta
    df = df_ini.copy()
    labels = list(dict.fromkeys(df[nome_alvo]))
    for i in range(0,len(labels)):
        if(i == 0):
            df_temp = df[df[nome_alvo] == labels[i]]
            df_train = df_temp.sample(frac = fracao, replace = False).copy()
            df_valid = df_temp.drop(df_train.index).copy()
        else:
            df_temp = df[df[nome_alvo] == labels[i]]
            df_temp_train = df_temp.sample(frac = fracao, replace = False).copy()
            df_temp_valid = df_temp.drop(df_temp_train.index).copy()
            df_train = pd.concat([df_train, df_temp_train], ignore_index = False, sort = False)
            df_valid = pd.concat([df_valid, df_temp_valid], ignore_index = False, sort = False)
    df_train = df_train.sort_index().copy()
    df_valid = df_valid.sort_index().copy()
    return df_train, df_valid

def randomizaPartes(df_ini, nome_alvo, num_partes, balanced):
    #Quebra em treino e validação mantendo a proporção da resposta
    dfs = []
    df = df_ini.copy()
    df = df.sample(frac = 1, replace = False).copy() #randomiza
    if(balanced == 0):
        num_rows = int(len(df)/num_partes)
        for i in range(0, num_partes):
            if(i < num_partes - 1):
                df_temp = df.iloc[:num_rows].copy() #Pega o número de linhas de uma amostra
                df = df.drop(df_temp.index).copy() #Remove as linhas que foram pegas
                dfs.append(df_temp)
            else:
                df_temp = df.copy()
                dfs.append(df_temp)
        return dfs
    elif(balanced == 1):
        labels = list(dict.fromkeys(df[nome_alvo]))
        for j in range(0, num_partes):
            if(j < num_partes - 1):
                for i in range(0,len(labels)):
                    if(i == 0):
                        df_temp = df[df[nome_alvo] == labels[i]]
                        num_rows = int(len(df_temp)/num_partes)
                        df_final = df_temp[:num_rows].copy()
                        df = df.drop(df_final.index).copy()
                    else:
                        df_temp = df[df[nome_alvo] == labels[i]]
                        num_rows = int(len(df_temp)/num_partes)
                        df_temp_final = df_temp[:num_rows].copy()
                        df_final = pd.concat([df_final, df_temp_final], ignore_index = False, sort = False)
                        df = df.drop(df_temp_final.index).copy()
                dfs.append(df_final)
            else:
                df_final = df.copy()
                dfs.append(df_final)
        return dfs
    elif(balanced == 2):
        for j in range(0, num_partes):
            if(j < num_partes - 1):
                fracao = 1/(num_partes - j)
                X_temp = df.drop(nome_alvo, axis = 1)
                y_temp = df[[nome_alvo]]
                X_rest, y_rest, X_frac, y_frac = iterative_train_test_split(X_temp.values, y_temp.values, test_size = fracao)
                df = pd.DataFrame(X_rest, columns = list(X_temp.columns))
                df[nome_alvo] = y_rest
                df_final = pd.DataFrame(X_frac, columns = list(X_temp.columns))
                df_final[nome_alvo] = y_frac
                dfs.append(df_final)
            else:
                df_final = df.copy()
                dfs.append(df_final)
        return dfs
    
#-------------------------------------------
#-------------------------------------------
#Função de visualização de modelos
#permite "modelo médio"
#-------------------------------------------
#-------------------------------------------

def avaliaModeloChute(df, nome_alvo, balanced, num_loop):
    lista_fracao = calculaListaFracoes(df[nome_alvo], 6)
    df_dinamico = criaDataFrameDinamico() 
    dh, prog_ant, tempo_ini = [], 0, 0
    for j in range(0, len(lista_fracao)):
        fracao = lista_fracao[j]
        df_parametros = criaDataFrameParametros() 
        for i in range(0, num_loop):
            dh, prog_ant, tempo_ini = display.updateProgress((j*num_loop + i)/(num_loop*len(lista_fracao)), 2, dh, prog_ant, tempo_ini)
            #Quebra em treino e validação
            if(balanced == 2):
                X_temp = df.drop(nome_alvo, axis = 1)
                y_temp = df[[nome_alvo]]
                X_train, y_train, X_valid, y_valid = iterative_train_test_split(X_temp.values, y_temp.values, test_size = 1 - fracao)
                df_train = pd.DataFrame(X_train, columns = list(X_temp.columns))
                df_train[nome_alvo] = y_train
                df_valid = pd.DataFrame(X_valid, columns = list(X_temp.columns))
                df_valid[nome_alvo] = y_valid
            elif(balanced == 1):
                df_train, df_valid = splitBalanceado(df, nome_alvo, fracao)
            elif(balanced == 0):
                df_train = df.sample(frac = fracao, replace = False).copy()
                df_valid = df.drop(df_train.index).copy() 
            #Faz as predições (chute aleatório)
            yt_prev = np.random.choice([0, 1], size = len(df_train), p = [0.5, 0.5])
            yv_prev = np.random.choice([0, 1], size = len(df_valid), p = [0.5, 0.5])
            yt_prob = np.array([0.5 for v in range(0, len(df_train))])
            yv_prob = np.array([0.5 for v in range(0, len(df_valid))])
            yt = df_train[nome_alvo]
            yv = df_valid[nome_alvo]
            #Adiciona os parâmetros para fazer uma análise estatística de tudo depois
            df_parametros = insereParametrosDataframe(yt, yt_prev, yt_prob, yv, yv_prev, yv_prob, df_parametros)
            if(i == 0 and j == 0):
                cf_matrix_train = metrics.confusion_matrix(yt, yt_prev)
                cf_matrix_valid = metrics.confusion_matrix(yv, yv_prev)
            elif(j == 0):
                cf_matrix_train, cf_matrix_valid = acrescentaMatrizConfusao(yt, yt_prev, yv, yv_prev, cf_matrix_train, cf_matrix_valid)
        if(j == 0):
            #Printa as estatísticas quando frac = 0.5
            y = df[nome_alvo]
            plotaParametrosMeioMeio(df_parametros, cf_matrix_train, cf_matrix_valid, y, num_loop) 
        df_dinamico = insereDadosDinamicos(df_dinamico, df_parametros, fracao)
    dh, prog_ant, tempo_ini = display.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    #Calcula as predições usando todo o dataset de treino
    y_prev = np.random.choice([0, 1], size = len(df), p = [0.5, 0.5])
    y_prob = np.array([0.5 for v in range(0, len(df))])
    y = df[nome_alvo]   
    #Organiza os dados
    df_dinamico = df_dinamico.sort_values(by = 'fracao')    
    #Calcula valores de metricas esperadas
    df_metricas = calculaMetricasEsperadas(y, y_prev, y_prob, df_dinamico)
    #Plota tendência com a dinâmica de frac
    plotaEvolucaoDinamica(df_dinamico)
    #Printa estatísticas esperadas com frac = 1
    printaResumoEsperado(df_metricas)

def predicaoProbModelos(models, X):
    y_prev = np.array([0 for x in range(0, len(X))])
    for m in models:
        try:
            y_prev = y_prev + m.predict_proba(X)[:,1]
        except:
            pred = m.predict(X)
            pred_aprox = np.where(pred == 1, 0.75, 0.25)
            y_prev = y_prev + pred_aprox
    y_prev = y_prev/len(models)
    return y_prev

def predicaoModelos(models, X, **kwargs):
    prob_corte = kwargs.get('prob_corte', 0.5)
    y_prev = predicaoProbModelos(models, X)
    y_prev = np.array([1 if prob >= prob_corte else 0 for prob in y_prev])
    return y_prev

def avaliaModelos(models_inp, df, nome_alvo, balanced, num_loop, **kwargs):
    models = []
    for m in models_inp:
        models.append(clone(m))
    flag_print = kwargs.get('flag_print', False)
    lista_fracao = calculaListaFracoes(df[nome_alvo], 6)
    df_dinamico = criaDataFrameDinamico()   
    df_vars = pd.DataFrame()
    dh, prog_ant, tempo_ini = [], 0, 0
    for j in range(0, len(lista_fracao)):
        fracao = lista_fracao[j]
        df_parametros = criaDataFrameParametros()
        df_vars_temp = pd.DataFrame()
        for i in range(0, num_loop):
            dh, prog_ant, tempo_ini = display.updateProgress((j*num_loop + i)/(num_loop*len(lista_fracao)), 2, dh, prog_ant, tempo_ini)
            
            #Quebra em treino e validação
            if(balanced == 2):
                X_train, y_train, X_valid, y_valid = iterative_train_test_split(df.drop(nome_alvo, axis = 1).values, df[[nome_alvo]].values, test_size = 1 - fracao)
                df_train = pd.DataFrame(X_train, columns = list(df.drop(nome_alvo, axis = 1).columns))
                df_train[nome_alvo] = y_train
                df_valid = pd.DataFrame(X_valid, columns = list(df.drop(nome_alvo, axis = 1).columns))
                df_valid[nome_alvo] = y_valid
                del X_train
                del y_train
                del X_valid
                del y_valid
            elif(balanced == 1):
                df_train, df_valid = splitBalanceado(df, nome_alvo, fracao)
            elif(balanced == 0):
                df_train = df.sample(frac = fracao, replace = False).copy()
                df_valid = df.drop(df_train.index).copy()   
            
            #Fita um clone dos modelos de referência em uma amostra do dataset    
            cv_fit = kwargs.get('cv_fit', False)
            models_temp = []
            df_vars_temp_count = pd.DataFrame()
            count = 0
            for m in models:
                model_temp, df_vars_temp_count, count = fitaModeloAmostra(m, df_train, nome_alvo, df_vars_temp_count, count = count, cv = cv_fit)
                models_temp.append(model_temp)
            if(count != 0):
                df_vars_temp['m' + str(i)] = df_vars_temp_count.mean(axis = 1)
                df_vars_temp['s' + str(i)] = df_vars_temp_count.std(axis = 1).fillna(0)
            
            #Faz as predições
            yt_prev = predicaoModelos(models_temp, df_train.drop(nome_alvo, axis = 1))
            yv_prev = predicaoModelos(models_temp, df_valid.drop(nome_alvo, axis = 1))
            yt_prob = predicaoProbModelos(models_temp, df_train.drop(nome_alvo, axis = 1))
            yv_prob = predicaoProbModelos(models_temp, df_valid.drop(nome_alvo, axis = 1))
            yt = df_train[nome_alvo]
            yv = df_valid[nome_alvo]
            
            #Adiciona os parâmetros para fazer uma análise estatística de tudo depois
            df_parametros = insereParametrosDataframe(yt, yt_prev, yt_prob, yv, yv_prev, yv_prob, df_parametros)
            if(i == 0 and j == 0):
                cf_matrix_train = metrics.confusion_matrix(yt, yt_prev)
                cf_matrix_valid = metrics.confusion_matrix(yv, yv_prev)
            elif(j == 0):
                cf_matrix_train, cf_matrix_valid = acrescentaMatrizConfusao(yt, yt_prev, yv, yv_prev, cf_matrix_train, cf_matrix_valid)
            
            del df_train
            del df_valid
            for m in models:
                del m
            del models_temp
            del df_vars_temp_count
            del yt_prev
            del yv_prev
            del yt_prob
            del yv_prob
            del yt
            del yv       
        
        if(j == 0 and flag_print == True):
            #Printa as estatísticas quando frac = 0.5
            y = df[nome_alvo]
            plotaParametrosMeioMeio(df_parametros, cf_matrix_train, cf_matrix_valid, y, num_loop)
            del y
        
        df_vars = ponderaImportanciasSub(df_vars, df_vars_temp, num_loop, j)
        df_dinamico = insereDadosDinamicos(df_dinamico, df_parametros, fracao)
        del df_parametros
        if(j == 0):
            del cf_matrix_train
            del cf_matrix_valid
    
    dh, prog_ant, tempo_ini = display.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    
    #Calcula as predições usando todo o dataset de treino
    try:
        y_prev = predicaoModelos(models, df.drop(nome_alvo, axis = 1))
        y_prob = predicaoProbModelos(models, df.drop(nome_alvo, axis = 1))
    except:
        for m in models:
            m.fit(df.drop(nome_alvo, axis = 1), df[nome_alvo])
        y_prev = predicaoModelos(models, df.drop(nome_alvo, axis = 1))
        y_prob = predicaoProbModelos(models, df.drop(nome_alvo, axis = 1))
    y = df[nome_alvo]    
        
    #Organiza os dados
    df_dinamico = df_dinamico.sort_values(by = 'fracao')
    df_vars = ponderaImportancias(df_vars, df_dinamico)
    
    #Calcula valores de metricas esperadas
    df_metricas = calculaMetricasEsperadas(y, y_prev, y_prob, df_dinamico)
    
    if(flag_print):
        #Plota tendência com a dinâmica de frac
        plotaEvolucaoDinamica(df_dinamico)
        #Printa estatísticas esperadas com frac = 1
        printaResumoEsperado(df_metricas)
        #Printa importâncias
        printaImportancias(df_vars, 15)
    
    return df_metricas, df_vars

#------------------------------------------------------

def predicaoProbModelosConjunto(models_nn, models_n, X_nn, X_n):
    #Prevê separa os nulos e não nulos em cada modelo
    y_prev_nn = predicaoProbModelos(models_nn, X_nn)
    y_prev_n = predicaoProbModelos(models_n, X_n)    
    
    ind_nn = list(X_nn.index)
    ind_n = list(X_n.index)
    
    #Junta a predição dos dois modelos
    ind_tot = list(ind_nn.copy())
    ind_tot.extend(list(ind_n.copy()))
    y_prev_tot = list(y_prev_nn.copy())
    y_prev_tot.extend(list(y_prev_n.copy()))

    df_res = pd.DataFrame(columns = ['ind', 'y_prev'])
    df_res['ind'] = ind_tot
    df_res['y_prev'] = y_prev_tot
    df_res = df_res.sort_values(by = 'ind', ascending = True).reset_index(drop = True)
    
    return df_res['y_prev'].values

def predicaoModelosConjunto(models_nn, models_n, X_nn, X_n, **kwargs):
    #Prevê separa os nulos e não nulos em cada modelo
    prob_corte = kwargs.get('prob_corte', 0.5)    
    y_prev_nn = predicaoModelos(models_nn, X_nn, prob_corte = prob_corte)
    y_prev_n = predicaoModelos(models_n, X_n, prob_corte = prob_corte)    
    
    ind_nn = list(X_nn.index)
    ind_n = list(X_n.index)
    
    #Junta a predição dos dois modelos
    ind_tot = list(ind_nn.copy())
    ind_tot.extend(list(ind_n.copy()))
    y_prev_tot = list(y_prev_nn.copy())
    y_prev_tot.extend(list(y_prev_n.copy()))

    df_res = pd.DataFrame(columns = ['ind', 'y_prev'])
    df_res['ind'] = ind_tot
    df_res['y_prev'] = y_prev_tot
    df_res = df_res.sort_values(by = 'ind', ascending = True).reset_index(drop = True)
    
    return df_res['y_prev'].values

def avaliaModelosConjunto(models_nn_inp, models_n_inp, df_nn_inp, df_n_inp, nome_alvo, balanced, num_loop, **kwargs):
    models_nn = []
    for m in models_nn_inp:
        models_nn.append(clone(m))    
    models_n = []
    for m in models_n_inp:
        models_n.append(clone(m))
    flag_print = kwargs.get('flag_print', False)
    df_nn = df_nn_inp.copy()
    df_n = df_n_inp.copy()
    df = pd.concat([df_nn, df_n], ignore_index = False, sort = False) #Junta os dois datasets
    lista_fracao = calculaListaFracoes(df[nome_alvo], 6)
    df_dinamico = criaDataFrameDinamico() 
    df_vars_nn = pd.DataFrame()
    df_vars_n = pd.DataFrame()
    dh, prog_ant, tempo_ini = [], 0, 0
    for j in range(0, len(lista_fracao)):
        fracao = lista_fracao[j]    
        df_parametros = criaDataFrameParametros() 
        df_vars_nn_temp = pd.DataFrame()
        df_vars_n_temp = pd.DataFrame()
        for i in range(0, num_loop):
            dh, prog_ant, tempo_ini = display.updateProgress((j*num_loop + i)/(num_loop*len(lista_fracao)), 2, dh, prog_ant, tempo_ini)
            
            #Quebra a base total em treino e validação
            if(balanced == 2):
                X_temp = df_nn.drop(nome_alvo, axis = 1)
                y_temp = df_nn[[nome_alvo]]
                X_train, y_train, X_valid, y_valid = iterative_train_test_split(X_temp.values, y_temp.values, test_size = 1 - fracao)
                df_nn_train = pd.DataFrame(X_train, columns = list(X_temp.columns))
                df_nn_train[nome_alvo] = y_train
                df_nn_valid = pd.DataFrame(X_valid, columns = list(X_temp.columns))
                df_nn_valid[nome_alvo] = y_valid
                X_temp = df_n.drop(nome_alvo, axis = 1)
                y_temp = df_n[[nome_alvo]]
                X_train, y_train, X_valid, y_valid = iterative_train_test_split(X_temp.values, y_temp.values, test_size = 1 - fracao)
                df_n_train = pd.DataFrame(X_train, columns = list(X_temp.columns))
                df_n_train[nome_alvo] = y_train
                df_n_valid = pd.DataFrame(X_valid, columns = list(X_temp.columns))
                df_n_valid[nome_alvo] = y_valid
            elif(balanced == 1):
                df_nn_train, df_nn_valid = splitBalanceado(df_nn, nome_alvo, fracao)
                df_n_train, df_n_valid = splitBalanceado(df_n, nome_alvo, fracao)
            elif(balanced == 0):
                df_nn_train = df_nn.sample(frac = fracao, replace = False).copy()
                df_nn_valid = df_nn.drop(df_nn_train.index).copy()
                df_n_train = df_n.sample(frac = fracao, replace = False).copy()
                df_n_valid = df_n.drop(df_n_train.index).copy()  
            df_train = pd.concat([df_nn_train, df_n_train], ignore_index = False, sort = False) #Junta os dois datasets
            
            #Pega a lista de colunas que não tem nenhum nulo
            colunas_sem_nulos = list(df_n_train.columns)
            
            #Note que aqui mandamos a base de treino toda para treinar, dropando os nulos antes
            #Fita um clone dos modelos de referência em uma amostra do dataset    
            models_n_temp = []
            df_vars_n_temp_count = pd.DataFrame()
            count = 0
            for m in models_n:
                model_n_temp, df_vars_n_temp_count, count = fitaModeloAmostra(m, df_train.dropna(axis = 1), nome_alvo, df_vars_n_temp_count, count = count)
                models_n_temp.append(model_n_temp) 
            if(count != 0):
                df_vars_n_temp['m' + str(i)] = df_vars_n_temp_count.mean(axis = 1)
                df_vars_n_temp['s' + str(i)] = df_vars_n_temp_count.std(axis = 1).fillna(0)            
            
            #Faz o Ensemble com o modelo de nulos nas bases de não-nulos
            #Note que temos que selecionar só as colunas que tem nos dois datasets (colunas_sem_nulos)
            df_nn_train['prob_model'] = predicaoProbModelos(models_n_temp, df_nn_train[colunas_sem_nulos].drop(nome_alvo, axis = 1))
            df_nn_valid['prob_model'] = predicaoProbModelos(models_n_temp, df_nn_valid[colunas_sem_nulos].drop(nome_alvo, axis = 1))
            
            #Agora fita o modelo dos não-nulos usando essa variavel a mais, vinda do ensemble
            #Fita um clone dos modelos de referência em uma amostra do dataset    
            models_nn_temp = []
            df_vars_nn_temp_count = pd.DataFrame()
            count = 0
            for m in models_nn:
                model_nn_temp, df_vars_nn_temp_count, count = fitaModeloAmostra(m, df_nn_train, nome_alvo, df_vars_nn_temp_count, count = count)
                models_nn_temp.append(model_nn_temp) 
            if(count != 0):
                df_vars_nn_temp['m' + str(i)] = df_vars_nn_temp_count.mean(axis = 1)
                df_vars_nn_temp['s' + str(i)] = df_vars_nn_temp_count.std(axis = 1).fillna(0) 
            
            #Faz as predições
            yt_prev = predicaoModelosConjunto(models_nn_temp, models_n_temp, df_nn_train.drop(nome_alvo, axis = 1), df_n_train.drop(nome_alvo, axis = 1))
            yv_prev = predicaoModelosConjunto(models_nn_temp, models_n_temp, df_nn_valid.drop(nome_alvo, axis = 1), df_n_valid.drop(nome_alvo, axis = 1))
            yt_prob = predicaoProbModelosConjunto(models_nn_temp, models_n_temp, df_nn_train.drop(nome_alvo, axis = 1), df_n_train.drop(nome_alvo, axis = 1))
            yv_prob = predicaoProbModelosConjunto(models_nn_temp, models_n_temp, df_nn_valid.drop(nome_alvo, axis = 1), df_n_valid.drop(nome_alvo, axis = 1))
            yt = pd.concat([df_nn_train, df_n_train], ignore_index = False, sort = False).sort_index()[nome_alvo].values
            yv = pd.concat([df_nn_valid, df_n_valid], ignore_index = False, sort = False).sort_index()[nome_alvo].values
            
            #Adiciona os parâmetros para fazer uma análise estatística de tudo depois
            df_parametros = insereParametrosDataframe(yt, yt_prev, yt_prob, yv, yv_prev, yv_prob, df_parametros)
            if(i == 0 and j == 0):
                cf_matrix_train = metrics.confusion_matrix(yt, yt_prev)
                cf_matrix_valid = metrics.confusion_matrix(yv, yv_prev)
            elif(j == 0):
                cf_matrix_train, cf_matrix_valid = acrescentaMatrizConfusao(yt, yt_prev, yv, yv_prev, cf_matrix_train, cf_matrix_valid)
            
        if(j == 0 and flag_print == True):
            #Printa as estatísticas quando frac = 0.5
            y = pd.concat([df_nn, df_n], ignore_index = False, sort = False).sort_index()[nome_alvo] 
            plotaParametrosMeioMeio(df_parametros, cf_matrix_train, cf_matrix_valid, y, num_loop) 
        
        df_vars_n = ponderaImportanciasSub(df_vars_n, df_vars_n_temp, num_loop, j)
        df_vars_nn = ponderaImportanciasSub(df_vars_nn, df_vars_nn_temp, num_loop, j)
        df_dinamico = insereDadosDinamicos(df_dinamico, df_parametros, fracao)
    
    dh, prog_ant, tempo_ini = display.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    
    #Calcula as predições usando todo o dataset de treino
    try:
        colunas_sem_nulos = list(df_n.columns)
        df_nn['prob_model'] = predicaoProbModelos(models_n, df_nn[colunas_sem_nulos].drop(nome_alvo, axis = 1))
        y_prev = predicaoModelosConjunto(models_nn, models_n, df_nn.drop(nome_alvo, axis = 1), df_n.drop(nome_alvo, axis = 1))
        y_prob = predicaoProbModelosConjunto(models_nn, models_n, df_nn.drop(nome_alvo, axis = 1), df_n.drop(nome_alvo, axis = 1))    
    except:
        for m in models_n:
            m.fit(df.dropna(axis = 1).drop(nome_alvo, axis = 1), df[nome_alvo])    
        colunas_sem_nulos = list(df_n.columns)
        df_nn['prob_model'] = predicaoProbModelos(models_n, df_nn[colunas_sem_nulos].drop(nome_alvo, axis = 1))
        for m in models_nn:
            m.fit(df_nn.drop(nome_alvo, axis = 1), df_nn[nome_alvo]) 
        y_prev = predicaoModelosConjunto(models_nn, models_n, df_nn.drop(nome_alvo, axis = 1), df_n.drop(nome_alvo, axis = 1))
        y_prob = predicaoProbModelosConjunto(models_nn, models_n, df_nn.drop(nome_alvo, axis = 1), df_n.drop(nome_alvo, axis = 1))
    y = pd.concat([df_nn, df_n], ignore_index = False, sort = False).sort_index()[nome_alvo]    
    
    #Organiza os dados
    df_dinamico = df_dinamico.sort_values(by = 'fracao')     
    df_vars_n = ponderaImportancias(df_vars_n, df_dinamico)
    df_vars_nn = ponderaImportancias(df_vars_nn, df_dinamico)
    
    #Calcula valores de metricas esperadas
    df_metricas = calculaMetricasEsperadas(y, y_prev, y_prob, df_dinamico)
    
    if(flag_print):
        #Plota tendência com a dinâmica de frac
        plotaEvolucaoDinamica(df_dinamico)
        #Printa estatísticas esperadas com frac = 1
        printaResumoEsperado(df_metricas)
        #Printa importâncias
        printaImportanciasConjuntas(df_vars_nn, df_vars_n, 15)
    
    return df_metricas, df_vars_nn, df_vars_n 
    
#------------------------------------------