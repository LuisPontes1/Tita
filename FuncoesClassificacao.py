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

#Usado para calcular Lift 10% e Lift 20%
def calculaKS_discretizado(y, y_prob):
    df = pd.DataFrame(columns = ['prob', 'alvo_1', 'alvo_0'])
    df['prob'] = y_prob
    df['alvo_1'] = y
    df['alvo_0'] = 1 - y
    df = df.sort_values(by = 'prob', ascending = False).reset_index(drop = True)
    tam_1 = df['alvo_1'].sum()
    tam_0 = df['alvo_0'].sum()
    div = int(len(y)/10)
    df_disc = pd.DataFrame(columns = ['decile', 'prob_max', 'prob_min', 'alvo_1', 'alvo_0'])
    for i in range(0, 9):
        df_aux = df.iloc[i*div:(i+1)*div,:]
        df_disc.loc[len(df_disc)] = [10-i, df_aux['prob'].max(), df_aux['prob'].min(), 
                                           df_aux['alvo_1'].sum()/tam_1, df_aux['alvo_0'].sum()/tam_0]
    df_aux = df.iloc[9*div:,:]
    df_disc.loc[len(df_disc)] = [1, df_aux['prob'].max(), df_aux['prob'].min(),
                                       df_aux['alvo_1'].sum()/tam_1, df_aux['alvo_0'].sum()/tam_0]
    df_disc['diff'] = df_disc['alvo_1'] - df_disc['alvo_0']
    df_disc['acum'] = df_disc['diff'].cumsum()
    ks = df_disc['acum'].max()
    return df_disc, ks
    
def calculaKS(y, y_prob):
    df = pd.DataFrame(columns = ['prob', 'alvo'])
    df['prob'] = y_prob
    df['alvo'] = y
    df = df.sort_values(by = 'prob', ascending = True).reset_index(drop = True)
    #ks = ks_2samp(df[df['alvo'] == 1]['prob'].values, df[df['alvo'] == 0]['prob'].values)[0]
    
    df_probs = df.reset_index()[['prob', 'index']].groupby(by = 'prob', axis = 0).max().reset_index()
    num_divs = min(1001, len(df_probs))
    df_probs['intervalo'] = pd.qcut(df_probs['prob'], q = num_divs)
    #display(df_probs)
    
    def count_1(x):
        return (x == 1).sum()
    def count_0(x):
        return (x == 0).sum()
    df = pd.merge(df, df_probs[['prob', 'intervalo']], how = 'left', left_on = 'prob', right_on = 'prob')
    df = df.groupby(by = 'intervalo', axis = 0).agg({'prob':'max', 'alvo':[count_1, count_0]})
    df.columns = ['_'.join(col) for col in df.columns]
    #df['count'] = df['alvo_count_1'] + df['alvo_count_0']
    df['alvo_acum_1'] = df['alvo_count_1'].cumsum()
    df['alvo_acum_0'] = df['alvo_count_0'].cumsum()
    #df['count_acum'] = df['count'].cumsum()
    df['frac_alvo_acum_1'] = df['alvo_acum_1']/df['alvo_acum_1'].values[-1]
    df['frac_alvo_acum_0'] = df['alvo_acum_0']/df['alvo_acum_0'].values[-1]
    df['diferenca'] = df['frac_alvo_acum_0'] - df['frac_alvo_acum_1']
    ks = df['diferenca'].max()
    #display(df)
    
    curva_1 = df[['prob_max', 'frac_alvo_acum_1']].copy()
    curva_1 = curva_1.rename({'prob_max': 'prob', 'frac_alvo_acum_1':'frac_alvo_acum'}, axis = 1)
    curva_0 = df[['prob_max', 'frac_alvo_acum_0']].copy()
    curva_0 = curva_0.rename({'prob_max': 'prob', 'frac_alvo_acum_0':'frac_alvo_acum'}, axis = 1)
    return curva_1, curva_0, ks

def calculaInfoGain(y, y_prob):
    df = pd.DataFrame(columns = ['prob', 'alvo'])
    df['prob'] = y_prob
    df['alvo'] = y
    df = df.sort_values(by = 'prob', ascending = True).reset_index(drop = True)
    
    tam = len(df)
    tam1 = df['alvo'].sum()
    tam0 = tam - tam1
    prob_1 = tam1/tam
    prob_0 = tam0/tam   
    entropia_inicial = entropy([prob_1, prob_0], base = 2)
    
    df_probs = df.reset_index()[['prob', 'index']].groupby(by = 'prob', axis = 0).max().reset_index()
    num_divs = min(1001, len(df_probs))
    df_probs['intervalo'] = pd.qcut(df_probs['prob'], q = num_divs)
    #display(df_probs)
    
    def count_1(x):
        return (x == 1).sum()
    def count_0(x):
        return (x == 0).sum()
    df = pd.merge(df, df_probs[['prob', 'intervalo']], how = 'left', left_on = 'prob', right_on = 'prob')
    df = df.groupby(by = 'intervalo', axis = 0).agg({'prob':'max', 'alvo':[count_1, count_0]})
    df.columns = ['_'.join(col) for col in df.columns]
    df['count'] = df['alvo_count_1'] + df['alvo_count_0']
    df['alvo_acum_1'] = df['alvo_count_1'].cumsum()
    df['alvo_acum_0'] = df['alvo_count_0'].cumsum()
    df['count_acum'] = df['count'].cumsum()
    df['prob_acum_1'] = df['alvo_acum_1']/df['count_acum']
    df['prob_acum_0'] = df['alvo_acum_0']/df['count_acum']
    df = df.iloc[:len(df)-1, :]
    df['prob_acum_1_compl'] = (tam1 - df['alvo_acum_1'])/(tam - df['count_acum'])
    df['prob_acum_0_compl'] = (tam0 - df['alvo_acum_0'])/(tam - df['count_acum'])
    df['entropia'] = df['prob_acum_1'].apply(lambda p: entropy([p, 1 - p], base = 2))
    df['entropia_compl'] = df['prob_acum_1_compl'].apply(lambda p: entropy([p, 1 - p], base = 2))
    df['entropia_geral'] = (df['entropia'].values*df['count_acum'].values + df['entropia_compl'].values*(tam - df['count_acum'].values))/tam
    df['info_gain'] = (entropia_inicial - df['entropia_geral'])/entropia_inicial
    
    df_ig = df[['prob_max', 'info_gain']].copy()
    df_ig = df_ig.rename({'prob_max': 'prob'}, axis = 1)
    info_gain_max = df_ig['info_gain'].max()
    prob_max = df_ig[df_ig['info_gain'] == info_gain_max]['prob'].mean()
    return df_ig, info_gain_max, prob_max
    
###############################

def calculaMetricasModelos(df_train_probs, df_test_probs, df_train_preds, df_test_preds, df_importance_vars, 
                           num_vars = 10, plot = False):    
    
    #Calcula a importância média e o desvio padrão das importâncias entre os modelos e amostras
    df_vars = pd.DataFrame()
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
    num_amostras = len(selectColumns(df_train_probs, 'v').columns)
    
    #Plota a Matriz de Confusão
    acuracia_train = []
    acuracia_balanceada_train = []
    acuracia_test = []
    acuracia_balanceada_test = []
    for i in range(0, num_amostras):
        y_train = df_train_probs['v'+str(i+1)].dropna(axis = 0)
        y_train_prob = selectColumns(df_train_preds, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
        y_train_pred = [1 if y_train_prob[j] >= 0.5 else 0 for j in range(0, len(y_train_prob))]
        acuracia_train.append(metrics.accuracy_score(y_train, y_train_pred))
        acuracia_balanceada_train.append(metrics.balanced_accuracy_score(y_train, y_train_pred))
        if(i == 0):
            cf_matrix_train = metrics.confusion_matrix(y_train, y_train_pred)
        else:
            cf_matrix_train = cf_matrix_train + metrics.confusion_matrix(y_train, y_train_pred)
        
        if(len(df_test_preds) > 0):
            y_test = df_test_probs['v'+str(i+1)].dropna(axis = 0)
            y_test_prob = selectColumns(df_test_preds, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
            y_test_pred = [1 if y_test_prob[j] >= 0.5 else 0 for j in range(0, len(y_test_prob))]
            acuracia_test.append(metrics.accuracy_score(y_test, y_test_pred))
            acuracia_balanceada_test.append(metrics.balanced_accuracy_score(y_test, y_test_pred))
            if(i == 0):
                cf_matrix_test = metrics.confusion_matrix(y_test, y_test_pred)
            else:
                cf_matrix_test = cf_matrix_test + metrics.confusion_matrix(y_test, y_test_pred)
        else:
            acuracia_test.append(np.nan)
            acuracia_balanceada_test.append(np.nan)
    
    labels = [0, 1] #Define os labels da matriz de confusão (padronizado)
    
    #Faz a média de todas as matrizes de confusão que somamos de todas as amostras
    cf_matrix_train = cf_matrix_train.astype('float')/num_amostras
    #Normaliza a matriz de confusão (NORMALIZA PELO VALOR REAL)
    cf_matrix_train_nr = cf_matrix_train.astype('float') / cf_matrix_train.sum(axis = 1)[:, np.newaxis]
    df_matrix_train_nr = pd.DataFrame(cf_matrix_train_nr, columns = labels, index = labels)
    #Normaliza a matriz de confusão (NORMALIZA PELO VALOR PREDITO)
    cf_matrix_train_np = cf_matrix_train.astype('float') / cf_matrix_train.sum(axis = 0)[np.newaxis, :]
    df_matrix_train_np = pd.DataFrame(cf_matrix_train_np, columns = labels, index = labels)
    df_matrix_train_nr = df_matrix_train_nr.fillna(0)
    df_matrix_train_np = df_matrix_train_np.fillna(0)
    if(len(df_test_preds) > 0):
        cf_matrix_test = cf_matrix_test.astype('float')/num_amostras
        cf_matrix_test_nr = cf_matrix_test.astype('float') / cf_matrix_test.sum(axis = 1)[:, np.newaxis]
        df_matrix_test_nr = pd.DataFrame(cf_matrix_test_nr, columns = labels, index = labels)
        cf_matrix_test_np = cf_matrix_test.astype('float') / cf_matrix_test.sum(axis = 0)[np.newaxis, :]
        df_matrix_test_np = pd.DataFrame(cf_matrix_test_np, columns = labels, index = labels)
        df_matrix_test_nr = df_matrix_test_nr.fillna(0)
        df_matrix_test_np = df_matrix_test_np.fillna(0)
    if(plot == True):
        fig, axs = plt.subplots(1, 2, sharey = 'row')
        disp = metrics.ConfusionMatrixDisplay(df_matrix_train_nr.values, display_labels = labels)
        disp.plot(ax = axs[0], values_format = '.2g')
        disp.ax_.set_title('Norm. Verdadeiro')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('Valor Verdadeiro')
        disp = metrics.ConfusionMatrixDisplay(df_matrix_train_np.values, display_labels = labels)
        disp.plot(ax = axs[1], values_format = '.2g')
        disp.ax_.set_title('Norm. Predito')
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        fig.suptitle('Matriz de Confusão: Treino')
        fig.text(0.4, 0.1, 'Valor Predito', ha = 'left')
        plt.subplots_adjust(wspace = 0.40, hspace = 0.1)
        fig.colorbar(disp.im_, ax = axs)
        plt.show()
        if(len(df_test_preds) > 0):
            fig, axs = plt.subplots(1, 2, sharey = 'row')
            disp = metrics.ConfusionMatrixDisplay(df_matrix_test_nr.values, display_labels = labels)
            disp.plot(ax = axs[0], values_format = '.2g')
            disp.ax_.set_title('Norm. Verdadeiro')
            disp.im_.colorbar.remove()
            disp.ax_.set_xlabel('')
            disp.ax_.set_ylabel('Valor Verdadeiro')
            disp = metrics.ConfusionMatrixDisplay(df_matrix_test_np.values, display_labels = labels)
            disp.plot(ax = axs[1], values_format = '.2g')
            disp.ax_.set_title('Norm. Predito')
            disp.im_.colorbar.remove()
            disp.ax_.set_xlabel('')
            disp.ax_.set_ylabel('')
            fig.suptitle('Matriz de Confusão: Teste')
            fig.text(0.4, 0.1, 'Valor Predito', ha = 'left')
            plt.subplots_adjust(wspace = 0.40, hspace = 0.1)
            fig.colorbar(disp.im_, ax = axs)
            plt.show()
    
    df_params['Acurácia_Treino'] = acuracia_train
    df_params['Acurácia_Teste'] = acuracia_test
    df_params['Acurácia_Balanceada_Treino'] = acuracia_balanceada_train
    df_params['Acurácia_Balanceada_Teste'] = acuracia_balanceada_test
    
    #Plota as curvas ROC
    if(plot == True):
        fig, axs = plt.subplots(1, 2, figsize = [12, 4])
    auc_train = []
    auc_test = []
    for i in range(0, num_amostras):
        y_train = df_train_probs['v'+str(i+1)].dropna(axis = 0)
        y_train_prob = selectColumns(df_train_probs, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
        fpr_train, tpr_train, thr_train = metrics.roc_curve(y_train, y_train_prob)
        auc_train.append(metrics.auc(fpr_train, tpr_train))
        if(len(df_test_probs) > 0):
            y_test = df_test_probs['v'+str(i+1)].dropna(axis = 0)
            y_test_prob = selectColumns(df_test_probs, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
            fpr_test, tpr_test, thr_test = metrics.roc_curve(y_test, y_test_prob)
            auc_test.append(metrics.auc(fpr_test, tpr_test))
        else:
            auc_test.append(np.nan)
        if(plot == True):
            axs[0].plot(fpr_train, tpr_train, color='darkorange', lw = 2)
            if(len(df_test_probs) > 0):
                axs[1].plot(fpr_test, tpr_test, color='darkorange', lw = 2)
    if(plot == True):
        for i in range(0, 2):
            axs[i].plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
            axs[i].set_xlabel('Taxa de Falso Positivo')
            axs[i].set_ylabel('Taxa de Verdadeiro Positivo')
            axs[i].set_xlim([0.0, 1.0])
            #axs[i].set_ylim([0.0, 1.05])
        plt.subplots_adjust(wspace = 0.9, hspace = 0.1)
        fig.suptitle('Curvas ROC: Treino / Teste')
        plt.show()
    
    df_params['AUC_Treino'] = auc_train
    df_params['AUC_Teste'] = auc_test
    
    #Plota as curvas KS
    if(plot == True):
        fig, axs = plt.subplots(1, 2, figsize = [12, 4])
    ks_train = []
    ks_test = []
    for i in range(0, num_amostras):
        y_train = df_train_probs['v'+str(i+1)].dropna(axis = 0).values
        y_train_prob = selectColumns(df_train_probs, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1).values
        curva_1, curva_0, ks = calculaKS(y_train, y_train_prob)
        #df_disc, _ = calculaKS_discretizado(y_test, y_test_prob)
        #print(df_disc[:2])
        ks_train.append(ks)
        if(len(df_test_probs) > 0):
            y_test = df_test_probs['v'+str(i+1)].dropna(axis = 0).values
            y_test_prob = selectColumns(df_test_probs, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1).values
            curva_1v, curva_0v, ksv = calculaKS(y_test, y_test_prob)
            ks_test.append(ksv)
        else:
            ks_test.append(np.nan)
        if(plot == True):
            if(i == 0):        
                axs[0].plot(curva_1['prob'], curva_1['frac_alvo_acum'], color = 'blue', lw = 2, label = 'Curva Alvo = 1')
                axs[0].plot(curva_0['prob'], curva_0['frac_alvo_acum'], color = 'red',lw = 2, label = 'Curva Alvo = 0')
                if(len(df_test_probs) > 0):
                    axs[1].plot(curva_1v['prob'], curva_1v['frac_alvo_acum'], color = 'blue', lw = 2, label = 'Curva Alvo = 1')
                    axs[1].plot(curva_0v['prob'], curva_0v['frac_alvo_acum'], color = 'red',lw = 2, label = 'Curva Alvo = 0')
            else:
                axs[0].plot(curva_1['prob'], curva_1['frac_alvo_acum'], color = 'blue', lw = 2)
                axs[0].plot(curva_0['prob'], curva_0['frac_alvo_acum'], color = 'red',lw = 2)
                if(len(df_test_probs) > 0):
                    axs[1].plot(curva_1v['prob'], curva_1v['frac_alvo_acum'], color = 'blue', lw = 2)
                    axs[1].plot(curva_0v['prob'], curva_0v['frac_alvo_acum'], color = 'red',lw = 2)
    if(plot == True):
        for i in range(0, 2):
            axs[i].set_xlabel('Probabilidade de Alvo = 1')
            axs[i].set_ylabel('Fração de Alvos Acumulados')
            axs[i].legend(loc = 'center left', ncol = 1, frameon = True, bbox_to_anchor = (1, 0.1))
            #axs[i].set_xlim([0.0, 1.0])
            #axs[i].set_ylim([0.0, 1.05])
        plt.subplots_adjust(wspace = 0.9, hspace = 0.1)
        fig.suptitle('Curvas KS: Treino / Teste')
        plt.show()

    df_params['KS_Treino'] = ks_train
    df_params['KS_Teste'] = ks_test

    #Plota as curvas de ganho de informação (Information Gain - IG)
    if(plot == True):
        fig, axs = plt.subplots(1, 2, figsize = [12, 4])
    info_train = []
    prob_train = []
    info_test = []
    prob_test = []
    for i in range(0, num_amostras):
        y_train = df_train_probs['v'+str(i+1)].dropna(axis = 0)
        y_train_prob = selectColumns(df_train_probs, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
        curva_train, info_gain_train, prob_max_train = calculaInfoGain(y_train, y_train_prob)
        info_train.append(info_gain_train)
        prob_train.append(prob_max_train)
        if(len(df_test_probs) > 0):
            y_test = df_test_probs['v'+str(i+1)].dropna(axis = 0)
            y_test_prob = selectColumns(df_test_probs, '_'+str(i+1)+'a').dropna(axis = 0).mean(axis = 1)
            curva_test, info_gain_test, prob_max_test = calculaInfoGain(y_test, y_test_prob)
            info_test.append(info_gain_test)
            prob_test.append(prob_max_test)
        else:
            info_test.append(np.nan)
            prob_test.append(np.nan)
        
        if(plot == True):
            axs[0].plot(curva_train['prob'], curva_train['info_gain'], color = 'black', lw = 2)
            if(len(df_test_probs) > 0):
                axs[1].plot(curva_test['prob'], curva_test['info_gain'], color = 'black', lw = 2)
    if(plot == True):
        for i in range(0, 2):
            axs[i].set_xlabel('Probabilidade de Corte')
            axs[i].set_ylabel('Ganho de Informação')
            #axs[i].set_xlim([0.0, 1.0])
            #axs[i].set_ylim([0.0, 1.05])
        plt.subplots_adjust(wspace = 0.9, hspace = 0.1)
        fig.suptitle('Curvas IG: Treino / Teste')
        plt.show()

    df_params['IG_Treino'] = info_train
    df_params['IG_Teste'] = info_test
    df_params['Prob_Treino'] = prob_train
    df_params['Prob_Teste'] = prob_test
    
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
    splits.get_n_splits(X, y)
    
    lista_vars = list(X.columns) #Pega a lista das variaveis de entrada
    
    #Inicia as bases que vai guardar os dados do modelo
    df_models = pd.DataFrame(columns = ['Nome_Modelo', 'Objeto'])
    df_train_probs = pd.DataFrame()
    df_test_probs = pd.DataFrame()
    df_train_preds = pd.DataFrame()
    df_test_preds = pd.DataFrame()
    df_importance_vars = pd.DataFrame()
    dh, prog_ant, tempo_ini = [], 0, 0
    i = 0
    
    for train_index, test_index in splits.split(X, y):
        dh, prog_ant, tempo_ini = mydisplay.updateProgress(i/amostras, 2, dh, prog_ant, tempo_ini)
        i = i + 1
        
        #Aplica a divisão de treino e teste
        if(so_tem_treino == False):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]
        else:
            X_train, y_train = X.iloc[df.index, :], y.iloc[df.index, :]
            X_test = pd.DataFrame(columns = X_train.columns)
            y_test = pd.DataFrame(columns = y_train.columns)
        
        #Treina os modelos e salva a importância das variaveis e probabilidades preditas de cada um
        j = 0
        for m in models:
            if(train_cv == 0):
                j = j + 1
                model_temp = clone(m)
                model_temp.fit(X_train, y_train[nome_alvo], verbose = 0)
                #Salva a importância das variáveis e predições
                try:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp.feature_importances_, index = lista_vars)
                except:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                df_train_probs['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict_proba(X_train)[:,1]
                df_train_preds['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp]
                if(len(X_test) > 0):
                    df_test_probs['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict_proba(X_test)[:,1]
                    df_test_preds['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_test)
                del model_temp
                gc.collect()
            else:
                #Faz as divisões para treino com validação cruzada (esse é bom que seja aleatório mesmo)
                splits2 = ShuffleSplit(n_splits = train_cv, test_size = 0.5)
                splits2.get_n_splits(X_train, y_train)
                for train_index1, train_index2 in splits2.split(X_train, y_train):
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
                    df_train_probs['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict_proba(X_train)[:,1]
                    df_train_preds['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp1]
                    if(len(X_test) > 0):
                        df_test_probs['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict_proba(X_test)[:,1]
                        df_test_preds['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_test)
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
                    df_train_probs['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict_proba(X_train)[:,1]
                    df_train_preds['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp2]
                    if(len(X_test) > 0):
                        df_test_probs['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict_proba(X_test)[:,1]
                        df_test_preds['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_test)
                    del eval_set2, model_temp2
                    del X_train1, X_train2, y_train1, y_train2
                    gc.collect()
                del splits2
                gc.collect()
        del X_train, X_test
        gc.collect()
        
        #Salva as respostas para poder comparar a predição com a resposta depois
        if(len(df_test_probs) > 0):
            df_train_probs['v'+str(i)] = y_train.reset_index(drop = True)
            df_train_preds['v'+str(i)] = y_train.reset_index(drop = True)
            df_test_probs['v'+str(i)] = y_test.reset_index(drop = True)
            df_test_preds['v'+str(i)] = y_test.reset_index(drop = True)
        else:
            df_train_probs['v'+str(i)] = y.reset_index(drop = True)
            df_train_preds['v'+str(i)] = y.reset_index(drop = True)
        del y_train, y_test
        gc.collect()
        
        #Salva os índices do split dos dados de treino e teste
        if(len(df_test_probs) > 0):
            df_train_probs['i'+str(i)] = train_index
            df_train_preds['i'+str(i)] = train_index
            df_test_probs['i'+str(i)] = test_index
            df_test_preds['i'+str(i)] = test_index
        else:
            df_train_probs['i'+str(i)] = df.index
            df_train_preds['i'+str(i)] = df.index
            df_test_probs = pd.DataFrame(columns = df_train_probs.columns)
            df_test_preds = pd.DataFrame(columns = df_train_preds.columns)
    
    for m in models:
        del m
    del models, X, y, lista_vars, splits
    gc.collect()
    dh, prog_ant, tempo_ini = mydisplay.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    return df_models, df_train_probs, df_test_probs, df_train_preds, df_test_preds, df_importance_vars
    
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
    df_train_probs = pd.DataFrame()
    df_test_probs = pd.DataFrame()
    df_train_preds = pd.DataFrame()
    df_test_preds = pd.DataFrame()
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
    
        df_train_probs_temp = pd.DataFrame()
        df_test_probs_temp = pd.DataFrame()
        df_train_preds_temp = pd.DataFrame()
        df_test_preds_temp = pd.DataFrame()
    
        #Treina os modelos e salva a importância das variaveis e probabilidades preditas de cada um
        j = 0
        for m in models:
            if(train_cv == 0):
                j = j + 1
                model_temp = clone(m)
                model_temp.fit(X_train, y_train[nome_alvo], verbose = 0)
                #Salva a importância das variáveis e predições
                try:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp.feature_importances_, index = lista_vars)
                except:
                    df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series([0 for v in lista_vars], index = lista_vars)
                df_train_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict_proba(X_train)[:,1]
                df_train_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp]
                if(len(X_test) > 0):
                    df_test_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict_proba(X_test)[:,1]
                    df_test_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp.predict(X_test)
                del model_temp
                gc.collect()
            elif(train_cv > 0):
                #Faz as divisões para treino com validação cruzada (esse é bom que seja aleatório mesmo)
                splits2 = ShuffleSplit(n_splits = train_cv, test_size = 0.5)
                splits2.get_n_splits(X_train, y_train)
                for train_index1, train_index2 in splits2.split(X_train, y_train):
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
                    df_train_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict_proba(X_train)[:,1]
                    df_train_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp1]
                    if(len(X_test) > 0):
                        df_test_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict_proba(X_test)[:,1]
                        df_test_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_test)
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
                    df_train_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict_proba(X_train)[:,1]
                    df_train_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_train)
                    df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp2]
                    if(len(X_test) > 0):
                        df_test_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict_proba(X_test)[:,1]
                        df_test_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_test)
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
                df_train_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict_proba(X_train)[:,1]
                df_train_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp1]
                if(len(X_test) > 0):
                    df_test_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict_proba(X_test)[:,1]
                    df_test_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp1.predict(X_test)
                del eval_set1, model_temp1
                gc.collect()

                j = j + 1
                eval_set2 = [(X_train2, y_train2[nome_alvo]), (X_train1, y_train1[nome_alvo])]
                model_temp2 = clone(m)
                model_temp2.fit(X_train2, y_train2[nome_alvo], eval_set = eval_set2, verbose = 0)
                #Salva a importância das variáveis e predições
                df_importance_vars['m'+str(j)+'_'+str(i)+'a'] = pd.Series(model_temp2.feature_importances_, index = lista_vars)
                df_train_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict_proba(X_train)[:,1]
                df_train_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_train)
                df_models.loc[len(df_models)] = ['m'+str(j)+'_'+str(i)+'a', model_temp2]
                if(len(X_test) > 0):
                    df_test_probs_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict_proba(X_test)[:,1]
                    df_test_preds_temp['m'+str(j)+'_'+str(i)+'a'] = model_temp2.predict(X_test)
                del eval_set2, model_temp2
                del X_train1, X_train2, y_train1, y_train2
                gc.collect()
                
        del X_train, X_test, df_train
        gc.collect()

        #Salva as respostas para poder comparar a predição com a resposta depois
        if(len(df_test_probs_temp) > 0):
            df_train_probs_temp['v'+str(i)] = y_train.reset_index(drop = True)
            df_train_preds_temp['v'+str(i)] = y_train.reset_index(drop = True)
            df_test_probs_temp['v'+str(i)] = y_test.reset_index(drop = True)
            df_test_preds_temp['v'+str(i)] = y_test.reset_index(drop = True)
        else:
            df_train_probs_temp['v'+str(i)] = y.reset_index(drop = True)
            df_train_preds_temp['v'+str(i)] = y.reset_index(drop = True)
        del y_train, y_test
        gc.collect()

        #Salva os índices do split dos dados de treino e teste
        if(len(df_test_probs_temp) > 0):
            df_train_probs_temp['i'+str(i)] = train_index
            df_train_preds_temp['i'+str(i)] = train_index
            df_test_probs_temp['i'+str(i)] = test_index
            df_test_preds_temp['i'+str(i)] = test_index
        else:
            df_train_probs_temp['i'+str(i)] = df.index
            df_train_preds_temp['i'+str(i)] = df.index
            df_test_probs_temp = pd.DataFrame(columns = df_train_probs_temp.columns)
            df_test_preds_temp = pd.DataFrame(columns = df_train_preds_temp.columns)
            
        df_train_probs = pd.concat([df_train_probs, df_train_probs_temp], sort = False, axis = 1)
        df_train_preds = pd.concat([df_train_preds, df_train_preds_temp], sort = False, axis = 1)
        df_test_probs = pd.concat([df_test_probs, df_test_probs_temp], sort = False, axis = 1)
        df_test_preds = pd.concat([df_test_preds, df_test_preds_temp], sort = False, axis = 1)
        del df_train_probs_temp, df_train_preds_temp, df_test_probs_temp, df_test_preds_temp
        gc.collect()
    
    for m in models:
        del m
    del models, X, y, lista_vars, df_loop
    gc.collect()
    dh, prog_ant, tempo_ini = mydisplay.updateProgress(1, 2, dh, prog_ant, tempo_ini)
    return df_models, df_train_probs, df_test_probs, df_train_preds, df_test_preds, df_importance_vars