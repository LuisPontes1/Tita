import numpy as np
import pandas as pd

from scipy.optimize import minimize

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations_with_replacement, combinations
from itertools import groupby
from operator import itemgetter

class LinearBoostRegressor:
    
    def __init__(self, max_termos = None, passo = 1):
        self.__max_termos = max_termos
        self.__passo = passo
        
        self.__nome_vars = None
        self.__X = None
        self.__y = None
        self.__num_linhas = None
        self.__num_cols = None
        
        self.__Xval = None
        self.__yval = None
        self.__tem_validacao = None
        self.__num_linhas_val = None
        self.__num_cols_val = None
    
    def __checa_X_e_y_treino(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.__X = X.values
            self.__nome_vars = X.columns.values
        else:
            try:
                if len(X.shape) == 2:
                    self.__X = X
                else:
                    print("Valores de entrada de treino não adequados")
                    return
            except:
                print("Valores de entrada de treino não adequados")
                return
        self.__num_linhas = self.__X.shape[0]
        self.__num_cols = self.__X.shape[1]
        
        if (isinstance(y, pd.DataFrame) and len(y.columns) == 1) or (isinstance(y, pd.Series)):
            self.__y = y.values
        else:
            try:
                if len(y.shape) == 1:
                    self.__y = y
                else:
                    print("Valores de alvo de treino não adequados")
                    return
            except:
                print("Valores de alvo de treino não adequados")
                return
        if(self.__y.size != self.__num_linhas):
            print("Quantidade de exemplos não coindicem em X e y")
            return
            
    def __checa_conjunto_validacao(self, conj_val):
        #Verifica se foi passado um conjunto de validação
        if conj_val != None:
            if isinstance(conj_val, tuple):
                if(len(conj_val) == 2):
                    if isinstance(conj_val[0], pd.DataFrame):
                        self.__Xval = conj_val[0].values
                    else:
                        try:
                            if len(conj_val[0].shape) == 2:
                                self.__Xval = conj_val[0]
                            else:
                                print("Valores de entrada de validação não adequados")
                                return
                        except:
                            print("Valores de entrada não adequados")
                            return
                    self.__num_linhas_val = self.__Xval.shape[0]
                    self.__num_cols_val = self.__Xval.shape[1]
                        
                    if (isinstance(conj_val[1], pd.DataFrame) and len(conj_val[1].columns) == 1) or (isinstance(conj_val[1], pd.Series)):
                        self.__yval = conj_val[1].values
                    else:
                        try:
                            if len(conj_val[1].shape) == 1:
                                self.__yval = conj_val[1]
                            else:
                                print("Valores de alvo de validação não adequados")
                                return
                        except:
                            print("Valores de alvo de validação não adequados")
                            return
                        
                else:
                    print("Valores de validação não adequados")
                    return
            else:
                print("Valores de validação não adequados")
                return
            self.__tem_validacao = True
            if(self.__yval.size != self.__num_linhas_val):
                print("Quantidade de exemplos não coindicem em Xval e yval")
                return
        else:
            self.__tem_validacao = False

    def __checa_X_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            try:
                if len(X.shape) != 2:
                    print("Valores de entrada de predição não adequados")
                    return
            except:
                print("Valores de entrada de predição não adequados")
                return
        return X 
      
    def __guarda_melhor_modelo(self, r2_val):
        #Guarda o modelo atual nas variáveis de melhor modelo
        self.__vars_selec_melhor = self.__vars_selec
        self.__num_termos_melhor = self.__num_termos
        self.__thetas_melhor = self.__thetas
        self.__r2_val_melhor = r2_val  
        
    def __calcula_r2(self, diff, mse_baseline):
        return 1 - np.mean(np.power(diff, 2))/mse_baseline
    
    def __derivadas_parciais_custo(self, diff, X):
        return np.dot(diff, X)
    
    def fit(self, X, y, conj_val = None, verbose = True):
        self.__checa_X_e_y_treino(X, y)
        if(self.__nome_vars is None):
            self.__nome_vars = np.array(['x' + str(i) for i in range(0, self.__num_cols)])
        self.__checa_conjunto_validacao(conj_val)
        
        #Faz a normalização (e filtra colunas com desvio padrão zero)
        #Facilita calculo de Inv. de Matriz e também facilita a Descida do Gradiente
        #(adaptado para lidar com NA)
        self.__means = np.nan_to_num(np.nanmean(self.__X, axis = 0), posinf = 0, neginf = 0)
        self.__stds = np.sqrt(np.nan_to_num(np.nanmean(np.power(self.__X - self.__means, 2), axis = 0), posinf = 0, neginf = 0))
        #self.__means = np.mean(self.__X, axis = 0)
        #self.__stds = np.sqrt(np.mean(np.power(self.__X - self.__means, 2), axis = 0))
        
        self.__flag_vars_validas = self.__stds > 0
        self.__num_cols = np.sum(self.__flag_vars_validas)
        self.__nome_vars = self.__nome_vars[self.__flag_vars_validas]
        self.__X = self.__X[:, self.__flag_vars_validas]
        self.__means = self.__means[self.__flag_vars_validas]
        self.__stds = self.__stds[self.__flag_vars_validas]
        if(self.__tem_validacao):
            self.__Xval = self.__Xval[:, self.__flag_vars_validas]
        
        self.__X = (self.__X - self.__means)/self.__stds
        self.__X = np.nan_to_num(self.__X, posinf = 0, neginf = 0) #Adaptação para ignorar os nulos
        if(self.__tem_validacao):
            self.__Xval = (self.__Xval - self.__means)/self.__stds
            self.__Xval = np.nan_to_num(self.__Xval, posinf = 0, neginf = 0) #Adaptação para ignorar os nulos
        
        #Adiciona a coluna do termo constante
        self.__X = np.insert(self.__X, 0, np.ones(self.__num_linhas), axis = 1)
        if(self.__tem_validacao):
            self.__Xval = np.insert(self.__Xval, 0, np.ones(self.__num_linhas_val), axis = 1)
        
        #Normaliza o y também: ajuda na Descida do Gradiente (Aqui não pode ter nulos mesmo!!)
        self.__media_y = np.mean(self.__y)
        self.__desvio_y = np.sqrt(np.mean(np.power(self.__y - self.__media_y, 2)))
        self.__y = (self.__y - self.__media_y)/self.__desvio_y
        if(self.__tem_validacao):
            self.__yval = (self.__yval - self.__media_y)/self.__desvio_y
        
        #Faz o modelo baseline: só o termo theta_0 -> cte = 0!!! (pq tá normalizado)
        self.__vars_selec = np.array([0]).astype(int) #Lista das variaveis selecionadas no regressor
        self.__num_termos = 1 #Número de termos do regressor
        self.__thetas = np.array([0]) #Valor dos thetas do regressor
        self.__diff = -self.__y # Diferença entre o valor predito e o valor do alvo
        if(self.__tem_validacao):
            self.__diff_val = -self.__yval # Diferença entre o valor predito e o valor do alvo
        
        #Salva o mse do modelo baseline (só constante = 0)
        self.__mse_baseline = 1 #É 1 pq é o desvio padrão (que é 1 pq y tá normalizado)
        if(self.__tem_validacao):
            self.__mse_baseline_val = np.mean(np.power(self.__yval, 2))
            
        #####
        ##### Algoritmo em loop incremental estilo Boosting para aumentar a complexidade #####
        #####
        
        #Funções para otimização da função de custo (MSE)
        def mse(thetas):
            return np.mean(np.power(np.dot(X, thetas) - self.__y, 2))/2
        def mse_der(thetas):
            return np.dot(np.dot(X, thetas), X)/self.__num_linhas - diff_cte
        def mse_hess(thetas):
            return matriz_hess
        
        #Inicializamos as variaveis para criar a curva viés-variância
        self.__curva_num_termos = np.array([self.__num_termos])
        self.__curva_r2 = np.array([0])
        if(self.__tem_validacao):
            r2_val = 0
            self.__curva_r2_val = np.array([r2_val])
            self.__guarda_melhor_modelo(r2_val)
            
        vars_disp = 1 + np.arange(self.__num_cols) #vetor com as variaveis disponíveis para escolher
        if(self.__max_termos == None):
            self.__max_termos = vars_disp.size + 1 #Considera todos os termos se não tiver dado limite
        else:
            self.__max_termos = min(self.__max_termos, self.__num_cols + 1)
            
        #Loop do processo incremental de adição de termos na regressão
        while(self.__num_termos < self.__max_termos and vars_disp.size > 0 and (self.__tem_validacao == False or r2_val >= 0)):
            
            #Calcula todos valores das derividas parciais dos termos que estão faltando na regressão
            derivadas_parciais = self.__derivadas_parciais_custo(self.__diff, self.__X[:, vars_disp])/self.__num_linhas
            derivadas_parciais_abs = np.abs(derivadas_parciais) #Pega os valores da derivada em módulo
            
            #Pega os indices para a ordenação decrescente
            inds_ordena_derivadas = np.argsort(derivadas_parciais_abs)[::-1]
            
            #Ordena também primeiro onde os sinais das derivadas são consistentes no conjunto de validação
            if(self.__tem_validacao):
                derivadas_parciais_val = self.__derivadas_parciais_custo(self.__diff_val, self.__Xval[:, vars_disp])/self.__num_linhas_val
                flag_mesmo_sinal = np.sign(derivadas_parciais) == np.sign(derivadas_parciais_val)
                flag_inds = flag_mesmo_sinal[inds_ordena_derivadas]
                inds_ordena_derivadas = np.append(inds_ordena_derivadas[flag_inds], inds_ordena_derivadas[~flag_inds])
            
            #Pega os indices dos termos que vamos adicionar (de acordo com o passo escolhido)
            passo = min(self.__passo, vars_disp.size)
            pos_der_add = inds_ordena_derivadas[:passo] 
            vars_der_add = vars_disp[pos_der_add] #Lista das respectivas vars para adicionar
            
            #Adiciona as variáveis novas
            self.__vars_selec = np.append(self.__vars_selec, vars_der_add)
            self.__num_termos = self.__num_termos + passo
            self.__thetas = np.append(self.__thetas, np.zeros(passo)) 
            vars_disp = np.delete(vars_disp, pos_der_add) #Remove essas variaveis das disponíveis
            
            #Note que: a ideia aqui foi incrementar o regressor com a variável que mais vai "bagunçar"
            # o mínimo que já haviamos encontrado para a Função de Custo: ou seja, deriv. parcial max.
            
            #Agora vamos encontrar os novos thetas que minimizam a função de custo
            #partindo dos thetas antigos acrescidos de valores zero para as novas features
            X = self.__X[:, self.__vars_selec]
            diff_cte = np.dot(self.__y, X)/self.__num_linhas
            matriz_hess = np.dot(np.transpose(X), X)/self.__num_linhas
            self.__thetas = minimize(mse, self.__thetas, method = 'Newton-CG', jac = mse_der, hess = mse_hess, 
                                     options = {'xtol': 1e-8, 'disp': False}).x
            self.__diff = np.dot(X, self.__thetas) - self.__y
            if(self.__tem_validacao):
                self.__diff_val = np.dot(self.__Xval[:, self.__vars_selec], self.__thetas) - self.__yval
            
            #Calcula os MSE e adiciona na curva de Viés-Variância
            self.__curva_num_termos = np.append(self.__curva_num_termos, self.__num_termos)
            r2 = self.__calcula_r2(self.__diff, self.__mse_baseline)
            self.__curva_r2 = np.append(self.__curva_r2, r2)
            if(self.__tem_validacao):
                r2_val = self.__calcula_r2(self.__diff_val, self.__mse_baseline_val)
                self.__curva_r2_val = np.append(self.__curva_r2_val, r2_val)
                #Vê se encontramos um novo melhor modelo e guarda ele
                if(r2_val > self.__r2_val_melhor):
                    self.__guarda_melhor_modelo(r2_val)
                if(verbose):
                    print(str(r2) + " / " + str(r2_val) + " (" + str(self.__num_termos_melhor) + ")")
            else:
                if(verbose):
                    print(str(r2) + " (" + str(self.__num_termos) + ")")

        if(r2_val <= 0):
            self.__curva_num_termos = self.__curva_num_termos[:-1]
            self.__curva_r2 = self.__curva_r2[:-1]
            if(self.__tem_validacao):
                self.__curva_r2_val = self.__curva_r2_val[:-1]
            if(self.__fit_total):
                self.__curva_r2_tot = self.__curva_r2_tot[:-1]

        self.__calcula_importancias()
        
    def predict(self, X):
        X = self.__checa_X_predict(X)
        X = X[:, self.__flag_vars_validas]
        X = (X - self.__means)/self.__stds
        X = np.nan_to_num(X, posinf = 0, neginf = 0) #Adaptação para ignorar os nulos
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        if(self.__tem_validacao):
            y_pred = np.dot(X[:, self.__vars_selec_melhor], self.__thetas_melhor)
        else:
            y_pred = np.dot(X[:, self.__vars_selec], self.__thetas)
        y_pred = self.__media_y + self.__desvio_y*y_pred 
        return y_pred
    
    def grafico_vies_variancia(self, pos_ini = None, pos_fim = None, figsize = [8, 6]):        
        #Prepara os valores e vetores de plot
        if(pos_ini == None):
            pos_ini = 0
        if(pos_fim == None):
            pos_fim = self.__curva_num_termos.size
        curva_num_termos = self.__curva_num_termos[pos_ini:pos_fim]
        curva_r2 = self.__curva_r2[pos_ini:pos_fim]
        if(self.__tem_validacao):
            curva_r2_val = self.__curva_r2_val[pos_ini:pos_fim]
            r2_val_melhor = self.__r2_val_melhor
            r2_min = min(min(curva_r2[np.isfinite(curva_r2)]), min(curva_r2_val[np.isfinite(curva_r2_val)]))
        #Plota as curvas e o ponto de parada do treinamento pela validação
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.plot(curva_num_termos, curva_r2, color = paleta_cores[0], label = 'Treino')
            if(self.__tem_validacao):
                axs.plot(curva_num_termos, curva_r2_val, color = paleta_cores[1], label = 'Validação')
                axs.vlines(self.__num_termos_melhor, r2_min, r2_val_melhor, color = 'k', 
                           linestyle = '--', label = 'Ponto de Parada')
            axs.set_xlabel('Número de Termos')
            axs.set_ylabel('R2 Ajustado')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def __calcula_importancias(self):
        if(self.__tem_validacao):
            vars_selec = self.__vars_selec_melhor[1:] - 1
            coef_abs = np.abs(self.__thetas_melhor[1:])
        else:
            vars_selec = self.__vars_selec[1:] - 1
            coef_abs = np.abs(self.__thetas[1:])
        self.feature_names_ = self.__nome_vars[vars_selec]
        self.feature_importances_ = coef_abs/np.sum(coef_abs)
        
    def grafico_importancias(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.__curva_num_termos.size
        vars_nomes = self.feature_names_
        vars_imp = self.feature_importances_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()
            
##############################

##############################
            
class LinearBoostClassifier:
    
    def __init__(self, max_termos = None, passo = 1):
        self.__max_termos = max_termos
        self.__passo = passo
        
        self.__nome_vars = None
        self.__X = None
        self.__y = None
        self.__num_linhas = None
        self.__num_cols = None
        
        self.__Xval = None
        self.__yval = None
        self.__tem_validacao = None
        self.__num_linhas_val = None
        self.__num_cols_val = None
    
    def __checa_X_e_y_treino(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.__X = X.values
            self.__nome_vars = X.columns.values
        else:
            try:
                if len(X.shape) == 2:
                    self.__X = X
                else:
                    print("Valores de entrada de treino não adequados")
                    return
            except:
                print("Valores de entrada de treino não adequados")
                return
        self.__num_linhas = self.__X.shape[0]
        self.__num_cols = self.__X.shape[1]
        
        if (isinstance(y, pd.DataFrame) and len(y.columns) == 1) or (isinstance(y, pd.Series)):
            self.__y = y.values
        else:
            try:
                if len(y.shape) == 1:
                    self.__y = y
                else:
                    print("Valores de alvo de treino não adequados")
                    return
            except:
                print("Valores de alvo de treino não adequados")
                return
        if(self.__y.size != self.__num_linhas):
            print("Quantidade de exemplos não coindicem em X e y")
            return
            
    def __checa_conjunto_validacao(self, conj_val):
        #Verifica se foi passado um conjunto de validação
        if conj_val != None:
            if isinstance(conj_val, tuple):
                if(len(conj_val) == 2):
                    if isinstance(conj_val[0], pd.DataFrame):
                        self.__Xval = conj_val[0].values
                    else:
                        try:
                            if len(conj_val[0].shape) == 2:
                                self.__Xval = conj_val[0]
                            else:
                                print("Valores de entrada de validação não adequados")
                                return
                        except:
                            print("Valores de entrada não adequados")
                            return
                    self.__num_linhas_val = self.__Xval.shape[0]
                    self.__num_cols_val = self.__Xval.shape[1]
                        
                    if (isinstance(conj_val[1], pd.DataFrame) and len(conj_val[1].columns) == 1) or (isinstance(conj_val[1], pd.Series)):
                        self.__yval = conj_val[1].values
                    else:
                        try:
                            if len(conj_val[1].shape) == 1:
                                self.__yval = conj_val[1]
                            else:
                                print("Valores de alvo de validação não adequados")
                                return
                        except:
                            print("Valores de alvo de validação não adequados")
                            return
                        
                else:
                    print("Valores de validação não adequados")
                    return
            else:
                print("Valores de validação não adequados")
                return
            self.__tem_validacao = True
            if(self.__yval.size != self.__num_linhas_val):
                print("Quantidade de exemplos não coindicem em Xval e yval")
                return
        else:
            self.__tem_validacao = False

    def __checa_X_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            try:
                if len(X.shape) != 2:
                    print("Valores de entrada de predição não adequados")
                    return
            except:
                print("Valores de entrada de predição não adequados")
                return
        return X 
      
    def __guarda_melhor_modelo(self, coef_logloss_val):
        #Guarda o modelo atual nas variáveis de melhor modelo
        self.__vars_selec_melhor = self.__vars_selec
        self.__num_termos_melhor = self.__num_termos
        self.__thetas_melhor = self.__thetas
        self.__coef_logloss_val_melhor = coef_logloss_val  
      
    def __calcula_coef_logloss(self, y, probs, logloss_baseline):
        #logloss = -1*np.mean(y*np.log(probs) + (1 - y)*np.log(1 - probs))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            logloss = -1*np.mean(np.where(y == 1, np.log(probs), np.log(1 - probs)))
            return 1 - logloss/logloss_baseline
    
    def __derivadas_parciais_custo(self, diff, X):
        return np.dot(diff, X)
    
    def fit(self, X, y, conj_val = None, verbose = True):
        self.__checa_X_e_y_treino(X, y)
        if(self.__nome_vars is None):
            self.__nome_vars = np.array(['x' + str(i) for i in range(0, self.__num_cols)])
        self.__checa_conjunto_validacao(conj_val)
        
        #Faz a normalização (e filtra colunas com desvio padrão zero)
        #Facilita calculo de Inv. de Matriz e também facilita a Descida do Gradiente
        #(adaptado para lidar com NA)
        self.__means = np.nan_to_num(np.nanmean(self.__X, axis = 0), posinf = 0, neginf = 0)
        self.__stds = np.sqrt(np.nan_to_num(np.nanmean(np.power(self.__X - self.__means, 2), axis = 0), posinf = 0, neginf = 0))
        #self.__means = np.mean(self.__X, axis = 0)
        #self.__stds = np.sqrt(np.mean(np.power(self.__X - self.__means, 2), axis = 0))
        
        self.__flag_vars_validas = self.__stds > 0
        self.__num_cols = np.sum(self.__flag_vars_validas)
        self.__nome_vars = self.__nome_vars[self.__flag_vars_validas]
        self.__X = self.__X[:, self.__flag_vars_validas]
        self.__means = self.__means[self.__flag_vars_validas]
        self.__stds = self.__stds[self.__flag_vars_validas]
        if(self.__tem_validacao):
            self.__Xval = self.__Xval[:, self.__flag_vars_validas]
        
        self.__X = (self.__X - self.__means)/self.__stds
        self.__X = np.nan_to_num(self.__X, posinf = 0, neginf = 0) #Adaptação para ignorar os nulos
        if(self.__tem_validacao):
            self.__Xval = (self.__Xval - self.__means)/self.__stds
            self.__Xval = np.nan_to_num(self.__Xval, posinf = 0, neginf = 0) #Adaptação para ignorar os nulos
        
        #Adiciona a coluna do termo constante
        self.__X = np.insert(self.__X, 0, np.ones(self.__num_linhas), axis = 1)
        if(self.__tem_validacao):
            self.__Xval = np.insert(self.__Xval, 0, np.ones(self.__num_linhas_val), axis = 1)
        
        self.__media_y = np.mean(self.__y)  #(Aqui não pode ter nulos mesmo!!)
        
        #Faz o modelo baseline: só o termo theta_0 -> cte!!! (prob média)
        self.__vars_selec = np.array([0]).astype(int) #Lista das variaveis selecionadas no regressor
        self.__num_termos = 1 #Número de termos do regressor
        theta_0 = np.log(self.__media_y/(1 - self.__media_y)) #Dedução da função logística
        self.__thetas = np.array([theta_0]) #Valor dos thetas do regressor
        self.__probs = np.repeat(self.__media_y, self.__num_linhas) #Valor predito de prob
        self.__diff = self.__media_y - self.__y # Diferença entre o valor predito e o valor do alvo
        if(self.__tem_validacao):
            self.__probs_val = np.repeat(self.__media_y, self.__num_linhas_val) #Valor predito de prob
            self.__diff_val = self.__media_y - self.__yval # Diferença entre o valor predito e o valor do alvo
        
        #Salva o loss do modelo baseline (só constante)
        def calcula_logloss(y, probs):
            #logloss = -1*np.mean(y*np.log(probs) + (1 - y)*np.log(1 - probs))
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                logloss = -1*np.mean(np.where(y == 1, np.log(probs), np.log(1 - probs)))
                return logloss
        self.__logloss_baseline = calcula_logloss(self.__y, self.__probs)
        if(self.__tem_validacao):
            probs_val = np.repeat(self.__media_y, self.__num_linhas_val)
            self.__logloss_baseline_val = calcula_logloss(self.__yval, probs_val)
            
        #####
        ##### Algoritmo em loop incremental estilo Boosting para aumentar a complexidade #####
        #####
        
        #Funções para otimização da função de custo (MSE)
        def logloss(thetas):
            probs = 1/(1 + np.exp(-np.dot(X, thetas)))
            #logloss = -1*np.mean(self.__y*np.log(probs) + (1 - self.__y)*np.log(1 - probs))
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                logloss = -1*np.mean(np.where(self.__y == 1, np.log(probs), np.log(1 - probs)))
                return logloss
        def logloss_der(thetas):
            probs = 1/(1 + np.exp(-np.dot(X, thetas)))
            return np.dot(probs, X)/self.__num_linhas - diff_cte
        #def logloss_hess(thetas):
        #    return matriz_hess
        
        #Inicializamos as variaveis para criar a curva viés-variância
        self.__curva_num_termos = np.array([self.__num_termos])
        self.__curva_coef_logloss = np.array([0])
        if(self.__tem_validacao):
            coef_logloss_val = 0
            self.__curva_coef_logloss_val = np.array([coef_logloss_val])
            self.__guarda_melhor_modelo(coef_logloss_val)
            
        vars_disp = 1 + np.arange(self.__num_cols) #vetor com as variaveis disponíveis para escolher
        if(self.__max_termos == None):
            self.__max_termos = vars_disp.size + 1 #Considera todos os termos se não tiver dado limite
        else:
            self.__max_termos = min(self.__max_termos, self.__num_cols + 1)
            
        #Loop do processo incremental de adição de termos na regressão
        while(self.__num_termos < self.__max_termos and vars_disp.size > 0 and (self.__tem_validacao == False or coef_logloss_val >= 0)):
            
            #Calcula todos valores das derividas parciais dos termos que estão faltando na regressão
            derivadas_parciais = self.__derivadas_parciais_custo(self.__diff, self.__X[:, vars_disp])/self.__num_linhas
            derivadas_parciais_abs = np.abs(derivadas_parciais) #Pega os valores da derivada em módulo
            
            #Pega os indices para a ordenação decrescente
            inds_ordena_derivadas = np.argsort(derivadas_parciais_abs)[::-1]
            
            #Ordena também primeiro onde os sinais das derivadas são consistentes no conjunto de validação
            if(self.__tem_validacao):
                derivadas_parciais_val = self.__derivadas_parciais_custo(self.__diff_val, self.__Xval[:, vars_disp])/self.__num_linhas_val
                flag_mesmo_sinal = np.sign(derivadas_parciais) == np.sign(derivadas_parciais_val)
                flag_inds = flag_mesmo_sinal[inds_ordena_derivadas]
                inds_ordena_derivadas = np.append(inds_ordena_derivadas[flag_inds], inds_ordena_derivadas[~flag_inds])
            
            #Pega os indices dos termos que vamos adicionar (de acordo com o passo escolhido)
            passo = min(self.__passo, vars_disp.size)
            pos_der_add = inds_ordena_derivadas[:passo] 
            vars_der_add = vars_disp[pos_der_add] #Lista das respectivas vars para adicionar
            
            #Adiciona as variáveis novas
            self.__vars_selec = np.append(self.__vars_selec, vars_der_add)
            self.__num_termos = self.__num_termos + passo
            self.__thetas = np.append(self.__thetas, np.zeros(passo)) 
            vars_disp = np.delete(vars_disp, pos_der_add) #Remove essas variaveis das disponíveis
            
            #Note que: a ideia aqui foi incrementar o regressor com a variável que mais vai "bagunçar"
            # o mínimo que já haviamos encontrado para a Função de Custo: ou seja, deriv. parcial max.
            
            #Agora vamos encontrar os novos thetas que minimizam a função de custo
            #partindo dos thetas antigos acrescidos de valores zero para as novas features
            X = self.__X[:, self.__vars_selec]
            diff_cte = np.dot(self.__y, X)/self.__num_linhas
            #matriz_hess = np.dot(np.transpose(X), X)/self.__num_linhas
            #self.__thetas = minimize(logloss, self.__thetas, method = 'Newton-CG', jac = logloss_der, hess = logloss_hess, 
            #                         options = {'xtol': 1e-8, 'disp': False}).x
            self.__thetas = minimize(logloss, self.__thetas, method = 'BFGS', jac = logloss_der, 
                                     options = {'disp': False}).x
            self.__probs = 1/(1 + np.exp(-np.dot(X, self.__thetas)))
            self.__diff = self.__probs - self.__y
            if(self.__tem_validacao):
                self.__probs_val = 1/(1 + np.exp(-np.dot(self.__Xval[:, self.__vars_selec], self.__thetas)))
                self.__diff_val = self.__probs_val - self.__yval
            
            #Calcula os Coef Logloss e adiciona na curva de Viés-Variância
            self.__curva_num_termos = np.append(self.__curva_num_termos, self.__num_termos)
            coef_logloss = self.__calcula_coef_logloss(self.__y, self.__probs, self.__logloss_baseline)
            self.__curva_coef_logloss = np.append(self.__curva_coef_logloss, coef_logloss)
            if(self.__tem_validacao):
                coef_logloss_val = self.__calcula_coef_logloss(self.__yval, self.__probs_val, self.__logloss_baseline_val)
                self.__curva_coef_logloss_val = np.append(self.__curva_coef_logloss_val, coef_logloss_val)
                #Vê se encontramos um novo melhor modelo e guarda ele
                if(coef_logloss_val > self.__coef_logloss_val_melhor):
                    self.__guarda_melhor_modelo(coef_logloss_val)
                if(verbose):
                    print(str(coef_logloss) + " / " + str(coef_logloss_val) + " (" + str(self.__num_termos_melhor) + ")")
            else:
                if(verbose):
                    print(str(coef_logloss) + " (" + str(self.__num_termos) + ")")

        if(coef_logloss_val <= 0):
            self.__curva_num_termos = self.__curva_num_termos[:-1]
            self.__curva_coef_logloss = self.__curva_coef_logloss[:-1]
            if(self.__tem_validacao):
                self.__curva_coef_logloss_val = self.__curva_coef_logloss_val[:-1]
            if(self.__fit_total):
                self.__curva_coef_logloss_tot = self.__curva_coef_logloss_tot[:-1]

        self.__calcula_importancias()
        
    def predict_proba(self, X):
        X = self.__checa_X_predict(X)
        X = X[:, self.__flag_vars_validas]
        X = (X - self.__means)/self.__stds
        X = np.nan_to_num(X, posinf = 0, neginf = 0) #Adaptação para ignorar os nulos
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        if(self.__tem_validacao):
            y_pred = 1/(1 + np.exp(-np.dot(X[:, self.__vars_selec_melhor], self.__thetas_melhor)))
        else:
            y_pred = 1/(1 + np.exp(-np.dot(X[:, self.__vars_selec], self.__thetas)))
        y_prob = np.dstack((1 - y_pred,y_pred))[0]
        return y_prob
    
    def predict(self, X):
        y_prob = self.predict_proba(X)
        y_pred = (y_prob[:, 1] >= self.__media_y).astype(int)
        return y_pred
    
    def grafico_vies_variancia(self, pos_ini = None, pos_fim = None, figsize = [8, 6]):        
        #Prepara os valores e vetores de plot
        if(pos_ini == None):
            pos_ini = 0
        if(pos_fim == None):
            pos_fim = self.__curva_num_termos.size
        curva_num_termos = self.__curva_num_termos[pos_ini:pos_fim]
        curva_coef_logloss = self.__curva_coef_logloss[pos_ini:pos_fim]
        if(self.__tem_validacao):
            curva_coef_logloss_val = self.__curva_coef_logloss_val[pos_ini:pos_fim]
            coef_logloss_val_melhor = self.__coef_logloss_val_melhor
            coef_logloss_min = min(min(curva_coef_logloss[np.isfinite(curva_coef_logloss)]), min(curva_coef_logloss_val[np.isfinite(curva_coef_logloss_val)]))
        #Plota as curvas e o ponto de parada do treinamento pela validação
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.plot(curva_num_termos, curva_coef_logloss, color = paleta_cores[0], label = 'Treino')
            if(self.__tem_validacao):
                axs.plot(curva_num_termos, curva_coef_logloss_val, color = paleta_cores[1], label = 'Validação')
                axs.vlines(self.__num_termos_melhor, coef_logloss_min, coef_logloss_val_melhor, color = 'k', 
                           linestyle = '--', label = 'Ponto de Parada')
            axs.set_xlabel('Número de Termos')
            axs.set_ylabel('Coeficiente LogLoss')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def __calcula_importancias(self):
        if(self.__tem_validacao):
            vars_selec = self.__vars_selec_melhor[1:] - 1
            coef_abs = np.abs(self.__thetas_melhor[1:])
        else:
            vars_selec = self.__vars_selec[1:] - 1
            coef_abs = np.abs(self.__thetas[1:])
        self.feature_names_ = self.__nome_vars[vars_selec]
        self.feature_importances_ = coef_abs/np.sum(coef_abs)
        
    def grafico_importancias(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.__curva_num_termos.size
        vars_nomes = self.feature_names_
        vars_imp = self.feature_importances_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()
            
 ########################
 
 ########################

#Faz as contas com matriz numpy e retorna o calcula do termo desejado quando é solicitado
#Ou seja, não ocupa memória mas custa processamento
class TaylorLaurentExpansion:

    def __init__(self, laurent = False, ordem = 2, apenas_interacoes = False, num_features = None):
        self.__laurent = laurent
        self.__apenas_interacoes = apenas_interacoes
        self.__num_features = num_features
        if(self.__apenas_interacoes):
            self.__ordem = min(ordem, self.__num_features)
        else:
            self.__ordem = ordem
        
        def cria_tupla(tupla):
            unique, count = np.unique(tupla, return_counts = True)
            return tuple((unique[i], count[i]) for i in range(0, unique.size))
        
        self.__lista_termos = []
        features = np.arange(0, self.__num_features)
        for i in range(1, self.__ordem + 1):
            if(self.__apenas_interacoes):
                comb = list(combinations(features, r = i)) #Não precisa de potencias das features
            else:
                comb = list(combinations_with_replacement(features, r = i))
            comb = [cria_tupla(v) for v in comb]
            self.__lista_termos.extend(comb)
        
        if(self.__laurent):
            if(self.__apenas_interacoes):
                def expande_laurent(tupla):
                    sinais = [[]]
                    tam = len(tupla)
                    for i in range(0, tam):
                        sinais_temp = sinais.copy()
                        for var in sinais_temp:
                            v_new1 = var.copy()
                            v_new2 = var.copy()
                            v_new1.append(1)
                            v_new2.append(-1)
                            sinais.pop(0)
                            sinais.append(v_new1)
                            sinais.append(v_new2)
                    #Remove os sinais que darão inversos multiplicativos
                    sinais_filtrados = []
                    while(len(sinais) > 1):
                        sinal = sinais[0]
                        sinais_filtrados.append(sinal)
                        sinais.remove(sinal)
                        try:
                            sinais.remove(list(-1*np.array(sinal)))
                        except:
                            pass
                    if(len(sinais) == 1):
                        sinais_filtrados.append(sinais[0])
                    sinais = sinais_filtrados
                    return [tuple((tupla[j][0], s[j]*tupla[j][1]) for j in range(0, tam)) for s in sinais]
            else:
                def expande_laurent(tupla):
                    sinais = [[]]
                    tam = len(tupla)
                    for i in range(0, tam):
                        sinais_temp = sinais.copy()
                        for var in sinais_temp:
                            v_new1 = var.copy()
                            v_new2 = var.copy()
                            v_new1.append(1)
                            v_new2.append(-1)
                            sinais.pop(0)
                            sinais.append(v_new1)
                            sinais.append(v_new2)
                    return [tuple((tupla[j][0], s[j]*tupla[j][1]) for j in range(0, tam)) for s in sinais]
            lista_aux = []
            for tupla in self.__lista_termos:
                lista_aux.extend(expande_laurent(tupla))
            self.__lista_termos = lista_aux
        
        self.__num_termos = len(self.__lista_termos)
    
    def numero_termos_expansao(self):
        return self.__num_termos
        
    def lista_termos(self):
        return self.__lista_termos
                
    def calcula_termo(self, X, pos_termo):
        termo = self.__lista_termos[pos_termo]
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            for i in range(0, len(termo)):
                if(i == 0):
                    valores = np.power(X[:, termo[0][0]], termo[0][1])
                else:
                    valores = valores * np.power(X[:, termo[i][0]], termo[i][0])
            return valores

##############################

##############################
            
class SeriesBoostRegressor:
    
    def __init__(self, max_termos = None, passo = 1, laurent = False, ordem = 1, apenas_interacoes = False, fit_total = False):
        self.__max_termos = max_termos
        self.__passo = passo
        
        self.__laurent = laurent
        self.__ordem = ordem
        self.__apenas_interacoes = apenas_interacoes
        self.__series = None
        
        self.__fit_total = fit_total
        
        self.__nome_vars = None
        self.__X = None
        self.__y = None
        self.__num_linhas = None
        self.__num_cols = None
        
        self.__Xval = None
        self.__yval = None
        self.__tem_validacao = None
        self.__num_linhas_val = None
        self.__num_cols_val = None
    
    def __checa_X_e_y_treino(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.__X = X.values
            self.__nome_vars = X.columns.values
        else:
            try:
                if len(X.shape) == 2:
                    self.__X = X
                else:
                    print("Valores de entrada de treino não adequados")
                    return
            except:
                print("Valores de entrada de treino não adequados")
                return
        self.__num_linhas = self.__X.shape[0]
        self.__num_cols = self.__X.shape[1]
        
        if (isinstance(y, pd.DataFrame) and len(y.columns) == 1) or (isinstance(y, pd.Series)):
            self.__y = y.values
        else:
            try:
                if len(y.shape) == 1:
                    self.__y = y
                else:
                    print("Valores de alvo de treino não adequados")
                    return
            except:
                print("Valores de alvo de treino não adequados")
                return
        if(self.__y.size != self.__num_linhas):
            print("Quantidade de exemplos não coindicem em X e y")
            return
            
    def __checa_conjunto_validacao(self, conj_val):
        #Verifica se foi passado um conjunto de validação
        if conj_val != None:
            if isinstance(conj_val, tuple):
                if(len(conj_val) == 2):
                    if isinstance(conj_val[0], pd.DataFrame):
                        self.__Xval = conj_val[0].values
                    else:
                        try:
                            if len(conj_val[0].shape) == 2:
                                self.__Xval = conj_val[0]
                            else:
                                print("Valores de entrada de validação não adequados")
                                return
                        except:
                            print("Valores de entrada não adequados")
                            return
                    self.__num_linhas_val = self.__Xval.shape[0]
                    self.__num_cols_val = self.__Xval.shape[1]
                        
                    if (isinstance(conj_val[1], pd.DataFrame) and len(conj_val[1].columns) == 1) or (isinstance(conj_val[1], pd.Series)):
                        self.__yval = conj_val[1].values
                    else:
                        try:
                            if len(conj_val[1].shape) == 1:
                                self.__yval = conj_val[1]
                            else:
                                print("Valores de alvo de validação não adequados")
                                return
                        except:
                            print("Valores de alvo de validação não adequados")
                            return
                        
                else:
                    print("Valores de validação não adequados")
                    return
            else:
                print("Valores de validação não adequados")
                return
            self.__tem_validacao = True
            if(self.__yval.size != self.__num_linhas_val):
                print("Quantidade de exemplos não coindicem em Xval e yval")
                return
        else:
            self.__tem_validacao = False

    def __checa_X_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            try:
                if len(X.shape) != 2:
                    print("Valores de entrada de predição não adequados")
                    return
            except:
                print("Valores de entrada de predição não adequados")
                return
        return X 
      
    def __guarda_melhor_modelo(self, r2_val):
        #Guarda o modelo atual nas variáveis de melhor modelo
        self.__vars_selec_melhor = self.__vars_selec
        self.__num_termos_melhor = self.__num_termos
        self.__thetas_melhor = self.__thetas
        self.__r2_val_melhor = r2_val
        if(self.__fit_total):   
            self.__thetas_melhor_tot = self.__thetas_tot        
        
    def __calcula_r2(self, diff, mse_baseline):
        return 1 - np.mean(np.power(diff, 2))/mse_baseline
    
    def __derivadas_parciais_custo(self, diff, X, vars):
        valores = np.array([])
        for i in range(0, vars.size):
            valores = np.append(valores, np.mean(diff*np.nan_to_num((self.__series.calcula_termo(X, vars[i]) - self.__means[vars[i]])/self.__stds[vars[i]], posinf = 0, neginf = 0)))
        return valores
    
    def __produto_matricial_normalizado(self, X, vars, thetas):
        valores = np.repeat(thetas[0], X.shape[0])
        for i in range(0, vars.size):
            valores = valores + np.nan_to_num((self.__series.calcula_termo(X, vars[i]) - self.__means[vars[i]])/self.__stds[vars[i]], posinf = 0, neginf = 0)*thetas[i+1]
        return valores
        
    def __derivada_parcial_normalizada(self, vetor, X, vars):
        valores = np.array([np.sum(vetor)])
        for i in range(0, vars.size):
            valores = np.append(valores, np.sum(vetor*np.nan_to_num((self.__series.calcula_termo(X, vars[i]) - self.__means[vars[i]])/self.__stds[vars[i]], posinf = 0, neginf = 0)))
        return valores
    
    def fit(self, X, y, conj_val = None, verbose = True):
        self.__checa_X_e_y_treino(X, y)
        if(self.__nome_vars is None):
            self.__nome_vars = np.array(['x' + str(i) for i in range(0, self.__num_cols)])
        self.__checa_conjunto_validacao(conj_val)
        
        #Filtro inicial de features sem variância
        self.__means = np.nan_to_num(np.nanmean(self.__X, axis = 0), posinf = 0, neginf = 0)
        self.__stds = np.sqrt(np.nan_to_num(np.nanmean(np.power(self.__X - self.__means, 2), axis = 0), posinf = 0, neginf = 0))
        self.__flag_vars_validas = self.__stds > 0
        self.__num_cols = np.sum(self.__flag_vars_validas)
        self.__nome_vars = self.__nome_vars[self.__flag_vars_validas]
        self.__X = self.__X[:, self.__flag_vars_validas]
        self.__means = self.__means[self.__flag_vars_validas]
        self.__stds = self.__stds[self.__flag_vars_validas]
        if(self.__tem_validacao):
            self.__Xval = self.__Xval[:, self.__flag_vars_validas]
        
        #Expansão em série
        self.__series = TaylorLaurentExpansion(self.__laurent, self.__ordem , self.__apenas_interacoes, self.__num_cols)
        self.__num_cols = self.__series.numero_termos_expansao()
        
        #Faz a normalização (e filtra colunas com desvio padrão zero)
        #Facilita calculo de Inv. de Matriz e também facilita a Descida do Gradiente
        #(adaptado para lidar com NA)
        self.__means = np.nan_to_num(np.array([np.nanmean(self.__series.calcula_termo(self.__X, i)) for i in range(0, self.__num_cols)]), posinf = 0, neginf = 0)
        self.__stds = np.nan_to_num(np.array([np.sqrt(np.nanmean(np.power(self.__series.calcula_termo(self.__X, i) - self.__means[i], 2))) for i in range(0, self.__num_cols)]), posinf = 0, neginf = 0)
        
        #Colunas efetivas (só as que tem desvio padrão não nulo)
        self.__cols_ef = np.arange(0, self.__num_cols)[self.__stds > 0]
        
        #Normaliza o y também: ajuda na Descida do Gradiente (Aqui não pode ter nulos mesmo!!)
        self.__media_y = np.mean(self.__y)
        self.__desvio_y = np.sqrt(np.mean(np.power(self.__y - self.__media_y, 2)))
        self.__y = (self.__y - self.__media_y)/self.__desvio_y
        if(self.__tem_validacao):
            self.__yval = (self.__yval - self.__media_y)/self.__desvio_y
        
        #Faz o modelo baseline: só o termo theta_0 -> cte!!! (prob média)
        self.__vars_selec = np.array([]).astype(int) #Lista das variaveis selecionadas no regressor
        self.__num_termos = 1 #Número de termos do regressor
        self.__thetas = np.array([0]) #Valor dos thetas do regressor
        self.__diff = -self.__y # Diferença entre o valor predito e o valor do alvo
        if(self.__tem_validacao):
            self.__diff_val = -self.__yval # Diferença entre o valor predito e o valor do alvo
        
        if(self.__fit_total):
            self.__thetas_tot = np.array([0]) #Valor dos thetas do regressor
        
        #Salva o mse do modelo baseline (só constante = 0)
        self.__mse_baseline = 1 #É 1 pq é o desvio padrão (que é 1 pq y tá normalizado)
        if(self.__tem_validacao):
            self.__mse_baseline_val = np.mean(np.power(self.__yval, 2))
            
        if(self.__fit_total):
            self.__mse_baseline_tot = (self.__mse_baseline*self.__num_linhas + self.__mse_baseline_val*self.__num_linhas_val)/(self.__num_linhas + self.__num_linhas_val)
            
        #####
        ##### Algoritmo em loop incremental estilo Boosting para aumentar a complexidade #####
        #####
        
        #Funções para otimização da função de custo (MSE)
        def mse(thetas):
            pred = self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)
            return np.mean(np.power(pred - self.__y, 2))/2
        def mse_der(thetas):
            pred = self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)
            return self.__derivada_parcial_normalizada(pred, self.__X, self.__vars_selec)/self.__num_linhas - diff_cte
        
        if(self.__fit_total):
            def mse_tot(thetas):
                pred = self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)
                mse = np.mean(np.power(pred - self.__y, 2))/2
                pred_val = self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, thetas)
                mse_val = np.mean(np.power(pred_val - self.__yval, 2))/2
                return (mse*self.__num_linhas + mse_val*self.__num_linhas_val)/(self.__num_linhas + self.__num_linhas_val)
            def mse_der_tot(thetas):
                pred = self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)
                der = self.__derivada_parcial_normalizada(pred, self.__X, self.__vars_selec)/self.__num_linhas - diff_cte
                pred_val = self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, thetas)
                der_val = self.__derivada_parcial_normalizada(pred_val, self.__Xval, self.__vars_selec)/self.__num_linhas_val - diff_cte_val
                return (der*self.__num_linhas + der_val*self.__num_linhas_val)/(self.__num_linhas + self.__num_linhas_val)
        
        #Inicializamos as variaveis para criar a curva viés-variância
        self.__curva_num_termos = np.array([self.__num_termos])
        self.__curva_r2 = np.array([0])
        if(self.__tem_validacao):
            r2_val = 0
            self.__curva_r2_val = np.array([r2_val])
            self.__guarda_melhor_modelo(r2_val)
            
        if(self.__fit_total):
            self.__curva_r2_tot = np.array([0])
            
        vars_disp = self.__cols_ef.copy()
        if(self.__max_termos == None):
            self.__max_termos = vars_disp.size + 1 #Considera todos os termos se não tiver dado limite
        else:
            self.__max_termos = min(self.__max_termos, vars_disp.size + 1)
            
        #Loop do processo incremental de adição de termos na regressão
        while(self.__num_termos < self.__max_termos and vars_disp.size > 0 and (self.__tem_validacao == False or r2_val >= 0)):
            
            #Calcula todos valores das derividas parciais dos termos que estão faltando na regressão
            derivadas_parciais = self.__derivadas_parciais_custo(self.__diff, self.__X, vars_disp)/self.__num_linhas
            derivadas_parciais_abs = np.abs(derivadas_parciais) #Pega os valores da derivada em módulo
            
            #Pega os indices para a ordenação decrescente
            inds_ordena_derivadas = np.argsort(derivadas_parciais_abs)[::-1]
            
            #Ordena também primeiro onde os sinais das derivadas são consistentes no conjunto de validação
            if(self.__tem_validacao):
                derivadas_parciais_val = self.__derivadas_parciais_custo(self.__diff_val, self.__Xval, vars_disp)/self.__num_linhas_val
                flag_mesmo_sinal = np.sign(derivadas_parciais) == np.sign(derivadas_parciais_val)
                flag_inds = flag_mesmo_sinal[inds_ordena_derivadas]
                inds_ordena_derivadas = np.append(inds_ordena_derivadas[flag_inds], inds_ordena_derivadas[~flag_inds])
            
            #Pega os indices dos termos que vamos adicionar (de acordo com o passo escolhido)
            passo = min(self.__passo, vars_disp.size)
            pos_der_add = inds_ordena_derivadas[:passo] 
            vars_der_add = vars_disp[pos_der_add] #Lista das respectivas vars para adicionar
            
            #Adiciona as variáveis novas
            self.__vars_selec = np.append(self.__vars_selec, vars_der_add)
            self.__num_termos = self.__num_termos + passo
            self.__thetas = np.append(self.__thetas, np.zeros(passo)) 
            vars_disp = np.delete(vars_disp, pos_der_add) #Remove essas variaveis das disponíveis
            
            if(self.__fit_total):
                self.__thetas_tot = np.append(self.__thetas_tot, np.zeros(passo))
            
            #Note que: a ideia aqui foi incrementar o regressor com a variável que mais vai "bagunçar"
            # o mínimo que já haviamos encontrado para a Função de Custo: ou seja, deriv. parcial max.
            
            #Agora vamos encontrar os novos thetas que minimizam a função de custo
            #partindo dos thetas antigos acrescidos de valores zero para as novas features
            diff_cte = self.__derivada_parcial_normalizada(self.__y, self.__X, self.__vars_selec)/self.__num_linhas
            self.__thetas = minimize(mse, self.__thetas, method = 'BFGS', jac = mse_der, options = {'disp': False}).x
            self.__diff = self.__produto_matricial_normalizado(self.__X, self.__vars_selec, self.__thetas) - self.__y
            if(self.__tem_validacao):
                self.__diff_val = self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, self.__thetas) - self.__yval
            
            if(self.__fit_total):
                diff_cte_val = self.__derivada_parcial_normalizada(self.__yval, self.__Xval, self.__vars_selec)/self.__num_linhas_val
                self.__thetas_tot = minimize(mse_tot, self.__thetas_tot, method = 'BFGS', jac = mse_der_tot, options = {'disp': False}).x
                preds1 = self.__produto_matricial_normalizado(self.__X, self.__vars_selec, self.__thetas_tot)
                preds2 = self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, self.__thetas_tot)
                coef_r2_tot = self.__calcula_r2(np.append(preds1, preds2) - np.append(self.__y, self.__yval), self.__mse_baseline_tot)
                self.__curva_r2_tot = np.append(self.__curva_r2_tot, coef_r2_tot)
            
            #Calcula os MSE e adiciona na curva de Viés-Variância
            self.__curva_num_termos = np.append(self.__curva_num_termos, self.__num_termos)
            r2 = self.__calcula_r2(self.__diff, self.__mse_baseline)
            self.__curva_r2 = np.append(self.__curva_r2, r2)
            if(self.__tem_validacao):
                r2_val = self.__calcula_r2(self.__diff_val, self.__mse_baseline_val)
                self.__curva_r2_val = np.append(self.__curva_r2_val, r2_val)
                #Vê se encontramos um novo melhor modelo e guarda ele
                if(r2_val > self.__r2_val_melhor):
                    self.__guarda_melhor_modelo(r2_val)
                if(verbose):
                    print(str(r2) + " / " + str(r2_val) + " (" + str(self.__num_termos_melhor) + ")")
            else:
                if(verbose):
                    print(str(r2) + " (" + str(self.__num_termos) + ")")

        if(r2_val <= 0):
            self.__curva_num_termos = self.__curva_num_termos[:-1]
            self.__curva_r2 = self.__curva_r2[:-1]
            if(self.__tem_validacao):
                self.__curva_r2_val = self.__curva_r2_val[:-1]
            if(self.__fit_total):
                self.__curva_r2_tot = self.__curva_r2_tot[:-1]

        self.__calcula_importancias()
        
    def predict(self, X):
        X = self.__checa_X_predict(X)
        X = X[:, self.__flag_vars_validas]
        if(self.__fit_total):
            y_pred = self.__produto_matricial_normalizado(X, self.__vars_selec_melhor, self.__thetas_melhor_tot)
        elif(self.__tem_validacao):
            y_pred = self.__produto_matricial_normalizado(X, self.__vars_selec_melhor, self.__thetas_melhor)
        else:
            y_pred = self.__produto_matricial_normalizado(X, self.__vars_selec, self.__thetas)
        y_pred = self.__media_y + self.__desvio_y*y_pred
        return y_pred
    
    def grafico_vies_variancia(self, pos_ini = None, pos_fim = None, figsize = [8, 6]):        
        #Prepara os valores e vetores de plot
        if(pos_ini == None):
            pos_ini = 0
        if(pos_fim == None):
            pos_fim = self.__curva_num_termos.size
        curva_num_termos = self.__curva_num_termos[pos_ini:pos_fim]
        curva_r2 = self.__curva_r2[pos_ini:pos_fim]
        if(self.__tem_validacao):
            curva_r2_val = self.__curva_r2_val[pos_ini:pos_fim]
            r2_val_melhor = self.__r2_val_melhor
            r2_min = min(min(curva_r2[np.isfinite(curva_r2)]), min(curva_r2_val[np.isfinite(curva_r2_val)]))
        if(self.__fit_total):
            curva_r2_tot = self.__curva_r2_tot[pos_ini:pos_fim]
        #Plota as curvas e o ponto de parada do treinamento pela validação
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.plot(curva_num_termos, curva_r2, color = paleta_cores[0], label = 'Treino')
            if(self.__tem_validacao):
                axs.plot(curva_num_termos, curva_r2_val, color = paleta_cores[1], label = 'Validação')
                axs.vlines(self.__num_termos_melhor, r2_min, r2_val_melhor, color = 'k', 
                           linestyle = '--', label = 'Ponto de Parada')
            if(self.__fit_total):
                axs.plot(curva_num_termos, curva_r2_tot, color = paleta_cores[2], label = 'Total')
            axs.set_xlabel('Número de Termos')
            axs.set_ylabel('R2 Ajustado')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def __calcula_importancias(self):
        if(self.__tem_validacao):
            vars_selec = self.__vars_selec_melhor
            coef_abs = np.abs(self.__thetas_melhor[1:])
        else:
            vars_selec = self.__vars_selec
            coef_abs = np.abs(self.__thetas[1:])
        termos = self.__series.lista_termos()
        termos = [termos[i] for i in vars_selec]
        def traduz_termo(termo, nome_vars):
            return tuple((nome_vars[v[0]], v[1]) for v in termo)
        self.feature_names_terms_ = np.array([str(traduz_termo(v, self.__nome_vars)) for v in termos])
        self.feature_importances_terms_ = coef_abs/np.sum(coef_abs)
        def pares_feature_coef_abs(termo, coef):
            return [(v[0], coef) for v in termo]
        lista_pesos = []
        for i in range(0, coef_abs.size):
            lista_pesos.extend(pares_feature_coef_abs(termos[i], coef_abs[i]))
        lista_agrupada = [(key, sum(map(itemgetter(1), ele))) for key, ele in groupby(sorted(lista_pesos, key = itemgetter(0)), key = itemgetter(0))]
        self.feature_names_ = np.array([self.__nome_vars[x[0]] for x in lista_agrupada])
        self.feature_importances_ = np.array([x[1] for x in lista_agrupada])
        self.feature_importances_ = self.feature_importances_/np.sum(self.feature_importances_)
        
    def grafico_importancias(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.__curva_num_termos.size
        vars_nomes = self.feature_names_
        vars_imp = self.feature_importances_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()
    
##############################

##############################
            
class SeriesBoostClassifier:
    
    def __init__(self, max_termos = None, passo = 1, laurent = False, ordem = 1, apenas_interacoes = False, fit_total = False):
        self.__max_termos = max_termos
        self.__passo = passo
        
        self.__laurent = laurent
        self.__ordem = ordem
        self.__apenas_interacoes = apenas_interacoes
        self.__series = None
        
        self.__fit_total = fit_total
        
        self.__nome_vars = None
        self.__X = None
        self.__y = None
        self.__num_linhas = None
        self.__num_cols = None
        
        self.__Xval = None
        self.__yval = None
        self.__tem_validacao = None
        self.__num_linhas_val = None
        self.__num_cols_val = None
    
    def __checa_X_e_y_treino(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.__X = X.values
            self.__nome_vars = X.columns.values
        else:
            try:
                if len(X.shape) == 2:
                    self.__X = X
                else:
                    print("Valores de entrada de treino não adequados")
                    return
            except:
                print("Valores de entrada de treino não adequados")
                return
        self.__num_linhas = self.__X.shape[0]
        self.__num_cols = self.__X.shape[1]
        
        if (isinstance(y, pd.DataFrame) and len(y.columns) == 1) or (isinstance(y, pd.Series)):
            self.__y = y.values
        else:
            try:
                if len(y.shape) == 1:
                    self.__y = y
                else:
                    print("Valores de alvo de treino não adequados")
                    return
            except:
                print("Valores de alvo de treino não adequados")
                return
        if(self.__y.size != self.__num_linhas):
            print("Quantidade de exemplos não coindicem em X e y")
            return
            
    def __checa_conjunto_validacao(self, conj_val):
        #Verifica se foi passado um conjunto de validação
        if conj_val != None:
            if isinstance(conj_val, tuple):
                if(len(conj_val) == 2):
                    if isinstance(conj_val[0], pd.DataFrame):
                        self.__Xval = conj_val[0].values
                    else:
                        try:
                            if len(conj_val[0].shape) == 2:
                                self.__Xval = conj_val[0]
                            else:
                                print("Valores de entrada de validação não adequados")
                                return
                        except:
                            print("Valores de entrada não adequados")
                            return
                    self.__num_linhas_val = self.__Xval.shape[0]
                    self.__num_cols_val = self.__Xval.shape[1]
                        
                    if (isinstance(conj_val[1], pd.DataFrame) and len(conj_val[1].columns) == 1) or (isinstance(conj_val[1], pd.Series)):
                        self.__yval = conj_val[1].values
                    else:
                        try:
                            if len(conj_val[1].shape) == 1:
                                self.__yval = conj_val[1]
                            else:
                                print("Valores de alvo de validação não adequados")
                                return
                        except:
                            print("Valores de alvo de validação não adequados")
                            return
                        
                else:
                    print("Valores de validação não adequados")
                    return
            else:
                print("Valores de validação não adequados")
                return
            self.__tem_validacao = True
            if(self.__yval.size != self.__num_linhas_val):
                print("Quantidade de exemplos não coindicem em Xval e yval")
                return
        else:
            self.__tem_validacao = False

    def __checa_X_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            try:
                if len(X.shape) != 2:
                    print("Valores de entrada de predição não adequados")
                    return
            except:
                print("Valores de entrada de predição não adequados")
                return
        return X 
      
    def __guarda_melhor_modelo(self, coef_logloss_val):
        #Guarda o modelo atual nas variáveis de melhor modelo
        self.__vars_selec_melhor = self.__vars_selec
        self.__num_termos_melhor = self.__num_termos
        self.__thetas_melhor = self.__thetas
        self.__coef_logloss_val_melhor = coef_logloss_val
        if(self.__fit_total):   
            self.__thetas_melhor_tot = self.__thetas_tot
      
    def __calcula_coef_logloss(self, y, probs, logloss_baseline):
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            logloss = -1*np.mean(np.where(y == 1, np.log(probs), np.log(1 - probs)))
            return 1 - logloss/logloss_baseline
    
    def __derivadas_parciais_custo(self, diff, X, vars):
        valores = np.array([])
        for i in range(0, vars.size):
            valores = np.append(valores, np.mean(diff*np.nan_to_num((self.__series.calcula_termo(X, vars[i]) - self.__means[vars[i]])/self.__stds[vars[i]], posinf = 0, neginf = 0)))
        return valores
    
    def __produto_matricial_normalizado(self, X, vars, thetas):
        valores = np.repeat(thetas[0], X.shape[0])
        for i in range(0, vars.size):
            valores = valores + np.nan_to_num((self.__series.calcula_termo(X, vars[i]) - self.__means[vars[i]])/self.__stds[vars[i]], posinf = 0, neginf = 0)*thetas[i+1]
        return valores
        
    def __derivada_parcial_normalizada(self, vetor, X, vars):
        valores = np.array([np.sum(vetor)])
        for i in range(0, vars.size):
            valores = np.append(valores, np.sum(vetor*np.nan_to_num((self.__series.calcula_termo(X, vars[i]) - self.__means[vars[i]])/self.__stds[vars[i]], posinf = 0, neginf = 0)))
        return valores
    
    def fit(self, X, y, conj_val = None, verbose = True):
        self.__checa_X_e_y_treino(X, y)
        if(self.__nome_vars is None):
            self.__nome_vars = np.array(['x' + str(i) for i in range(0, self.__num_cols)])
        self.__checa_conjunto_validacao(conj_val)
        
        #Filtro inicial de features sem variância
        self.__means = np.nan_to_num(np.nanmean(self.__X, axis = 0), posinf = 0, neginf = 0)
        self.__stds = np.sqrt(np.nan_to_num(np.nanmean(np.power(self.__X - self.__means, 2), axis = 0), posinf = 0, neginf = 0))
        self.__flag_vars_validas = self.__stds > 0
        self.__num_cols = np.sum(self.__flag_vars_validas)
        self.__nome_vars = self.__nome_vars[self.__flag_vars_validas]
        self.__X = self.__X[:, self.__flag_vars_validas]
        self.__means = self.__means[self.__flag_vars_validas]
        self.__stds = self.__stds[self.__flag_vars_validas]
        if(self.__tem_validacao):
            self.__Xval = self.__Xval[:, self.__flag_vars_validas]
        
        #Expansão em série
        self.__series = TaylorLaurentExpansion(self.__laurent, self.__ordem , self.__apenas_interacoes, self.__num_cols)
        self.__num_cols = self.__series.numero_termos_expansao()
        
        #Faz a normalização (e filtra colunas com desvio padrão zero)
        #Facilita calculo de Inv. de Matriz e também facilita a Descida do Gradiente
        #(adaptado para lidar com NA)
        self.__means = np.nan_to_num(np.array([np.nanmean(self.__series.calcula_termo(self.__X, i)) for i in range(0, self.__num_cols)]), posinf = 0, neginf = 0)
        self.__stds = np.nan_to_num(np.array([np.sqrt(np.nanmean(np.power(self.__series.calcula_termo(self.__X, i) - self.__means[i], 2))) for i in range(0, self.__num_cols)]), posinf = 0, neginf = 0)
        
        #Colunas efetivas (só as que tem desvio padrão não nulo)
        self.__cols_ef = np.arange(0, self.__num_cols)[self.__stds > 0]
        
        self.__media_y = np.mean(self.__y)  #(Aqui não pode ter nulos mesmo!!)
        
        #Faz o modelo baseline: só o termo theta_0 -> cte!!! (prob média)
        self.__vars_selec = np.array([]).astype(int) #Lista das variaveis selecionadas no regressor
        self.__num_termos = 1 #Número de termos do regressor
        theta_0 = np.log(self.__media_y/(1 - self.__media_y)) #Dedução da função logística
        self.__thetas = np.array([theta_0]) #Valor dos thetas do regressor
        self.__probs = np.repeat(self.__media_y, self.__num_linhas) #Valor predito de prob
        self.__diff = self.__media_y - self.__y # Diferença entre o valor predito e o valor do alvo
        if(self.__tem_validacao):
            self.__probs_val = np.repeat(self.__media_y, self.__num_linhas_val) #Valor predito de prob
            self.__diff_val = self.__media_y - self.__yval # Diferença entre o valor predito e o valor do alvo
        
        if(self.__fit_total):
            self.__media_y_tot = (self.__media_y*self.__num_linhas + np.sum(self.__yval))/(self.__num_linhas + self.__num_linhas_val)
            theta_0 = np.log(self.__media_y_tot/(1 - self.__media_y_tot)) #Dedução da função logística
            self.__thetas_tot = np.array([theta_0]) #Valor dos thetas do regressor
        
        #Salva o loss do modelo baseline (só constante)
        def calcula_logloss(y, probs):
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                logloss = -1*np.mean(np.where(y == 1, np.log(probs), np.log(1 - probs)))
                return logloss
        self.__logloss_baseline = calcula_logloss(self.__y, self.__probs)
        if(self.__tem_validacao):
            probs_val = np.repeat(self.__media_y, self.__num_linhas_val)
            self.__logloss_baseline_val = calcula_logloss(self.__yval, probs_val)
        
        if(self.__fit_total):
            self.__logloss_baseline_tot = (self.__logloss_baseline*self.__num_linhas + self.__logloss_baseline_val*self.__num_linhas_val)/(self.__num_linhas + self.__num_linhas_val)
            
        #####
        ##### Algoritmo em loop incremental estilo Boosting para aumentar a complexidade #####
        #####
        
        #Funções para otimização da função de custo (MSE)
        def logloss(thetas):
            probs = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)))
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                logloss = -1*np.mean(np.where(self.__y == 1, np.log(probs), np.log(1 - probs)))
                return logloss
        def logloss_der(thetas):
            probs = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)))
            return self.__derivada_parcial_normalizada(probs, self.__X, self.__vars_selec)/self.__num_linhas - diff_cte
        
        if(self.__fit_total):
            def logloss_tot(thetas):
                probs = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)))
                probs_val = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, thetas)))
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    logloss = -1*np.mean(np.where(self.__y == 1, np.log(probs), np.log(1 - probs)))
                    logloss_val = -1*np.mean(np.where(self.__yval == 1, np.log(probs_val), np.log(1 - probs_val)))
                    return (logloss*self.__num_linhas + logloss_val*self.__num_linhas_val)/(self.__num_linhas + self.__num_linhas_val)
            def logloss_der_tot(thetas):
                probs = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__X, self.__vars_selec, thetas)))
                der = self.__derivada_parcial_normalizada(probs, self.__X, self.__vars_selec)/self.__num_linhas - diff_cte
                probs_val = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, thetas)))
                der_val = self.__derivada_parcial_normalizada(probs_val, self.__Xval, self.__vars_selec)/self.__num_linhas_val - diff_cte_val
                return (der*self.__num_linhas + der_val*self.__num_linhas_val)/(self.__num_linhas + self.__num_linhas_val)
        
        #Inicializamos as variaveis para criar a curva viés-variância
        self.__curva_num_termos = np.array([self.__num_termos])
        self.__curva_coef_logloss = np.array([0])
        if(self.__tem_validacao):
            coef_logloss_val = 0
            self.__curva_coef_logloss_val = np.array([coef_logloss_val])
            self.__guarda_melhor_modelo(coef_logloss_val)
        
        if(self.__fit_total):
            self.__curva_coef_logloss_tot = np.array([0])
        
        vars_disp = self.__cols_ef.copy()
        if(self.__max_termos == None):
            self.__max_termos = vars_disp.size + 1 #Considera todos os termos se não tiver dado limite
        else:
            self.__max_termos = min(self.__max_termos, vars_disp.size + 1)
            
        #Loop do processo incremental de adição de termos na regressão
        while(self.__num_termos < self.__max_termos and vars_disp.size > 0 and (self.__tem_validacao == False or coef_logloss_val >= 0)):
            
            #Calcula todos valores das derividas parciais dos termos que estão faltando na regressão
            derivadas_parciais = self.__derivadas_parciais_custo(self.__diff, self.__X, vars_disp)/self.__num_linhas
            derivadas_parciais_abs = np.abs(derivadas_parciais) #Pega os valores da derivada em módulo
            
            #Pega os indices para a ordenação decrescente
            inds_ordena_derivadas = np.argsort(derivadas_parciais_abs)[::-1]
            
            #Ordena também primeiro onde os sinais das derivadas são consistentes no conjunto de validação
            if(self.__tem_validacao):
                derivadas_parciais_val = self.__derivadas_parciais_custo(self.__diff_val, self.__Xval, vars_disp)/self.__num_linhas_val
                flag_mesmo_sinal = np.sign(derivadas_parciais) == np.sign(derivadas_parciais_val)
                flag_inds = flag_mesmo_sinal[inds_ordena_derivadas]
                inds_ordena_derivadas = np.append(inds_ordena_derivadas[flag_inds], inds_ordena_derivadas[~flag_inds])
            
            #Pega os indices dos termos que vamos adicionar (de acordo com o passo escolhido)
            passo = min(self.__passo, vars_disp.size)
            pos_der_add = inds_ordena_derivadas[:passo] 
            vars_der_add = vars_disp[pos_der_add] #Lista das respectivas vars para adicionar
            
            #Adiciona as variáveis novas
            self.__vars_selec = np.append(self.__vars_selec, vars_der_add)
            self.__num_termos = self.__num_termos + passo
            self.__thetas = np.append(self.__thetas, np.zeros(passo)) 
            vars_disp = np.delete(vars_disp, pos_der_add) #Remove essas variaveis das disponíveis
            
            if(self.__fit_total):
                self.__thetas_tot = np.append(self.__thetas_tot, np.zeros(passo)) 
            
            #Note que: a ideia aqui foi incrementar o regressor com a variável que mais vai "bagunçar"
            # o mínimo que já haviamos encontrado para a Função de Custo: ou seja, deriv. parcial max.
            
            #Agora vamos encontrar os novos thetas que minimizam a função de custo
            #partindo dos thetas antigos acrescidos de valores zero para as novas features
            diff_cte = self.__derivada_parcial_normalizada(self.__y, self.__X, self.__vars_selec)/self.__num_linhas
            self.__thetas = minimize(logloss, self.__thetas, method = 'BFGS', jac = logloss_der, options = {'disp': False}).x
            self.__probs = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__X, self.__vars_selec, self.__thetas)))
            self.__diff = self.__probs - self.__y
            if(self.__tem_validacao):
                self.__probs_val = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, self.__thetas)))
                self.__diff_val = self.__probs_val - self.__yval
            
            if(self.__fit_total):
                diff_cte_val = self.__derivada_parcial_normalizada(self.__yval, self.__Xval, self.__vars_selec)/self.__num_linhas_val
                self.__thetas_tot = minimize(logloss_tot, self.__thetas_tot, method = 'BFGS', jac = logloss_der_tot, options = {'disp': False}).x
                probs1 = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__X, self.__vars_selec, self.__thetas_tot)))
                probs2 = 1/(1 + np.exp(-self.__produto_matricial_normalizado(self.__Xval, self.__vars_selec, self.__thetas_tot)))
                coef_logloss_tot = self.__calcula_coef_logloss(np.append(self.__y, self.__yval), np.append(probs1, probs2), self.__logloss_baseline_tot)
                self.__curva_coef_logloss_tot = np.append(self.__curva_coef_logloss_tot, coef_logloss_tot)
            
            #Calcula os Coef Logloss e adiciona na curva de Viés-Variância
            self.__curva_num_termos = np.append(self.__curva_num_termos, self.__num_termos)
            coef_logloss = self.__calcula_coef_logloss(self.__y, self.__probs, self.__logloss_baseline)
            self.__curva_coef_logloss = np.append(self.__curva_coef_logloss, coef_logloss)
            if(self.__tem_validacao):
                coef_logloss_val = self.__calcula_coef_logloss(self.__yval, self.__probs_val, self.__logloss_baseline_val)
                self.__curva_coef_logloss_val = np.append(self.__curva_coef_logloss_val, coef_logloss_val)
                #Vê se encontramos um novo melhor modelo e guarda ele
                if(coef_logloss_val > self.__coef_logloss_val_melhor):
                    self.__guarda_melhor_modelo(coef_logloss_val)
                if(verbose):
                    print(str(coef_logloss) + " / " + str(coef_logloss_val) + " (" + str(self.__num_termos_melhor) + ")")
            else:
                if(verbose):
                    print(str(coef_logloss) + " (" + str(self.__num_termos) + ")")
        
        if(coef_logloss_val <= 0):
            self.__curva_num_termos = self.__curva_num_termos[:-1]
            self.__curva_coef_logloss = self.__curva_coef_logloss[:-1]
            if(self.__tem_validacao):
                self.__curva_coef_logloss_val = self.__curva_coef_logloss_val[:-1]
            if(self.__fit_total):
                self.__curva_coef_logloss_tot = self.__curva_coef_logloss_tot[:-1]
        
        self.__calcula_importancias()
        
    def predict_proba(self, X):
        X = self.__checa_X_predict(X)
        X = X[:, self.__flag_vars_validas]
        if(self.__fit_total):
            y_pred = 1/(1 + np.exp(-self.__produto_matricial_normalizado(X, self.__vars_selec_melhor, self.__thetas_melhor_tot)))
        elif(self.__tem_validacao):
            y_pred = 1/(1 + np.exp(-self.__produto_matricial_normalizado(X, self.__vars_selec_melhor, self.__thetas_melhor)))
        else:
            y_pred = 1/(1 + np.exp(-self.__produto_matricial_normalizado(X, self.__vars_selec, self.__thetas)))
        y_prob = np.dstack((1 - y_pred,y_pred))[0]
        return y_prob
        
    def predict(self, X):
        y_prob = self.predict_proba(X)
        y_pred = (y_prob[:, 1] >= self.__media_y).astype(int)
        return y_pred
    
    def grafico_vies_variancia(self, pos_ini = None, pos_fim = None, figsize = [8, 6]):        
        #Prepara os valores e vetores de plot
        if(pos_ini == None):
            pos_ini = 0
        if(pos_fim == None):
            pos_fim = self.__curva_num_termos.size
        curva_num_termos = self.__curva_num_termos[pos_ini:pos_fim]
        curva_coef_logloss = self.__curva_coef_logloss[pos_ini:pos_fim]
        if(self.__tem_validacao):
            curva_coef_logloss_val = self.__curva_coef_logloss_val[pos_ini:pos_fim]
            coef_logloss_val_melhor = self.__coef_logloss_val_melhor
            coef_logloss_min = min(min(curva_coef_logloss[np.isfinite(curva_coef_logloss)]), min(curva_coef_logloss_val[np.isfinite(curva_coef_logloss_val)]))
        if(self.__fit_total):
            curva_coef_logloss_tot = self.__curva_coef_logloss_tot[pos_ini:pos_fim]
        #Plota as curvas e o ponto de parada do treinamento pela validação
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.plot(curva_num_termos, curva_coef_logloss, color = paleta_cores[0], label = 'Treino')
            if(self.__tem_validacao):
                axs.plot(curva_num_termos, curva_coef_logloss_val, color = paleta_cores[1], label = 'Validação')
                axs.vlines(self.__num_termos_melhor, coef_logloss_min, coef_logloss_val_melhor, color = 'k', 
                            linestyle = '--', label = 'Ponto de Parada')
            if(self.__fit_total):
                axs.plot(curva_num_termos, curva_coef_logloss_tot, color = paleta_cores[2], label = 'Total')
            axs.set_xlabel('Número de Termos')
            axs.set_ylabel('Coeficiente LogLoss')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def __calcula_importancias(self):
        if(self.__tem_validacao):
            vars_selec = self.__vars_selec_melhor
            coef_abs = np.abs(self.__thetas_melhor[1:])
        else:
            vars_selec = self.__vars_selec
            coef_abs = np.abs(self.__thetas[1:])
        termos = self.__series.lista_termos()
        termos = [termos[i] for i in vars_selec]
        def traduz_termo(termo, nome_vars):
            return tuple((nome_vars[v[0]], v[1]) for v in termo)
        self.feature_names_terms_ = np.array([str(traduz_termo(v, self.__nome_vars)) for v in termos])
        self.feature_importances_terms_ = coef_abs/np.sum(coef_abs)
        def pares_feature_coef_abs(termo, coef):
            return [(v[0], coef) for v in termo]
        lista_pesos = []
        for i in range(0, coef_abs.size):
            lista_pesos.extend(pares_feature_coef_abs(termos[i], coef_abs[i]))
        lista_agrupada = [(key, sum(map(itemgetter(1), ele))) for key, ele in groupby(sorted(lista_pesos, key = itemgetter(0)), key = itemgetter(0))]
        self.feature_names_ = np.array([self.__nome_vars[x[0]] for x in lista_agrupada])
        self.feature_importances_ = np.array([x[1] for x in lista_agrupada])
        self.feature_importances_ = self.feature_importances_/np.sum(self.feature_importances_)
        
    def grafico_importancias(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.__curva_num_termos.size
        vars_nomes = self.feature_names_
        vars_imp = self.feature_importances_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()