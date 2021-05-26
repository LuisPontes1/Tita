import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re

import random

from Transforms import *
from DatasetAnalysis import *
from Metrics import *

class AvaliaClassificacao:

    def __init__(self, df, col_alvo, col_prob = None, num_div_prob = None, p_corte = None, p01_corte = [0, 0], num_div = None, frac_cat = 0, unit = None):
        self.__df_tratado = df.copy()
        self.distribuicoes = DistribuicoesDataset(df, num_div = num_div, frac_cat = frac_cat, unit = unit, autorun = False)
        self.__dict_intervs, self.__dict_filtroscat = self.distribuicoes.retorna_trata_dataset().retorna_instancias_tratamento()
        
        self.__y = df[col_alvo].values
        self.__col_alvo = col_alvo
        self.__col_prob = col_prob
        
        if(col_prob != None):
            self.__y_prob = df[col_prob].values
            self.__num_div_prob = num_div_prob
            
            #Calculas as métricas gerais do dataset
            self.metricas_gerais = AletricasClassificacao(self.__y, self.__y_prob, num_div = self.__num_div_prob, p_corte = p_corte, p01_corte = p01_corte)
            #Probabilidades de Corte para Avaliação de Tomada de Decisão (Usa do Ganho de Informação se não for passado)
            probs_ig = self.metricas_gerais.valor_prob_ig()
            if(p_corte == None):
                self.__p_corte = probs_ig['Prob_Corte']
            else:
                self.__p_corte = p_corte
            if(np.sum(np.array(p01_corte)) == 0):
                self.__p01_corte = [probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']]
            else:
                self.__p01_corte = p01_corte
            
            #OBS: Note que, dessa forma, se for o dataset de treino, podemos não passar as probs e usar como corte o decidido pelo IG no próprio dataset
            #Porém, se for um dataset de Validação ou Teste, podemos passar as probs de corte que foram obtidas na avaliação do dataset de Treino
            
        else:
            self.__y_prob = None
            self.__num_div_prob = None
            self.__p_corte = None
            self.__p01_corte = [0, 0]
        
        self.__dict_valores = {}
        self.__dict_qtds = {}
        
        self.__dict_qtds1 = {}
        self.__dict_prob1 = {}
        self.__dict_ig = {}
        self.__dict_ig_por_bit = {} #Ganho de Informação por Bit (quantidade de valores únicos em log2)
        
        self.__dict_soma_prob = {}
        self.__dict_media_prob = {}
        self.__dict_metricas = {}
    
    def colunas_metricas_condicionais_prontas(self):
        return self.__dict_valores.keys()
    
    def calcula_metricas_condicionais(self, col_ref = []):
        num_linhas, _, colunas = self.distribuicoes.info_dataset()
        
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            self.distribuicoes.trata_coluna(col_ref)
            self.__df_tratado[col_ref] = self.distribuicoes.retorna_valores(col_ref)
            self.__dict_intervs, self.__dict_filtroscat = self.distribuicoes.retorna_trata_dataset().retorna_instancias_tratamento()
        
            flag_na = self.distribuicoes.retorna_flag_na(col_ref)
            valores = self.__df_tratado.loc[~flag_na, col_ref].values
            y = self.__y[~flag_na]
            
            inds_ordenado = np.argsort(valores)
            valores_unico, primeira_ocorrencia, qtds = np.unique(valores[inds_ordenado], return_index = True, return_counts = True)
            self.__dict_valores[col_ref] = valores_unico
            self.__dict_qtds[col_ref] = qtds
            
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            qtds1 = np.array([np.sum(v) for v in y_agrup])
            probs1 = qtds1/qtds
            self.__dict_qtds1[col_ref] = qtds1
            self.__dict_prob1[col_ref] = probs1
            
            #Calcula a Entropia de Shannon
            def entropia_shannon(p1):
                p0 = 1 - p1
                if p0 == 0 or p1 == 0:
                    return 0
                else:
                    return -p0*np.log2(p0) - p1*np.log2(p1)
            
            num_linhas_sem_na = num_linhas - np.sum(flag_na)
            
            entropia_ini = entropia_shannon(np.sum(qtds1)/num_linhas_sem_na)
            entropias_parciais = np.array([entropia_shannon(x) for x in probs1])
            entropia = np.sum(entropias_parciais*qtds)/num_linhas_sem_na
            ig = (entropia_ini - entropia)/entropia_ini
            self.__dict_ig[col_ref] = ig
            qtd_unicos = valores_unico.size
            if(qtd_unicos > 1):
                self.__dict_ig_por_bit[col_ref] = ig/np.log2(qtd_unicos)
            else:
                self.__dict_ig_por_bit[col_ref] = 0
            
            if(self.__col_prob != None):
                y_prob = self.__y_prob[~flag_na]
                y_prob_agrup = np.split(y_prob[inds_ordenado], primeira_ocorrencia[1:])
                soma_prob = np.array([np.sum(v) for v in y_prob_agrup])
                self.__dict_soma_prob[col_ref] = soma_prob
                self.__dict_media_prob[col_ref] = soma_prob/qtds
                self.__dict_metricas[col_ref] = np.array([AletricasClassificacao(y_agrup[i], y_prob_agrup[i], num_div = self.__num_div_prob,
                                                          p_corte = self.__p_corte, p01_corte = self.__p01_corte) for i in range(valores_unico.size)])
                                                          
        #Ordena os Ganhos de Informação
        self.__dict_ig = dict(reversed(sorted(self.__dict_ig.items(), key = lambda x: x[1])))
        self.__dict_ig_por_bit = dict(reversed(sorted(self.__dict_ig_por_bit.items(), key = lambda x: x[1])))
    
    def ganho_info(self):
        return pd.Series(self.__dict_ig, index = self.__dict_ig.keys())

    def ganho_info_por_bit(self):
        return pd.Series(self.__dict_ig_por_bit, index = self.__dict_ig_por_bit.keys())
    
    def valor_metricas_condicionais(self, col_ref):
        df = pd.DataFrame()
        if(col_ref in self.__dict_valores.keys()):
            
            df['Valores'] = self.__dict_valores[col_ref]
            if(col_ref in self.__dict_intervs.keys()):
                df['Labels'] = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            elif(col_ref in self.__dict_filtroscat.keys()):
                df['Labels'] = self.__dict_filtroscat[col_ref].strings_categorias()
            df['QTD'] = self.__dict_qtds[col_ref]
            df['QTD_0'] = self.__dict_qtds[col_ref] - self.__dict_qtds1[col_ref]
            df['QTD_1'] = self.__dict_qtds1[col_ref]
            df['Frac_0'] = 1 - self.__dict_prob1[col_ref]
            df['Frac_1'] = self.__dict_prob1[col_ref]
            
            if(self.__col_prob != None):
                df['Soma_Prob'] = self.__dict_soma_prob[col_ref]
                df['Media_Prob'] = self.__dict_media_prob[col_ref]
                df['Metricas'] = self.__dict_metricas[col_ref]
                df = pd.concat([df, df['Metricas'].apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                df = df.drop('Metricas', axis = 1)

        return df
    
    def curva_probabilidade_condicional(self, col_ref):
        valores = self.__dict_valores[col_ref]
        if(col_ref in self.__dict_intervs.keys()):
            labels = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            tipo = 'intervalo'
        elif(col_ref in self.__dict_filtroscat.keys()):
            labels = self.__dict_filtroscat[col_ref].strings_categorias()
            tipo = 'categoria'
        else:
            labels = None
            tipo = None
        prob1 = self.__dict_prob1[col_ref]
        if(self.__col_prob != None):
            media_prob = self.__dict_media_prob[col_ref]
        else:
            media_prob = None
            
        #Ordena por média do alvo se forem categorias (e podemos rearranjar a ordem)
        if(tipo == 'categoria'):
            inds_ordenado = np.argsort(prob1)
            valores = valores[inds_ordenado]
            labels = labels[inds_ordenado]
            prob1 = prob1[inds_ordenado]
            if(self.__col_prob != None):
                media_prob = media_prob[inds_ordenado]
            
        return valores, labels, prob1, media_prob, tipo
        
    def curvas_metricas_condicionais(self, col_ref, metricas):
        valores = self.__dict_valores[col_ref]
        if(col_ref in self.__dict_intervs.keys()):
            labels = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            tipo = 'intervalo'
        elif(col_ref in self.__dict_filtroscat.keys()):
            labels = self.__dict_filtroscat[col_ref].strings_categorias()
            tipo = 'categoria'
        else:
            labels = None
            tipo = None
            
        df = self.valor_metricas_condicionais(col_ref)            
        valores_metricas = []
        for metrica in metricas:
            valores_metricas.append(df[metrica].values)
            
        return valores, labels, valores_metricas, tipo
    
    def grafico_probabilidade_condicional(self, col_ref, ymax = 0, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_valores.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                valores, labels, prob1, media_prob, tipo = self.curva_probabilidade_condicional(col_ref)
                ig = self.__dict_ig[col_ref]
                ig_por_bit = self.__dict_ig_por_bit[col_ref]
                if(labels is not None):
                    if(tipo == 'intervalo'):
                        axs.bar(valores, prob1, color = paleta_cores[0], label = 'Real')
                        axs.set_xticks(valores)
                        axs.set_xticklabels(labels, rotation = 90)
                        if(media_prob is not None):
                            axs.plot(valores, media_prob, color = paleta_cores[1], linewidth = 2, label = 'Classificador')
                    elif(tipo == 'categoria'):
                        if(ticks_chars is not None):
                            labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                        axs.bar(labels, prob1, color = paleta_cores[0], label = 'Real')
                        if(media_prob is not None):
                            axs.plot(labels, media_prob, color = paleta_cores[1], linewidth = 2, label = 'Classificador')
                else:
                    if(conv_str):
                        valores = valores.astype(str)
                        if(ticks_chars is not None):
                            valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    axs.bar(valores, prob1, color = paleta_cores[0], label = 'Real')
                    if(media_prob is not None):
                        axs.plot(valores, media_prob, color = paleta_cores[1], linewidth = 2, label = 'Classificador')
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig + '\n' + 'IG/Bit = ' + '%.2g' % ig_por_bit, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Probabilidade de 1')
                if(ymax > 0 and ymax <= 1):
                    axs.set_ylim([0, ymax])
                axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                plt.show()

    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_qtds1.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico 
            #Plot a curva de métrica em função da coluna de referência
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                valores, labels, valores_metricas, tipo = self.curvas_metricas_condicionais(col_ref, metricas)
                for i in range(len(metricas)):
                    valores_metrica = valores_metricas[i]
                    if(labels is not None):
                        if(tipo == 'intervalo'):
                            if(valores.size > 1):
                                axs.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                            else:
                                axs.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                            axs.set_xticks(valores)
                            axs.set_xticklabels(labels, rotation = 90)
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None and i == 0):
                                labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                            if(valores.size > 1):
                                axs.plot(labels, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                            else:
                                axs.scatter(labels, valores_metrica, color = paleta_cores[i], label = metricas[i])
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        if(valores.size > 1):
                            axs.plot(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                        else:
                            axs.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Metricas')
                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                plt.show()

##############################

##############################

class AvaliaDatasetsClassificacao:

    def __init__(self, dict_dfs, col_alvo, col_prob = None, num_div_prob = None, num_div = None, frac_cat = 0, unit = None, chave_treino = 'Treino'):
        self.__dict_dfs = dict_dfs
        self.__num_dfs = len(dict_dfs)
        self.__chave_treino = chave_treino
        
        self.__dict_avaliaclf = {}
        if(self.__chave_treino in self.__dict_dfs.keys()):
            avaliaclf_treino = AvaliaClassificacao(self.__dict_dfs[self.__chave_treino], col_alvo, col_prob, num_div_prob = num_div_prob, 
                                                   num_div = num_div, frac_cat = frac_cat, unit = unit)
            #Probabilidades de Corte para Avaliação de Tomada de Decisão
            probs_ig = avaliaclf_treino.metricas_gerais.valor_prob_ig()
            p_corte = probs_ig['Prob_Corte']
            p01_corte = [probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']]
            self.__dict_avaliaclf[self.__chave_treino] = avaliaclf_treino
        for chave in self.__dict_dfs.keys():
            if(chave != self.__chave_treino):
                self.__dict_avaliaclf[chave] = AvaliaClassificacao(self.__dict_dfs[chave], col_alvo, col_prob, num_div_prob, p_corte, p01_corte, 
                                                                   num_div, frac_cat, unit)
        
    def avaliadores_individuais(self):
        return self.__dict_avaliaclf
        
    def calcula_metricas_condicionais(self, col_ref = []):
        for chave in self.__dict_dfs.keys():
            self.__dict_avaliaclf[chave].calcula_metricas_condicionais(col_ref)
    
    def valor_metricas(self, estatisticas_globais = True, probs_corte = True, probs_condicionais = True, lifts = True):
        df = pd.DataFrame(self.__dict_avaliaclf[self.__chave_treino].metricas_gerais.valor_metricas(estatisticas_globais, probs_corte, probs_condicionais, lifts), columns = [self.__chave_treino])
        for chave in self.__dict_dfs.keys():
            if(chave != self.__chave_treino):
                df[chave] = self.__dict_avaliaclf[chave].metricas_gerais.valor_metricas(estatisticas_globais, probs_corte, probs_condicionais, lifts)
        return df
    
    def grafico_roc(self, roc_usual = True, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__dict_dfs.keys():
                curva_tnp, curva_tvp, auc = self.__dict_avaliaclf[chave].metricas_gerais.curva_roc()
                if(roc_usual):
                    axs.plot(1-curva_tnp, curva_tvp, color = paleta_cores[i], label = chave)
                else:
                    axs.plot(curva_tnp, curva_tvp, color = paleta_cores[i], label = chave)
                i = i + 1
            if(roc_usual):
                axs.plot([0, 1], [0, 1], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Falso Positivo')
            else:
                axs.plot([0, 1], [1, 0], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Verdadeiro Negativo')
            axs.set_ylabel('Taxa de Verdadeiro Positivo')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

    def grafico_revocacao(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__dict_dfs.keys():
                y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks = self.__dict_avaliaclf[chave].metricas_gerais.curva_revocacao()
                axs.plot(y_prob_plot, curva_revoc0_plot, color = paleta_cores[i], alpha = 0.6)
                axs.plot(y_prob_plot, curva_revoc1_plot, color = paleta_cores[i], alpha = 0.4)
                axs.vlines(pos_max, 0, 1, color = paleta_cores[i], linestyle = '--', label = chave)
                i = i + 1
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Revocação')
            plt.show()
            
    def grafico_informacao(self, mostrar_ig_2d = False, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__dict_dfs.keys():
                y_prob_plot, curva_ig_plot, pos_max, ig, p0_corte, p1_corte, ig_2d = self.__dict_avaliaclf[chave].metricas_gerais.curva_informacao()
                axs.plot(y_prob_plot, curva_ig_plot, color = paleta_cores[i], label = chave)
                axs.vlines(pos_max, 0, ig, color = paleta_cores[i], linestyle = '--')
                if(mostrar_ig_2d and ig_2d != np.nan):
                    axs.vlines(p0_corte, 0, ig_2d, color = paleta_cores[i], alpha = 0.5, linestyle = '--')
                    axs.vlines(p1_corte, 0, ig_2d, color = paleta_cores[i], alpha = 0.5, linestyle = '--')
                i = i + 1
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Ganho de Informação')
            plt.show()
            
    def grafico_informacao_2d(self, plot_3d = True, figsize = [7, 6]):
        paleta_cores = sns.color_palette("colorblind")
        if(plot_3d):
            with sns.axes_style("whitegrid"):
                fig = plt.figure(figsize = figsize)
                axs = fig.add_subplot(111, projection='3d')
                i = 0
                hlds = []
                for chave in self.__dict_dfs.keys():
                    x, y, z, p0_corte, p1_corte, ig_2d = self.__dict_avaliaclf[chave].metricas_gerais.curva_informacao_2d()
                    N = 256
                    vals = np.ones((N, 4)) #A última componente (quarta) é o alpha que é o índice de transparência
                    cor = paleta_cores[i]
                    #Define as Cores RGB pelas componentes (no caso é o azul -> 0,0,255)
                    vals[:, 0] = np.linspace(cor[0], 1, N)
                    vals[:, 1] = np.linspace(cor[1], 1, N)
                    vals[:, 2] = np.linspace(cor[2], 1, N)
                    cmap = mpl.colors.ListedColormap(vals[::-1])
                    axs.scatter(x, y, z, c = z, marker = 'o', cmap = cmap)
                    hlds.append(mpl.patches.Patch(color = cor, label = chave))
                    i = i + 1
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_zlabel('Ganho de Informação')
                axs.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()
        else:
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                i = 0
                hlds = []
                mini = 1
                maxi = 0
                for chave in self.__dict_dfs.keys():
                    x, y, z, p0_corte, p1_corte, ig_2d = self.__dict_avaliaclf[chave].metricas_gerais.curva_informacao_2d()
                    mini = min(mini, min(x[0], y[0]))
                    maxi = max(maxi, max(x[-1], y[-1]))
                    N = 256
                    cor = paleta_cores[i]
                    vals = np.ones((N, 4))
                    vals[:, 0] = cor[0]
                    vals[:, 1] = cor[1]
                    vals[:, 2] = cor[2]
                    cmap_linhas = mpl.colors.ListedColormap(vals[::-1])
                    vals[:, 3] = np.linspace(0, 1, N)[::-1]
                    cmap = mpl.colors.ListedColormap(vals[::-1])
                    axs.tricontour(x, y, z, levels = 14, linewidths = 1.0, cmap = cmap_linhas)
                    #cntr = axs.tricontourf(x, y, z, levels = 14, cmap = cmap)
                    axs.scatter(p0_corte, p1_corte, color = cor)
                    axs.vlines(p0_corte, 0, p1_corte, color = cor, alpha = 0.5, linestyle = '--')
                    axs.hlines(p1_corte, 0, p0_corte, color = cor, alpha = 0.5, linestyle = '--')
                    hlds.append(mpl.patches.Patch(color = cor, label = chave))
                    i = i + 1
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_xlim([mini, maxi])
                axs.set_ylim([mini, maxi])
                axs.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()
                
    def grafico_distribuicao(self, col_ref = [], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        _, _, colunas = self.__dict_avaliaclf[self.__chave_treino].distribuicoes.info_dataset()
    
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
    
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_prontas()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__dict_dfs.keys():
                        valores, frac, tipo = self.__dict_avaliaclf[chave].distribuicoes.curva_distribuicao(col_ref)
                        if(tipo == 'intervalo'):
                            axs.fill_between(valores, frac, color = paleta_cores[i], alpha = 0.5)
                            axs.plot(valores, frac, color = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração/L')
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                            axs.bar(valores, frac, color = paleta_cores[i], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração')
                        else:
                            if(conv_str):
                                valores = valores.astype(str)
                                if(ticks_chars is not None):
                                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                            axs.bar(valores, frac, color = paleta_cores[i], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração')
                        i = i + 1
                    axs.set_xlabel(col_ref)
                    axs.set_ylim(bottom = 0.0)
                    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.show()
                
    def grafico_probabilidade_condicional(self, col_ref, ymax = 0, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_prontas()):
        
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                if(ymax > 0 and ymax <= 1):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                i = 0
                hlds = []             
                
                for chave in self.__dict_dfs.keys():
                    if(self.__num_dfs > 1):
                        ax = axs[i]
                    else:
                        ax = axs
                    valores, labels, prob1, media_prob, tipo = self.__dict_avaliaclf[chave].curva_probabilidade_condicional(col_ref)
                    if(labels is not None):
                        if(tipo == 'intervalo'):
                            ax.bar(valores, prob1, color = paleta_cores[i], label = chave)
                            ax.set_xticks(valores)
                            ax.set_xticklabels(labels, rotation = 90)
                            if(media_prob is not None):
                                if(valores.size > 1):
                                    ax.plot(valores, media_prob, color = 'black', linewidth = 2)
                                else:
                                    ax.scatter(valores, media_prob, color = 'black')
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None):
                                labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                            ax.bar(labels, prob1, color = paleta_cores[i], label = chave)
                            if(media_prob is not None):
                                if(valores.size > 1):
                                    ax.plot(labels, media_prob, color = 'black', linewidth = 2)
                                else:
                                    ax.scatter(labels, media_prob, color = 'black')
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        ax.bar(valores, prob1, color = paleta_cores[i], label = chave)
                        if(media_prob is not None):
                            if(valores.size > 1):
                                ax.plot(valores, media_prob, color = 'black', linewidth = 2)
                            else:
                                ax.scatter(valores, media_prob, color = 'black')      
                    if(ymax > 0 and ymax <= 1):
                        ax.set_ylim([0, ymax])
                        if(i == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Probabilidade de 1')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Probabilidade de 1')
                    ax.set_title(chave)
                    hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                    i = i + 1
                hlds.append(mpl.patches.Patch(color = 'k', label = 'Classificador'))
                plt.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_prontas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                j = 0
                hlds = []
                for chave in self.__dict_dfs.keys():
                    if(self.__num_dfs > 1):
                        ax = axs[j]
                    else:
                        ax = axs

                    valores, labels, valores_metricas, tipo = self.__dict_avaliaclf[chave].curvas_metricas_condicionais(col_ref, metricas)
                    for i in range(len(metricas)):
                        valores_metrica = valores_metricas[i]
                        if(labels is not None):
                            if(tipo == 'intervalo'):
                                if(valores.size > 1):
                                    ax.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                                else:
                                    ax.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                                ax.set_xticks(valores)
                                ax.set_xticklabels(labels, rotation = 90)
                            elif(tipo == 'categoria'):
                                if(ticks_chars is not None and i == 0):
                                    labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                                if(valores.size > 1):
                                    ax.plot(labels, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                                else:
                                    ax.scatter(labels, valores_metrica, color = paleta_cores[i], label = metricas[i])
                        else:
                            if(conv_str):
                                valores = valores.astype(str)
                                if(ticks_chars is not None):
                                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                            if(valores.size > 1):
                                ax.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                            else:
                                ax.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                                    
                        if(chave == self.__chave_treino):
                            hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = metricas[i]))
                    
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(j == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Metricas')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Metricas')
                    ax.set_title(chave)
                    j = j + 1
                plt.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()

###########################################

###########################################

class AvaliaRegressao:

    def __init__(self, df, col_alvo, col_pred = None, y_ref = None, num_kendalltau = 10000, num_div = None, frac_cat = 0, unit = None):
        self.__df_tratado = df.copy()
        self.distribuicoes = DistribuicoesDataset(df, num_div = num_div, frac_cat = frac_cat, unit = unit, autorun = False)
        self.__dict_intervs, self.__dict_filtroscat = self.distribuicoes.retorna_trata_dataset().retorna_instancias_tratamento()
        
        self.__y = df[col_alvo].values
        self.__col_alvo = col_alvo
        self.__col_pred = col_pred
        self.__num_kendalltau = num_kendalltau
        
        if(col_pred != None):
            self.__y_pred = df[col_pred].values
            
            #Calculas as métricas gerais do dataset
            self.metricas_gerais = AletricasRegressao(self.__y, self.__y_pred, y_ref = y_ref, num_kendalltau = self.__num_kendalltau)
            
            #Valor de referência para as métricas
            if(y_ref == None):
                self.__y_ref = self.metricas_gerais.valor_media_alvo()
            else:
                self.__y_ref = y_ref
            
        else:
            self.__y_pred = None
            self.__y_ref = None
        
        self.__dict_valores = {}
        self.__dict_qtds = {}
        
        self.__dict_soma_alvo = {}
        self.__dict_media_alvo = {}
        self.__dict_desvio_norm = {}
        self.__dict_desvio_norm_por_bit = {}
        
        self.__dict_soma_pred = {}
        self.__dict_media_pred = {}
        self.__dict_metricas = {}
    
    def colunas_metricas_condicionais_prontas(self):
        return self.__dict_valores.keys()
    
    def calcula_metricas_condicionais(self, col_ref = []):
        num_linhas, _, colunas = self.distribuicoes.info_dataset()
        
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            self.distribuicoes.trata_coluna(col_ref)
            self.__df_tratado[col_ref] = self.distribuicoes.retorna_valores(col_ref)
            self.__dict_intervs, self.__dict_filtroscat = self.distribuicoes.retorna_trata_dataset().retorna_instancias_tratamento()
        
            flag_na = self.distribuicoes.retorna_flag_na(col_ref)
            valores = self.__df_tratado.loc[~flag_na, col_ref].values
            y = self.__y[~flag_na]
            
            inds_ordenado = np.argsort(valores)
            valores_unico, primeira_ocorrencia, qtds = np.unique(valores[inds_ordenado], return_index = True, return_counts = True)
            self.__dict_valores[col_ref] = valores_unico
            self.__dict_qtds[col_ref] = qtds
            
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            soma = np.array([np.sum(v) for v in y_agrup])
            self.__dict_soma_alvo[col_ref] = soma
            media = soma/qtds
            self.__dict_media_alvo[col_ref] = media
            
            #Ideia para ser o equivalente do Ganho de Informação
            desvio_ini = np.std(y)
            desvio_norm = np.std(media)/desvio_ini
            self.__dict_desvio_norm[col_ref] = desvio_norm
            qtd_unicos = valores_unico.size
            if(qtd_unicos > 1):
                self.__dict_desvio_norm_por_bit[col_ref] = desvio_norm/np.log2(qtd_unicos)
            else:
                self.__dict_desvio_norm_por_bit[col_ref] = 0
            
            if(self.__col_pred != None):
                y_pred = self.__y_pred[~flag_na]
                y_pred_agrup = np.split(y_pred[inds_ordenado], primeira_ocorrencia[1:])
                soma_pred = np.array([np.sum(v) for v in y_pred_agrup])
                self.__dict_soma_pred[col_ref] = soma_pred
                self.__dict_media_pred[col_ref] = soma_pred/qtds
                self.__dict_metricas[col_ref] = np.array([AletricasRegressao(y_agrup[i], y_pred_agrup[i], 
                                                          y_ref = self.__y_ref, num_kendalltau = self.__num_kendalltau) for i in range(valores_unico.size)])
                                                          
        #Ordena por Desvio
        self.__dict_desvio_norm = dict(reversed(sorted(self.__dict_desvio_norm.items(), key = lambda x: x[1])))
        self.__dict_desvio_norm_por_bit = dict(reversed(sorted(self.__dict_desvio_norm_por_bit.items(), key = lambda x: x[1])))
    
    def desvio_normalizado(self):
        return pd.Series(self.__dict_desvio_norm, index = self.__dict_desvio_norm.keys())

    def desvio_normalizado_por_bit(self):
        return pd.Series(self.__dict_desvio_norm_por_bit, index = self.__dict_desvio_norm_por_bit.keys())
    
    def valor_metricas_condicionais(self, col_ref):
        df = pd.DataFrame()
        if(col_ref in self.__dict_soma_alvo.keys()):
        
            df['Valores'] = self.__dict_valores[col_ref]
            if(col_ref in self.__dict_intervs.keys()):
                df['Labels'] = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            elif(col_ref in self.__dict_filtroscat.keys()):
                df['Labels'] = self.__dict_filtroscat[col_ref].strings_categorias()
            df['QTD'] = self.__dict_qtds[col_ref]
            df['Soma_Alvo'] = self.__dict_soma_alvo[col_ref]
            df['Media_Alvo'] = self.__dict_media_alvo[col_ref]
            
            if(self.__col_pred != None):
                df['Soma_Pred'] = self.__dict_soma_pred[col_ref]
                df['Media_Pred'] = self.__dict_media_pred[col_ref]
                df['Metricas'] = self.__dict_metricas[col_ref]
                df = pd.concat([df, df['Metricas'].apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                df = df.drop('Metricas', axis = 1)

        return df
    
    def curva_media_condicional(self, col_ref):
        valores = self.__dict_valores[col_ref]
        if(col_ref in self.__dict_intervs.keys()):
            labels = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            tipo = 'intervalo'
        elif(col_ref in self.__dict_filtroscat.keys()):
            labels = self.__dict_filtroscat[col_ref].strings_categorias()
            tipo = 'categoria'
        else:
            labels = None
            tipo = None
        media_alvo = self.__dict_media_alvo[col_ref]
        if(self.__col_pred != None):
            media_pred = self.__dict_media_pred[col_ref]
        else:
            media_pred = None
            
        #Ordena por média do alvo se forem categorias (e podemos rearranjar a ordem)
        if(tipo == 'categoria'):
            inds_ordenado = np.argsort(media_alvo)
            valores = valores[inds_ordenado]
            labels = labels[inds_ordenado]
            media_alvo = media_alvo[inds_ordenado]
            if(self.__col_pred != None):
                media_pred = media_pred[inds_ordenado]
            
        return valores, labels, media_alvo, media_pred, tipo
    
    def curvas_metricas_condicionais(self, col_ref, metricas):
        valores = self.__dict_valores[col_ref]
        if(col_ref in self.__dict_intervs.keys()):
            labels = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            tipo = 'intervalo'
        elif(col_ref in self.__dict_filtroscat.keys()):
            labels = self.__dict_filtroscat[col_ref].strings_categorias()
            tipo = 'categoria'
        else:
            labels = None
            tipo = None
            
        df = self.valor_metricas_condicionais(col_ref)            
        valores_metricas = []
        for metrica in metricas:
            valores_metricas.append(df[metrica].values)
            
        return valores, labels, valores_metricas, tipo
    
    def grafico_media_condicional(self, col_ref, ylim = [0, 0], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_soma_alvo.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                valores, labels, media_alvo, media_pred, tipo = self.curva_media_condicional(col_ref)
                desvio_norm = self.__dict_desvio_norm[col_ref]
                desvio_norm_por_bit = self.__dict_desvio_norm_por_bit[col_ref]
                if(labels is not None):
                    if(tipo == 'intervalo'):
                        axs.bar(valores, media_alvo, color = paleta_cores[0], label = 'Real')
                        axs.set_xticks(valores)
                        axs.set_xticklabels(labels, rotation = 90)
                        if(media_pred is not None):
                            axs.plot(valores, media_pred, color = paleta_cores[1], linewidth = 2, label = 'Regressor')
                    elif(tipo == 'categoria'):
                        if(ticks_chars is not None):
                            labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                        axs.bar(labels, media_alvo, color = paleta_cores[0], label = 'Real')
                        if(media_pred is not None):
                            axs.plot(labels, media_pred, color = paleta_cores[1], linewidth = 2, label = 'Regressor')
                else:
                    if(conv_str):
                        valores = valores.astype(str)
                        if(ticks_chars is not None):
                            valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    axs.bar(valores, media_alvo, color = paleta_cores[0], label = 'Real')
                    if(media_pred is not None):
                        axs.plot(valores, media_pred, color = paleta_cores[1], linewidth = 2, label = 'Classificador')
                plt.gcf().text(1, 0.5, 'DESV = ' + '%.2g' % desvio_norm + '\n' + 'DESV/Bit = ' + '%.2g' % desvio_norm_por_bit, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Média dos Valores')
                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                plt.show()

    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_soma_alvo.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            #Plot a curva de métrica em função da coluna de referência
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                valores, labels, valores_metricas, tipo = self.curvas_metricas_condicionais(col_ref, metricas)
                for i in range(len(metricas)):
                    valores_metrica = valores_metricas[i]
                    if(labels is not None):
                        if(tipo == 'intervalo'):
                            if(valores.size > 1):
                                axs.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                            else:
                                axs.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                            axs.set_xticks(valores)
                            axs.set_xticklabels(labels, rotation = 90)
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None and i == 0):
                                labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                            if(valores.size > 1):
                                axs.plot(labels, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                            else:
                                axs.scatter(labels, valores_metrica, color = paleta_cores[i], label = metricas[i])
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        if(valores.size > 1):
                            axs.plot(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                        else:
                            axs.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Metricas')
                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                plt.show()
                
##############################

##############################

class AvaliaDatasetsRegressao:

    def __init__(self, dict_dfs, col_alvo, col_pred = None, num_kendalltau = 10000, num_div = None, frac_cat = 0, unit = None, chave_treino = 'Treino'):
        self.__dict_dfs = dict_dfs
        self.__num_dfs = len(dict_dfs)
        self.__chave_treino = chave_treino
        
        self.__dict_avaliargs = {}
        if(self.__chave_treino in self.__dict_dfs.keys()):
            avaliargs_treino = AvaliaRegressao(self.__dict_dfs[self.__chave_treino], col_alvo, col_pred, num_kendalltau = num_kendalltau, 
                                               num_div = num_div, frac_cat = frac_cat, unit = unit)
            #Pega o y_ref do treino
            y_ref = avaliargs_treino.metricas_gerais.valor_media_alvo()
            self.__dict_avaliargs[self.__chave_treino] = avaliargs_treino
        for chave in self.__dict_dfs.keys():
            if(chave != self.__chave_treino):
                self.__dict_avaliargs[chave] = AvaliaRegressao(self.__dict_dfs[chave], col_alvo, col_pred, y_ref, num_kendalltau, 
                                                               num_div, frac_cat, unit)
        
    def avaliadores_individuais(self):
        return self.__dict_avaliargs
        
    def calcula_metricas_condicionais(self, col_ref = []):
        for chave in self.__dict_dfs.keys():
            self.__dict_avaliargs[chave].calcula_metricas_condicionais(col_ref)
    
    def valor_metricas(self, estatisticas_globais = True, metricas_ref = True, alga_signif = 0, conv_str = False):
        df = pd.DataFrame(self.__dict_avaliargs[self.__chave_treino].metricas_gerais.valor_metricas(estatisticas_globais, metricas_ref, alga_signif, conv_str), columns = [self.__chave_treino])
        for chave in self.__dict_dfs.keys():
            if(chave != self.__chave_treino):
                df[chave] = self.__dict_avaliargs[chave].metricas_gerais.valor_metricas(estatisticas_globais, metricas_ref, alga_signif, conv_str)
        return df
                
    def grafico_distribuicao(self, col_ref = [], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        _, _, colunas = self.__dict_avaliaclf[self.__chave_treino].distribuicoes.info_dataset()
    
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
    
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_prontas()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__dict_dfs.keys():
                        valores, frac, tipo = self.__dict_avaliaclf[chave].distribuicoes.curva_distribuicao(col_ref)
                        if(tipo == 'intervalo'):
                            axs.fill_between(valores, frac, color = paleta_cores[i], alpha = 0.5)
                            axs.plot(valores, frac, color = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração/L')
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                            axs.bar(valores, frac, color = paleta_cores[i], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração')
                        else:
                            if(conv_str):
                                valores = valores.astype(str)
                                if(ticks_chars is not None):
                                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                            axs.bar(valores, frac, color = paleta_cores[i], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração')
                        i = i + 1
                    axs.set_xlabel(col_ref)
                    axs.set_ylim(bottom = 0.0)
                    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.show()
                
    def grafico_media_condicional(self, col_ref, ylim = [0, 0], conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_prontas()):
        
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                i = 0
                hlds = []
                for chave in self.__dict_dfs.keys():
                    if(self.__num_dfs > 1):
                        ax = axs[i]
                    else:
                        ax = axs
                        
                    valores, labels, media_alvo, media_pred, tipo = self.__dict_avaliargs[chave].curva_media_condicional(col_ref)
                    
                    if(labels is not None):
                        if(tipo == 'intervalo'):
                            ax.bar(valores, media_alvo, color = paleta_cores[i], label = chave)
                            ax.set_xticks(valores)
                            ax.set_xticklabels(labels, rotation = 90)
                            if(media_pred is not None):
                                if(valores.size > 1):
                                    ax.plot(valores, media_pred, color = 'black', linewidth = 2)
                                else:
                                    ax.scatter(valores, media_pred, color = 'black', label = 'Regressor')
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None):
                                labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                            ax.bar(labels, media_alvo, color = paleta_cores[i], label = chave)
                            if(media_pred is not None):
                                if(valores.size > 1):
                                    ax.plot(labels, media_pred, color = 'black', linewidth = 2)
                                else:
                                    ax.scatter(labels, media_pred, color = 'black')
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        ax.bar(valores, media_alvo, color = paleta_cores[i], label = chave)
                        if(media_pred is not None):
                            if(valores.size > 1):
                                ax.plot(valores, media_pred, color = 'black', linewidth = 2)
                            else:
                                ax.scatter(valores, media_pred, color = 'black')
                        
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(i == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Média dos Valores')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Média dos Valores')
                    ax.set_title(chave)
                    hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                    i = i + 1
                hlds.append(mpl.patches.Patch(color = 'k', label = 'Regressor'))
                plt.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_prontas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                j = 0
                hlds = []
                for chave in self.__dict_dfs.keys():
                    if(self.__num_dfs > 1):
                        ax = axs[j]
                    else:
                        ax = axs

                    valores, labels, valores_metricas, tipo = self.__dict_avaliargs[chave].curvas_metricas_condicionais(col_ref, metricas)
                    for i in range(len(metricas)):
                        valores_metrica = valores_metricas[i]
                        if(labels is not None):
                            if(tipo == 'intervalo'):
                                if(valores.size > 1):
                                    ax.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                                else:
                                    ax.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                                ax.set_xticks(valores)
                                ax.set_xticklabels(labels, rotation = 90)
                            elif(tipo == 'categoria'):
                                if(ticks_chars is not None and i == 0):
                                    labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                                if(valores.size > 1):
                                    ax.plot(labels, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                                else:
                                    ax.scatter(labels, valores_metrica, color = paleta_cores[i], label = metricas[i])
                        else:
                            if(conv_str):
                                valores = valores.astype(str)
                                if(ticks_chars is not None):
                                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                            if(valores.size > 1):
                                ax.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                            else:
                                ax.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                                    
                        if(chave == self.__chave_treino):
                            hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = metricas[i]))
                    
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(j == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Metricas')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Metricas')
                    ax.set_title(chave)
                    j = j + 1
                plt.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()

##############################

##############################

class ImportanciaVariaveisClassificacao:

    def __init__(self, clf, df, col_alvo, cols_features = None, num_loop = 5, random_state = None):
        self.__df = df.copy()
        self.__y = df[col_alvo].values
        self.__tam = self.__y.size
        self.__clf = clf
        if(cols_features is None):
            self.__features = np.array([col for col in list(df.columns) if col != col_alvo])
        else:
            self.__features = np.array(cols_features)
        self.__col_alvo = col_alvo
        self.__num_loop = num_loop
        
        self.__importancias = None
        self.__incerteza = None
        self.__dict_imp = None
        
        if(random_state is not None):
            random.seed(random_state)
            
        self.__calcula_importancias()
        
    def __calcula_importancias(self):
        def logloss(y, y_prob):
            logloss = -1*np.mean(y*np.log2(y_prob) + (1 - y)*np.log2(1 - y_prob))
            return logloss
        
        y_prob = self.__clf.predict_proba(self.__df)[:, 1]
        logloss_ini = logloss(self.__y, y_prob)
        
        vetor_logloss = np.array([])
        vetor_incerteza = np.array([])
        for feature in self.__features:
            logloss_steps = np.array([])
            df_temp = self.__df.copy()
            valores_feature = list(df_temp[feature].values)
            for i in range(0, self.__num_loop):
                df_temp[feature] = random.sample(valores_feature, self.__tam)
                y_prob = self.__clf.predict_proba(df_temp)[:, 1]
                logloss_steps = np.append(logloss_steps, logloss(self.__y, y_prob))
            vetor_logloss = np.append(vetor_logloss, np.mean(logloss_steps))
            vetor_incerteza = np.append(vetor_incerteza, np.std(logloss_steps))
        
        self.__importancias = np.abs(vetor_logloss/logloss_ini - 1)
        soma = np.sum(self.__importancias)
        self.__importancias = self.__importancias/soma
        
        self.__incerteza = vetor_incerteza/(logloss_ini*soma)
        
        self.__dict_imp = {self.__features[i]:self.__importancias[i] for i in range(0, self.__features.size)}
        self.__dict_imp = dict(reversed(sorted(self.__dict_imp.items(), key = lambda x: x[1])))
    
    def retorna_importancias(self):
        return self.__dict_imp
        
    def retorna_incerteza(self):
        return self.__incerteza
        
##############################

##############################

class ImportanciaVariaveisRegressao:

    def __init__(self, rgs, df, col_alvo, cols_features = None, num_loop = 5, random_state = None):
        self.__df = df.copy()
        self.__y = df[col_alvo].values
        self.__tam = self.__y.size
        self.__rgs = rgs
        if(cols_features is None):
            self.__features = np.array([col for col in list(df.columns) if col != col_alvo])
        else:
            self.__features = np.array(cols_features)
        self.__col_alvo = col_alvo
        self.__num_loop = num_loop
        
        self.__importancias = None
        self.__incerteza = None
        self.__dict_imp = None
        
        if(random_state is not None):
            random.seed(random_state)
            
        self.__calcula_importancias()
        
    def __calcula_importancias(self):
        def mse(y, y_pred):
            mse = np.mean(np.power(y - y_pred, 2))
            return mse
        
        y_pred = self.__rgs.predict(self.__df)
        mse_ini = mse(self.__y, y_pred)
        
        vetor_mse = np.array([])
        vetor_incerteza = np.array([])
        for feature in self.__features:
            mse_steps = np.array([])
            df_temp = self.__df.copy()
            valores_feature = list(df_temp[feature].values)
            for i in range(0, self.__num_loop):
                df_temp[feature] = random.sample(valores_feature, self.__tam)
                y_pred = self.__rgs.predict(df_temp)
                mse_steps = np.append(mse_steps, mse(self.__y, y_pred))
            vetor_mse = np.append(vetor_mse, np.mean(mse_steps))
            vetor_incerteza = np.append(vetor_incerteza, np.std(mse_steps))
        
        self.__importancias = np.abs(vetor_mse/mse_ini - 1)
        soma = np.sum(self.__importancias)
        self.__importancias = self.__importancias/soma
        
        self.__incerteza = vetor_incerteza/(mse_ini*soma)
        
        self.__dict_imp = {self.__features[i]:self.__importancias[i] for i in range(0, self.__features.size)}
        self.__dict_imp = dict(reversed(sorted(self.__dict_imp.items(), key = lambda x: x[1])))
    
    def retorna_importancias(self):
        return self.__dict_imp
        
    def retorna_incerteza(self):
        return self.__incerteza