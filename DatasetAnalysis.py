import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re

from Transforms import *

#Para funcionar direito, é preciso transformar os valores numéricos todos em float (não pode ter int, por exemplo)
class DistribuicoesDataset:

    def __init__(self, df, num_div = None, frac_cat = 0, unit = None, autorun = True):
        self.__df = df.copy()
        self.__num_linhas = len(self.__df)
        self.__colunas = self.__df.columns.values
        self.__num_colunas = self.__colunas.size
        self.__autorun = autorun
        
        self.__num_div = num_div
        self.__frac_cat = frac_cat
        
        self.__dict_flag_na = {}
        self.__dict_frac_na = {}
        self.__tipo_numerico = np.array([])
        self.__tipo_categorico = np.array([])
        self.__tipo_temporal = np.array([])
        
        self.__colunas_dropar = np.array([])
        self.__colunas_tratadas = np.array([])
        
        for col in self.__colunas:
            self.__encontra_na_e_tipo(col)
        
        if(self.__colunas_dropar.size > 0):
            self.__df = self.__df.drop(self.__colunas_dropar, axis = 1)
            self.__colunas = self.__df.columns.values
            self.__num_colunas = self.__colunas.size
        
        self.__tratadf = TrataDataset(self.__df, num_div = self.__num_div, frac_cat = self.__frac_cat, 
                                      features_numericas = list(self.__tipo_numerico), 
                                      features_categoricas = list(self.__tipo_categorico),
                                      features_temporais = list(self.__tipo_temporal),
                                      unit = unit, autorun = self.__autorun)        
        self.__dict_intervs, self.__dict_filtroscat = self.__tratadf.retorna_instancias_tratamento()
        
        if(self.__autorun):
            self.__colunas_tratadas = self.__colunas
    
    def trata_coluna(self, feature):
        self.__tratadf.trata_coluna(feature)
        self.__dict_intervs, self.__dict_filtroscat = self.__tratadf.retorna_instancias_tratamento()
        if(self.__colunas_tratadas.size == 0 or np.sum(np.where(self.__colunas_tratadas == feature)) == 0):
            self.__colunas_tratadas = np.append(self.__colunas_tratadas, feature)
    
    def info_dataset(self):
        return self.__num_linhas, self.__num_colunas, self.__colunas
        
    def retorna_trata_dataset(self):
        return self.__tratadf
    
    def retorna_flag_na(self, col_ref):
        return self.__dict_flag_na[col_ref]
    
    def retorna_valores(self, col_ref):
        if(self.__colunas_tratadas.size > 0 and col_ref in self.__colunas_tratadas):
            if(col_ref in self.__dict_intervs.keys()):
                valores = self.__dict_intervs[col_ref].aplica_discretizacao(self.__df[col_ref].values, usar_ponto_medio = False)
                tipo = 'intervalo'
            elif(col_ref in self.__dict_filtroscat.keys()):
                valores = self.__dict_filtroscat[col_ref].aplica_filtro_categorias(self.__df[col_ref].values, considera_resto = True, usar_str = False)
                tipo = 'categoria'
            else:
                valores = self.__df[col_ref].values
                tipo = 'discreto'
            return valores
    
    #Enquantra a quantidade e posições dos NA na coluna e separa as colunas por tipo
    def __encontra_na_e_tipo(self, col_ref):
        valores = self.__df[col_ref].values
        if(valores.dtype in [np.number, 'int64', 'float64']):
            flag_na = np.isnan(valores, where = True)
            frac_na = np.sum(flag_na)/self.__num_linhas
            if(frac_na == 1):
                self.__colunas_dropar = np.append(self.__colunas_dropar, col_ref)
            else:
                self.__tipo_numerico = np.append(self.__tipo_numerico, col_ref)
        elif(valores.dtype in ['<M8[ns]']):
            flag_na = pd.isna(self.__df[col_ref]).values
            frac_na = np.sum(flag_na)/self.__num_linhas
            if(frac_na == 1):
                self.__colunas_dropar = np.append(self.__colunas_dropar, col_ref)
            else:
                self.__tipo_temporal = np.append(self.__tipo_temporal, col_ref)
        else:
            flag_na = pd.isna(self.__df[col_ref]).values
            frac_na = np.sum(flag_na)/self.__num_linhas
            if(frac_na == 1):
                self.__colunas_dropar = np.append(self.__colunas_dropar, col_ref)
            else:
                self.__tipo_categorico = np.append(self.__tipo_categorico, col_ref)
        self.__dict_flag_na[col_ref] = flag_na
        self.__dict_frac_na[col_ref] = frac_na
    
    def info_distribuicao(self, col_ref):
        if(self.__colunas_tratadas.size > 0 and col_ref in self.__colunas_tratadas):
            if(col_ref in self.__dict_intervs.keys()):
                df_info = self.__dict_intervs[col_ref].info_discretizacao()
                tipo = 'intervalo'
                
            elif(col_ref in self.__dict_filtroscat.keys()):
                df_info = self.__dict_filtroscat[col_ref].info_categorias()
                tipo = 'categoria'
                
            else:
                valores, qtds = np.unique(self.__df[col_ref].dropna().values, return_counts = True)
                frac = qtds/self.__num_linhas
                df_info = pd.DataFrame(zip(qtds, frac, valores), columns = ['QTD', 'Frac', 'Valor'])
                tipo = 'discreto'
            return df_info
    
    def curva_distribuicao(self, col_ref):
        if(self.__colunas_tratadas.size > 0 and col_ref in self.__colunas_tratadas):
            if(col_ref in self.__dict_intervs.keys()):
                valores, frac = self.__dict_intervs[col_ref].curva_distribuicao()
                tipo = 'intervalo'
                
            elif(col_ref in self.__dict_filtroscat.keys()):
                valores, frac = self.__dict_filtroscat[col_ref].curva_distribuicao()
                tipo = 'categoria'
                
            else:
                valores, qtds = np.unique(self.__df[col_ref].dropna().values, return_counts = True)
                frac = qtds/self.__num_linhas
                tipo = 'discreto'
            return valores, frac, tipo
    
    def grafico_distribuicao(self, col_ref = [], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        #Se não passar nada plota o gráfico de todas as colunas
        if(len(col_ref) == 0):
            colunas = self.__colunas
        else:
            colunas = col_ref
        
        #Para cada coluna
        for col_ref in colunas:
            if(self.__colunas_tratadas.size > 0 and col_ref in self.__colunas_tratadas):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                frac_na = self.__dict_frac_na[col_ref]
                
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    valores, frac, tipo = self.curva_distribuicao(col_ref)
                    if(tipo == 'intervalo'):
                        axs.fill_between(valores, frac, color = paleta_cores[0], alpha = 0.5)
                        axs.plot(valores, frac, color = paleta_cores[0])
                        axs.set_ylabel('Fração/L')
                    elif(tipo == 'categoria'):
                        if(ticks_chars is not None):
                            valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        axs.bar(valores, frac, color = paleta_cores[0], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[0])
                        axs.set_ylabel('Fração')
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        axs.bar(valores, frac, color = paleta_cores[0], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[0])
                        axs.set_ylabel('Fração')
                    plt.gcf().text(1, 0.8, 'Fração de NA = ' + '%.2g' % frac_na, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                    axs.set_xlabel(col_ref)
                    axs.set_ylim(bottom = 0.0)
                    plt.show()
                    
##############################

##############################

class AvaliaDatasetsDistribuicoes:

    def __init__(self, dict_dfs, num_div = None, frac_cat = 0, unit = None):
        self.__dict_dfs = dict_dfs
        self.__num_dfs = len(dict_dfs)
        
        self.__dict_distribuicoes = {}
        for chave in self.__dict_dfs.keys():
            self.__dict_distribuicoes[chave] = DistribuicoesDataset(self.__dict_dfs[chave], num_div = num_div, frac_cat = frac_cat, unit = unit, autorun = False)
        
        self.__colunas_tratadas = np.array([])
        _, _, self.__colunas = self.__dict_distribuicoes[list(self.__dict_dfs.keys())[0]].info_dataset()
        
    def trata_coluna(self, col_ref = []):
        colunas = self.__colunas
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
        for col_ref in colunas:
            for chave in self.__dict_dfs.keys():
                self.__dict_distribuicoes[chave].trata_coluna(col_ref)
            if(self.__colunas_tratadas.size == 0 or np.sum(np.where(self.__colunas_tratadas == col_ref)) == 0):
                self.__colunas_tratadas = np.append(self.__colunas_tratadas, col_ref)
                
    def grafico_distribuicao(self, col_ref = [], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        colunas = self.__colunas
    
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
    
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in self.__colunas_tratadas):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__dict_dfs.keys():
                        valores, frac, tipo = self.__dict_distribuicoes[chave].curva_distribuicao(col_ref)
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
                    
##############################

##############################

class TendenciasAlvoDataset:

    def __init__(self, df, col_alvo, num_div = None, frac_cat = 0, unit = None, considera_sinal = False, dict_pesos = None):
        
        self.distribuicoes = DistribuicoesDataset(df.drop(col_alvo, axis = 1), num_div = num_div, frac_cat = frac_cat, unit = unit)
        self.__dict_intervs, self.__dict_filtroscat = self.distribuicoes.retorna_trata_dataset().retorna_instancias_tratamento()
        self.__df_tratado = self.distribuicoes.retorna_trata_dataset().aplica_transformacao(df, usar_ponto_medio = False, considera_resto = True, usar_str = False)
        
        self.__y = df[col_alvo].values
        self.__col_alvo = col_alvo
        self.__considera_sinal = considera_sinal
        
        self.__dict_valores = {}
        self.__dict_qtds = {}
        
        self.__dict_soma_alvo = {}
        self.__dict_media_alvo = {}
        self.__dict_impacto = {}
        self.__dict_impacto_por_bit = {}
        
        self.__dict_tendencia = {}
        
        self.__dict_media_norm = {}
        
        #Idealmente, os pesos devem ser as importâncias das variáveis se tivermos um modelo e o alvo for a prob dada pelo modelo
        _, _, colunas = self.distribuicoes.info_dataset()
        colunas = np.array([col for col in colunas if col != self.__col_alvo])
        if(dict_pesos is None):
            self.__dict_pesos = None
        else:
            if(isinstance(dict_pesos, dict) and len(list(set(dict_pesos.keys()) & set(colunas))) == colunas.size):
                peso_maximo = np.max(list(dict_pesos.values()))
                peso_minimo = np.min(list(dict_pesos.values()))
                if(peso_maximo <= 1 and peso_minimo >= 0):
                    self.__peso_max = peso_maximo
                    pesos = list(dict_pesos.values())
                    self.__dict_pesos = dict(zip(dict_pesos.keys(), np.array(pesos)))
                else:
                    print('Lista de pesos não foi passada corretamente')
            else:
                print('Lista de pesos não foi passada corretamente')
        
        self.__calcula_metricas_condicionais()
        self.__calcula_tendencias()
        self.__calcula_medias_normalizadas()
    
    def __calcula_metricas_condicionais(self):
        _, _, colunas = self.distribuicoes.info_dataset()
        colunas = [col for col in colunas if col != self.__col_alvo]
        for col_ref in colunas:
            flag_na = self.distribuicoes.retorna_flag_na(col_ref)
            valores = self.__df_tratado.loc[~flag_na, col_ref].values
            y = self.__y[~flag_na]
            
            inds_ordenado = np.argsort(valores)
            valores_unico, primeira_ocorrencia, qtds = np.unique(valores[inds_ordenado], return_index = True, return_counts = True)
            self.__dict_valores[col_ref] = valores_unico
            self.__dict_qtds[col_ref] = qtds
            
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            soma_alvo = np.array([np.sum(v) for v in y_agrup])
            self.__dict_soma_alvo[col_ref] = soma_alvo
            medias_alvo = soma_alvo/qtds
            self.__dict_media_alvo[col_ref] = medias_alvo
            
            impacto = np.std(medias_alvo)
            self.__dict_impacto[col_ref] = impacto
            if(valores_unico.size > 1):
                self.__dict_impacto_por_bit[col_ref] = impacto/np.log2(valores_unico.size)
            else:
                self.__dict_impacto_por_bit[col_ref] = 0
        
        #Normaliza pelo impacto da coluna alvo no próprio alvo
        std_alvo = np.std(self.__df_tratado[self.__col_alvo])
        for col_ref in colunas:
            self.__dict_impacto[col_ref] = self.__dict_impacto[col_ref]/std_alvo
            self.__dict_impacto_por_bit[col_ref] = self.__dict_impacto_por_bit[col_ref]/std_alvo
            
        #Ordena os Impactos
        self.__dict_impacto = dict(reversed(sorted(self.__dict_impacto.items(), key = lambda x: x[1])))
        self.__dict_impacto_por_bit = dict(reversed(sorted(self.__dict_impacto_por_bit.items(), key = lambda x: x[1])))
        
        self.__impacto_max = list(self.__dict_impacto.values())[0]

    def valores_impactos(self):
        return pd.Series(self.__dict_impacto, index = self.__dict_impacto.keys())

    def valores_impactos_por_bit(self):
        return pd.Series(self.__dict_impacto_por_bit, index = self.__dict_impacto_por_bit.keys())

    def valor_metricas_condicionais(self, col_ref):
        _, _, colunas = self.distribuicoes.info_dataset()
        df = pd.DataFrame()
        if(col_ref in colunas):
            df['Valores'] = self.__dict_valores[col_ref]
            if(col_ref in self.__dict_intervs.keys()):
                df['Labels'] = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            elif(col_ref in self.__dict_filtroscat.keys()):
                df['Labels'] = self.__dict_filtroscat[col_ref].strings_categorias()
            df['QTD'] = self.__dict_qtds[col_ref]
            df['Soma_Alvo'] = self.__dict_soma_alvo[col_ref]
            df['Media_Alvo'] = self.__dict_media_alvo[col_ref]       
        return df
    
    def __curva_medias_condicional(self, col_ref):
        valores = self.__dict_valores[col_ref]
        if(col_ref in self.__dict_intervs.keys()):
            labels = self.__dict_intervs[col_ref].strings_intervalos_discretizacao()
            tipo = 'intervalo'
        elif(col_ref in self.__dict_filtroscat.keys()):
            labels = self.__dict_filtroscat[col_ref].strings_categorias()
            tipo = 'categoria'
        else:
            labels = None
            tipo = 'discreto'
        medias_alvo = self.__dict_media_alvo[col_ref]
        impacto = self.__dict_impacto[col_ref]
        impacto_por_bit = self.__dict_impacto_por_bit[col_ref]
        
        #Ordena por média do alvo se forem categorias (e podemos rearranjar a ordem)
        if(tipo == 'categoria'):
            inds_ordenado = np.argsort(medias_alvo)
            valores = valores[inds_ordenado]
            labels = labels[inds_ordenado]
            medias_alvo = medias_alvo[inds_ordenado]
        
        return valores, labels, tipo, medias_alvo, impacto, impacto_por_bit
    
    def grafico_medias_condicional(self, col_ref = [], conv_str = True, ticks_chars = None, figsize = [6, 4]):
        _, _, colunas = self.distribuicoes.info_dataset()
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in colunas):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    valores, labels, tipo, medias_alvo, impacto, impacto_por_bit = self.__curva_medias_condicional(col_ref)
                    if(labels is not None):
                        if(tipo == 'intervalo'):
                            axs.bar(valores, medias_alvo, color = paleta_cores[0])
                            axs.set_xticks(valores)
                            axs.set_xticklabels(labels, rotation = 90)
                        elif(tipo == 'categoria'):
                            if(ticks_chars is not None):
                                labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                            axs.bar(labels, medias_alvo, color = paleta_cores[0])
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                            if(ticks_chars is not None):
                                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        axs.bar(valores, medias_alvo, color = paleta_cores[0])
                        
                    plt.gcf().text(1, 0.5, 'Impacto = ' + '%.2g' % impacto + '\n' +  'Impacto/Bit = ' + '%.2g' % impacto_por_bit, 
                                   bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                        
                    axs.set_xlabel(col_ref)
                    axs.set_ylabel('Média Alvo')
                    plt.show()
    
    def __calcula_tendencias(self):
    
        #Calculamos a derivada por aproximação de uma parábola (com os 2 pontos mais próximos)
        def calc_diff(vetor):
            if(vetor.size > 2):
                diferenca = np.array([(vetor[i+1] - vetor[i-1])*0.5 for i in range(1, vetor.size-1)])
            else:
                diferenca = np.array([])
            if(vetor.size > 1):
                diferenca = np.insert(diferenca, 0, (vetor[1] - vetor[0])*0.5)
                diferenca = np.append(diferenca, (vetor[-1] - vetor[-2])*0.5)
            else:
                diferenca = np.array([0])
            return diferenca
        
        _, _, colunas = self.distribuicoes.info_dataset()
        colunas = [col for col in colunas if col != self.__col_alvo]
        
        #Calcula um número que resume a tendência e a derivada dos gráficos de média para ver a tendência por valor da variável
        for col_ref in colunas:
            valores, labels, tipo, medias_alvo, impacto, impacto_por_bit = self.__curva_medias_condicional(col_ref)
            self.__dict_tendencia[col_ref] = calc_diff(medias_alvo)
        
        #Normaliza a derivada pelo impacto ou peso (se for passado)
        for col_ref in colunas:
            tend = self.__dict_tendencia[col_ref]
            minimo = np.min(tend)
            maximo = np.max(tend)
            if(maximo > 0 and minimo < 0):
                self.__dict_tendencia[col_ref] = np.array([v/maximo if v > 0 else v/np.abs(minimo) for v in self.__dict_tendencia[col_ref]])
            elif(maximo > 0 and minimo >= 0):
                self.__dict_tendencia[col_ref] = self.__dict_tendencia[col_ref]/maximo
            elif(maximo <= 0 and minimo < 0):
                self.__dict_tendencia[col_ref] = self.__dict_tendencia[col_ref]/np.abs(minimo)
                
            if(self.__dict_pesos is None):
                self.__dict_tendencia[col_ref] = self.__dict_tendencia[col_ref]*self.__dict_impacto[col_ref]/self.__impacto_max
            else:
                self.__dict_tendencia[col_ref] = self.__dict_tendencia[col_ref]*self.__dict_pesos[col_ref]/self.__peso_max
        
    def __curva_tendencia(self, col_ref):
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
        medias_alvo = self.__dict_media_alvo[col_ref]
        tendencia = self.__dict_tendencia[col_ref]
        impacto = self.__dict_impacto[col_ref]
        impacto_por_bit = self.__dict_impacto_por_bit[col_ref]
        
        #Ordena por média do alvo se forem categorias (e podemos rearranjar a ordem)
        if(tipo == 'categoria'):
            inds_ordenado = np.argsort(medias_alvo)
            valores = valores[inds_ordenado]
            labels = labels[inds_ordenado]
            tendencia = tendencia[inds_ordenado]
            medias_alvo = medias_alvo[inds_ordenado]
            
        return valores, labels, tipo, tendencia, impacto, impacto_por_bit
                    
    def grafico_tendencias(self, ticks_chars = None):      
        if(self.__dict_pesos is None):
            colunas = dict(reversed(sorted(self.__dict_impacto.items(), key = lambda x: x[1]))).keys()
        else:
            colunas = dict(reversed(sorted(self.__dict_pesos.items(), key = lambda x: x[1]))).keys()
        colunas = [col for col in colunas if col != self.__col_alvo]
        
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        N = 256
        cor_neg = paleta_cores[0]
        cor_pos = paleta_cores[1]
        vals = np.ones((N, 4))
        regua = np.linspace(-1, 1, N)
        vals[:, 0] = np.array([cor_pos[0] if v > 0 else cor_neg[0] for v in regua])
        vals[:, 1] = np.array([cor_pos[1] if v > 0 else cor_neg[1] for v in regua])
        vals[:, 2] = np.array([cor_pos[2] if v > 0 else cor_neg[2] for v in regua])
        vals[:, 3] = np.array([(v**2)**(1/2) for v in regua]) #Aqui podemos alterar a velocidade com que o alpha muda
        cmap = mpl.colors.ListedColormap(vals)
        cores = cmap(np.arange(cmap.N))
        
        num_cols = len(colunas)
        fig, axs = plt.subplots(num_cols, 1, figsize = [7, 2*num_cols], constrained_layout = True)
        fig.suptitle('Tendência das Variáveis:')
        
        i = 0
        for col_ref in colunas:
            valores, labels, tipo, tendencia, impacto, impacto_por_bit = self.__curva_tendencia(col_ref)
            cores_plot = cores[np.floor((tendencia + 1)*(N-1)/2).astype(int)]
            axs[i].imshow([cores_plot], aspect = 0.5*(valores.size/10), interpolation = 'spline16')
            axs[i].set_yticks([])
            
            if(labels is not None):
                if(tipo == 'intervalo'):
                    axs[i].set_xticks(valores)
                    axs[i].set_xticklabels(labels, rotation = 90)
                elif(tipo == 'categoria'):
                    if(ticks_chars is not None):
                        labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                    axs[i].set_xticks(range(0, valores.size))
                    axs[i].set_xticklabels(labels)
            else:
                axs[i].set_xticks(range(0, valores.size))
                axs[i].set_xticklabels(valores.astype(str))

            if(self.__dict_pesos is None):
                axs[i].set_title(col_ref + ': ' + '%.2g' % impacto, loc = 'left')
            else:
                axs[i].set_title(col_ref + ': ' + '%.2g' % self.__dict_pesos[col_ref], loc = 'left')
                
            i = i + 1
        plt.show()
        
    def __calcula_medias_normalizadas(self):
        _, _, colunas = self.distribuicoes.info_dataset()
        colunas = [col for col in colunas if col != self.__col_alvo]
        
        for col_ref in colunas:
            media = self.__dict_media_alvo[col_ref]
            minimo = np.min(media)
            maximo = np.max(media)
        
            if(self.__considera_sinal):
                if(maximo > 0 and minimo < 0):
                    self.__dict_media_norm[col_ref] = np.array([v/maximo if v > 0 else v/np.abs(minimo) for v in self.__dict_media_alvo[col_ref]])
                elif(maximo > 0 and minimo >= 0):
                    self.__dict_media_norm[col_ref] = self.__dict_media_alvo[col_ref]/maximo
                elif(maximo <= 0 and minimo < 0):
                    self.__dict_media_norm[col_ref] = self.__dict_media_alvo[col_ref]/np.abs(minimo)
            else:
                if(maximo == minimo):
                    self.__dict_media_norm[col_ref] = np.array([0 for v in self.__dict_media_alvo[col_ref]]).astype(float)
                else:
                    self.__dict_media_norm[col_ref] = (2*self.__dict_media_alvo[col_ref] - maximo - minimo)/(maximo - minimo)
            
            if(self.__dict_pesos is None):
                self.__dict_media_norm[col_ref] = self.__dict_media_norm[col_ref]*self.__dict_impacto[col_ref]/self.__impacto_max
            else:
                self.__dict_media_norm[col_ref] = self.__dict_media_norm[col_ref]*self.__dict_pesos[col_ref]/self.__peso_max
                
    def __curva_medias_normalizada(self, col_ref):
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
        medias_alvo = self.__dict_media_alvo[col_ref]
        medias_norm = self.__dict_media_norm[col_ref]
        impacto = self.__dict_impacto[col_ref]
        impacto_por_bit = self.__dict_impacto_por_bit[col_ref]
        
        #Ordena por média do alvo se forem categorias (e podemos rearranjar a ordem)
        if(tipo == 'categoria'):
            inds_ordenado = np.argsort(medias_alvo)
            valores = valores[inds_ordenado]
            labels = labels[inds_ordenado]
            medias_alvo = medias_alvo[inds_ordenado]
            medias_norm = medias_norm[inds_ordenado]
        
        return valores, labels, tipo, medias_norm, impacto, impacto_por_bit
                
    def mapa_calor(self, ticks_chars = None):
        if(self.__dict_pesos is None):
            colunas = dict(reversed(sorted(self.__dict_impacto.items(), key = lambda x: x[1]))).keys()
        else:
            colunas = dict(reversed(sorted(self.__dict_pesos.items(), key = lambda x: x[1]))).keys()
        colunas = [col for col in colunas if col != self.__col_alvo]
        
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        N = 256
        cor_neg = paleta_cores[0]
        cor_pos = paleta_cores[1]
        vals = np.ones((N, 4))
        regua = np.linspace(-1, 1, N)
        vals[:, 0] = np.array([cor_pos[0] if v > 0 else cor_neg[0] for v in regua])
        vals[:, 1] = np.array([cor_pos[1] if v > 0 else cor_neg[1] for v in regua])
        vals[:, 2] = np.array([cor_pos[2] if v > 0 else cor_neg[2] for v in regua])
        vals[:, 3] = np.array([(v**2)**(1/2) for v in regua]) #Aqui podemos alterar a velocidade com que o alpha muda
        cmap = mpl.colors.ListedColormap(vals)
        cores = cmap(np.arange(cmap.N))
        
        num_cols = len(colunas)
        fig, axs = plt.subplots(num_cols, 1, figsize = [7, 2*num_cols], constrained_layout = True)
        fig.suptitle('Mapa de Calor das Variáveis:')
        
        i = 0
        for col_ref in colunas:
            valores, labels, tipo, medias_norm, impacto, impacto_por_bit = self.__curva_medias_normalizada(col_ref)
            cores_plot = cores[np.floor((medias_norm + 1)*(N-1)/2).astype(int)]
            axs[i].imshow([cores_plot], aspect = 0.5*(valores.size/10), interpolation = 'spline16')
            axs[i].set_yticks([])
            
            if(labels is not None):
                if(tipo == 'intervalo'):
                    axs[i].set_xticks(valores)
                    axs[i].set_xticklabels(labels, rotation = 90)
                elif(tipo == 'categoria'):
                    if(ticks_chars is not None):
                        labels = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in labels]
                    axs[i].set_xticks(range(0, valores.size))
                    axs[i].set_xticklabels(labels)
            else:
                axs[i].set_xticks(range(0, valores.size))
                axs[i].set_xticklabels(valores.astype(str))
            
            if(self.__dict_pesos is None):
                axs[i].set_title(col_ref + ': ' + '%.2g' % impacto, loc = 'left')
            else:
                axs[i].set_title(col_ref + ': ' + '%.2g' % self.__dict_pesos[col_ref], loc = 'left')
                
            i = i + 1
        plt.show()
    
    def calcula_ordenacao_indices_dataset(self, df):
        colunas = list(set(list(df.columns)) & set([col for col in self.__dict_impacto.keys() if col != self.__col_alvo]))
        df_calor = self.distribuicoes.retorna_trata_dataset().aplica_transformacao(df[colunas], usar_ponto_medio = False, considera_resto = True, usar_str = False)
        df_calor = df_calor[colunas]
        for col in colunas:
            valores, labels, tipo, medias_norm, impacto, impacto_por_bit = self.__curva_medias_normalizada(col)
            df_calor[col] = df_calor[col].replace({valores[i]: medias_norm[i] for i in range(0, valores.size)})
        df_calor['Score'] = df_calor.mean(axis = 1, skipna = True)
        indices_ordenados = list(df_calor.sort_values('Score', ascending = False).index)
        return indices_ordenados
    
    def aplica_mapa_calor_dataset(self, df):
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        N = 256
        cor_neg = paleta_cores[0]
        cor_pos = paleta_cores[1]
        vals = np.ones((N, 4))
        regua = np.linspace(-1, 1, N)
        vals[:, 0] = np.array([cor_pos[0] if v > 0 else cor_neg[0] for v in regua])
        vals[:, 1] = np.array([cor_pos[1] if v > 0 else cor_neg[1] for v in regua])
        vals[:, 2] = np.array([cor_pos[2] if v > 0 else cor_neg[2] for v in regua]) 
        vals[:, 3] = np.array([(v**2)**(1/2) for v in regua]) #Aqui podemos alterar a velocidade com que o alpha muda
        cmap = mpl.colors.ListedColormap(vals)
        cores = cmap(np.arange(cmap.N))
        
        colunas = set(list(df.columns)).intersection(set([col for col in self.__dict_impacto.keys() if col != self.__col_alvo]))
        colunas_ausente = [col for col in df.columns if col not in colunas]
        df_calor = self.distribuicoes.retorna_trata_dataset().aplica_transformacao(df[colunas], usar_ponto_medio = False, considera_resto = True, usar_str = False)
        for col in colunas:
            valores, labels, tipo, medias_norm, impacto, impacto_por_bit = self.__curva_medias_normalizada(col)
            cores_col = cores[np.floor((medias_norm + 1)*(N-1)/2).astype(int)]
            cores_col = np.rint(255*cores_col).astype(int)
            cores_col_hex = ['background-color: ' + '#{:02x}{:02x}{:02x}{:02x}'.format(*cores_col[i,:]) + '; opacity: 1.0' for i in range(0, cores_col.shape[0])] #Guarda a cor em hexadecimal (com o alpha)
            mapa = dict(zip(valores, cores_col_hex))
            df_calor = df_calor.replace({col: mapa})
            df_calor = df_calor.fillna('background-color: black')
        if(len(colunas_ausente) > 0):
            for col in list(colunas_ausente):
                df_calor[colunas_ausente] = np.nan
        df_calor = df_calor.loc[:, df.columns]
        df_calor = df_calor.fillna('background-color: white')
        
        def apply_color(x):
            return df_calor
        styles = [dict(selector="th", props=[("background-color", "white")]),
                  dict(selector="tr", props=[("background-color", "white")])]
        styler = df.style.set_table_styles(styles).apply(apply_color, axis = None)
        return styler
        