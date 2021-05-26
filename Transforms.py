import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from itertools import combinations_with_replacement, combinations

#TEM ALGUM BUG QUE FICA INTERVALO VAZIO E NÃO SEI AINDA QUANDO ACONTECE
#OBS: Tem que ser um tipo que possa ser convertido para float
#OBS: Não pode ter NA, mas pode ter -Inf e +Inf
class CortaIntervalosQuasiUniforme:

    def __init__(self, vetor, num_div, eh_temporal = False, unit = None):
        
        #OBS: Precisa estar em float o vetor para funcionar direito (se não ele força as contas a retornar inteiro)
        self.__eh_temporal = eh_temporal
        self.__unit = unit
        if(eh_temporal == True):
            self.__vetor = np.array([float(v) for v in vetor]).astype(float)
        else:
            self.__vetor = vetor.astype(float) #vetor que queremos dividir em intervalos
        
        self.__qtd_tot = vetor.size #Tamanho do vetor que vamos discretizar
        self.__num_div = num_div #número de divisões que vamos tentar fazer
        
        self.__vetor_disc = None #Vetor discretizado
        self.__pontos_corte = None #Armazena os pontos de corte (min,max] obtidos na discretização por quantils
        
        self.__qtds = None #Quantidade de elementos por intervalo da discretização
        self.__min_ajust = None #Armazena a informacao de menor valor no intervalo ajustado
        self.__max_ajust = None #Armazena a informacao de maior valor no intervalo ajustado
        self.__qtds_por_compr = None #Quantidade de elementos por comprimento do intervalo (mede concentração)
        self.__densidade_valores = None #Fração de elementos do total do vetor por comprimento do intervalo (mede densidade)
        #OBS: Ajustado significa que ajustamos para que o final de um intervalo seja o início do outro (sem buracos) de forma simétrica
        
        self.__alga_signif = None #Armazena o melhor valor de algarismos significativos para diferenciar os intervalos
        self.__min_signif = None #Armazena a informacao de menor valor no intervalo com melhor alga_signif
        self.__max_signif = None #Armazena a informacao de maior valor no intervalo com melhor alga_signif
        #OBS: Vamos usar esses valores para criar as strings (a, b] que representa os intervalos
        
        self.__strings_intervalos = None #Strings sugeridas para identificar cada intervalo
        self.__pontos_medios = None #Pontos médios dos intervalos da discretização
        self.__pares_minimo_maximo = None #Vetor com os pares [min, max] de cada intervalo
        
        self.__calcula_intervalos()
        
    def __calcula_discretizacao(self):
        #Pega as divisões por quantis
        quantils = np.linspace(0, 100, self.__num_div + 1)
        valores_divisao = np.asarray(np.percentile(self.__vetor, quantils, interpolation = 'nearest'))
        
        #Encontra a primeira e última ocorrência de cada divisor de quantil
        valores_unicos, primeira_ocorrencia = np.unique(valores_divisao, return_index = True)
        valores_unicos, ultima_ocorrencia = np.unique(valores_divisao[::-1], return_index = True)
        ultima_ocorrencia = valores_divisao.size - 1 - ultima_ocorrencia
        
        #checa os divisores que aparecem mais de uma vez (significa que eles sozinhos já formam um Intervalo)
        flag_prim_igual_ult = (primeira_ocorrencia != ultima_ocorrencia).astype(int) + 1
        #Cria os intervalos considerando os intervalos que podem conter só um valor
        valores_divisao = np.array([valores_unicos[i] for i in range(valores_unicos.size) for j in range(flag_prim_igual_ult[i])])
        
        vetor_unicos = np.unique(self.__vetor) #Pega os valores únicos do vetor que estamos discretizando
        diff_zero = np.append(np.diff(valores_divisao) == 0, False) #Flag com as posições em que o próximo elemento é igual ao atual (intervalos com um só valor)
        
        #Simetriza os intervalos colocando o corte no meio do "vácuo de valores"
        flag_simetrizar = np.append(np.insert(~diff_zero[1:-1], 0, False), False) #Não tem pq fazer isso nas pontas (nem no inicio de um intervalo de um só valor)
        #Encontramos o valor superior ao corte por busca binária
        valores_sup = np.array([])
        vetor_unicos_aux = vetor_unicos.copy()
        for valor in valores_divisao[flag_simetrizar]:
            ind_sup = np.searchsorted(vetor_unicos_aux, valor) + 1
            valores_sup = np.append(valores_sup, vetor_unicos_aux[min(ind_sup, vetor_unicos_aux.size - 1)])
            vetor_unicos_aux = vetor_unicos_aux[ind_sup:] #Já descartamos os valores do vetor que não tem mais pq buscar
        valores_divisao[flag_simetrizar] = (valores_divisao[flag_simetrizar] + valores_sup)/2 #Ajustamos o corte para o meio dos dois valores
        
        #Precisamos encontrar os intervalos com só um valor e simetrizar o inicio (que antes foi desconsiderado na simetrização)
        flag_simetrizar = np.append(np.insert(diff_zero[1:-1], 0, False), False) #Novamente, não tem necessidade de fazer isso nas pontas
        #Encontramos o valor inferior ao corte por busca binária
        valores_inf = np.array([])
        vetor_unicos_aux = vetor_unicos.copy()
        for valor in valores_divisao[flag_simetrizar]:
            ind_inf = np.searchsorted(vetor_unicos_aux, valor) - 1
            valores_inf = np.append(valores_inf, vetor_unicos_aux[max(ind_inf, 0)])
            vetor_unicos_aux = vetor_unicos_aux[ind_inf:] #Já descartamos os valores do vetor que não tem mais pq buscar
        valores_divisao[flag_simetrizar] = (valores_inf + valores_divisao[flag_simetrizar])/2 #Ajustamos o corte para o meio dos dois valores
        
        #Remove valores de corte repetidos caso tenha aparecido algum após a simetrização
        if(valores_divisao.size > 2): #Só é necessário se tivermos mais de um intervalo: para não dar problema se só tiver um intervalo (0, 0], por exemplo
            valores_divisao = np.unique(valores_divisao) 
            
        self.__pontos_corte = valores_divisao #Guarda os pontos de corte
        self.__vetor_disc = np.digitize(self.__vetor, self.__pontos_corte[1:], right = True) #Faz a discretização pelos pontos de corte
    
    def __calcula_info_discretizacao(self):
        #Calcula as informações da discretização feita
        #1)Quantidade de elementos em cada intervalo
        #2)Os valores de mínimo e máximo do começo e final de cada intervalo
        #3)Densidade/Concentração de valores em cada intervalo
        
        ind_unico, qtds = np.unique(self.__vetor_disc, return_counts = True)
        self.__qtds = np.zeros(self.__pontos_corte.size - 1)
        self.__qtds[ind_unico] = qtds
        
        #Remove intervalos que ficaram vazios
        flag_zeros = self.__qtds == 0
        if(np.sum(flag_zeros) > 0):
            self.__pontos_corte = np.append(self.__pontos_corte[0], self.__pontos_corte[1:][~flag_zeros])
            self.__vetor_disc = np.digitize(self.__vetor, self.__pontos_corte[1:], right = True)
            ind_unico, qtds = np.unique(self.__vetor_disc, return_counts = True)
            self.__qtds = np.zeros(self.__pontos_corte.size - 1)
            self.__qtds[ind_unico] = qtds
        
        self.__min_ajust = self.__pontos_corte[:-1]
        self.__max_ajust = self.__pontos_corte[1:]
        
        L = self.__max_ajust - self.__min_ajust
        self.__qtds_por_compr = np.zeros(self.__pontos_corte.size - 1)
        flag_L0_Inf = (L == 0)&(self.__qtds > 0)
        flag_L0_Zero = (L == 0)&(self.__qtds == 0)
        flag_resto = (~flag_L0_Inf)&(~flag_L0_Zero)
        self.__qtds_por_compr[flag_L0_Inf] = np.inf
        self.__qtds_por_compr[flag_L0_Zero] = 0
        self.__qtds_por_compr[flag_resto] = self.__qtds[flag_resto]/L[flag_resto]
        self.__densidade_valores = self.__qtds_por_compr/self.__qtd_tot
        
    def __calcula_melhor_alga_signif(self):
        #Calcula o melhor algarismo significativo para utilizar na string (a,b] do intervalo
        #Defiminos o melhor algarismo significativo como o menor valor que consegue expressar todas as separações
        
        #Se os intervalos forem todos distintos
        if(np.sum(np.diff(self.__pontos_corte)) > 0):
            alga_signif = 1
            str_conv = '%.' + str(alga_signif) + 'g'
            cortes_interv = np.array([float(str_conv%self.__pontos_corte[i]) for i in range(self.__pontos_corte.size)])
            if(cortes_interv.size > 1): #Se houver mais de um corte de intervalos (ou seja, mais de 2 intervalos)
                flag = np.min(np.diff(cortes_interv)) #A diferença entre os pontos de corte não pode dar zero (não pode ter pontos de corte iguais)
                while flag == 0:
                    alga_signif += 1
                    str_conv = '%.' + str(alga_signif) + 'g'
                    cortes_interv = np.array([float(str_conv%self.__pontos_corte[i]) for i in range(self.__pontos_corte.size)])
                    flag = np.min(np.diff(cortes_interv))
        #Exemplo: self.__pontos_corte = [1.5, 1.5] -> só tem um intervalo e todos os elementos tem o mesmo valor
        else:
            alga_signif = 1
            str_conv = '%.' + str(alga_signif) + 'g'
            cortes_interv = np.array([float(str_conv%self.__pontos_corte[i]) for i in range(self.__pontos_corte.size)])
        
        self.__alga_signif = alga_signif
        self.__min_signif = cortes_interv[:-1]
        self.__max_signif = cortes_interv[1:]
    
    def __calcula_strings_intervalos(self):
        min_str = [str(v) for v in self.__min_signif]
        max_str = [str(v) for v in self.__max_signif]
        min_str = [v if v[-2:] != '.0' else v[:-2] for v in min_str]
        max_str = [v if v[-2:] != '.0' else v[:-2] for v in max_str]
        self.__strings_intervalos = np.array(['('+min_str[i]+', '+max_str[i]+']' for i in range(self.__qtds.size)])
    
    def __calcula_intervalos(self):
        self.__calcula_discretizacao()
        self.__calcula_info_discretizacao()
        self.__calcula_melhor_alga_signif()
        self.__calcula_strings_intervalos()
        self.__pontos_medios = (self.__min_ajust + self.__max_ajust)/2
        
        if(self.__eh_temporal):
            self.__min_ajust = self.__min_ajust.astype('<M8[ns]')
            self.__max_ajust = self.__max_ajust.astype('<M8[ns]')
            self.__pontos_medios = self.__pontos_medios.astype('<M8[ns]')
            self.__min_signif = self.__min_ajust
            self.__max_signif = self.__max_ajust
            if(self.__unit is None):
                self.__calcula_strings_intervalos()
            else:
                min_str = [np.datetime_as_string(v, unit = self.__unit) for v in self.__min_signif]
                max_str = [np.datetime_as_string(v, unit = self.__unit) for v in self.__max_signif]
                self.__strings_intervalos = np.array(['('+min_str[i]+', '+max_str[i]+']' for i in range(self.__qtds.size)])
        
        self.__pares_minimo_maximo = np.array(list(zip(self.__min_ajust, self.__max_ajust)))
    
    def vetor_discretizado(self, usar_ponto_medio = False):
        if(usar_ponto_medio):
            return self.__pontos_medios[self.__vetor_disc]
        else:
            return self.__vetor_disc
    
    def pares_minimo_maximo_discretizacao(self):
        return self.__pares_minimo_maximo

    def pontos_medios_discretizacao(self):
        return self.__pontos_medios

    def strings_intervalos_discretizacao(self):
        return self.__strings_intervalos
    
    def info_discretizacao(self):
        #Retorna um dataframe com as informações relevantes de cada intervalo da discretização
        df = pd.DataFrame(zip(self.__qtds, self.__densidade_valores, self.__min_ajust, self.__max_ajust, self.__strings_intervalos), 
                          columns = ['QTD', 'Frac/L', 'Min', 'Max', 'Str'])
        return df
    
    def curva_distribuicao(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        fracL = np.repeat(self.__densidade_valores, 2)
        return valores, fracL
    
    def grafico_distribuicao(self, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            valores, fracL = self.curva_distribuicao()
            axs.fill_between(valores, fracL, color = paleta_cores[0], alpha = 0.5)
            axs.plot(valores, fracL, color = paleta_cores[0])
            axs.set_xlabel('Valores')
            axs.set_ylabel('Fração/L')
            axs.set_ylim(bottom = 0.0)
            plt.show()
    
    def aplica_discretizacao(self, v, usar_ponto_medio = False):
        if(self.__eh_temporal):
            def converte_data(val):
                try:
                    return float(val)
                except:
                    return np.nan
            v = np.array([converte_data(val) for val in v]).astype(float)
        else:
            v = v.astype(float)
    
        #Se for maior que o máximo do último intervalo, também coloca no último intervalo
        max_ajust = self.__max_ajust[-1]
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            v = np.where(v > max_ajust, max_ajust, v)
        disc = np.digitize(v, self.__pontos_corte[1:], right = True)
        
        if(usar_ponto_medio):
            def pega_elemento_vetor(vetor, i):
                try:
                    return vetor[i]
                except:
                    return np.nan
            return np.array([pega_elemento_vetor(self.__pontos_medios, i) for i in disc])
        else:
            disc = disc.astype(float)
            flag_na = disc == self.__qtds.size
            if(np.sum(flag_na) > 0):
                disc[flag_na] = np.nan
            return disc

########################
 
########################
 
#OBS: Tem que ser um tipo que possa ser convertido para string
#OBS: Não pode ter NA
class FiltraCategoriasRelevantes:
 
    def __init__(self, vetor, frac_cat):
        self.__vetor = vetor.astype(str) #vetor que queremos filtrar as categorias
        
        self.__qtd_tot = vetor.size #Tamanho do vetor que vamos filtrar
        self.__frac_cat = frac_cat #fração mínima da categoria no vetor para ser considerada relevante
        
        self.__vetor_filtrado = None #Vetor filtrado
        self.__dict_cat = None #Dicionário para aplicação do filtro
        self.__vetor_cat = None #Vetor para aplicação do filtro
        
        self.__cats = None #Lista de todas as categorias encontradas (ordenado por quantidade na base)
        self.__qtds = None #Quantidade de elementos por categoria
        self.__frac_cats = None #Fração de cada categoria no vetor
        self.__flag_rel = None #Marca se a categoria foi considerada relevante ou não
        
        self.__cats_rel = None #Vetor com as categorias relevantes (em ordem de relevância)
        self.__cats_rest = None #Vetor com todas as categorias que foram separadas como "restante" (última categoria)
        self.__num_cats_rel = None #Número de categorias relevantes
        self.__num_cats_rest = None #Número de categorias relevantes
        
        self.__calcula_categorias_relevantes()
    
    def __calcula_quantidade_categorias(self):
        self.__cats, self.__qtds = np.unique(self.__vetor, return_counts = True)
        inds_ordenado = np.argsort(self.__qtds)[::-1]
        self.__cats = self.__cats[inds_ordenado]
        self.__qtds = self.__qtds[inds_ordenado]
        self.__frac_cats = self.__qtds/self.__qtd_tot
        
    def __filtra_categorias_relevantes(self):
        self.__flag_rel = self.__frac_cats >= self.__frac_cat
        self.__cats_rel = self.__cats[self.__flag_rel]
        self.__cats_rest = self.__cats[~self.__flag_rel]
        self.__num_cats_rel = self.__cats_rel.size
        self.__num_cats_rest = self.__cats_rest.size
    
    def __cria_dicionario_aplicacao(self):
        dict_cats = dict(zip(self.__cats_rel, np.arange(0, self.__num_cats_rel)))
        if(self.__num_cats_rest > 0):
            dict_cat_rest = dict(zip(self.__cats_rest, np.repeat(self.__num_cats_rel, self.__num_cats_rest)))
            dict_cats.update(dict_cat_rest)
        self.__dict_cat = dict_cats
        
    def __cria_vetor_aplicacao_str(self):
        if(self.__num_cats_rest > 1):
            vetor_cats = np.append(self.__cats_rel, 'Resto')
        elif(self.__num_cats_rest == 1):
            vetor_cats = np.append(self.__cats_rel, self.__cats_rest)
        else:
            vetor_cats = self.__cats_rel
        self.__vetor_cat = vetor_cats
    
    def __calcula_categorias_relevantes(self):
        self.__calcula_quantidade_categorias()
        self.__filtra_categorias_relevantes()
        self.__cria_dicionario_aplicacao()
        self.__cria_vetor_aplicacao_str()
    
    def strings_categorias(self):
        return self.__vetor_cat
    
    def info_categorias(self):
        if(self.__num_cats_rest > 0):
            qtds_filtro = np.append(self.__qtds[self.__flag_rel], np.sum(self.__qtds[~self.__flag_rel]))
            frac_filtro = qtds_filtro/self.__qtd_tot
            flag_resto = np.append(np.repeat(False, self.__num_cats_rel), True)
            categorias_filtro = np.append(self.__cats_rel, ', '.join(self.__cats_rest))
            df = pd.DataFrame(zip(qtds_filtro, frac_filtro, flag_resto, categorias_filtro), 
                              columns = ['QTD', 'Fração', 'Flag_Resto', 'Categoria'])
        else:
            flag_resto = np.repeat(False, self.__num_cats_rel)
            df = pd.DataFrame(zip(self.__qtds, self.__frac_cats, flag_resto, self.__cats), 
                              columns = ['QTD', 'Fração', 'Flag_Resto', 'Categoria'])
        return df
    
    def curva_distribuicao(self):
        valores = self.__vetor_cat
        frac = self.__frac_cats[self.__flag_rel]
        if(self.__num_cats_rest > 0):
            frac = np.append(frac, 1-np.sum(frac))
        return valores, frac
    
    def grafico_distribuicao(self, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            valores, frac = self.curva_distribuicao()
            axs.bar(valores, frac, color = paleta_cores[0], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[0])
            axs.set_xlabel('Valores')
            axs.set_ylabel('Fração')
            axs.set_ylim(bottom = 0.0)
            plt.show()
    
    def aplica_filtro_categorias(self, vetor, considera_resto = True, usar_str = False):
        #vetor = vetor.astype(str)
        def pega_valor_dicionario(dicionario, v):
            try:
                return dicionario[str(v)]
            except:
                return np.nan
        vetor_filtrado = np.array([pega_valor_dicionario(self.__dict_cat, v) for v in vetor]).astype(float)
        if(considera_resto == False):
            if(self.__num_cats_rest > 0):
                vetor_filtrado[vetor_filtrado == self.__num_cats_rel] = np.nan
        if(usar_str):
            def pega_elemento_vetor(vetor, i):
                try:
                    return vetor[int(i)]
                except:
                    return np.nan
            vetor_filtrado = [pega_elemento_vetor(self.__vetor_cat, i) for i in vetor_filtrado]
            #vetor_filtrado = np.array([pega_elemento_vetor(self.__vetor_cat, i) for i in vetor_filtrado], dtype = str) #Problema com os NaN
        return vetor_filtrado
  
 ########################
 
 ########################

class TrataDataset:
 
    def __init__(self, df, num_div = 20, frac_cat = 0.05, features_numericas = None, features_categoricas = None, features_temporais = None, unit = None, autorun = True):
        self.__df = df
        self.__num_div = num_div
        self.__frac_cat = frac_cat
        self.__features_numericas = features_numericas
        self.__features_categoricas = features_categoricas
        self.__features_temporais = features_temporais
        self.__autorun = autorun
        self.__unit = unit
        
        self.__features_numericas_tratadas = np.array([])
        self.__features_temporais_tratadas = np.array([])
        self.__dict_intervs = {}
        self.__dict_filtroscat = {}
        
        if(self.__autorun):
            if(isinstance(self.__features_numericas, list) and self.__num_div != None):
                for feature in self.__features_numericas:
                    if(np.unique(self.__df[feature].dropna().values).size > self.__num_div):
                        self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(self.__df[feature].dropna().values, num_div = self.__num_div)
                        self.__features_numericas_tratadas = np.append(self.__features_numericas_tratadas, feature)
            
            if(isinstance(self.__features_temporais, list) and self.__num_div != None):
                for feature in self.__features_temporais:
                    if(np.unique(self.__df[feature].dropna().values).size > self.__num_div):
                        self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(self.__df[feature].dropna().values, num_div = self.__num_div, 
                                                                                    eh_temporal = True, unit = self.__unit)
                        self.__features_temporais_tratadas = np.append(self.__features_temporais_tratadas, feature)
            
            if(isinstance(self.__features_categoricas, list) and self.__frac_cat != None):
                for feature in self.__features_categoricas:
                    self.__dict_filtroscat[feature] = FiltraCategoriasRelevantes(self.__df[feature].dropna().values, frac_cat = self.__frac_cat)
        
        self.__encoder = None
        self.__considera_resto_ohe = None
        self.__usar_str_ohe = None
    
    def trata_coluna(self, feature):
        if(isinstance(self.__features_numericas, list) and self.__num_div != None and feature in self.__features_numericas):
            if(np.unique(self.__df[feature].dropna().values).size > self.__num_div):
                self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(self.__df[feature].dropna().values, num_div = self.__num_div)
                if(self.__features_numericas_tratadas.size == 0 or np.sum(np.where(self.__features_numericas_tratadas == feature)) == 0):
                    self.__features_numericas_tratadas = np.append(self.__features_numericas_tratadas, feature)
                    
        if(isinstance(self.__features_temporais, list) and self.__num_div != None and feature in self.__features_temporais):
            if(np.unique(self.__df[feature].dropna().values).size > self.__num_div):
                self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(self.__df[feature].dropna().values, num_div = self.__num_div, 
                                                                            eh_temporal = True, unit = self.__unit)
                if(self.__features_temporais_tratadas.size == 0 or np.sum(np.where(self.__features_temporais_tratadas == feature)) == 0):
                    self.__features_temporais_tratadas = np.append(self.__features_temporais_tratadas, feature)
                    
        if(isinstance(self.__features_categoricas, list) and self.__frac_cat != None and self.__frac_cat != None and feature in self.__features_categoricas):
            self.__dict_filtroscat[feature] = FiltraCategoriasRelevantes(self.__df[feature].dropna().values, frac_cat = self.__frac_cat)
    
    def retorna_instancias_tratamento(self):
        return self.__dict_intervs, self.__dict_filtroscat
        
    def cria_one_hot_encoder(self, considera_resto = True, usar_str = False):
        if(isinstance(self.__features_categoricas, list)):
            if(self.__frac_cat != None):
                lista_cats = []
                for feature in self.__features_categoricas:
                    transf = self.__dict_filtroscat[feature].aplica_filtro_categorias(self.__df[feature].values, considera_resto = considera_resto, usar_str = usar_str)
                    lista_cats.append(transf)
                df_cat = pd.DataFrame(zip(*lista_cats), columns = self.__features_categoricas, index = self.__df.index)
            else:
                df_cat = self.__df[self.__features_categoricas]
            valores = df_cat.values
            if(usar_str):
                flag_na = pd.isna(df_cat).values
                valores[flag_na] = '-1'
            else:
                flag_na = np.isnan(valores, where = True)
                valores[flag_na] = -1
                valores = valores.astype(int)
            self.__encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False).fit(valores)
            self.__considera_resto_ohe = considera_resto
            self.__usar_str_ohe = usar_str
    
    def aplica_transformacao(self, df_inp, usar_ponto_medio = False, considera_resto = True, usar_str = False):
        df_aplic = df_inp.copy()
        if(isinstance(self.__features_numericas, list) and self.__num_div != None and self.__features_numericas_tratadas.size > 0):
            for feature in self.__features_numericas_tratadas:
                df_aplic[feature] = self.__dict_intervs[feature].aplica_discretizacao(df_aplic[feature].values, usar_ponto_medio = usar_ponto_medio)
        if(isinstance(self.__features_temporais, list) and self.__num_div != None and self.__features_temporais_tratadas.size > 0):
            for feature in self.__features_temporais_tratadas:
                df_aplic[feature] = self.__dict_intervs[feature].aplica_discretizacao(df_aplic[feature].values, usar_ponto_medio = usar_ponto_medio)
        if(isinstance(self.__features_categoricas, list) and self.__frac_cat != None):
            for feature in self.__features_categoricas:
                df_aplic[feature] = self.__dict_filtroscat[feature].aplica_filtro_categorias(df_aplic[feature].values, 
                                                                                           considera_resto = considera_resto, usar_str = usar_str)                                                                                  
        return df_aplic
        
    def aplica_transformacao_ohe(self, df_inp, usar_ponto_medio = False):
        df_aplic = df_inp.copy()
        df_aplic = self.aplica_transformacao(df_inp, usar_ponto_medio = usar_ponto_medio, 
                                             considera_resto = self.__considera_resto_ohe, usar_str = self.__usar_str_ohe)
                                             
        if(isinstance(self.__features_categoricas, list)):
            valores = df_aplic[self.__features_categoricas].values
            if(self.__usar_str_ohe):
                flag_na = pd.isna(df_aplic[self.__features_categoricas]).values
                valores[flag_na] = '-1'
            else:
                flag_na = np.isnan(valores, where = True)
                valores[flag_na] = -1
                valores = valores.astype(int)
            matriz_cat = self.__encoder.transform(valores)
            nome_colunas = self.__encoder.get_feature_names(self.__features_categoricas)
            df_cat = pd.DataFrame(matriz_cat, columns = nome_colunas, index = df_aplic.index)
            df_cat = df_cat.drop([col for col in nome_colunas if col[-3:] == '_-1'], axis = 1).astype(int)
            df_aplic = df_aplic.drop(self.__features_categoricas, axis = 1)
            df_aplic = pd.concat([df_aplic, df_cat], axis = 1)
            
        return df_aplic
        
 ########################
 
 ########################

#Faz as contas com o Dataset em Pandas e retorna o Dataset em Pandas com tudo calculado
#Ou seja, ocupa memória
class TaylorLaurentSeries:

    def __init__(self, laurent = False, ordem = 2, apenas_interacoes = False, features_numericas = None):
        self.__laurent = laurent
        self.__apenas_interacoes = apenas_interacoes
        self.__features_numericas = features_numericas
        if(self.__apenas_interacoes):
            self.__ordem = min(ordem, len(self.__features_numericas))
        else:
            self.__ordem = ordem
        
        if(laurent == False):
            self.__lista_features = [self.__features_numericas]
        else:
            #Para evitar de fazer a conta (1/x)*x = 1 ou (1/x)(1/x)*x = x
            combs = [[]]
            for feature in self.__features_numericas:
                combs_temp = combs.copy()
                for var in combs_temp:
                    v_new1 = var.copy()
                    v_new2 = var.copy()
                    v_new1.append(feature)
                    v_new2.append('1/' + feature)
                    combs.pop(0)
                    combs.append(v_new1)
                    combs.append(v_new2)
            self.__lista_features = combs
        
        #Pega todas as colunas que vamos ter que calcular, removendo as repetições
        self.__lista_combs = []
        for i in range(2, self.__ordem + 1):
            conjunto_combs = set() #Remove as combinações repetidas no update por sem um conjunto
            for features in self.__lista_features:
                if(self.__apenas_interacoes):
                    #Não precisa de potencias das features
                    comb = set(combinations(features, r = i))
                else:
                    comb = set(combinations_with_replacement(features, r = i))
                conjunto_combs.update(comb)
            conjunto_combs = list(conjunto_combs)
            if(self.__apenas_interacoes):
                #remove as combinações que são inversos multiplicativos
                def inverso_multiplicativo(tupla):
                    return tuple(v[2:] if v[:2] == '1/' else '1/' + v for v in tupla)
                conjunto_filtrado = []
                while(len(conjunto_combs) > 1):
                    tupla = conjunto_combs[0]
                    conjunto_filtrado.append(tupla)
                    conjunto_combs.remove(tupla)
                    try:
                        conjunto_combs.remove(inverso_multiplicativo(tupla))
                    except:
                        pass
                if(len(conjunto_combs) == 1):
                    conjunto_filtrado.append(conjunto_combs[0])
                conjunto_combs = conjunto_filtrado
                #Inverte quando todos as features da tupla estão invertidas
                def eh_tudo_inverso(tupla):
                    return np.sum(np.array([False if v[:2] == '1/' else True for v in tupla])) == 0
                conjunto_combs = [inverso_multiplicativo(v) if eh_tudo_inverso(v) else v for v in conjunto_combs]
            self.__lista_combs.append(conjunto_combs) #Popula a lista de combinações por ordem de número de features (posição 0 => pares 2 a 2)
            
        self.__variaveis_criadas = []
        if(self.__laurent and self.__apenas_interacoes == False):
            self.__variaveis_criadas.extend(['1/' + v for v in self.__features_numericas])
        for combs in self.__lista_combs:
            self.__variaveis_criadas.extend([str(col).replace("'","") for col in combs])
            
    def variaveis_criadas(self):
        return self.__variaveis_criadas
                
    def aplica_transformacao(self, df):
        def inverso_multiplicativo(tupla):
            tupla_inverso = []
            for v in tupla:
                if(v[:2] == '1/'):
                    tupla_inverso.append(v[2:])
                else:
                    tupla_inverso.append('1/' + v)
            return tuple(tupla_inverso)
        
        X = df[self.__features_numericas].copy()
        if(self.__laurent):
            X_inv = (1/X).add_prefix('1/')
            
        if(self.__apenas_interacoes or self.__laurent == False):
            X_res = pd.DataFrame(index = df.index)
        else:
            X_res = X_inv.copy()
        
        if(self.__laurent):
            for i in range(0, len(self.__lista_combs)):
                X_temp = pd.DataFrame(index = df.index)
                if(i == 0):
                    for comb in self.__lista_combs[i]:
                        if(comb[0][:2] == '1/' and comb[1][:2] == '1/'):
                            X_temp[comb] = X_inv[comb[0]].values * X_inv[comb[1]].values
                        elif(comb[0][:2] == '1/'):
                            X_temp[comb] = X_inv[comb[0]].values * X[comb[1]].values
                        elif(comb[1][:2] == '1/'):
                            X_temp[comb] = X[comb[0]].values * X_inv[comb[1]].values
                        else:
                            X_temp[comb] = X[comb[0]].values * X[comb[1]].values
                else:
                    for comb in self.__lista_combs[i]:
                        if(comb[0][:2] == '1/'):
                            try:
                                X_temp[comb] = X_inv[comb[0]].values * X_ant[comb[1:]].values
                            except:
                                X_temp[comb] = X_inv[comb[0]].values * X_ant[inverso_multiplicativo(comb[1:])].values
                        else:
                            try:
                                X_temp[comb] = X[comb[0]].values * X_ant[comb[1:]].values
                            except:
                                X_temp[comb] = X[comb[0]].values * X_ant[inverso_multiplicativo(comb[1:])].values
                X_res = pd.concat([X_res, X_temp], axis = 1, sort = False)
                X_ant = X_temp
        else:
            for i in range(0, len(self.__lista_combs)):
                X_temp = pd.DataFrame(index = df.index)
                if(i == 0):
                    for comb in self.__lista_combs[i]:
                        X_temp[comb] = X[comb[0]].values * X[comb[1]].values
                else:
                    for comb in self.__lista_combs[i]:
                        X_temp[comb] = X[comb[0]].values * X_ant[comb[1:]].values
                X_res = pd.concat([X_res, X_temp], axis = 1, sort = False)
                X_ant = X_temp
        
        X_res.columns = self.__variaveis_criadas
        X_res = pd.concat([df, X_res], axis = 1, sort = False)
        return X_res