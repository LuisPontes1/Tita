import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#OBS: Precisa estar em float o vetor para funcionar direito (se não ele força as contas a retornar inteiro)
#OBS: Não pode ter NA, mas pode ter -Inf e +Inf
class CortaIntervalosQuasiUniforme:

    def __init__(self, vetor, num_div):
        self.__vetor = vetor #vetor que queremos dividir em intervalos
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
            valores_sup = np.append(valores_sup, vetor_unicos_aux[ind_sup])
            vetor_unicos_aux = vetor_unicos_aux[ind_sup:] #Já descartamos os valores do vetor que não tem mais pq buscar
        valores_divisao[flag_simetrizar] = (valores_divisao[flag_simetrizar] + valores_sup)/2 #Ajustamos o corte para o meio dos dois valores
        
        #Precisamos encontrar os intervalos com só um valor e simetrizar o inicio (que antes foi desconsiderado na simetrização)
        flag_simetrizar = np.append(np.insert(diff_zero[1:-1], 0, False), False) #Novamente, não tem necessidade de fazer isso nas pontas
        #Encontramos o valor inferior ao corte por busca binária
        valores_inf = np.array([])
        vetor_unicos_aux = vetor_unicos.copy()
        for valor in valores_divisao[flag_simetrizar]:
            ind_inf = np.searchsorted(vetor_unicos_aux, valor) - 1
            valores_inf = np.append(valores_inf, vetor_unicos_aux[ind_inf])
            vetor_unicos_aux = vetor_unicos_aux[ind_inf:] #Já descartamos os valores do vetor que não tem mais pq buscar
        valores_divisao[flag_simetrizar] = (valores_inf + valores_divisao[flag_simetrizar])/2 #Ajustamos o corte para o meio dos dois valores
        
        #Remove valores de corte repetidos caso tenha aparecido algum após a simetrização
        if(valores_divisao.size > 2): #Só é necessário se tivermos mais de um intervalo: para não dar problema se só tiver um intervalo (0, 0], por exemplo
            valores_divisao = np.unique(valores_divisao) 
            
        self.__pontos_corte = valores_divisao #Guarda os pontos de corte
        self.__vetor_disc = np.digitize(self.__vetor, valores_divisao[1:], right = True) #Faz a discretização pelos pontos de corte
    
    def __calcula_info_discretizacao(self):
        #Calcula as informações da discretização feita
        #1)Quantidade de elementos em cada intervalo
        #2)Os valores de mínimo e máximo do começo e final de cada intervalo
        #3)Densidade/Concentração de valores em cada intervalo
        
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
        self.__pares_minimo_maximo = np.array(list(zip(self.__min_ajust, self.__max_ajust)))
    
    def vetor_discretizado(self):
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

###################################################

###################################################

#Para funcionar direito, não pode haver nulos em y ou y_prob
class AletricasClassificacao:
    
    def __init__(self, y, y_prob, num_div = None, p_corte = None, p01_corte = [0, 0]):
        self.__y = y
        self.__y_prob = y_prob
        self.__y_prob_inicial = y_prob
        
        #Probabilidades de Corte para Avaliação de Tomada de Decisão
        #OBS: Quando nenhuma prob de corte é passada, 
        #usamos a prob de ganho de informação para a tomada de decisão
        self.__p_corte = p_corte
        self.__p01_corte = np.array(p01_corte)
        
        #Variaveis caso queira fazer as contas por intervalos de prob
        self.__num_div = num_div
        self.__interv = None
        
        self.__qtd_tot = None
        self.__soma_probs = None
        self.__qtd1_tot = None
        self.__qtd0_tot = None
        
        self.__y_prob_unico = None
        self.__qtds = None
        self.__qtds1 = None
        
        self.__qtds_acum = None
        self.__qtds1_acum = None
        self.__qtds0_acum = None
        
        #_c indica o conjunto complementar (o que ainda não foi somado)
        self.__qtds_acum_c = None
        self.__qtds1_acum_c = None 
        self.__qtds0_acum_c = None 
        
        #vp: verdadeiro positivo, p_tot: total de positivos
        #vn: verdadeiro negativo, n_tot: total de negativos
        self.__curva_tvp = None #Armazena a curva de taxa de verdadeiro positivo (vp / p_tot)
        self.__curva_tvn = None #Armazena a curva de taxa verdadeiro negativo (vn / n_tot)
        self.__auc = None
        
        #Curva Operacional de Probabilidades Condicionais (COPC)
        self.__curva_p00 = None
        self.__curva_p11 = None
        
        self.__curva_revoc1 = None #Armazena a curva de revocacao de 1
        self.__curva_revoc0 = None #Armazena a curva de revocacao de 0
        self.__pos_max_dif = None
        self.__ks = None
        
        self.__curva_ig = None #Armazena a curva de ganho de informação
        self.__pos_max_ig = None
        self.__ig = None
        
        self.__liftF_10 = None
        self.__alavF_10 = None
        self.__liftF_20 = None
        self.__alavF_20 = None
        self.__liftV_10 = None
        self.__alavV_10 = None
        self.__liftV_20 = None
        self.__alavV_20 = None
        
        self.__vetor_p0_ig_2d = None
        self.__vetor_p1_ig_2d = None
        self.__vetor_ig_2d = None
        self.__pos_max_ig_2d = None
        self.__ig_2d = None
        
        self.matriz_confusao = None
        self.matriz_confusao_2d = None
        self.__frac_incerto_2d = None
        self.__p00 = None #Prob de ser 0 dado que o modelo disse que é 0
        self.__p11 = None #Prob de ser 1 dado que o modelo disse que é 1
        self.__p00_2d = None #Prob de ser 0 dado que o modelo disse que é 0 (corte com p0 e p1)
        self.__p11_2d = None #Prob de ser 1 dado que o modelo disse que é 1 (corte com p0 e p1)
        self.__acuracia = None
        self.__acuracia_balanceada = None
        self.__acuracia_2d = None 
        self.__acuracia_balanceada_2d = None
        
        self.__calcula_metricas()
        
    def __ordena_probs(self):
        self.__qtd_tot = self.__y.size
        self.__soma_probs = np.sum(self.__y_prob)
        
        if(self.__num_div != None):
            self.__interv = CortaIntervalosQuasiUniforme(self.__y_prob, num_div = self.__num_div)
            self.__y_prob = self.__interv.vetor_discretizado()
        
        inds_ordenado = np.argsort(self.__y_prob)
        self.__y_prob_unico, primeira_ocorrencia, self.__qtds = np.unique(self.__y_prob[inds_ordenado], 
                                                                      return_index = True, return_counts = True)
        y_agrup = np.split(self.__y[inds_ordenado], primeira_ocorrencia[1:])
        self.__qtds1 = np.array([np.sum(v) for v in y_agrup])
        
        self.__qtds_acum = np.cumsum(self.__qtds) 
        self.__qtds1_acum = np.cumsum(self.__qtds1)
        self.__qtds0_acum = self.__qtds_acum - self.__qtds1_acum
        
        self.__qtd1_tot = self.__qtds1_acum[-1]
        self.__qtd0_tot = self.__qtd_tot - self.__qtd1_tot
        
        self.__qtds_acum_c = self.__qtd_tot - self.__qtds_acum
        self.__qtds1_acum_c = self.__qtd1_tot - self.__qtds1_acum
        self.__qtds0_acum_c = self.__qtd0_tot - self.__qtds0_acum
    
    def __calcula_roc(self):
        #Estima a área abaixo da curva por Soma de Riemann
        def area(x,y):
            dx = np.diff(x)
            h = (y[:-1] + y[1:])/2
            A = np.sum(h*dx)
            return A
        
        self.__curva_tnp = self.__qtds0_acum/self.__qtd0_tot
        self.__curva_tvp = self.__qtds1_acum_c/self.__qtd1_tot
        
        #Coloca na mão o valor inicial da curva ROC
        self.__curva_tnp = np.insert(self.__curva_tnp, 0, 0)
        self.__curva_tvp = np.insert(self.__curva_tvp, 0, 1)
            
        self.__auc = area(self.__curva_tnp, self.__curva_tvp)

    def __calcula_copc(self):
        #Curva Operacional de Probabilidades Condicionais
        self.__curva_p00 = self.__qtds0_acum[:-1]/self.__qtds_acum[:-1]
        self.__curva_p11 = self.__qtds1_acum_c[:-1]/self.__qtds_acum_c[:-1]
    
    def __calcula_ks(self):
        self.__curva_revoc0 = self.__qtds0_acum/self.__qtd0_tot
        self.__curva_revoc1 = self.__qtds1_acum/self.__qtd1_tot
        
        curva_dif = self.__curva_revoc0 - self.__curva_revoc1
        self.__pos_max_dif = np.argmax(curva_dif) #Pega as posições em que encontrou o máximo
        
        #Pega o valor máximo (tenta ver se pos_max é um vetor ou um número)
        try:
            self.__ks = curva_dif[self.__pos_max_dif[0]]
        except:
            self.__ks = curva_dif[self.__pos_max_dif]
        #OBS: 
        #self.__ks = np.max(curva_dif[self.__pos_max_dif]) seria mais genérico mas o try nesse caso é mais rápido
    
    def __calcula_lift_alavancagem(self, decrescente = False, frac = 0.5):
        qtd_ref = frac*self.__qtd_tot
        if(decrescente == False):
            pos_ini = np.sum(self.__qtds_acum <= qtd_ref) - 1
            if(self.__qtds_acum[pos_ini] == qtd_ref or pos_ini == self.__qtds_acum.size - 1):
                lift = self.__qtds0_acum[pos_ini]/qtd_ref
                alav = lift/frac
            else:
                qtd_ref_medio = (self.__qtds_acum[pos_ini] + self.__qtds_acum[pos_ini+1])/2
                qtd0_medio = (self.__qtds0_acum[pos_ini] + self.__qtds0_acum[pos_ini+1])/2
                lift = qtd0_medio/self.__qtd0_tot
                alav = lift*self.__qtd_tot/qtd_ref_medio
        else:
            pos_ini = self.__qtds_acum_c.size - np.sum(self.__qtds_acum_c <= qtd_ref)
            if(self.__qtds_acum_c[pos_ini] == qtd_ref or pos_ini == 0):
                lift = self.__qtds1_acum_c[pos_ini]/qtd_ref
                alav = lift/frac
            else:
                qtd_ref_medio = (self.__qtds_acum_c[pos_ini] + self.__qtds_acum_c[pos_ini-1])/2
                qtd1_medio = (self.__qtds1_acum_c[pos_ini] + self.__qtds1_acum_c[pos_ini-1])/2
                lift = qtd1_medio/self.__qtd1_tot
                alav = lift*self.__qtd_tot/qtd_ref_medio
        return lift, alav
                
    def __calcula_ig(self):
        #Calcula a Entropia de Shannon
        def entropia_shannon(p1):
            p0 = 1 - p1
            if p0 == 0 or p1 == 0:
                return 0
            else:
                return -p0*np.log2(p0) - p1*np.log2(p1)
        
        p1 = self.__qtd1_tot/self.__qtd_tot
        entropia_ini = entropia_shannon(p1)

        #O último corte por definição não dá informação nenhuma, então nem faz a conta (por isso o [:-1])
        qtds_acum = self.__qtds_acum[:-1]
        qtds1_acum = self.__qtds1_acum[:-1]
        p1_acum = qtds1_acum/qtds_acum
        entropia_parcial = np.array([entropia_shannon(x) for x in p1_acum])

        qtds_acum_c = self.__qtds_acum_c[:-1]
        qtds1_acum_c = self.__qtds1_acum_c[:-1]
        p1c_acum = qtds1_acum_c/qtds_acum_c
        entropia_parcial_c = np.array([entropia_shannon(x) for x in p1c_acum])

        entropia = (entropia_parcial*qtds_acum + entropia_parcial_c*qtds_acum_c)/self.__qtd_tot
        #Coloca o valor [-1] que removemos no começo do calcula da entropia
        entropia = np.append(entropia, entropia_ini)
        self.__curva_ig = (entropia_ini - entropia)/entropia_ini
        
        self.__pos_max_ig = np.argmax(self.__curva_ig) #Pega as posições em que encontrou o máximo
        #Pega o valor máximo (tenta ver se pos_max é um vetor ou um número)
        try:
            self.__ig = self.__curva_ig[self.__pos_max_ig[0]] 
        except:
            self.__ig = self.__curva_ig[self.__pos_max_ig]
        #OBS: Note que o try except também poderia ser trocado aqui por self.__ig = np.max(self.__curva_ig[self.__pos_max_ig])
        
    def __calcula_ig_2d(self):
        #Calcula a Entropia de Shannon
        def entropia_shannon(p1):
            p0 = 1 - p1
            if p0 == 0 or p1 == 0:
                return 0
            else:
                return -p0*np.log2(p0) - p1*np.log2(p1)
        
        p1_ini = self.__qtd1_tot/self.__qtd_tot
        entropia_ini = entropia_shannon(p1_ini) 
        
        vetor_p0 = np.array([])
        vetor_p1 = np.array([])
        vetor_entropia = np.array([])
        vetor_ig = np.array([])
        #Temos a subtração -1 pois como já discutido, o último corte por definição não trás ganho de informação
        num_loop = self.__y_prob_unico.size-1
        #Subtrai mais um aqui pq queremos garantir que todo o loop tem um intervalo de resto
        for i in range(num_loop-1):
            start_loop2 = i + 1 #O segundo loop começa sempre 1 a frente pq queremos que sobre um intervalo de resto
            vetor_p0 = np.append(vetor_p0, np.repeat(self.__y_prob_unico[i], num_loop - start_loop2))
            qtd_acum = self.__qtds_acum[i]
            qtd1_acum = self.__qtds1_acum[i]
            p1 = qtd1_acum/qtd_acum
            entropia_parcial = entropia_shannon(p1)
            
            entropia_aux = entropia_parcial*qtd_acum/self.__qtd_tot
            
            #Segundo loop implicito nos vetores
            vetor_p1 = np.append(vetor_p1, self.__y_prob_unico[start_loop2:num_loop])
            qtd_acum_c = self.__qtds_acum_c[start_loop2:num_loop]
            qtd1_acum_c = self.__qtds1_acum_c[start_loop2:num_loop]
            p1c = qtd1_acum_c/qtd_acum_c
            entropia_parcial_c = np.array([entropia_shannon(x) for x in p1c])
            
            qtd_resto = self.__qtd_tot - qtd_acum - qtd_acum_c
            qtd1_acum_resto = self.__qtd1_tot - qtd1_acum - qtd1_acum_c
            p1r = qtd1_acum_resto/qtd_resto
            entropia_parcial_r = np.array([entropia_shannon(x) for x in p1r])
            
            entropia = entropia_aux + (entropia_parcial_c*qtd_acum_c + entropia_parcial_r*qtd_resto)/self.__qtd_tot
            vetor_entropia = np.append(vetor_entropia, entropia)
                
        self.__vetor_ig_2d = (entropia_ini - vetor_entropia)/entropia_ini
        self.__vetor_p0_ig_2d = vetor_p0
        self.__vetor_p1_ig_2d = vetor_p1
        
        self.__pos_max_ig_2d = np.argmax(self.__vetor_ig_2d) #Pega as posições em que encontrou o máximo
        try:
            self.__ig_2d = self.__vetor_ig_2d[self.__pos_max_ig_2d[0]] 
        except:
            self.__ig_2d = self.__vetor_ig_2d[self.__pos_max_ig_2d]
    
    def calcula_matriz_confusao(self, p0, p1, normalizado = False):
        if(p0 == p1):
            y_pred = np.array([0 if p <= p0 else 1 for p in self.__y_prob_inicial])
            y = self.__y
            frac_incerto = 0
        else:
            y_pred = np.array([0 if p <= p0 else 1 if p > p1 else np.nan for p in self.__y_prob_inicial])
            flag_na = np.isnan(y_pred, where = True)
            y_pred = y_pred[~flag_na]
            y = self.__y[~flag_na]
            frac_incerto = np.sum(flag_na)/self.__y_prob_inicial.size
        flag_vp = (y == y_pred)&(y_pred == 1)
        flag_vn = (y == y_pred)&(y_pred == 0)
        flag_fp = (y != y_pred)&(y_pred == 1)
        flag_fn = (y != y_pred)&(y_pred == 0)
        #Linhas: Preditos, Colunas: Labels
        #Normalização: Tem como objetivo obter as probabilidades condicionais
        #Isto é, dado que o modelo prediz um certo label, qual a prob de ser esse label mesmo e qual a prob de ser o outro label
        if(normalizado):
            vn = np.sum(flag_vn)
            fn = np.sum(flag_fn)
            norm_n = vn + fn
            fp = np.sum(flag_fp)
            vp = np.sum(flag_vp)
            norm_p = vp + fp
            matriz = np.matrix([[vn/norm_n, fn/norm_n], [fp/norm_p, vp/norm_p]])
        else:
            matriz = np.matrix([[np.sum(flag_vn), np.sum(flag_fn)], [np.sum(flag_fp), np.sum(flag_vp)]])
        return matriz, frac_incerto
    
    def __calcula_probabilidades_condicionais(self, matriz):
        soma = np.sum(matriz[0, :])
        if soma > 0:
            p00 = matriz[0, 0]/soma
        else:
            p00 = np.nan
        soma = np.sum(matriz[1, :])
        if soma > 0:
            p11 = matriz[1, 1]/soma
        else:
            p11 = np.nan
        return p00, p11
        
    def __calcula_acuracia(self, matriz):
        soma = np.sum(matriz)
        if soma > 0:
            acuracia = (matriz[0, 0] + matriz[1, 1])/soma
        else:
            acuracia = np.nan
        return acuracia
        
    def __calcula_acuracia_balanceada(self, matriz):
        soma_0 = matriz[0, 0] + matriz[1, 0]
        soma_1 = matriz[0, 1] + matriz[1, 1]
        if soma_0 > 0 and soma_1 > 0:
            acuracia_0 = matriz[0, 0]/soma_0
            acuracia_1 = matriz[1, 1]/soma_1
            acuracia_bal = (acuracia_0 + acuracia_1)*0.5
        else:
            acuracia_bal = np.nan
        return acuracia_bal
    
    def __calcula_metricas(self):
        self.__ordena_probs()
        if(self.__qtd0_tot*self.__qtd1_tot > 0):
            if(self.__y_prob_unico.size > 2):
                self.__calcula_roc()
                self.__calcula_copc()
                self.__calcula_ks()
                self.__liftF_10, self.__alavF_10 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.1)
                self.__liftF_20, self.__alavF_20 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.2)
                self.__liftV_10, self.__alavV_10 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.1)
                self.__liftV_20, self.__alavV_20 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.2)
                self.__calcula_ig()
                self.__calcula_ig_2d()
                probs_ig = self.valor_prob_ig()
                if(self.__p_corte == None):
                    self.__p_corte = probs_ig['Prob_Corte']
                if(np.sum(self.__p01_corte) == 0):
                    self.__p01_corte = np.array([probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']])
                self.matriz_confusao, _ = self.calcula_matriz_confusao(p0 = self.__p_corte, p1 = self.__p_corte)
                self.matriz_confusao_2d, self.__frac_incerto_2d = self.calcula_matriz_confusao(p0 = self.__p01_corte[0], p1 = self.__p01_corte[1])
                self.__p00, self.__p11 = self.__calcula_probabilidades_condicionais(self.matriz_confusao)
                self.__p00_2d, self.__p11_2d = self.__calcula_probabilidades_condicionais(self.matriz_confusao_2d)
                self.__acuracia = self.__calcula_acuracia(self.matriz_confusao)
                self.__acuracia_balanceada = self.__calcula_acuracia_balanceada(self.matriz_confusao)
                self.__acuracia_2d = self.__calcula_acuracia(self.matriz_confusao_2d)
                self.__acuracia_balanceada_2d = self.__calcula_acuracia_balanceada(self.matriz_confusao_2d)
            elif(self.__y_prob_unico.size > 1):
                self.__calcula_roc()
                self.__calcula_copc()
                self.__calcula_ks()
                self.__liftF_10, self.__alavF_10 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.1)
                self.__liftF_20, self.__alavF_20 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.2)
                self.__liftV_10, self.__alavV_10 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.1)
                self.__liftV_20, self.__alavV_20 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.2)
                self.__calcula_ig()
                probs_ig = self.valor_prob_ig()
                if(self.__p_corte == None):
                    self.__p_corte = probs_ig['Prob_Corte']
                self.matriz_confusao, _ = self.calcula_matriz_confusao(p0 = self.__p_corte, p1 = self.__p_corte)
                self.__p00, self.__p11 = self.__calcula_probabilidades_condicionais(self.matriz_confusao)
                self.__acuracia = self.__calcula_acuracia(self.matriz_confusao)
                self.__acuracia_balanceada = self.__calcula_acuracia_balanceada(self.matriz_confusao)
    
    def valor_prob_ig(self):
        #Retorna um pd.Series com as probs de corte encontradas no ganho de informação
        d = {}
        prob_corte = None
        p0_corte = None
        p1_corte = None
        if(self.__num_div != None):
            if(self.__pos_max_ig != None):
                prob_corte = self.__interv.pontos_medios_discretizacao()[self.__pos_max_ig]
            if(self.__pos_max_ig_2d != None):
                pos_p0_aux = int(self.__vetor_p0_ig_2d[self.__pos_max_ig_2d])
                pos_p1_aux = int(self.__vetor_p1_ig_2d[self.__pos_max_ig_2d])
                p0_corte = self.__interv.pontos_medios_discretizacao()[pos_p0_aux]
                p1_corte = self.__interv.pontos_medios_discretizacao()[pos_p1_aux]
        else:
            if(self.__pos_max_ig != None):
                prob_corte = self.__y_prob_unico[self.__pos_max_ig]
            if(self.__pos_max_ig_2d != None):
                p0_corte = self.__vetor_p0_ig_2d[self.__pos_max_ig_2d]
                p1_corte = self.__vetor_p1_ig_2d[self.__pos_max_ig_2d]
        d['Prob_Corte'] = prob_corte
        d['Prob0_Corte'] = p0_corte
        d['Prob1_Corte'] = p1_corte
        return d
    
    def valor_metricas(self, estatisticas_globais = True, probs_corte = True, probs_condicionais = True, lifts = True):
        #Retorna um pd.Series com as metricas calculadas
        #Esse formato é bom para criar dataframes
        d = {}
        if(estatisticas_globais):
            d['QTD'] = self.__qtd_tot
            d['QTD_0'] = self.__qtd_tot - self.__qtd1_tot
            d['QTD_1'] = self.__qtd1_tot
            d['Frac_0'] = (self.__qtd_tot - self.__qtd1_tot)/self.__qtd_tot
            d['Frac_1'] = self.__qtd1_tot/self.__qtd_tot
            d['Soma_Probs'] = self.__soma_probs
            d['Prob_Media'] = self.__soma_probs/self.__qtd_tot 
        d['AUC'] = self.__auc
        d['KS'] = self.__ks
        if(lifts):
            d['LiftF_10'] = self.__liftF_10
            d['LiftV_10'] = self.__liftV_10
            d['LiftF_20'] = self.__liftF_20
            d['LiftV_20'] = self.__liftV_20
            d['AlavF_10'] = self.__alavF_10
            d['AlavV_10'] = self.__alavV_10
            d['AlavF_20'] = self.__alavF_20
            d['AlavV_20'] = self.__alavV_20
        d['IG'] = self.__ig
        d['IG_2D'] = self.__ig_2d
        d['Frac_Incerto_2D'] = self.__frac_incerto_2d
        if(probs_corte):
            d.update(self.valor_prob_ig())
        d['Acurácia'] = self.__acuracia
        d['Acurácia_Balanceada'] = self.__acuracia_balanceada
        d['Acurácia_2D'] = self.__acuracia_2d
        d['Acurácia_Balanceada_2D'] = self.__acuracia_balanceada_2d
        d['Acurácia_Balanceada_Cond'] = (self.__p00 + self.__p11)*0.5
        d['Acurácia_Balanceada_Cond_2D'] = (self.__p00_2d + self.__p11_2d)*0.5
        if(probs_condicionais):
            d['P(0|0)'] = self.__p00
            d['P(1|1)'] = self.__p11
            d['P_2D(0|0)'] = self.__p00_2d
            d['P_2D(1|1)'] = self.__p11_2d
        return pd.Series(d, index = d.keys())
    
    def curva_roc(self):
        if(self.__y_prob_unico.size > 1):
            curva_tnp = self.__curva_tnp
            curva_tvp = self.__curva_tvp
            auc = self.__auc
        else:
            curva_tnp = np.array([])
            curva_tvp = np.array([])
            auc = np.nan
        return curva_tnp, curva_tvp, auc
        
    def curva_copc(self):  
        if(self.__y_prob_unico.size > 1):
            if(self.__num_div != None):
                y_prob_plot = [x for y in self.__interv.pares_minimo_maximo_discretizacao()[self.__y_prob_unico[:-1]] for x in y]
                curva_p00 = np.repeat(self.__curva_p00, 2)
                curva_p11 = np.repeat(self.__curva_p11, 2)
            else:
                y_prob_plot = self.__y_prob_unico[:-1]
                curva_p00 = self.__curva_p00
                curva_p11 = self.__curva_p11
        else:
            y_prob_plot = np.array([])
            curva_p00 = np.array([])
            curva_p11 = np.array([])
        return y_prob_plot, curva_p00, curva_p11
        
    def curva_revocacao(self):
        if(self.__y_prob_unico.size > 1):
            if(self.__num_div != None):
                y_prob_plot = [x for y in self.__interv.pares_minimo_maximo_discretizacao()[self.__y_prob_unico] for x in y] 
                curva_revoc0_plot = np.repeat(self.__curva_revoc0, 2)
                curva_revoc1_plot = np.repeat(self.__curva_revoc1, 2)
                pos_max = self.__interv.pontos_medios_discretizacao()[self.__pos_max_dif]
            else:
                y_prob_plot = self.__y_prob_unico
                curva_revoc0_plot = self.__curva_revoc0
                curva_revoc1_plot = self.__curva_revoc1
                pos_max = self.__y_prob_unico[self.__pos_max_dif]
            ks = self.__ks
        else:
            y_prob_plot = np.array([])
            curva_revoc0_plot = np.array([])
            curva_revoc1_plot = np.array([])
            pos_max = np.nan
            ks = np.nan
        return y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks
        
    def curva_informacao(self):
        if(self.__y_prob_unico.size > 1):
            if(self.__num_div != None):
                y_prob_plot = [x for y in self.__interv.pares_minimo_maximo_discretizacao()[self.__y_prob_unico] for x in y]
                curva_ig_plot = np.repeat(self.__curva_ig, 2)
                pos_max = self.__interv.pontos_medios_discretizacao()[self.__pos_max_ig]
                ig = self.__ig
                #Se eu quiser plotar o intervalo de prob "confiável" calculado com a informação 2D
                if(self.__y_prob_unico.size > 2):
                    pos_p0_aux = int(self.__vetor_p0_ig_2d[self.__pos_max_ig_2d])
                    pos_p1_aux = int(self.__vetor_p1_ig_2d[self.__pos_max_ig_2d])
                    p0_corte = self.__interv.pontos_medios_discretizacao()[pos_p0_aux]
                    p1_corte = self.__interv.pontos_medios_discretizacao()[pos_p1_aux]
                    ig_2d = self.__ig_2d
                else:
                    p0_corte = np.nan
                    p1_corte = np.nan
                    ig_2d = np.nan   
            else:
                y_prob_plot = self.__y_prob_unico
                curva_ig_plot = self.__curva_ig
                pos_max = self.__y_prob_unico[self.__pos_max_ig]
                ig = self.__ig
                if(self.__y_prob_unico.size > 2):
                    p0_corte = self.__vetor_p0_ig_2d[self.__pos_max_ig_2d]
                    p1_corte = self.__vetor_p1_ig_2d[self.__pos_max_ig_2d]
                    ig_2d = self.__ig_2d
                else:
                    p0_corte = np.nan
                    p1_corte = np.nan
                    ig_2d = np.nan
        else:
            y_prob_plot = np.array([])
            curva_ig_plot = np.array([])
            pos_max = np.nan
            ig = np.nan
            p0_corte = np.nan
            p1_corte = np.nan
            ig_2d = np.nan
        return y_prob_plot, curva_ig_plot, pos_max, ig, p0_corte, p1_corte, ig_2d
        
    def curva_informacao_2d(self):
        if(self.__y_prob_unico.size > 2):
            if(self.__num_div != None):
                x = [i for j in self.__interv.pares_minimo_maximo_discretizacao()[self.__vetor_p0_ig_2d.astype(int)] for i in j]
                y = [i for j in self.__interv.pares_minimo_maximo_discretizacao()[self.__vetor_p1_ig_2d.astype(int)] for i in j]
                x.extend(list(self.__interv.pontos_medios_discretizacao()[self.__vetor_p0_ig_2d.astype(int)]))
                y.extend(list(self.__interv.pontos_medios_discretizacao()[self.__vetor_p1_ig_2d.astype(int)]))
                z = np.repeat(self.__vetor_ig_2d, 2)
                z = np.append(z, self.__vetor_ig_2d)
                pos_p0_aux = int(self.__vetor_p0_ig_2d[self.__pos_max_ig_2d])
                pos_p1_aux = int(self.__vetor_p1_ig_2d[self.__pos_max_ig_2d])
                p0_corte = self.__interv.pontos_medios_discretizacao()[pos_p0_aux]
                p1_corte = self.__interv.pontos_medios_discretizacao()[pos_p1_aux]
                ig_2d = self.__ig_2d
            else:
                x = self.__vetor_p0_ig_2d
                y = self.__vetor_p1_ig_2d
                z = self.__vetor_ig_2d
                p0_corte = self.__vetor_p0_ig_2d[self.__pos_max_ig_2d]
                p1_corte = self.__vetor_p1_ig_2d[self.__pos_max_ig_2d]
                ig_2d = self.__ig_2d
            
        else:
            x = np.array([])
            y = np.array([])
            z = np.array([])
            p0_corte = np.nan
            p1_corte = np.nan
            ig_2d = np.nan
        return x, y, z, p0_corte, p1_corte, ig_2d
    
    def grafico_roc(self, roc_usual = False, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            curva_tnp, curva_tvp, auc = self.curva_roc()
            if(roc_usual):
                axs.plot(1-curva_tnp, curva_tvp, color = paleta_cores[0], label = 'Curva ROC')
                axs.plot([0, 1], [0, 1], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Falso Positivo')
            else:
                axs.plot(curva_tnp, curva_tvp, color = paleta_cores[0], label = 'Curva ROC')
                axs.plot([0, 1], [1, 0], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Verdadeiro Negativo')
            plt.gcf().text(1, 0.5, 'AUC = ' + '%.2g' % auc, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.set_ylabel('Taxa de Verdadeiro Positivo')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def grafico_copc(self, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_prob_plot, curva_p00, curva_p11 = self.curva_copc()
            axs.plot(y_prob_plot, curva_p00, color = paleta_cores[0], label = 'P(0|0)')
            axs.plot(y_prob_plot, curva_p11, color = paleta_cores[1], label = 'P(1|1)')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Probalidade Condicional')
            plt.show()
        
    def grafico_revocacao(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks = self.curva_revocacao()
            axs.plot(y_prob_plot, curva_revoc0_plot, color = paleta_cores[0], alpha = 1.0, label = 'Revocação 0')
            axs.plot(y_prob_plot, curva_revoc1_plot, color = paleta_cores[1], alpha = 1.0, label = 'Revocação 1')
            axs.vlines(pos_max, 0, 1, color = 'k', linestyle = '--', label = 'Ponto KS')
            plt.gcf().text(1, 0.5, 'KS = ' + '%.2g' % ks, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Revocação')
            plt.show()
    
    def grafico_informacao(self, mostrar_ig_2d = False, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_prob_plot, curva_ig_plot, pos_max, ig, p0_corte, p1_corte, ig_2d = self.curva_informacao()
            axs.plot(y_prob_plot, curva_ig_plot, color = paleta_cores[0], label = 'Curva IG')
            axs.vlines(pos_max, 0, ig, color = 'k', linestyle = '--', label = 'Ganho Máx.')
            if(mostrar_ig_2d and ig_2d != np.nan):
                axs.vlines(p0_corte, 0, ig_2d, color = 'k', alpha = 0.5, linestyle = '--', label = 'Ganho Máx. 2D')
                axs.vlines(p1_corte, 0, ig_2d, color = 'k', alpha = 0.5, linestyle = '--')
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig + '\n' + 'IG 2D = ' + '%.2g' % ig_2d, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                plt.gcf().text(1, 0.3, 'Prob Corte = ' + '%.2g' % pos_max + '\n\n' + 'Prob0 Corte = ' + '%.2g' % p0_corte + '\n' + 'Prob1 Corte = ' + '%.2g' % p1_corte, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            else:
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                plt.gcf().text(1, 0.3, 'Prob Corte = ' + '%.2g' % pos_max, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Ganho de Informação')
            plt.show()
        
    def grafico_informacao_2d(self, plot_3d = True, figsize = [7, 6]):
        paleta_cores = sns.color_palette("colorblind")
        x, y, z, p0_corte, p1_corte, ig_2d = self.curva_informacao_2d()
        if(plot_3d):
            with sns.axes_style("whitegrid"):
                fig = plt.figure(figsize = figsize)
                axs = fig.add_subplot(111, projection='3d')
                #Constrói gradiente de uma cor até o branco (1,1,1) -> Lembrar que em RGB a mistura de todas as cores é que é o branco
                N = 256
                vals = np.ones((N, 4)) #A última componente (quarta) é o alpha que é o índice de transparência
                cor = paleta_cores[0]
                #Define as Cores RGB pelas componentes (no caso é o azul -> 0,0,255)
                vals[:, 0] = np.linspace(cor[0], 1, N)
                vals[:, 1] = np.linspace(cor[1], 1, N)
                vals[:, 2] = np.linspace(cor[2], 1, N)
                cmap = mpl.colors.ListedColormap(vals[::-1])
                axs.scatter(x, y, z, c = z, marker = 'o', cmap = cmap)
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_zlabel('Ganho de Informação')
                plt.show()
        else:
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                #Faz uma mapa de cores com base em uma cor e mudando a transparência
                N = 256
                cor = paleta_cores[0]
                vals = np.ones((N, 4))
                vals[:, 0] = cor[0]
                vals[:, 1] = cor[1]
                vals[:, 2] = cor[2]
                cmap_linhas = mpl.colors.ListedColormap(vals[::-1])
                vals[:, 3] = np.linspace(0, 1, N)[::-1]
                cmap = mpl.colors.ListedColormap(vals[::-1])
                axs.tricontour(x, y, z, levels = 14, linewidths = 0.5, cmap = cmap_linhas)
                cntr = axs.tricontourf(x, y, z, levels = 14, cmap = cmap)
                cbar = plt.colorbar(cntr, ax = axs)
                cbar.ax.set_title('Ganho Info.')
                axs.scatter(p0_corte, p1_corte, color = 'k')
                axs.vlines(p0_corte, 0, p1_corte, color = 'k', alpha = 0.5, linestyle = '--')
                axs.hlines(p1_corte, 0, p0_corte, color = 'k', alpha = 0.5, linestyle = '--')
                plt.gcf().text(1, 0.8, 'IG 2D = ' + '%.2g' % ig_2d, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                plt.gcf().text(1, 0.7, 'Prob0 Corte = ' + '%.2g' % p0_corte + '\n' + 'Prob1 Corte = ' + '%.2g' % p1_corte, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_xlim([min(x[0], y[0]), max(x[-1], y[-1])])
                axs.set_ylim([min(x[0], y[0]), max(x[-1], y[-1])])
                plt.show()

##############################

##############################

#Para funcionar direito, é preciso transformar os valores numéricos todos em float (não pode ter int, por exemplo)
class AvaleDistribuicoes:

    def __init__(self, df, num_div = None):
        self.__df = df
        self.__num_linhas = len(self.__df)
        self.__colunas = df.columns.values
        self.__num_colunas = self.__colunas.size
        self.__num_div = num_div
        
        self.__dict_flag_na = {}
        self.__dict_frac_na = {}
        self.__dict_num_div = {}
        self.__dict_interv = {}
        self.__dict_inds_ordenado = {}
        self.__dict_valores_unicos = {}
        self.__dict_primeira_ocorrencia = {}
        self.__dict_qtds = {}
    
    def info_dataset(self):
        return self.__num_linhas, self.__num_colunas, self.__colunas
    
    #Trata o tipo do vetor e os valores NA
    def __trata_tipo_e_na_vetor(self, valores, col_ref):
        #valores = np.where(valores == np.inf, np.nan, valores)
        #valores = np.where(valores == -np.inf, np.nan, valores)
        if(valores.dtype in [np.number, 'int64', 'float64']):
            flag_na = np.isnan(valores, where = True)
            valores = valores[~flag_na]
            if(self.__num_div != None and np.unique(valores).size > self.__num_div):
                num_div = self.__num_div
            else:
                num_div = None
        else:
            flag_na = np.array([x is np.nan for x in valores])
            valores = valores[~flag_na]
            num_div = None
            
        self.__dict_flag_na[col_ref] = flag_na
        self.__dict_frac_na[col_ref] = np.sum(flag_na)/self.__num_linhas
        self.__dict_num_div[col_ref] = num_div
        
        return valores, num_div
    
    def calcula_distribuicao(self, col_ref = []):
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        
        if(len(col_ref) == 0):
            colunas = self.__colunas
        else:
            colunas = col_ref
        
        for col_ref in colunas:
            valores = self.__df[col_ref].values
            valores, num_div = self.__trata_tipo_e_na_vetor(valores, col_ref)
            
            #Divide a variável em intervalos se for passado num_div
            if(num_div != None):
                interv = CortaIntervalosQuasiUniforme(valores, num_div = num_div)
                valores = interv.vetor_discretizado()
                self.__dict_interv[col_ref] = interv
            else:
                #Se não, tenta deletar o interv se existir para essa col_ref
                try:
                    del self.__dict_interv[col_ref]
                except:
                    pass
            
            inds_ordenado = np.argsort(valores)        
            valores_unico, primeira_ocorrencia, qtds = np.unique(valores[inds_ordenado], 
                                                                 return_index = True, return_counts = True)
            self.__dict_inds_ordenado[col_ref] = inds_ordenado
            self.__dict_valores_unicos[col_ref] = valores_unico
            self.__dict_primeira_ocorrencia[col_ref] = primeira_ocorrencia
            self.__dict_qtds[col_ref] = qtds
    
    def info_distribuicao(self, col_ref):
        if(col_ref in self.__dict_valores_unicos.keys()):
            num_div = self.__dict_num_div[col_ref]
            if(num_div != None):
                interv = self.__dict_interv[col_ref]
            else:
                interv = None
            return self.__dict_flag_na[col_ref], self.__dict_frac_na[col_ref], num_div, interv, self.__dict_inds_ordenado[col_ref], self.__dict_valores_unicos[col_ref], self.__dict_primeira_ocorrencia[col_ref], self.__dict_qtds[col_ref]
    
    def curva_distribuicao(self, col_ref):
        if(col_ref in self.__dict_valores_unicos.keys()):
            if(col_ref in self.__dict_interv.keys()):
                valores, fracL = self.__dict_interv[col_ref].curva_distribuicao()
                eh_intervalo = True
            else:
                valores = self.__dict_valores_unicos[col_ref]
                qtds = self.__dict_qtds[col_ref]
                fracL = qtds/np.sum(qtds)
                eh_intervalo = False
            #Ordena se for categorico
            if(valores.dtype not in [np.number, 'int64', 'float64']):
                bool_ord = np.argsort(fracL)[::-1]
                valores = valores[bool_ord]
                fracL = fracL[bool_ord]
            return valores, fracL, eh_intervalo
    
    def grafico_distribuicao(self, col_ref = [], conv_str = True, figsize = [6, 4]):
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
    
        if(len(col_ref) == 0):
            colunas = self.__colunas
        else:
            colunas = col_ref
        for col_ref in colunas:
            if(col_ref in self.__dict_valores_unicos.keys()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    valores, fracL, eh_intervalo = self.curva_distribuicao(col_ref)
                    frac_na = self.__dict_frac_na[col_ref]
                    if(eh_intervalo):
                        axs.fill_between(valores, fracL, color = paleta_cores[0], alpha = 0.5)
                        axs.plot(valores, fracL, color = paleta_cores[0])
                        axs.set_ylabel('Fração/L')
                    else:
                        if(conv_str):
                            valores = valores.astype(str)
                        axs.bar(valores, fracL, color = paleta_cores[0], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[0])
                        #axs.plot(valores, fracL, color = paleta_cores[0], linewidth = 2)
                        axs.set_ylabel('Fração')
                    plt.gcf().text(1, 0.8, 'Fração de NA = ' + '%.2g' % frac_na, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                    axs.set_xlabel(col_ref)
                    axs.set_ylim(bottom = 0.0)
                    plt.show()

##############################

##############################

class AvaleClassificacao:

    def __init__(self, df, col_alvo, col_prob = None, num_div_prob = None, p_corte = None, p01_corte = [0, 0], num_div = None):
        self.distribuicoes = AvaleDistribuicoes(df, num_div)
        
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
        
        self.__dict_qtds1 = {}
        self.__dict_prob1 = {}
        self.__dict_ig = {}
        self.__dict_ig_por_bit = {} #Ganho de Informação por Bit (quantidade de valores únicos em log2)
        
        self.__dict_somaprobs = {}
        self.__dict_mediaprobs = {}
        self.__dict_metricas = {}
    
    def colunas_metricas_condicionais_prontas(self):
        return self.__dict_qtds1.keys()
    
    def calcula_metricas_condicionais(self, col_ref = []):
        num_linhas, _, colunas = self.distribuicoes.info_dataset()
        
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            self.distribuicoes.calcula_distribuicao(col_ref)
            flag_na, frac_na, num_div, interv, inds_ordenado, valores_unicos, primeira_ocorrencia, qtds = self.distribuicoes.info_distribuicao(col_ref)
            num_linhas_sem_na = num_linhas*(1 - frac_na)
            
            y = self.__y[~flag_na]
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            qtds1 = np.array([np.sum(v) for v in y_agrup])
            
            self.__dict_qtds1[col_ref] = qtds1
            probs1 = qtds1/qtds
            self.__dict_prob1[col_ref] = probs1
            
            #Calcula a Entropia de Shannon
            def entropia_shannon(p1):
                p0 = 1 - p1
                if p0 == 0 or p1 == 0:
                    return 0
                else:
                    return -p0*np.log2(p0) - p1*np.log2(p1)
            
            entropia_ini = entropia_shannon(np.sum(qtds1)/num_linhas_sem_na)
            entropias_parciais = np.array([entropia_shannon(x) for x in probs1])
            entropia = np.sum(entropias_parciais*qtds)/num_linhas_sem_na
            ig = (entropia_ini - entropia)/entropia_ini
            self.__dict_ig[col_ref] = ig
            qtd_unicos = valores_unicos.size
            if(qtd_unicos > 1):
                self.__dict_ig_por_bit[col_ref] = ig/np.log2(qtd_unicos)
            else:
                self.__dict_ig_por_bit[col_ref] = 0
            
            if(self.__col_prob != None):
                y_prob = self.__y_prob[~flag_na]
                y_prob_agrup = np.split(y_prob[inds_ordenado], primeira_ocorrencia[1:])
                soma_probs = np.array([np.sum(v) for v in y_prob_agrup])
                self.__dict_somaprobs[col_ref] = soma_probs
                self.__dict_mediaprobs[col_ref] = soma_probs/qtds
                self.__dict_metricas[col_ref] = np.array([AletricasClassificacao(y_agrup[i], y_prob_agrup[i], num_div = self.__num_div_prob,
                                                          p_corte = self.__p_corte, p01_corte = self.__p01_corte) for i in range(valores_unicos.size)])
                                                          
        #Ordena os Ganhos de Informação
        self.__dict_ig = dict(reversed(sorted(self.__dict_ig.items(), key = lambda x: x[1])))
        self.__dict_ig_por_bit = dict(reversed(sorted(self.__dict_ig_por_bit.items(), key = lambda x: x[1])))
    
    def ganho_info(self):
        return pd.Series(self.__dict_ig, index = self.__dict_ig.keys())

    def ganho_info_por_bit(self):
        return pd.Series(self.__dict_ig_por_bit, index = self.__dict_ig_por_bit.keys())
    
    def valor_metricas_condicionais(self, col_ref):
        df = pd.DataFrame()
        if(col_ref in self.__dict_qtds1.keys()):
        
            #flag_na, frac_na, num_div, interv, inds_ordenado, valores_unicos, primeira_ocorrencia, qtds = self.distribuicoes.info_distribuicao(col_ref)
            _, _, num_div, interv, _, valores, _, qtds = self.distribuicoes.info_distribuicao(col_ref)
            qtds1 = self.__dict_qtds1[col_ref]
            prob1 = self.__dict_prob1[col_ref]
            
            df['Valores'] = valores
            if(num_div != None):
                df['Labels'] = interv.strings_intervalos_discretizacao()
            df['QTD'] = qtds
            df['QTD_0'] = qtds - qtds1
            df['QTD_1'] = qtds1
            df['Frac_0'] = (qtds - qtds1)/qtds
            df['Frac_1'] = prob1
            
            if(self.__col_prob != None):
                soma_probs = self.__dict_somaprobs[col_ref]
                media_probs = self.__dict_mediaprobs[col_ref]
                vetor_metricas = self.__dict_metricas[col_ref]
                df['Soma_Probs'] = soma_probs
                df['Prob_Media'] = media_probs
                
                df['Metricas'] = vetor_metricas
                df = pd.concat([df, df['Metricas'].apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                df = df.drop('Metricas', axis = 1)
            
            #Ordena se for categorico
            if(valores.dtype not in [np.number, 'int64', 'float64']):
                df = df.sort_values('Frac_1', ascending = False).reset_index(drop = True)

        return df
    
    def curva_probabilidade_condicional(self, col_ref):
        if(col_ref in self.__dict_qtds1.keys()):
            _, _, num_div, interv, _, valores, _, _ = self.distribuicoes.info_distribuicao(col_ref)
            prob1 = self.__dict_prob1[col_ref]
            if(self.__col_prob != None):
                media_probs = self.__dict_mediaprobs[col_ref]
                tem_clf = True
            else:
                media_probs = None
                tem_clf = False
            if(num_div != None):
                labels = interv.strings_intervalos_discretizacao()
                eh_intervalo = True
            else:
                labels = None
                eh_intervalo = False
            
            #Ordena se for categorico
            if(valores.dtype not in [np.number, 'int64', 'float64']):
                bool_ord = np.argsort(prob1)[::-1]
                valores = valores[bool_ord]
                prob1 = prob1[bool_ord]
                if(tem_clf):
                    media_probs = media_probs[bool_ord]            
            
            return valores, prob1, media_probs, labels, tem_clf, eh_intervalo
    
    def grafico_probabilidade_condicional(self, col_ref, figsize = [6, 4]):
        if(col_ref in self.__dict_qtds1.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                valores, prob1, media_probs, labels, tem_clf, eh_intervalo = self.curva_probabilidade_condicional(col_ref)
                ig = self.__dict_ig[col_ref]
                ig_por_bit = self.__dict_ig_por_bit[col_ref]
                axs.bar(valores, prob1, color = paleta_cores[0], label = 'Real')
                #axs.plot(valores, prob1, color = paleta_cores[0], linewidth = 2, label = 'Real')
                if(eh_intervalo):
                    axs.set_xticks(valores)
                    axs.set_xticklabels(labels, rotation = 90)
                if(tem_clf):
                    if(valores.size > 1):
                        axs.plot(valores, media_probs, color = paleta_cores[1], linewidth = 2, label = 'Classificador')
                    else:
                        axs.scatter(valores, media_probs, color = paleta_cores[1], label = 'Classificador')
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig + '\n' + 'IG/Bit = ' + '%.2g' % ig_por_bit, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Probabilidade de 1')
                axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                plt.show()

    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], conv_str = True, figsize = [6, 4]):
        if(col_ref in self.__dict_qtds1.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            df = self.valor_metricas_condicionais(col_ref)
            valores = df['Valores'].values
            if(df.columns[1] == 'Labels'):
                labels = df['Labels'].values
                eh_intervalo = True
            else:
                eh_intervalo = False
            if(conv_str):
                valores = valores.astype(str)
                
            valores_metricas = []
            for metrica in metricas:
                valores_metricas.append(df[metrica].values)
            
            #Plot a curva de métrica em função da coluna de referência
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                for i in range(len(metricas)):
                    valores_metrica = valores_metricas[i]
                    if(valores.size > 1):
                        axs.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                    else:
                        axs.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                if(eh_intervalo):
                    axs.set_xticks(valores)
                    axs.set_xticklabels(labels, rotation = 90)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Metricas')
                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                plt.show()

##############################

##############################

class AvaleDatasetsClassificacao:

    def __init__(self, dict_dfs, col_alvo, col_prob = None, num_div_prob = None, num_div = None, chave_treino = 'Treino'):
        self.__dict_dfs = dict_dfs
        self.__num_dfs = len(dict_dfs)
        self.__chave_treino = chave_treino
        
        self.__dict_avaliaclf = {}
        if(self.__chave_treino in self.__dict_dfs.keys()):
            avaliaclf_treino = AvaleClassificacao(self.__dict_dfs[self.__chave_treino], col_alvo, col_prob, num_div_prob = num_div_prob, num_div = num_div)
            #Probabilidades de Corte para Avaliação de Tomada de Decisão
            probs_ig = avaliaclf_treino.metricas_gerais.valor_prob_ig()
            p_corte = probs_ig['Prob_Corte']
            p01_corte = [probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']]
            self.__dict_avaliaclf[self.__chave_treino] = avaliaclf_treino
        for chave in self.__dict_dfs.keys():
            if(chave != self.__chave_treino):
                self.__dict_avaliaclf[chave] = AvaleClassificacao(self.__dict_dfs[chave], col_alvo, col_prob, num_div_prob, p_corte, p01_corte, num_div)
        
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
    
    def grafico_roc(self, roc_usual = False, figsize = [6, 4]):
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
            
    def grafico_copc(self, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__dict_dfs.keys():
                y_prob_plot, curva_p00, curva_p11 = self.__dict_avaliaclf[chave].metricas_gerais.curva_copc()
                axs.plot(y_prob_plot, curva_p00, color = paleta_cores[i], alpha = 0.6, label = 'P(0|0)' + ' - ' + chave)
                axs.plot(y_prob_plot, curva_p11, color = paleta_cores[i], alpha = 0.4, label = 'P(1|1)' + ' - ' + chave)
                i = i + 1
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Probalidade Condicional')
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
                
    def grafico_distribuicao(self, col_ref = [], conv_str = True, figsize = [6, 4]):
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
                        valores, fracL, eh_intervalo = self.__dict_avaliaclf[chave].distribuicoes.curva_distribuicao(col_ref)
                        if(eh_intervalo):
                            axs.fill_between(valores, fracL, color = paleta_cores[i], alpha = 0.5)
                            axs.plot(valores, fracL, color = paleta_cores[i], label = chave)
                            axs.set_ylabel('Fração/L')
                        else:
                            if(conv_str):
                                valores = valores.astype(str)
                            axs.bar(valores, fracL, color = paleta_cores[i], alpha = 0.5, width = 1, linewidth = 3, edgecolor = paleta_cores[i], label = chave)
                            #axs.plot(valores, fracL, color = paleta_cores[i], linewidth = 2, label = chave)
                            axs.set_ylabel('Fração')
                        i = i + 1
                    axs.set_xlabel(col_ref)
                    axs.set_ylim(bottom = 0.0)
                    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.show()
                
    def grafico_probabilidade_condicional(self, col_ref, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_prontas()):
        
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                i = 0
                hlds = []
                for chave in self.__dict_dfs.keys():
                    if(self.__num_dfs > 1):
                        ax = axs[i]
                    else:
                        ax = axs
                    valores, prob1, media_probs, labels, tem_clf, eh_intervalo = self.__dict_avaliaclf[chave].curva_probabilidade_condicional(col_ref)
                    ax.bar(valores, prob1, color = paleta_cores[i])
                    #ax.plot(valores, prob1, color = paleta_cores[0], linewidth = 2, label = 'Real')
                    if(eh_intervalo):
                        ax.set_xticks(valores)
                        ax.set_xticklabels(labels, rotation = 90)
                    if(tem_clf):
                        if(valores.size > 1):
                            ax.plot(valores, media_probs, color = 'k', linewidth = 2)
                        else:
                            ax.scatter(valores, media_probs, color = 'k')
                    ax.set_xlabel(col_ref)
                    ax.set_ylabel('Probabilidade de 1')
                    ax.set_title(chave)
                    hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                    i = i + 1
                hlds.append(mpl.patches.Patch(color = 'k', label = 'Classificador'))
                plt.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], conv_str = True, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_prontas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                j = 0
                hlds = []
                for chave in self.__dict_dfs.keys():
                    if(self.__num_dfs > 1):
                        ax = axs[j]
                    else:
                        ax = axs
                    df = self.__dict_avaliaclf[chave].valor_metricas_condicionais(col_ref)
                    valores = df['Valores'].values
                    if(df.columns[1] == 'Labels'):
                        labels = df['Labels'].values
                        eh_intervalo = True
                    else:
                        eh_intervalo = False
                    valores_metricas = []
                    for metrica in metricas:
                        valores_metricas.append(df[metrica].values)
                    if(conv_str):
                        valores = valores.astype(str)

                    for i in range(len(metricas)):
                        valores_metrica = valores_metricas[i]
                        if(valores.size > 1):
                            ax.plot(valores, valores_metrica, color = paleta_cores[i], linewidth = 2, label = metricas[i])
                        else:
                            ax.scatter(valores, valores_metrica, color = paleta_cores[i], label = metricas[i])
                        if(chave == self.__chave_treino):
                            hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = metricas[i]))
                    if(eh_intervalo):
                        ax.set_xticks(valores)
                        ax.set_xticklabels(labels, rotation = 90)
                    ax.set_xlabel(col_ref)
                    ax.set_ylabel('Metricas')
                    ax.set_title(chave)
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                    j = j + 1
                plt.legend(handles = hlds, bbox_to_anchor = (1.3, 1), loc = 'upper left')
                plt.show()
                
##############################

##############################

class AvaleVariaveis:

    def __init__(self, df, col_alvo, col_pred = None, num_div = None):
        
        self.distribuicoes = AvaleDistribuicoes(df, num_div)
        
        self.__y = df[col_alvo].values
        self.__col_alvo = col_alvo
        self.__col_pred = col_pred
        if(col_pred != None):
            self.__y_pred = df[col_pred].values
            
        else:
            self.__y_pred = None
        
        self.__dict_somaalvos = {}
        self.__dict_mediaalvos = {}
        self.__dict_somapreds = {}
        self.__dict_mediapreds = {}
        
        self.__dict_inds_ordena_real = {}
        self.__dict_inds_ordena_pred = {}
        
        self.__dict_tendencia_real = {}
        self.__dict_tendencia_pred = {}
        
        self.__dict_imp_real = {}
        self.__dict_imp_pred = {}
    
    def colunas_metricas_condicionais_prontas(self):
        return self.__dict_somaalvo.keys()
    
    def calcula_metricas_condicionais(self, col_ref = []):
        _, _, colunas = self.distribuicoes.info_dataset()
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
        for col_ref in colunas:
            self.distribuicoes.calcula_distribuicao(col_ref)
            flag_na, frac_na, num_div, interv, inds_ordenado, valores_unicos, primeira_ocorrencia, qtds = self.distribuicoes.info_distribuicao(col_ref)
            
            y = self.__y[~flag_na]
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            somaalvos = np.array([np.sum(v) for v in y_agrup])
            self.__dict_somaalvos[col_ref] = somaalvos
            self.__dict_mediaalvos[col_ref] = somaalvos/qtds
            if(self.__col_pred != None):
                y_pred = self.__y_pred[~flag_na]
                y_pred_agrup = np.split(y_pred[inds_ordenado], primeira_ocorrencia[1:])
                somapreds = np.array([np.sum(v) for v in y_pred_agrup])
                self.__dict_somapreds[col_ref] = somapreds
                self.__dict_mediapreds[col_ref] = somapreds/qtds
    
    def valor_metricas_condicionais(self, col_ref):
        df = pd.DataFrame()
        if(col_ref in self.__dict_somaalvos.keys()):
            _, _, num_div, interv, _, valores, _, qtds = self.distribuicoes.info_distribuicao(col_ref)
            somaalvos = self.__dict_somaalvos[col_ref]
            mediaalvos = self.__dict_mediaalvos[col_ref]
            df['Valores'] = valores
            if(num_div != None):
                df['Labels'] = interv.strings_intervalos_discretizacao()
            df['QTD'] = qtds
            df['Soma_Alvos'] = somaalvos
            df['Media_Alvos'] = mediaalvos
            if(self.__col_pred != None):
                somapreds = self.__dict_ssomapreds[col_ref]
                mediapreds = self.__dict_mediapreds[col_ref]
                df['Soma_Preds'] = soma_probs
                df['Media_Preds'] = media_probs
            
            #Ordena se for categorico
            if(valores.dtype not in [np.number, 'int64', 'float64']):
                df = df.sort_values('Media_Alvos', ascending = False).reset_index(drop = True)            
            
        return df
    
    def curva_medias_condicional(self, col_ref):
        if(col_ref in self.__dict_somaalvos.keys()):
            _, _, num_div, interv, _, valores, _, _ = self.distribuicoes.info_distribuicao(col_ref)
            mediaalvos = self.__dict_mediaalvos[col_ref]
            if(self.__col_pred != None):
                mediapreds = self.__dict_mediapreds[col_ref]
                tem_pred = True
            else:
                mediapreds = None
                tem_pred = False
            if(num_div != None):
                labels = interv.strings_intervalos_discretizacao()
                eh_intervalo = True
            else:
                labels = None
                eh_intervalo = False
                
            #Ordena se for categorico
            if(valores.dtype not in [np.number, 'int64', 'float64']):
                bool_ord = np.argsort(mediaalvos)[::-1]
                valores = valores[bool_ord]
                mediaalvos = mediaalvos[bool_ord]
                if(tem_pred):
                    mediapreds = mediapreds[bool_ord] 
            
            return valores, mediaalvos, mediapreds, labels, tem_pred, eh_intervalo
    
    def grafico_medias_condicional(self, col_ref = [], figsize = [6, 4]):
        _, _, colunas = self.distribuicoes.info_dataset()
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
        for col_ref in colunas:
            if(col_ref in self.__dict_somaalvos.keys()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                
                #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = figsize)
                    valores, mediaalvos, mediapreds, labels, tem_pred, eh_intervalo = self.curva_medias_condicional(col_ref)
                    axs.bar(valores, mediaalvos, color = paleta_cores[0], label = 'Real')
                    if(eh_intervalo):
                        axs.set_xticks(valores)
                        axs.set_xticklabels(labels, rotation = 90)
                    if(tem_pred):
                        if(valores.size > 1):
                            axs.plot(valores, mediapreds, color = paleta_cores[1], linewidth = 2, label = 'Predição')
                        else:
                            axs.scatter(valores, mediapreds, color = paleta_cores[1], label = 'Predição')
                    axs.set_xlabel(col_ref)
                    axs.set_ylabel('Média')
                    axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
                    plt.show()
    
    def calcula_tendencias(self):
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
        colunas = [col for col in colunas if col not in (self.__col_alvo, self.__col_pred)]
        #Calcula um número que resume a tendência e a derivada dos gráficos de média para ver a tendência por valor da variável
        for col_ref in colunas:
            _, _, num_div, _, _, valores, _, _ = self.distribuicoes.info_distribuicao(col_ref)
            medias = self.__dict_mediaalvos[col_ref]
            if(num_div == None and valores.dtype not in [np.number, 'int64', 'float64']):
                inds_ordena = np.argsort(medias)
                medias = medias[inds_ordena]
                self.__dict_inds_ordena_real[col_ref] = inds_ordena
            self.__dict_imp_real[col_ref] = np.std(medias) #np.max(medias) - np.min(medias)
            self.__dict_tendencia_real[col_ref] = calc_diff(medias)
            if(self.__col_pred != None):
                medias = self.__dict_mediapreds[col_ref]
                if(num_div == None and valores.dtype not in [np.number, 'int64', 'float64']):
                    inds_ordena = np.argsort(medias)
                    medias = medias[inds_ordena]
                    self.__dict_inds_ordena_pred[col_ref] = inds_ordena
                self.__dict_imp_pred[col_ref] = np.std(medias) #np.max(medias) - np.min(medias)
                self.__dict_tendencia_pred[col_ref] = calc_diff(medias)
        
        #Normaliza a derivada (o menor valor será -1 e o maior 1)
        minimos_alvo = np.array([])
        maximos_alvo = np.array([])
        minimos_pred = np.array([])
        maximos_pred = np.array([])
        for col_ref in colunas:
            imp = self.__dict_tendencia_real[col_ref]
            minimos_alvo = np.append(minimos_alvo, np.min(imp))
            maximos_alvo = np.append(maximos_alvo, np.max(imp))
            if(self.__col_pred != None):
                imp = self.__dict_tendencia_pred[col_ref]
                minimos_pred = np.append(minimos_pred, np.min(imp))
                maximos_pred = np.append(maximos_pred, np.max(imp))
        minimo_alvo = np.min(minimos_alvo)
        maximo_alvo = np.max(maximos_alvo)
        if(self.__col_pred != None):
            minimo_pred = np.min(minimos_pred)
            maximo_pred = np.max(maximos_pred)
        if(self.__col_pred != None):
            for col_ref in colunas:
                self.__dict_tendencia_real[col_ref] = np.array([v/maximo_alvo if v > 0 else -v/minimo_alvo for v in self.__dict_tendencia_real[col_ref]])
                self.__dict_tendencia_pred[col_ref] = np.array([v/maximo_pred if v > 0 else -v/minimo_pred for v in self.__dict_tendencia_pred[col_ref]])
        else:
            for col_ref in colunas:
                self.__dict_tendencia_real[col_ref] = np.array([v/maximo_alvo if v > 0 else -v/minimo_alvo for v in self.__dict_tendencia_real[col_ref]])
        
        #Normaliza a tendência para que a soma de todas as tendências seja 1
        valores = list(self.__dict_imp_real.values())
        self.__dict_imp_real = dict(zip(list(self.__dict_imp_real.keys()), valores/np.sum(valores))) 
        valores = list(self.__dict_imp_pred.values())
        self.__dict_imp_pred = dict(zip(list(self.__dict_imp_pred.keys()), valores/np.sum(valores))) 
        
    def curva_tendencia(self, col_ref):
        if(col_ref in self.__dict_somaalvos.keys()):
            _, _, num_div, interv, _, valores, _, _ = self.distribuicoes.info_distribuicao(col_ref)
            tend_real = self.__dict_tendencia_real[col_ref]
            imp_real = self.__dict_imp_real[col_ref]
            if(num_div == None and valores.dtype not in [np.number, 'int64', 'float64']):
                inds_ordena = self.__dict_inds_ordena_real[col_ref]
                valores_real = valores[inds_ordena]
            else:
                valores_real = valores
            
            if(self.__col_pred != None):
                tend_pred = self.__dict_tendencia_pred[col_ref]
                imp_pred = self.__dict_imp_pred[col_ref]
                if(num_div == None and valores.dtype not in [np.number, 'int64', 'float64']):
                    inds_ordena = self.__dict_inds_ordena_pred[col_ref]
                    valores_pred = valores[inds_ordena]
                else:
                    valores_pred = valores
                tem_pred = True
            else:
                tend_pred = None
                imp_pred = None
                valores_pred = None
                tem_pred = False
            
            if(num_div != None):
                labels = interv.strings_intervalos_discretizacao()
                eh_intervalo = True
            else:
                labels = None
                eh_intervalo = False
            
            return valores_real, valores_pred, tend_real, tend_pred, imp_real, imp_pred, labels, tem_pred, eh_intervalo
                    
    def grafico_tendencias(self):      
        colunas = dict(reversed(sorted(self.__dict_imp_real.items(), key = lambda x: x[1]))).keys()
        
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
        if(self.__col_pred != None):
            fig, axs = plt.subplots(num_cols, 2, figsize = [14, 2*num_cols], constrained_layout = True)
            fig.suptitle('Tendência das Variáveis: Real/Predito')
        else:
            fig, axs = plt.subplots(num_cols, 1, figsize = [7, 2*num_cols], constrained_layout = True)
            fig.suptitle('Tendência das Variáveis:')
        
        if(self.__col_pred != None):
            i = 0
            for col_ref in colunas:
                valores_real, valores_pred, tend_real, tend_pred, imp_real, imp_pred, labels, tem_pred, eh_intervalo = self.curva_tendencia(col_ref)
                cores_plot = cores[np.floor((tend_real + 1)*(N-1)/2).astype(int)]
                axs[i][0].imshow([cores_plot], aspect = 0.5*(valores_real.size/10), interpolation = 'spline16')
                axs[i][0].set_yticks([])
                if(eh_intervalo):
                    axs[i][0].set_xticks(valores_real)
                    axs[i][0].set_xticklabels(labels, rotation = 90)
                else:
                    axs[i][0].set_xticks(range(0, valores_real.size))
                    axs[i][0].set_xticklabels(valores_real.astype(str))
                axs[i][0].set_title(col_ref + ': ' + '%.2g' % imp_real, loc = 'left')
                i = i + 1
            
            i = 0            
            colunas = dict(reversed(sorted(self.__dict_imp_pred.items(), key = lambda x: x[1]))).keys()
            for col_ref in colunas:
                valores_real, valores_pred, tend_real, tend_pred, imp_real, imp_pred, labels, tem_pred, eh_intervalo = self.curva_tendencia(col_ref)
                cores_plot = cores[np.floor((tend_pred + 1)*(N-1)/2).astype(int)]
                axs[i][1].imshow([cores_plot], aspect = 0.5*(valores_pred.size/10), interpolation = 'spline16')
                axs[i][1].set_yticks([])
                if(eh_intervalo):
                    axs[i][1].set_xticks(valores_pred)
                    axs[i][1].set_xticklabels(labels, rotation = 90)
                else:
                    axs[i][1].set_xticks(range(0, valores_pred.size))
                    axs[i][1].set_xticklabels(valores_pred.astype(str))
                axs[i][1].set_title(col_ref + ': ' + '%.2g' % imp_pred, loc = 'left')
                i = i + 1
                
        else:
            i = 0
            for col_ref in colunas:
                valores_real, valores_pred, tend_real, tend_pred, imp_real, imp_pred, labels, tem_pred, eh_intervalo = self.curva_tendencia(col_ref)
                cores_plot = cores[np.floor((tend_real + 1)*(N-1)/2).astype(int)]
                axs[i].imshow([cores_plot], aspect = 0.5*(valores_real.size/10), interpolation = 'spline16')
                axs[i].set_yticks([])
                if(eh_intervalo):
                    axs[i].set_xticks(valores_real)
                    axs[i].set_xticklabels(labels, rotation = 90)
                else:
                    axs[i].set_xticks(range(0, valores_real.size))
                    axs[i].set_xticklabels(valores_real.astype(str))
                axs[i].set_title(col_ref + ': ' + '%.2g' % imp_real, loc = 'left')
                i = i + 1
        plt.show()