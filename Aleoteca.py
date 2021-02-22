import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go

class CortaIntervalosQuasiUniforme:
    
    def __init__(self, vetor, num_por_div):
        self.vetor = vetor
        self.num_por_div = num_por_div
        
        self.valores = None #Armazena os valores unicos
        self.qtds = None #Armazena as quantidades de cada valor unico
        self.inds_rec = None #Armazena aos indices de reconstroção do vetor self.vetor
        
        self.ind_cres = None #Armazena o indice dos intervalos pela ordenação crescente (e depois o tratado conflito tb)
        self.ind_decr = None #Armazena o indice dos intervalos pela ordenação decrescente
        self.ind_max = None #Armazena o número de intervalos que vamos ter no final
        
        self.qtds_sc = None #Armazena as quantidades em cada intervalo que inicialmente coincidiram cres. e decr.
        self.flag_sc = None #Armazena um booleano indicando as posições em self.valores que inicialmente não teve conflito
        #OBS: No final, self.qtds_sc já vai conter a quantidade de elementos em cada indice dos intervalos
        
        self.max_min_ajust = None #Armazena os pontos de corte max_min ajustados (vai servir pra encontrar algo_signif)
        self.min_ajust = None #Armazena a informacao de menor valor no intervalo
        self.max_ajust = None #Armazena a informacao de maior valor no intervalo
        
        self.algo_signif = None #Armazena o melhor valor de algarismos significativos para diferenciar os intervalos
        self.max_min_signif = None #Armazena os pontos de corte com melhor numero de algarismos significativos
        self.min_signif = None #Armazena a informacao de menor valor no intervalo com melhor algo_signif
        self.max_signif = None #Armazena a informacao de maior valor no intervalo com melhor algo_signif
        
        self.calcula_intervalos()
        
    def separa_intervalos_cres_decr(self):
        
        #Faz virar um contador (inteiros consecutivos)
        def transforma_em_inteiros_consecutivos(div):
            div_unico = np.unique(div) #Pega todos os valores unicos
            mp = np.arange(0, div[-1] + 1) #Cria um vetor contador com o tamanho do maior elemento do vetor original
            mp[div_unico] = np.arange(0, div_unico.size) #Nas pos. dos valores únicos, trocar por outro vetor contador
            ind = mp[div] #Nosso vetor final é pegar o vetor mp nas posição dos valores de div
            #OBS: Se div tem o msm valor duas vezes, vamos pegar o mesmo valor de mp duas vezes (é o que queremos)
            return ind
        
        #Pega os valores unicos, indices para reconstruir o vetor e a quantidade de cada valor unico
        self.valores, self.inds_rec, self.qtds = np.unique(self.vetor, return_inverse = True, return_counts = True)
        #OBS: o código valores[inds_rec] reconstrói o vetor self.vetor
        
        div = np.cumsum(self.qtds) // self.num_por_div #Faz a soma acumulada e as divisoes inteiras para separar intervalos
        div_cres = div - np.insert(np.where(np.diff(div) > 1, 1, 0), 0, 0) #Em saltos nas divisões, gera um interv a mais
        self.ind_cres = transforma_em_inteiros_consecutivos(div_cres)
        #Terminei de fazer o primeiro indice (separação crescente)
        
        div = np.cumsum(self.qtds[::-1]) // self.num_por_div #Soma acum e divisao inteira em order descrescente
        div_decr = div - np.insert(np.where(np.diff(div) > 1, 1, 0), 0, 0) #Novamente trata os saltos da divisão inteira
        div_decr = div[-1] - div_decr[::-1] #Desinverte o vetor e corrige pela divisão máxima para ficar crescente
        self.ind_decr = transforma_em_inteiros_consecutivos(div_decr)
        #Terminei de fazer o segundo indice (separação decrescente)
        
        #O maior indice que existe entre as duas separações de intervalo
        self.ind_max = max(self.ind_cres[-1], self.ind_decr[-1]) 
        
    def conta_qtds_intervalos_sem_conflito(self):
        self.flag_sc = self.ind_cres == self.ind_decr #vetor de booleanos para saber onde os indices batem
        qtds_sc = self.qtds[self.flag_sc]
        ind_sc = self.ind_cres[self.flag_sc]
        
        #Faz um group by na mão
        #Pega um vetor ordenado que será a coluna para agrupamento, pega os indices onde o valor dele muda
        #Splita o outro vetor nos pontos onde o vetor de groupby muda de valor
        ind_sc_unico, primeira_ocorrencia = np.unique(ind_sc, return_index = True)
        vetor_agrupado = np.split(qtds_sc, primeira_ocorrencia[1:])
        qtds_agrupado = np.array([np.sum(v) for v in vetor_agrupado])
        
        #Cria um vetor em que o indice é o indice do intervalo e o valor a quantidade já sem conflito no intervalo
        #Dessa forma se tiver intervalos ainda não preenchidos (sem conflito) eu vou saber (pois vai estar zerado)
        self.qtds_sc = np.zeros(self.ind_max + 1)
        self.qtds_sc[ind_sc_unico] = qtds_agrupado
    
    def trata_intervalos_conflitantes(self):
        flag_cc = ~self.flag_sc #vetor de booleanos para saber onde os indices não batem
        qtds_cc = self.qtds[flag_cc]
        ind_cres_cc = self.ind_cres[flag_cc]
        ind_decr_cc = self.ind_decr[flag_cc]
        
        ind_cc = np.array(list(zip(ind_cres_cc, ind_decr_cc))) #Agrupa em um vetor em que cada elemento é um par zipado
        #Passa a dimensao (axis = 0) para mantem o par
        ind_cc_unico, primeira_ocorrencia, qtd_oc = np.unique(ind_cc, return_index = True, return_counts = True, axis = 0) 
        #Vamos usar primeira_ocorrencia para saber onde começa e termina um agrupamento, então adicionamos qtd_oc[-1]
        primeira_ocorrencia = np.append(primeira_ocorrencia, primeira_ocorrencia[-1] + qtd_oc[-1])
        
        #Para cada par com conflito, pegamos a posição do começo do conflito e do final do conflito
        #Vamos tratando o começo e o final simultaneamente até que as posições se encontrem
        for i in range(qtd_oc.size):
            ind_cres, ind_decr = ind_cc_unico[i]
            pos_ini = primeira_ocorrencia[i]
            pos_fim = primeira_ocorrencia[i+1]-1
            while pos_ini <= pos_fim:
                #Dá preferência pelo indice que tiver menor número de elementos até o momento
                if self.qtds_sc[ind_cres] < self.qtds_sc[ind_decr]:
                    ind_cres_cc[pos_ini] = ind_cres
                    self.qtds_sc[ind_cres] += qtds_cc[pos_ini]
                    pos_ini += 1
                elif self.qtds_sc[ind_cres] > self.qtds_sc[ind_decr]:
                    ind_cres_cc[pos_fim] = ind_decr
                    self.qtds_sc[ind_decr] += qtds_cc[pos_fim]
                    pos_fim -= 1
                #Se os dois indices tiver com os mesmos valores
                else:
                    #Se pos_ini e pos_fim ainda não se encontraram, então trata os dois lados juntos
                    if pos_ini != pos_fim:
                        ind_cres_cc[pos_ini] = ind_cres
                        self.qtds_sc[ind_cres] += qtds_cc[pos_ini]
                        pos_ini += 1
                        ind_cres_cc[pos_fim] = ind_decr
                        self.qtds_sc[ind_decr] += qtds_cc[pos_fim]
                        pos_fim -= 1
                    #Se pos_ini e pos_fim já se encontraram, escolhe de forma aleatória como será o último tratamento
                    else:
                        ind = int(round((ind_cres + ind_decr)/2)) #Faz a conta do indice médio arredondado
                        #atribui o elemento para o indice em que o ind ficou mais próximo (em módulo)
                        if abs(ind - ind_cres) < abs(ind - ind_decr):
                            ind_cres_cc[pos_ini] = ind_cres
                            self.qtds_sc[ind_cres] += qtds_cc[pos_ini]
                            pos_ini += 1
                        else:
                            ind_cres_cc[pos_fim] = ind_decr
                            self.qtds_sc[ind_decr] += qtds_cc[pos_fim]
                            pos_fim -= 1
        
        #troca os indices em ind_cres que tinham conflitos pelos indices com conflitos tratados
        self.ind_cres[flag_cc] = ind_cres_cc
    
    def calcula_info_discretizacao(self):
        ind_unico, primeira_ocorrencia = np.unique(self.ind_cres, return_index = True)
        vetor_agrupado = np.split(self.valores, primeira_ocorrencia[1:])
        max_min_agrupado = np.array([np.array([np.min(v), np.max(v)]) for v in vetor_agrupado])
        
        self.max_min_ajust = (max_min_agrupado[1:, 0] + max_min_agrupado[:-1, 1])/2
        
        self.min_ajust = np.insert(self.max_min_ajust, 0, max_min_agrupado[0, 0])
        self.max_ajust = np.append(self.max_min_ajust, max_min_agrupado[-1, 1])
    
    def calcula_melhor_algo_signif(self):
        algo_signif = 1
        str_conv = '%.' + str(algo_signif) + 'g'
        cortes_interv = np.array([float(str_conv%self.max_min_ajust[i]) for i in range(self.max_min_ajust.size)])
        if(cortes_interv.size > 1): #Se houver mais de um corte de intervalos (ou seja, mais de 2 intervalos)
            flag = np.min(np.diff(cortes_interv))
            while flag == 0:
                algo_signif += 1
                str_conv = '%.' + str(algo_signif) + 'g'
                cortes_interv = np.array([float(str_conv%self.max_min_ajust[i]) for i in range(self.max_min_ajust.size)])
                flag = np.min(np.diff(cortes_interv))
            
        self.algo_signif = algo_signif
        self.max_min_signif = cortes_interv
        
        self.min_signif = np.insert(self.max_min_signif, 0, float(str_conv%self.min_ajust[0]))
        self.max_signif = np.append(self.max_min_signif, float(str_conv%self.max_ajust[-1]))
    
    def calcula_intervalos(self):
        self.separa_intervalos_cres_decr()
        self.conta_qtds_intervalos_sem_conflito()
        if(np.sum(self.flag_sc) !=  self.flag_sc.size): #Se tiver conflito
            self.trata_intervalos_conflitantes()
        self.calcula_info_discretizacao()
        self.calcula_melhor_algo_signif()
    
    def vetor_discretizado(self):
        return self.ind_cres[self.inds_rec]
    
    def df_info_discretizacao(self):
        min_str = [str(v) for v in self.min_signif]
        max_str = [str(v) for v in self.max_signif]
        min_str = [v if v[-2:] != '.0' else v[:-2] for v in min_str]
        max_str = [v if v[-2:] != '.0' else v[:-2] for v in max_str]
        string_interv = ['('+min_str[i]+', '+max_str[i]+']' for i in range(self.qtds_sc.size)]
        df = pd.DataFrame(zip(self.qtds_sc, self.min_ajust, self.max_ajust, string_interv), 
                          columns = ['QTD', 'Min', 'Max', 'Str'])
        return df
    
    #vetor_discretizado() retorna o vetor dado como entrada com a discretização feita
    #df_info_discretizacao() retorna as informações sobre os intervalos da discretização

#######################################

class Aletricas:
    
    def __init__(self, y, y_prob, num_por_div = None):
        self.y = y
        self.y_prob = y_prob
        
        #Variaveis caso queira fazer as contas por intervalos de prob
        self.num_por_div = num_por_div
        self.interv = None
        
        self.qtd_tot = None
        self.qtd1_tot = None
        self.qtd0_tot = None
        
        self.y_prob_unico = None
        self.qtds = None
        self.qtds1 = None
        
        self.qtds_acum = None
        self.qtds1_acum = None
        self.qtds0_acum = None
        
        #_c indica o conjunto complementar (o que ainda não foi somado)
        self.qtds_acum_c = None
        self.qtds1_acum_c = None 
        self.qtds0_acum_c = None 
        
        #vp: verdadeiro positivo, p_tot: total de positivos
        #vn: verdadeiro negativo, n_tot: total de negativos
        self.curva_tvp = None #Armazena a curva de taxa de verdadeiro positivo (vp / p_tot)
        self.curva_tvn = None #Armazena a curva de taxa verdadeiro negativo (vn / n_tot)
        self.auc = None
        
        self.curva_revoc1 = None #Armazena a curva de revocacao de 1
        self.curva_revoc0 = None #Armazena a curva de revocacao de 0
        self.pos_max_dif = None
        self.ks = None
        
        self.curva_ig = None #Armazena a curva de ganho de informação
        self.pos_max_ig = None
        self.ig = None
        
        self.vetor_p0_ig_2d = None
        self.vetor_p1_ig_2d = None
        self.vetor_ig_2d = None
        self.pos_max_ig_2d = None
        self.ig_2d = None
        
        self.calcula_metricas()
        
    def ordena_probs(self):
        self.qtd_tot = self.y.size
        
        if(self.num_por_div != None):
            self.interv = CortaIntervalosQuasiUniforme(self.y_prob, num_por_div = self.num_por_div)
            self.y_prob = self.interv.vetor_discretizado()
        
        inds_ordenado = np.argsort(self.y_prob)
        self.y_prob_unico, primeira_ocorrencia, self.qtds = np.unique(self.y_prob[inds_ordenado], 
                                                                      return_index = True, return_counts = True)
        y_agrup = np.split(self.y[inds_ordenado], primeira_ocorrencia[1:])
        self.qtds1 = np.array([np.sum(v) for v in y_agrup])
        
        self.qtds_acum = np.cumsum(self.qtds) 
        self.qtds1_acum = np.cumsum(self.qtds1)
        self.qtds0_acum = self.qtds_acum - self.qtds1_acum
        
        self.qtd1_tot = self.qtds1_acum[-1]
        self.qtd0_tot = self.qtd_tot - self.qtd1_tot
        
        self.qtds_acum_c = self.qtd_tot - self.qtds_acum
        self.qtds1_acum_c = self.qtd1_tot - self.qtds1_acum
        self.qtds0_acum_c = self.qtd0_tot - self.qtds0_acum
    
    def calcula_roc(self):
        #Estima a área abaixo da curva por Soma de Riemann
        def area(x,y):
            dx = np.diff(x)
            h = (y[:-1] + y[1:])/2
            A = np.sum(h*dx)
            return A
        
        if(self.qtd0_tot == 0):
            self.curva_tnp = np.repeat(1, self.qtds0_acum.size)
        else:
            self.curva_tnp = self.qtds0_acum/self.qtd0_tot
            
        if(self.qtd1_tot == 0):
            self.curva_tvp = np.repeat(1, self.qtds1_acum.size)
        else:
            self.curva_tvp = self.qtds1_acum_c/self.qtd1_tot
            
        self.auc = area(self.curva_tnp, self.curva_tvp)
    
    def calcula_ks(self):
    
        if(self.qtd0_tot == 0):
            self.curva_revoc0 = np.repeat(1, self.qtds0_acum.size)
        else:
            self.curva_revoc0 = self.qtds0_acum/self.qtd0_tot

        if(self.qtd1_tot == 0):
            self.curva_revoc1 = np.repeat(1, self.qtd1_tot.size)
        else:
            self.curva_revoc1 = self.qtds1_acum/self.qtd1_tot
        
        curva_dif = self.curva_revoc0 - self.curva_revoc1
        self.pos_max_dif = np.argmax(curva_dif) #Pega as posições em que encontrou o máximo
        
        #self.ks = np.max(curva_dif[self.pos_max_dif])
        #Pega o valor máximo (tenta ver se pos_max é um vetor ou um número)
        try:
            self.ks = curva_dif[self.pos_max_dif[0]]
        except:
            self.ks = curva_dif[self.pos_max_dif]
    
    def calcula_ig(self):
        #Calcula a Entropia de Shannon
        def entropia_shannon(p0, p1):
            if p0 == 0 or p1 == 0:
                return 0
            else:
                return -p0*np.log2(p0) - p1*np.log2(p1)
        
        p1 = self.qtd1_tot/self.qtd_tot
        entropia_ini = entropia_shannon(1-p1, p1)

        if(entropia_ini == 0):
            self.curva_ig = np.repeat(1, self.qtds_acum.size)
        else:
            #O último corte por definição não dá informação nenhuma, então nem faz a conta (por isso o [:-1])
            qtds_acum = self.qtds_acum[:-1]
            qtds1_acum = self.qtds1_acum[:-1]
            p1_acum = qtds1_acum/qtds_acum
            p0_acum = 1 - p1_acum
            entropia_parcial = np.array([entropia_shannon(p0_acum[i], p1_acum[i]) for i in range(qtds_acum.size)]) 

            qtds_acum_c = self.qtds_acum_c[:-1]
            qtds1_acum_c = self.qtds1_acum_c[:-1]
            p1c_acum = qtds1_acum_c/qtds_acum_c
            p0c_acum = 1 - p1c_acum
            entropia_parcial_c = np.array([entropia_shannon(p0c_acum[i], p1c_acum[i]) for i in range(qtds_acum_c.size)])

            entropia = (entropia_parcial*qtds_acum + entropia_parcial_c*qtds_acum_c)/self.qtd_tot
            #Coloca o valor [-1] que removemos no começo do calcula da entropia
            entropia = np.append(entropia, entropia_ini)
            self.curva_ig = (entropia_ini - entropia)/entropia_ini
        
        self.pos_max_ig = np.argmax(self.curva_ig) #Pega as posições em que encontrou o máximo
        #self.ig = np.max(self.curva_ig[self.pos_max_ig])
        #Pega o valor máximo (tenta ver se pos_max é um vetor ou um número)
        try:
            self.ig = self.curva_ig[self.pos_max_ig[0]] 
        except:
            self.ig = self.curva_ig[self.pos_max_ig]
        
    def calcula_ig_2d(self):
        #Calcula a Entropia de Shannon
        def entropia_shannon(p0, p1):
            if p0 == 0 or p1 == 0:
                return 0
            else:
                return -p0*np.log2(p0) - p1*np.log2(p1)
        
        p1_ini = self.qtd1_tot/self.qtd_tot
        entropia_ini = entropia_shannon(1-p1_ini, p1_ini) 
        
        if(entropia_ini == 0):
            vetor_p0_p1 = np.array([np.array([u, v]) for u in self.y_prob_unico[:-1] for v in self.y_prob_unico[:-1] if u < v])
            self.vetor_p0_ig_2d = vetor_p0_p1[:, 0]
            self.vetor_p1_ig_2d = vetor_p0_p1[:, 1]
            self.vetor_ig_2d = np.repeat(1, self.vetor_p0_ig_2d.size)
        else:
            vetor_p0 = np.array([])
            vetor_p1 = np.array([])
            vetor_entropia = np.array([])
            vetor_ig = np.array([])
            # -1 pois como já discutido, o último corte por definição não trás ganho de informação
            num_loop = self.y_prob_unico.size-1
            #Subtrai mais um aqui pq queremos garantir que todo o loop tem um intervalo de resto
            for i in range(num_loop-1):
                start_loop2 = i + 1 #O segundo loop começa sempre 1 a frente pq queremos que sobre um intervalo de resto
                vetor_p0 = np.append(vetor_p0, np.repeat(self.y_prob_unico[i], num_loop - start_loop2))
                qtd_acum = self.qtds_acum[i]
                qtd1_acum = self.qtds1_acum[i]
                p1 = qtd1_acum/qtd_acum
                entropia_parcial = entropia_shannon(1-p1, p1)
                
                entropia_aux = entropia_parcial*qtd_acum/self.qtd_tot
                
                vetor_p1 = np.append(vetor_p1, self.y_prob_unico[start_loop2:num_loop])
                qtd_acum_c = self.qtds_acum_c[start_loop2:num_loop]
                qtd1_acum_c = self.qtds1_acum_c[start_loop2:num_loop]
                p1c = qtd1_acum_c/qtd_acum_c
                p0c = 1 - p1c
                entropia_parcial_c = np.array([entropia_shannon(1-p1c[i], p1c[i]) for i in range(qtd_acum_c.size)])
                
                qtd_resto = self.qtd_tot - qtd_acum - qtd_acum_c
                qtd1_acum_resto = self.qtd1_tot - qtd1_acum - qtd1_acum_c
                p1r = qtd1_acum_resto/qtd_resto
                p0r = 1 - p1r
                entropia_parcial_r = np.array([entropia_shannon(1-p1r[i], p1r[i]) for i in range(qtd_resto.size)])
                
                entropia = entropia_aux + (entropia_parcial_c*qtd_acum_c + entropia_parcial_r*qtd_resto)/self.qtd_tot
                vetor_entropia = np.append(vetor_entropia, entropia)
                    
            self.vetor_ig_2d = (entropia_ini - vetor_entropia)/entropia_ini
            self.vetor_p0_ig_2d = vetor_p0
            self.vetor_p1_ig_2d = vetor_p1
        
        if(self.vetor_ig_2d.size > 0): #Se tem pelo menos um valor de ganho de informação
            self.pos_max_ig_2d = np.argmax(self.vetor_ig_2d) #Pega as posições em que encontrou o máximo
            try:
                self.ig_2d = self.vetor_ig_2d[self.pos_max_ig_2d[0]] 
            except:
                self.ig_2d = self.vetor_ig_2d[self.pos_max_ig_2d]
    
    def calcula_metricas(self):
        self.ordena_probs()
        if(self.y_prob_unico.size > 1):
            self.calcula_roc()
            self.calcula_ks()
            self.calcula_ig()
            if(self.y_prob_unico.size > 2):
                self.calcula_ig_2d()
    
    def valor_metricas(self):
        d = {}
        d['AUC'] = self.auc
        d['KS'] = self.ks
        d['IG'] = self.ig
        d['IG_2D'] = self.ig_2d
        return pd.Series(d, index = d.keys())
    
    def valor_prob_ig(self):
        d = {}
        prob_corte = None
        p0_corte = None
        p1_corte = None
        if(self.num_por_div != None):
            if(self.pos_max_ig != None):
                prob_corte = (self.interv.min_ajust[self.pos_max_ig] + self.interv.max_ajust[self.pos_max_ig])/2
            if(self.pos_max_ig_2d != None):
                pos_p0_aux = int(self.vetor_p0_ig_2d[self.pos_max_ig_2d])
                pos_p1_aux = int(self.vetor_p1_ig_2d[self.pos_max_ig_2d])
                p0_corte = (self.interv.min_ajust[pos_p0_aux] + self.interv.max_ajust[pos_p0_aux])/2
                p1_corte = (self.interv.min_ajust[pos_p1_aux] + self.interv.max_ajust[pos_p1_aux])/2
        else:
            if(self.pos_max_ig != None):
                prob_corte = self.y_prob_unico[self.pos_max_ig]
            if(self.pos_max_ig_2d != None):
                p0_corte = self.vetor_p0_ig_2d[self.pos_max_ig_2d]
                p1_corte = self.vetor_p1_ig_2d[self.pos_max_ig_2d]
        d['Prob_Corte'] = prob_corte
        d['Prob0_Corte'] = p0_corte
        d['Prob1_Corte'] = p1_corte
        return pd.Series(d, index = d.keys())
    
    def grafico_roc(self):
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = [6, 4])
            if(self.y_prob_unico.size > 1):
                axs.plot(self.curva_tnp, self.curva_tvp, color = 'blue', label = 'Curva ROC')
                axs.plot([0, 1], [1, 0], color='k', linestyle='--', label = 'Linha de Ref.')
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Taxa de Verdadeiro Negativo')
            axs.set_ylabel('Taxa de Verdadeiro Positivo')
            plt.show()
        
    def grafico_revocacao(self):
        if(self.num_por_div != None):
            y_prob_plot = [x for y in zip(self.interv.min_ajust, self.interv.max_ajust) for x in y]
            curva_revoc0_plot = np.repeat(self.curva_revoc0, 2)
            curva_revoc1_plot = np.repeat(self.curva_revoc1, 2)
            pos_max = (self.interv.min_ajust[self.pos_max_dif] + self.interv.max_ajust[self.pos_max_dif])/2
        else:
            y_prob_plot = self.y_prob_unico
            curva_revoc0_plot = self.curva_revoc0
            curva_revoc1_plot = self.curva_revoc1
            pos_max = self.y_prob_unico[self.pos_max_dif]
        
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = [6, 4])
            if(self.y_prob_unico.size > 1):
                axs.plot(y_prob_plot, curva_revoc0_plot, color = 'blue', label = 'Revocação_0')
                axs.plot(y_prob_plot, curva_revoc1_plot, color = 'red', label = 'Revocação_1')
                axs.vlines(pos_max, 0, 1, linestyle='--')
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Revocação')
            plt.show()
    
    def grafico_informacao(self):
        if(self.num_por_div != None):
            y_prob_plot = [x for y in zip(self.interv.min_ajust, self.interv.max_ajust) for x in y]
            curva_ig_plot = np.repeat(self.curva_ig, 2)
            pos_max = (self.interv.min_ajust[self.pos_max_ig] + self.interv.max_ajust[self.pos_max_ig])/2
        else:
            y_prob_plot = self.y_prob_unico
            curva_ig_plot = self.curva_ig
            pos_max = self.y_prob_unico[self.pos_max_ig]
        
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = [6, 4])
            if(self.y_prob_unico.size > 1):
                axs.plot(y_prob_plot, curva_ig_plot, color = 'blue', label = 'Ganho de Informação')
                axs.vlines(pos_max, 0, self.ig, linestyle='--')
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Ganho de Informação')
            plt.show()
        
    def grafico_informacao_2d(self):
        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize = [8, 6])
            axs = fig.add_subplot(111, projection='3d')
            if(self.y_prob_unico.size > 2):
                cmap = plt.get_cmap("Blues") #Mapa de cores
                col = np.arange(self.vetor_ig_2d.size)
                #plot_trisurf
                axs.scatter(self.vetor_p0_ig_2d, self.vetor_p1_ig_2d, self.vetor_ig_2d, 
                            c = self.vetor_ig_2d, marker = 'o', cmap = cmap, alpha = 1)
            axs.set_xlabel('Probabilidade de Corte 0')
            axs.set_ylabel('Probabilidade de Corte 1')
            axs.set_zlabel('Ganho de Informação')
            plt.show()
        
    def grafico_informacao_2d_plotly(self):
        if(self.y_prob_unico.size > 2):
            data = [go.Scatter3d(x = self.vetor_p0_ig_2d, y = self.vetor_p1_ig_2d, z = self.vetor_ig_2d,
                                 mode = 'markers', 
                                 marker = dict(size=5,color=self.vetor_ig_2d, colorscale='Viridis',opacity=0.8))]
            layout = go.Layout(
                showlegend = True,
                scene = go.layout.Scene(
                    xaxis = go.layout.scene.XAxis(title='Prob0'),
                    yaxis = go.layout.scene.YAxis(title='Prob1'),
                    zaxis = go.layout.scene.ZAxis(title='IG_2D')
                )
            )
            fig = go.Figure(data = data, layout = layout)
            fig.show()

##############################