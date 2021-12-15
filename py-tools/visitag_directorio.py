import os
from distutils.dir_util import copy_tree
import re
import subprocess
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
import sklearn.cluster as cluster

import scipy.stats as st
from tkinter import Tk
from tkinter import messagebox

import configparser

# returns confidence interval of mean
def confIntMean(a, conf=0.95):
  mean, sem, m = np.mean(a), st.sem(a), st.t.ppf((1+conf)/2., len(a)-1)
  return mean - m*sem, mean + m*sem


# Funciones para ver etiquetas en 3D
#############################################
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
        
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)       
        
# Funciones auxiliares #
########################

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]

def f(x): 
    if (x > 14):
        return np.inf
    elif (x < 2):
        return 0.0000001
    else:
        return (x**50)

# Función que busca el trayecto utiliza las variables globales.
# Trabaj sobre el dataframe dfn
     
def trayecto(zona, dfn):
    
    global color
    
    # Cogemos la zona, y una matriz X, Y, Z
    dfz = dfn[dfn['Site'] == zona];
    puntos = dfz[['X','Y','Z']];   
    if(len(puntos) < 2):
        return
    
    #dfn['Color'] = 1; 
        
    # Calculamos la distancia de todos con todos
    distancias = cdist(puntos, puntos);     
 
    # Buscamos los puntos mas alejados
    m,mm = np.where(distancias == np.amax(distancias))
    inicio = m[0]
    fin    = m[1]
    
    # Del primer punto de ablación al más alejado de este
    #inicio = 0
    #dd = distancias[0]
    #fin = np.where(dd == np.amax(dd))[0][0]
    
    # Elevar la distancia al cuadrado o al cubo, para que merezca la pena pasar por los puntos intermedios. 
    # Tenemos que poner distancia infinita si esta por encima de 8mm
    distancias = np.vectorize(f)(distancias)    
    
    # Buscamos los caminos más cortos    
    dist_matrix, predecessors = shortest_path(csgraph=distancias, directed=True, return_predecessors = True, method = 'FW')
    
    '''
    # Buscamos un punto inicial en los comentarios
    inicio = -1
    n = 0
    for index, row in dfz.iterrows():
        if ( not pd.isnull(row['Comment']) and len(row['Comment'].split('-')) > 1):
            if (row['Comment'].split('-')[1] == 'inicio'):
                inicio = n;                
            if (row['Comment'].split('-')[1] == 'fin'):
                fin = n;
        n = n + 1
      

    # Si no hay punto inicial cogemos los dos puntos mas alejados. 
    if (inicio == -1):
    ''' 
      
    camino = get_path(predecessors, inicio, fin);
    color = color + 1
    # Si es mayor que 1
    # Asignamos el valor de la secuencia. 
    n = 1;
    for index in camino:  
        dfn.loc[dfn['SiteIndex'] == dfz.iloc[index,2], ['Seq']] = n;   
        dfn.loc[dfn['SiteIndex'] == dfz.iloc[index,2], ['Color']] = color;                  
        n = n + 1;
    n = n - 1;
 
  #### Esto es para buscar el camino de vuelta en un bucle circular. ####
 
    # Borramos todos los puntos que hemos recorrido 
    dfz = dfn[dfn['Site'] == zona]
    
    for index, row in dfz.iterrows():
        if ( (row['Seq'] > 1) and (row['Seq'] < n ) ):
            dfz = dfz.drop(index)
                
    # Cogemos el inicio y el fin para el camino de vuelta
    p = -1
    for index, row in dfz.iterrows():
        p = p + 1     
        if (row['Seq'] == 1):
            inicio = p       
        if (row['Seq'] == n):
            fin = p                

    # Y buscamos el camino de vuelta circular. 
    puntos =  dfz[['X','Y','Z']];   
    distancias = cdist(puntos, puntos);
    distancias = np.vectorize(f)(distancias)    
    dist_matrix, predecessors = shortest_path(csgraph=distancias, directed=True, return_predecessors = True, method = 'FW') 
    camino = get_path(predecessors, fin, inicio);
    
    color = color + 1
    if (len(camino) > 1):    
        print("Cierra camino de zona: ", zona)
        for index in camino:  
            dfn.loc[dfn['SiteIndex'] == dfz.iloc[index,2], ['Seq']] = n;
            dfn.loc[dfn['SiteIndex'] == dfz.iloc[index,2], ['Color']] = color;              
            n = n + 1;    
    
def segmenta(dfn, N):
    
    global labels,color
    
    elementos = dfn['Comment'].value_counts().index.tolist();  

    if ((N == 0) and (len(elementos) > 1)):
    
        # Creamos las secciones en función de los comentarios y ponemos un color
        ultimo = "any"
       
        for index, row in dfn.iterrows():   
            if (pd.isnull(row['Comment'])):
                dfn.at[index,'Site'] = ultimo
                dfn.at[index,'Color'] = color
            elif (ultimo == row['Comment'].split('-')[0]):
                dfn.at[index,'Color'] = color
                dfn.at[index,'Site'] = ultimo
            elif (row['Comment'] == 'o'):
                dfn.at[index,'Color'] = 1
                dfn.at[index,'Site'] = 'o'
            else:
                ultimo = row['Comment'].split('-')[0]
                color = color + 1
                dfn.at[index,'Site'] = ultimo
                dfn.at[index,'Color'] = color
    else:
        # CLUSTERING   
        X = dfn[['X','Y','Z']].to_numpy();        
        kmeans = cluster.KMeans(n_clusters=N)
        kmeans = kmeans.fit(X)
        labels = kmeans.labels_
        for index, row in dfn.iterrows(): 
            dfn.at[index,'Site']  = "z" + str(labels[index])
            dfn.at[index,'Color'] = labels[index]
          
    return dfn

def procesa(dir_paciente, dfResumen, dfTotal, dfRfs, dflistado):
    
    global dfx, color;         
    color = 1
                 
 # Cargamos los datos
           
    input_file   = dir_paciente + "\\Sites.txt";
    input_file2  = dir_paciente + "\\VisiTagSessions.txt";    
    output_file   = dir_paciente + '\\..\\' + os.path.basename(os.path.dirname(input_file)) + '.csv';

    print("#######################################################")
    print(input_file)
    
    try:
        df  = pd.read_csv(input_file, sep='\s+', index_col=0, engine='python')
        df2 = pd.read_csv(input_file2, sep='\s+', index_col=0, engine='python')
        dfn = df.join(df2)
        dfn.reset_index(level=0, inplace=True)
        del df
        del df2
    except:
        print('Error procesando: ' + input_file)
        return dfResumen, dfTotal, dfRfs, dflistado
       
    if (confirm):
        nsegmentos = int(segmentos)
        
        while (True):
            try:
                dfn = segmenta(dfn,nsegmentos)
                # Mostramos las secciones. 
                try:
                    plt.close()   
                except Exception as e:
                    print(e)               
                    
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title("All points " + dir_paciente )
                ax.scatter(dfn['X'].to_numpy(),dfn['Y'].to_numpy(),dfn['Z'].to_numpy(), c = dfn['Color'].to_numpy())
                plt.draw()
                plt.pause(1)
            except Exception as e: 
                print("Error con la segmentación Automática")
                print(e)
                dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'No'}, ignore_index=True)
                plt.close()
                return dfResumen, dfTotal, dfRfs, dflistado            
            
            question = messagebox.askyesnocancel(
                  title="¿Aceptar Segmentación?",
                  message="Si - Aceptar \nNo - Reintentar con más segmentos \nCancelar - Descartar",
                  default=messagebox.YES)
                       
            if question:
                break;
            elif question is None:
                dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'No'}, ignore_index=True)
                plt.close()
                return dfResumen, dfTotal, dfRfs, dflistado 
            else:
                nsegmentos = nsegmentos + 1;
                if (nsegmentos > 5):
                    nsegmentos = 1
    else:
        try:
            dfn = segmenta(dfn,int(segmentos))
        except:
            print("Error con la segmentación Automática")
            dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'No'}, ignore_index=True)
            return dfResumen, dfTotal, dfRfs, dflistado            
    
    plt.close()
    # Descartamos los elementos con errores
    dfn = dfn[dfn['RFIndex'] > 0]
    dfn = dfn[dfn['ImpedanceDrop'] > 0]
    dfn = dfn[dfn['MaxTemperature'] > 0]
    dfn = dfn[dfn['MaxPower'] > 0]
    
    # Contamos Visitags totales
    visitags_totales = len(dfn.index)

    # Contamos Visitags que cumplen tagindex
    dfnd = dfn[dfn['RFIndex'] > int(tagindex)]    
    visitags_cumplen = len(dfnd.index)

    # Descartamos los elementos por Ablation Index
    dfn = dfn[dfn['RFIndex'] > int(tagindexd)]
    
    # Miramos cuantas secciones hay
    elementos = dfn['Site'].value_counts().index.tolist();
    
    # Ejecutamos el algoritmo en cada zona
    for zona in elementos:   
        trayecto(zona,dfn);

    # Dibujamos los trayectos            
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dfn['X'].to_numpy(),dfn['Y'].to_numpy(),dfn['Z'].to_numpy(), c = dfn['Color'].to_numpy())
    ax.set_title(zona + " " + dir_paciente.replace("\DIL-Navarra\Patient ","").replace("..\\","") )
    
    xyzn = zip(dfn['X'].to_numpy(),dfn['Y'].to_numpy(),dfn['Z'].to_numpy())
    labels = dfn['Seq'].to_numpy()

    for j, xyz_ in enumerate(xyzn): 
        annotate3D(ax, s = str(labels[j]), xyz=xyz_, fontsize=10, xytext=(-3,3),
        textcoords='offset points', ha='right',va='bottom')        
    plt.draw()
    plt.pause(1)

    question = messagebox.askyesnocancel(
                  title="¿Aceptar Secuencia de tags?",
                  message="Si - Aceptar \nNo - Mandar para revisión \nCancelar - Descartar",
                  default=messagebox.YES)
    print(question)
    if question:
        plt.close()
        print("Aceptado")
    elif question is None:    
        dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'R'}, ignore_index=True)
        plt.close()
        print("Descartado")
        return dfResumen, dfTotal, dfRfs, dflistado       
    else:
        dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'No'}, ignore_index=True)
        plt.close()
        print("copiar")
        print(dir_paciente)    
        print()
        print(dstdir)
        print("copiar-------------")
        copy_tree(dir_paciente, dstdir +  "\\revision\\" + os.path.basename(os.path.normpath(dir_paciente)))
        return dfResumen, dfTotal, dfRfs, dflistado 
            
    # Calculamos las Distancias
    try:
        for index, row in dfn.iterrows():    
            p1 = dfn.loc[(dfn['Seq'] == row['Seq'])     & (dfn['Site'] == row['Site']), ['X','Y','Z']]
            p2 = dfn.loc[(dfn['Seq'] == row['Seq'] + 1) & (dfn['Site'] == row['Site']), ['X','Y','Z']]
            d = cdist(p1,p2)            
            if (d.size > 0):
                dfn.loc[(dfn['Seq'] == row['Seq']) & (dfn['Site'] == row['Site']), ['Distancia']] = d[0][0]          
    except:
        print("Error calculando trayecto")
        dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'No'}, ignore_index=True)
        return dfResumen, dfTotal, dfRfs, dflistado
        
  
    if ('Distancia' in dfn.columns):
        
        dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'Si'}, ignore_index=True)
        
        # Calculamos el resumen de datos
        for zona in elementos:           
            dfx = dfn[dfn['Site'] == zona]
            
            visitags_secuencia    = dfx[dfx.Seq > 0].shape[0]
            visitags_gaps_55      = dfx[dfx.Distancia > 6].shape[0]
            visitags_gaps_6       = dfx[dfx.Distancia > 6].shape[0]
            visitags_gaps_65      = dfx[dfx.Distancia > 6].shape[0]
            visitags_totales_zona = len(dfx.index)
            # Contamos los que cumplen en zona
            dfxd = dfx[dfx['RFIndex'] > int(tagindex)]
            visitags_cumplen_zona = len(dfxd.index)
                
            # Intervalos de confianza          
            try:
                dfmm =  dfx[dfx["Distancia"] > 0]
                data = dfmm["Distancia"].to_numpy()
                intervalo95 = confIntMean(data, conf=0.99)
                intervalo90 = confIntMean(data, conf=0.95)
                intervalo85 = confIntMean(data, conf=0.90)
            except:
                print('Error con los intervalos de confianza')     
    
            
            print()
            print(zona)
            print('---------')
            print('Distancia Media ', dfx['Distancia'].mean()) 
            print('Distancia maxima ',dfx['Distancia'].max()) 
        
            newline = {'file': input_file.replace(rootdir,"").replace("Sites.txt",""),
                       'zona': zona,
                       'Distancia Media ':    dfx['Distancia'].mean(),
                       'Distancia Mediana ':  dfx['Distancia'].median(),
                       'Desviacion estandar ':dfx['Distancia'].std(),
                       'Distancia maxima ':   dfx['Distancia'].max(), 
                       '# Visitags': visitags_totales,         
                       '# Visitags >300' : visitags_cumplen,  
                       '# Visitags en Zona': visitags_totales_zona, 
                       '# Visitags en Zona >300' : visitags_cumplen_zona,                       
                       '# Visitags en Zona y linea': visitags_secuencia,
                       '# Gaps >5,5': visitags_gaps_55,
                       '# Gaps >6': visitags_gaps_6,
                       '# Gaps >6,5': visitags_gaps_65,
                       '# 99%': intervalo95, 
                       '# 95%': intervalo90, 
                       '# 90%': intervalo85}            
            
            dfResumen  = dfResumen.append(newline, ignore_index=True)        
        
            if ((zona.upper() == "DER") or (zona.upper() == "IZQ") or (zona == "z0") or (zona == "z1") ):               
                col = dir_paciente.replace(rootdir,"") + "_" + zona.upper()
                columna = dfx['Distancia'].dropna().reset_index(drop=True)
                dfTotal = pd.concat([dfTotal, columna], axis=1)   
                dfTotal.columns = [*dfTotal.columns[:-1], col]
    else:
        
        print("Error calculando trayecto")
        dflistado = dflistado.append({'directorio' : input_file, 'procesado' : 'No'}, ignore_index=True)
        return dfResumen, dfTotal, dfRfs, dflistado      
                        
    # Exportamos a csv.
    dfRfs = pd.concat([dfRfs, dfn])
    dfn.to_csv(output_file,sep = ';', decimal=',');
    return dfResumen, dfTotal, dfRfs, dflistado

#############################################################################
# INICIO DE PROGRAMA
#############################################################################

root = Tk()
root.withdraw()

config = configparser.ConfigParser()
config.read('visitag.ini')
rootdir      = config['DEFAULT']['Carpeta_Origen']
dstdir       = config['DEFAULT']['Carpeta_Destino']
tagindex     = config['DEFAULT']['Ablation_Index']
tagindexd    = config['DEFAULT']['Ablation_Index_Discard']
Study_Name   = config['DEFAULT']['Study_Name']
segmentos    = config['DEFAULT']['Segmentos']

if (config['DEFAULT']['Confirmar'] == "Si"):
    confirm      = True;
else: 
    confirm      = False;
    
if (config['DEFAULT']['Solo_cargar_datos'] == "Si"):
    procesar     = False;
else: 
    procesar      = True;    

if (config['DEFAULT']['Cargar_desde_Carto'] == "Si"):
    rootdir      = "F:\\CartoData\\Patients\\"
#   rootdir      = "F:\\"

print("Directorio Origen:  " + rootdir)
print("Directorio Destino: " + dstdir)
#userInput = input("Enter para continuar... ");

try:
    if(not os.path.isdir(dstdir)):                                            
        os.mkdir(dstdir)
except:
    print('Error Creando directorio')             
                            
'''
# Config Variables
rootdir      = "F:\\CartoData\\Patients\\"
dstdir       = "..\\Visitag-Estudios\\"
rootdir      = "..\\..\\Estudios"
dstdir       = "..\\..\\Estudios\\"
'''

# Variables
resumefile   = dstdir + "per-tag-statics.csv"
alldatafile  = dstdir + "alldata.csv"
allrfsfile   = dstdir + "allrfs.csv"
procesfiles  = dstdir + "procesed.csv"

subdir       = '\\Studies\\'
datadir      = '\\RawData\\'
mapsdir      = '\\Maps\\'
patientdata  = '\\Label.INI'
lista_pacientes = [];
dflistado = pd.DataFrame([], columns = ['directorio','procesado'])

if (rootdir      == "F:\\CartoData\\Patients\\"):
#if (rootdir      == "F:\\"):    
    print("Accediendo a los directorios de Carto")
    for paciente in os.listdir(rootdir):
        item = os.path.join(rootdir, paciente)        
        if (not os.path.isfile(item) and re.match('Patient(.*)', paciente)):        
            for estudio in os.listdir(rootdir + paciente + subdir):    
                item = rootdir + paciente + subdir + estudio + datadir;           
                mapd  = rootdir + paciente + subdir + estudio + mapsdir;                            
                if os.path.isdir(item):                                                                
                    for zipfile in os.listdir(item):
                        print(zipfile)
                        try:
                            found = re.search('Export_(.*)-(.*)-(.*)-(.*)-(.*)\\.zip$', zipfile)    
                            if (re.match(Study_Name, found[1])):
                                cf  = pd.read_csv(mapd + os.listdir(mapd)[0] + patientdata, sep='=')                   
                                finaldir = cf.iloc[2,1].strip() + '-' + cf.iloc[4,1].strip()
                                item = rootdir + paciente + subdir + estudio + datadir + zipfile;
                                try:
                                    if(not os.path.isdir(dstdir + finaldir)):                                            
                                        os.mkdir(dstdir + finaldir)
                                    lista_pacientes.append(dstdir + finaldir)
                                    # dflistado = dflistado.append({'directorio' : finaldir}, ignore_index=True)
                                    
                                except:
                                    print('Error Creando directorio')                        
                                try:
                                    cmd = ['7z', 'e', '-y', '-o' + dstdir + finaldir, item, 'VisitagExport\Sites.txt', 'VisitagExport\VisiTagSessions.txt']                                                                                                
                                    sp = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)                                                                     
                                    #copyfile(mapd + os.listdir(mapd)[0] + patientdata, dstdir + finaldir + patientdata)
                                except:
                                    print('Error descomprimiendo')                        
                        except:
                            print('Error con la carpeta')                                                            
else: 
    for paciente in os.listdir(rootdir):
        item = os.path.join(rootdir, paciente)        
        lista_pacientes.append(item)                                                                            
        # dflistado = dflistado.append({'directorio' : item, 'procesado' : 'Si'}, ignore_index=True)                                       

if (procesar):
    print("Procesando visitags...")                                 
    # Data Frame resumen
    dfResumen = pd.DataFrame([], columns = ['file','zona','Distancia Media ',
                                            'Distancia Mediana ','Desviacion estandar ','Distancia maxima ',
                                            '# Visitags','# Visitags >300',
                                            '# Visitags en Zona','# Visitags en Zona >300', '# Visitags en Zona y linea',
                                            '# Gaps >5,5', '# Gaps >6','# Gaps >6,5',
                                            '# 99%', '# 95%','# 90%'])
                      
    dfTotal = pd.DataFrame([], columns = [])                                       
    dfRfs   = pd.DataFrame([], columns = [])  

    for dir_paciente in lista_pacientes:
        print(dir_paciente)
        if (os.path.isfile(dir_paciente + '\\Sites.txt')):
            dfResumen, dfTotal, dfRfs, dflistado = procesa(dir_paciente, dfResumen, dfTotal, dfRfs, dflistado)
        
    dfResumen.to_csv(resumefile, sep = ';', decimal=',');
    dfTotal.to_csv(alldatafile, sep = ';', decimal=',');
    dfRfs.to_csv(allrfsfile, sep = ';', decimal=',');
    dflistado.to_csv(procesfiles, sep = ';', decimal=',');
    

#userInput = input("Fin del script... ");
