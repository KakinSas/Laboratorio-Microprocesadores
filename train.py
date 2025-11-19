#librerias 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats
import statsmodels.formula.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

#datasets 
#dataset1=csv_egresos.csv= data1.csv 
cat_dataset = pd.read_csv('data1.csv', sep =';', encoding='latin1')
#dataset2=Delitos_CSV.csv = data2.csv 
cat_MINPUB = pd.read_csv('data2.csv', sep =';', encoding='latin1')
df_del =pd.read_csv('data2.csv', sep =';')
#dataset3=Egresos_gendarmeria.csv= data3.csv
df_egr = pd.read_csv('data3.csv', sep =';', encoding='latin1')
df_fin = pd.merge(df_del, df_egr, on ='COD_DELITO')


df_fin.columns
df_fin.info()
df_fin2 = df_fin
df_final = df_fin2.drop_duplicates()

df_final["COD_PERS"].value_counts()
df_final.groupby(['MES_EGRESO', 'COD_PERS','Codigo', 'COD_DELITO'])['SCORE'].sum()
df_fin2.isnull().sum()
df_fin2.nunique()
df_fin2['COD_DELITO'].unique()
df_fin2['MES_EGRESO'].unique()

df_fin2['MES_EGRESO'].value_counts()
df_fin2['MES_EGRESO'] = df_fin2['MES_EGRESO'].astype(str)
df_fin2.groupby(['MES_EGRESO'])['SCORE'].sum()

score = df_fin2.groupby(['MES_EGRESO','COD_PERS']).agg({'SCORE': lambda x: x.sum()})
score.reset_index(inplace=True)

col =['COD_PERS', 'MES_EGRESO', 'SCORE', 'COD_DELITO','Codigo']
rfm = df_fin2[col]

rfm['MES_EGRESO'] = pd.to_datetime(rfm['MES_EGRESO'],errors ='coerce')
rfm['MES_EGRESO'].max()
f_corte = dt.datetime(2022,7,1)
rfm = rfm.drop_duplicates()
RFM1 = rfm.groupby('COD_PERS').agg({'MES_EGRESO': lambda x: (f_corte - x.max()).days})
RFM1['Frecuencia'] = (rfm.groupby(by=['COD_PERS'])['Codigo'].count()).astype(float)
RFM1['ScoreTotal'] = rfm.groupby(by=['COD_PERS']).agg({'SCORE': 'sum'})

RFM1.rename(columns={'MES_EGRESO': 'Egreso más reciente'}, inplace=True)

RFM1[RFM1['Egreso más reciente'] == 0]
RFM1[RFM1['Frecuencia'] == 0]
RFM1[RFM1['ScoreTotal'] == 0]
RFM1 = RFM1[RFM1['Egreso más reciente'] > 0]
RFM1.reset_index(drop=True,inplace=True)
RFM1 = RFM1[RFM1['Frecuencia'] > 0]
RFM1.reset_index(drop=True,inplace=True)
RFM1 = RFM1[RFM1['ScoreTotal'] > 0]
RFM1.reset_index(drop=True,inplace=True)

Data_RFM1 = RFM1[['Egreso más reciente','Frecuencia','ScoreTotal']]

data_log = np.log(Data_RFM1)
scaler = StandardScaler()
scaler.fit(data_log)
data_sc = scaler.transform(data_log)
df_norm = pd.DataFrame(data_sc, columns=Data_RFM1.columns)


#plots 
def plots_model():    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for x in RFM1.grupos.unique():        
        xs = RFM1[RFM1.grupos == x]['Egreso más reciente']
        zs = RFM1[RFM1.grupos == x]['Frecuencia']
        ys = RFM1[RFM1.grupos == x]['ScoreTotal']
        ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', label = x)

    plt.legend()
    plt.title('Clusters del Modelo KMeans')
    plt.savefig('clusters_plot.png')  # Guardar el gráfico como un archivo PNG


model = KMeans(n_clusters=4, init='k-means++', max_iter=301)
grupos = model.fit_predict(df_norm)
df_norm['grupos'] = grupos
RFM1['grupos'] = grupos
plots_model()
plt.show()
# segundo push