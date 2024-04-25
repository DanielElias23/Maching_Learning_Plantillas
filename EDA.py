import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler


##########################################################################################################################

                                                   #IMPORTANCION ARCHIVOS

housing = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv", sep='\t')

Hou = pd.DataFrame(housing)


##########################################################################################################################

                                                         #LIMPIEZA

print(Hou.shape)

#Muestra la cantidad de valores nulos por columnas solo los que si tienen valores
#print(Hou.isnull().sum()[Hou.isnull().sum() > 0])  

#Mustra todos los valores de las columnas para no equivocarse en el nombre
#print(Hou.columns.tolist())

#Elimina las columna con muchos valores nulos
Hou_subsample1 = Hou.drop(["Alley","Mas Vnr Type","Fireplace Qu","Pool QC","Fence","Misc Feature"], axis=1)

print(Hou_subsample1.isnull().sum()[Hou_subsample1.isnull().sum() > 0])



#Muestra la cantidad de filas que estan duplicadas
#print(Hou.duplicated().sum())

#Muestra un resumen completo de todas las columnas
#print(Hou.info())  

#Muestra todo el resudemn estadisitco de todas las columnas, pero no deben haber columnas string, solo numericas sino especifica
#que columna quieres saber poniendola dentro de describe, round es para redondiar en 3 digitos
#print(Hou.describe().round(3))

#Muestra resumen de esa columna en particular
#Hou["Sale Type"].describe()  

#Un subconjunto de solo numeros, para poder ver alguna correlacion
hous_num = Hou_subsample1.select_dtypes(include = ['float64', 'int64']) 

#Cambia todos los valores nulos, por el promedio de cada columna
Hou_new1 = hous_num.fillna(hous_num.mean())

#Muestra todas las correlaciones de todas las columnas
#for i in Hou_new1.corr().columns.tolist():
#           Hou_paso=Hou_new1.corr()[i]
#           Hou_paso2=(Hou_paso[(abs(Hou_paso)>0.5) & (abs(Hou_paso)<1)])
#           print(Hou_paso2)

#matriz de correlacion
#print(hous_num.corr())  

#Muestra la ultima columna SalePrice, menos el ultimo valor que es 1 consigo misma
Hou_new1_corr = Hou_new1.corr()['SalePrice'][:-1] 
print(Hou_new1_corr)


#print(hous_num_corr)

#solamente mostrara los valores de la columna mayores a 0.5 de mayor a menor
top_features = Hou_new1_corr[abs(Hou_new1_corr) > 0.5].sort_values(ascending=False)

#print(top_features)


#Explorando estas columnas que tienen un correlacion alta
#sns.boxplot(Hou_new1["SalePrice"])
#plt.show()
#sns.boxplot(Hou_new1["Overall Qual"])
#plt.show()

#plt.scatter(Hou_new1["SalePrice"], Hou_new1["Overall Qual"])
#plt.show()

q25, q50, q75 = np.percentile(Hou_new1["SalePrice"], [25, 50, 70])
resto= q75 - q25

min = q25 - 1.5*(resto)
max = q75 + 1.5*(resto)

#limites
print(len(Hou_new1["SalePrice"]))
print(len(Hou_new1["SalePrice"][Hou_new1["SalePrice"]>max]))
print(min, q25, q50, q75, max)

#Para eliminar outlier
Hou_new2 = Hou_new1[Hou_new1["SalePrice"]<max]
print(Hou_new2.shape)

#Lo muestra explicitamente
#print([x for x in Hou_new1["SalePrice"] if x > max])

#Lo de las correlaciones tiene muchos usos sobretodo para reducir dimensiones y encontrar regresiones



#######################################################################################################

#                                          CODIFICACIONES

###One hot encoder

#Es para columnas categoricas, divide una columna categorica en cierta cantidad de columnas nuevas,
#dependiendo de la cantidad de categorias unicas de la columnas, cada columna pondra 0 o 1 si es que
#la fila tiene esa nueva categoria o no


titanic = sns.load_dataset("titanic")
print(titanic.head())

#Dice la cantidad de categorias de la columnas
print(titanic['embark_town'].unique().tolist())

#Hace la codiciacion, pero es una data por separado
data = pd.get_dummies(titanic.embark_town)

#Para saber cuales filas no tienen categoria explicitamente
print(titanic.loc[titanic.embarked[titanic.embarked.isna()].index])

#Para hacer la codificacion añadiendo otra columna extra categoria NaN, para los que no tienen categoria
data = pd.get_dummies(titanic.embarked, dummy_na = True)

#Para sustituir la columna categorica inicial(se elimina), por la nueva estructura codificada
titanic = pd.concat([titanic.drop("embark_town", axis = 1), data], axis = 1)
titanic.head()


#Otro ejemplo en este caso son todas columnas categoricas
print(Hou_new1['MS SubClass'].unique().tolist())
print(Hou_new1['Mo Sold'].unique().tolist())
print(Hou_new1['Yr Sold'].unique().tolist())

#De esta forma se agregan automaticamente al final de la estructura todas las columans creadas
data1 = pd.get_dummies(data=Hou_new1, columns = ['MS SubClass', 'Mo Sold', 'Yr Sold'])
print(data1.head())"""

"""

#otra forma de hacerlo es con OneHotEncoder de sklearn, se hace de la siguiente forma para que el
#output sea un dataframe, este metodo tiene tambien la opcion de poner cuantas categorias, etc
OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")

#transforma valores nulos en objetos para que los considere la codificacion
titanic.embarked.fillna("nan", inplace = True)

data2 = OHE.fit_transform(titanic[["embarked"]])

#para unir las codificaciones creadas anterior mente al dataframe original
titanic2 = pd.concat([titanic, data2], axis=1).drop(columns=["embarked"])

print(titanic2.info())



###LabelEncoder

#Lo que hace esta codificacion es cambiar una columna con categorias en string, a pasar esas categorias
#a numeros, puede hacerse de dos formas

titanic = sns.load_dataset("titanic")

print(titanic['embark_town'].unique().tolist())

titanic.embarked.fillna("nan", inplace = True)

titanic_LE = titanic["embark_town"].replace({"Southampton":0, "Cherbourg":1, "Queenstown":2,"nan":3})

titanic = pd.concat([titanic.drop("embark_town", axis = 1), titanic_LE], axis = 1)


#La otra forma es con sklearn LabelEncoder

titanic = sns.load_dataset("titanic")

LB = LabelEncoder()

titanic.embark_town.fillna("nan", inplace = True)

titanic.embark_town=LB.fit_transform(titanic.embark_town)




#para codificar fechas se hace de la siguiente forma, es por trayectos de horas
data1['dep_timezone'] = pd.cut(data1.Dep_Hour, [0,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'])

data1['Month']= pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.month

data1['day_of_week'] = pd.to_datetime(data1['Date_of_Journey']).dt.day_name()


###OrdinalEncoder

titanic = sns.load_dataset("titanic")

OE = OrdinalEncoder()

titanic.embark_town.fillna("nan", inplace = True)

#Hace lo mismo que LabelEncoder, pero puede hacerlo con numeros y con string, con varias columnas simultaneamente, en este caso el
#remplazo por otra clasificacion es por orden, si es string es por abecedario de la inicial de la palabra, lo remplazara por 0, 1,
#,2, etc segun abecedario y si es por numero de menor a mayor, ejemplo 1000:2, 100:1, 1:0 de remplazo de clasificacion. 
titanic.embark_town=OE.fit_transform(titanic[["embark_town"]])

#En caso de que se haga fit y transform por separado, fit para practicar el modelo y transform para probar lo que quiere cambiar
#titanic.embark_town=OE.inverse_transform(titanic["embark_town"])



###########################################################################################################################

#                                               ESCALAMIENTOS
 
 
###scale

#este metodo estandariza, los hace oscilar creca de 0, puede contener valores negativos y positivos, puede ser mayores que 1 o 
#menores que -1    
                                               
titanic = sns.load_dataset("titanic")

titanic.age.fillna(25, inplace=True)                                               

#La formula que ocupa para escalar es:  (x-promedio)/desviacion_standar

print(titanic["age"].mean())
print(titanic["age"].std())
print(len(titanic["age"]))


edad = scale(titanic["age"], axis=0)


print(edad)

###StandardScaler

#este metodo estandariza        
                                               
titanic = sns.load_dataset("titanic")

titanic.age.fillna(25, inplace=True)                                               

#La formula que ocupa para escalar es:  (x-promedio)/desviacion_standar
#Scale con StandardScaler son iguales pero se ocupan diferentes en sintaxis

print(titanic["age"].mean())
print(titanic["age"].std())
print(len(titanic["age"]))

SS=StandardScaler()
edad = SS.fit_transform(titanic[["age"]])

print(edad)

###MinMaxScaler

#Este metodo es normalizador, pone los valores entre 0 y 1

titanic = sns.load_dataset("titanic")

titanic.age.fillna(25, inplace=True) 

#La formula que se ocupa es:  (x-min(x)) / (max(x)-min(x))

print(titanic["age"].mean())
print(titanic["age"].std())
print(titanic["age"].min())
print(titanic["age"].max())

MMS = MinMaxScaler()

edad = MMS.fit_transform(titanic[["age"]])

print(edad)




