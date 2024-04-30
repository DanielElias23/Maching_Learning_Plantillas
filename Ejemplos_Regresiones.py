                                         ___________________________________
                                         |                                  |
                                         |             INDICE               |
                                         |__________________________________|

        1. CORRELACION Y TEST DE DISTRIBUCION              "Ocupar matriz de correlacion y como hacer test de normalidad"
        2. REGRESION LINEAL                                "Un ejemplo de como ocupar la regresion lineal"
        3. REGRESION POLINOMICA                            "Como hacer descomposicion polinomia de las caracterisitcas" 
        4. REGRESION LINEAL MULTIPLE                       "Modelo mas complejo de regresion lineal con mas variables"
        5. REGRESION LINEAL MULTIPLE SOBREAJUSTADO         "Problema de tener muchas caracteristicas"
        6. VALIDACION CRUZADA                              "Ejemplo de como ocupar la validacion cruzada" 
        7. REGRESIONES CON VALIDACION CRUZADA              "Regresiones con validacion cruzada incluida"
        8. K-FOLD                                          "Ocupar K-FOLD"
        9. GRIDSEARCH                                      "Gridsearch busca hiperparametros de forma facil"  
       10. MODELOS CON REGULACION Y GRADIENTE DESCENDIENTE "DEMOSTRACION DE MODELOS DE REGRESION CON REGULACION"
       11. COMPARACIONES DE LOS MODELO DE REGRESION        "COMO ELEGIR EL MODELO DE REGRESION Y SUS HIPERPARAMETROS"
       12. MODELOS DE REGULACION Y SOBREAJUSTE             "TODO SOBRE MODELOS COMPLEJOS DE REGULACION"


#Cada seccion tiene ejemplo completos sobre el tema

###########################################################################################################################

import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pylab as plt



from scipy.stats import boxcox
from scipy.stats.mstats import normaltest

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline



data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')
print(data.head())
print(data.shape)
print(data.columns.tolist())
print(data.info())
print(data.describe())


###########################################################################################################################

                                      #ELECCION DE MODELOS Y CONFIGURACIONES

#Para poder encontrar un grafico que sea una regresion lineal, podemos ver todos los graficos juntos con el metodo pairplot()
#Muestra graficos para todas las columnas contra todas las otras, graficos de dispercion, en el caso que le toque con sigo
#mismo hace un grafico de barra tomando 1 columna de datos, OJO SOLO LO HARA CON COLUMNA QUE TENGAN DATOS NUMERICOS,las
#columna con datos string no las mostrara las saltara, el orden es como se muestra la tabla y hacia abajo el mismo orden
#de izquierda a derecha, el misma de arriba hacia abajo
#sns.pairplot(data)
#plt.show()


#Para graficar el grafico scatter y le regresion con un posible rango
fig = plt.subplots(figsize = (12,8), ncols=1,sharey=False)
sns.scatterplot( x = data.enginesize, y = data.price)
sns.regplot(x=data.enginesize, y=data.price)
 
#sns.scatterplot(x = data.horsepower,y = data.price, ax=ax2)
#sns.regplot(x=data.horsepower, y=data.price, ax=ax2)
plt.show()

#residplot grafica los residuos o el llamado error proyectado en un nuevo grafico mostrado solo en "y", cuanto se distancia
#de la regresion anterior cada dato
plt.subplots(figsize = (12,8))
sns.residplot(data, x=data["enginesize"], y=data["price"])
plt.show()

#En este caso se le agrega una linea que representa de la mejor forma los datos
plt.subplots(figsize = (12,8))
sns.residplot(data, x=data["enginesize"], y=data["price"], lowess=True, line_kws=dict(color="r"))
plt.show()
#En este caso no se cumple la homocedasticidad

def plotting_3_chart(data, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(data.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(data.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(data.loc[:,feature], orient='v', ax = ax3);
    
plotting_3_chart(data, 'price')
plt.show()

previous_data = data.copy()

#se hace un test de normalizacion de la siguiente forma
print(normaltest(data.price.values))

data2 = np.log(data['price'])
plt.hist(data2)
plt.show()

print(normaltest(data2.values))



data3 = np.sqrt(data['price'])
plt.hist(data3)
plt.show()

print(normaltest(data3.values))

data4 = boxcox(data['price'])[0]
print(data4.shape)
data5 = pd.DataFrame(data4)
plt.hist(data5)
plt.show()

print(normaltest(data5.values))

fig = plt.subplots(figsize = (12,8), ncols=1,sharey=False)
sns.scatterplot( x =data2, y = data.enginesize)
sns.regplot(x=data2, y=data.enginesize)

plt.show()

fig = plt.subplots(figsize = (12,8), ncols=1,sharey=False)
sns.scatterplot( x = data.price, y = data.enginesize)
sns.regplot(x=data.price, y=data.enginesize)
plt.show()

#Si p-value es menor que 0.05 es que no es una distribucion normal, debe ser mayor

#Para ver la matriz de correlacion de manera de grafico se puede ver con heatmap, nos ayudara a encontrar
#posibles regresiones lineales y cuales estan sobreajustadas (que tienen muchas correlacion)
num = data.select_dtypes(include = ['int64', 'float64']) 
plt.figure(figsize = (30, 25))
sns.heatmap(num.corr(), annot = True, cmap="YlGnBu")
plt.show()


###########################################################################################################################

                                               #REGRESIONES LINEALES

###Separacion de datos de entrenamiento y de prueba

from sklearn.model_selection import train_test_split

#Aqui se separan por completo todos los datos, x de entrenamiento, x de prueba, y de entrenamiento, y de prueba,
#primero se pone el x, puede ser una o mas columnas, pero deben ser numericas, luego la columna que queremos de etiqueta
#para predecir, test_size indica que sera 20% los datos de prueba, random state es la forma que se distribuyen los datos
#(puede influir en algun calculo) cada numero es un estado
X_train, X_test, y_train, y_test = train_test_split(data[["enginesize"]], data[["price"]], test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

X_train=ss.fit_transform(X_train)

#Ajustando el modelo con los datos de entrenamiento
LR = LinearRegression()
LR.fit(X_train,y_train)

X_test=ss.transform(X_test)
car_price_predictions = LR.predict(X_test)


from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(y_test, car_price_predictions)

LR.score(X_test,y_test)

r2_1= r2_score(y_test, car_price_predictions)

print(mse1)
print(r2_1)


########################################################################################################################

                                            #Hacerlo con Pipeline


#El mismo resultado anterior
steps=[('scaler', StandardScaler()), ('lm',  LinearRegression())]

pipe = Pipeline(steps=steps)

pipe.fit(X_train,y_train)

car_price_predictions = pipe.predict(X_test)
mse = mean_squared_error(y_test, car_price_predictions)
rmse = np.sqrt(mse)
rmse
r2_score(car_price_predictions, y_test)

######################################################################################################################

                                             #REGRESION POLINOMICA
                                             
#Se trata de una tecnica para hacer modelos no lineales que deberian tener otro tipo de regresiones, pero se pueden
#hacer con regresiones lineales

import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Funcion que sirve para ver las puntuaciones de cada columna
def get_R2_features(model,test=True): 
    #X: global  
    features=list(X)
    features.remove("three")
    
    R_2_train=[]
    R_2_test=[]

    for feature in features:
        model.fit(X_train[[feature]],y_train)
        
        R_2_test.append(model.score(X_test[[feature]],y_test))
        R_2_train.append(model.score(X_train[[feature]],y_train))
        
    plt.bar(features,R_2_train,label="Train")
    plt.bar(features,R_2_test,label="Test")
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)),str(np.mean(R_2_test))) )
    print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)),str(np.max(R_2_test))) )

#Funcion para comparar dos histogramas de dos columnas
def  plot_dis(y,yhat):
    
    plt.figure()
    ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
    sns.distplot(yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
    plt.legend()

    plt.title('Actual vs Fitted Values')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv')
print(data.head())
 
print(data.info()) 

#Muestra que los datos de estas columnas diguen una relacion no lineal                                              
sns.lmplot(x = 'curbweight', y = 'price', data = data, order=2)
plt.show()

#Muestra tambien que estas columnas siguen una relacion no lineal
sns.lmplot(x = 'carlength', y = 'price', data = data, order=2)
plt.show()

#Muestra estas otra columnas que tiene una relacion no lineal
sns.lmplot(x = 'horsepower', y = 'price', data = data, order=2)
plt.show()

X = data.drop('price', axis=1)
y = data.price

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
print("Number of test samples:", X_test.shape[0])
print("Number of training samples:", X_train.shape[0])

lm = LinearRegression()

lm.fit(X_train, y_train)

predicted = lm.predict(X_test)

#Puntuaciones de la regresion lineal, en general muestra buenas puntuaciones
print("R^2 on training  data ",lm.score(X_train, y_train))
print("R^2 on testing data ",lm.score(X_test,y_test))

#Muestra la relacion de autos con su precio
plot_dis(y_test,predicted)

{col:coef for col,coef in zip(X.columns, lm.coef_)}

#Muestra todas las caracteristicas y su valores de sus coeficientes, las mas importantes a priori, pero que sea alto el
#coeficiente no significa que puede explicar la prediccion
plt.bar(X.columns[2:],abs(lm.coef_[2:]))
plt.xticks(rotation=90)
plt.ylabel("$coefficients$")
plt.show()


#Muestra las mejores puntuaciones R2 de las caracteristicas, mostrando tanto entrenamiento como los datos de prueba
get_R2_features(lm)

pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])
pipe.fit(X_train,y_train)

#Muestra que las puntuaciones en general de cada caracteristicas son bajas, pero es lo normal
print("R^2 on training data ", pipe.score(X_train, y_train))
print("R^2 on testing data ", pipe.score(X_test,y_test)) 
predicted = pipe.predict(X_test)
plot_dis(y_test,predicted)

#Muestra las puntuaciones explicitamente, los datos de pruba en general mantienen la puntuacion
#pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])
get_R2_features(pipe)

#REGRESION POLINOMICA

#La regresion aumenta la complejidad del modelo, esto debe hacerse con cuidado ya que lo puede sobreajustar

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

print(X_train_poly.shape)

print(X_test_poly.shape)

lm = LinearRegression()
lm.fit(X_train_poly, y_train)

predicted = lm.predict(X_train_poly)

#Los datos de prueba muestran una puntacion es negativa en los datos de prueba, se sobreajusto
print("R^2 on training data:", lm.score(X_train_poly, y_train))
print("R^2 on testing data:", lm.score(X_test_poly,y_test))


Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)), ('model', LinearRegression())]

pipe=Pipeline(Input)
pipe.fit(X_train, y_train)

print("R^2 on training  data:", pipe.score(X_train, y_train))
print("R^2 on testing data:", pipe.score(X_test,y_test))

get_R2_features(pipe)


Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)), ('model',LinearRegression())]
pipe=Pipeline(Input)

param_grid = {
    "polynomial__degree": [1, 2, 3],
    #"model__normalize":[True, False]
    
}

search = GridSearchCV(pipe, param_grid, n_jobs=1)

pipe.fit(X_train, y_train)
search.fit(X_test, y_test)

best=search.best_estimator_
print(best)

best.score(X_test,y_test)

predicted=best.predict(X_test)
plot_dis(y_test,predicted)

features=list(X)
   
    
R_2_train=[]
R_2_test=[]

for feature in features:
    param_grid = {
    "polynomial__degree": [ 1, 2,3,4,5],
    "model__positive":[True, False]}
    Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)), ('model',LinearRegression())]
    pipe=Pipeline(Input)
    print(feature)
    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X_test[[feature]], y_test)
    best=search.best_estimator_
        
    R_2_test.append(best.score(X_test[[feature]],y_test))
    R_2_train.append(best.score(X_train[[feature]],y_train))
    

#Se puede ver que la puntuacion de los datos de entrenamiento con los datos de pruebas se parecen bastante
#El modelo no esta sobre ajustado        
plt.bar(features,R_2_train,label="Train")
plt.bar(features,R_2_test,label="Test")
plt.xticks(rotation=90)
plt.ylabel("$R^2$")
plt.legend()
plt.show()
print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)),str(np.mean(R_2_test))) )
print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)),str(np.max(R_2_test))) )



######################################################################################################################

                                          #REGRESION LINEAL MULTIPLE

from ISLP import load_data
Boston = load_data('Boston')

print(Boston.head(15))

#Una visualizacion de la etiqueta, se ve que no se distribuye normalmente
import matplotlib.pyplot as plt
Boston.medv.hist()
plt.show()

from scipy.stats.mstats import normaltest

#El test dice que no es una distribucion normal con los datos no trasformados
print(normaltest(Boston.medv.values))

sqrt_medv = np.sqrt(Boston.medv)
plt.hist(sqrt_medv)
plt.show()

#mejoro con la transformacion cuadratica, pero aun no es una distribucion normal
print(normaltest(sqrt_medv))

from scipy.stats import boxcox

#aplica la trasnformacion boxcox a la columna etiqueta
bc_result = boxcox(Boston.medv)
boxcox_medv = bc_result[0]
lam = bc_result[1]


plt.hist(boxcox_medv)
plt.show()

#Ahora es una distribucion normal con la trasnformacion boxcox
print(normaltest(boxcox_medv))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, 
                                   PolynomialFeatures)
        
        
#definicion de la aplicacion de regresion lineal                           
lr = LinearRegression()

#X son todas las columnas, menos la etiqueta, la etiqueta es medv
X = Boston.drop("medv", axis=1)
y = Boston["medv"]

print(X)
print(X.shape)

#Aqui lo que se hace es como se hara una regresion lineal multiple, amplifica la cantidad de datos para que
#la regrsion sea mas precisa agrega muchas mas columnas
pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)

print(X_pf)
print(X_pf.shape)

#Se aplica la cantidad de columnas amplificadas para entrenar con ellas
X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.3, 
                                                    random_state=72018)

#Se normalizan todos los datos del datasets, menos la etiqueta y los datos de prueba
s = StandardScaler()
X_train_s = s.fit_transform(X_train)

#Le aplica la trasnformacion boxcox a las etiquetas de entrenamiento
bc_result2 = boxcox(y_train)
y_train_bc = bc_result2[0]

#solo toma el valor de lamda
lam2 = bc_result2[1]

y_train_bc.shape

#Ajusta la regresion lineal multiple con los datos menos etiqueta estandarizados
#y la etiqueta con una transformacion boxcox
lr.fit(X_train_s, y_train_bc)

#Hace una estandarizacion de los datos menos la etiqueta de prueba. recien empieza la prueba
X_test_s = s.transform(X_test)

#Hace la prediccion con los datos de prueba estandarizados
y_pred_bc = lr.predict(X_test_s)

#EL MODELO ESTA FRABRICADO SOLO PARA DATOS ESTANDARIZADOS PRONOSTICANDO "Y" CON TRASNFORMACION BOXCOX


#Invierte una trasfomacion boxcox, estos valores son iguales, ojo que el lamda lo entrega al hacer el boxcox
from scipy.special import inv_boxcox
inv_boxcox(boxcox_medv, lam)[:10]
Boston['medv'].values[:10]

#Se hace la prueba de r2, y_test es la etiqueta sin modificaciones, y_pred_tran es el pronostico con la trasnformacion
#boxcox a la inversa, se obtuvo con "X de prueba", OJO QUE EL LAMDA SE CALCULO CON "Y DE ENTRENAMIENTO" se le
#habia hecho la transformacion boxcox
y_pred_tran = inv_boxcox(y_pred_bc,lam2)

print(r2_score(y_test,y_pred_tran))


##########################################################################################################################

                                     #REGRESION LINEAL MULTIPLE CON SOBREAJUSTE


import pandas as pd
import numpy as np

# Import the data using the file path
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/Ames_Housing_Sales.csv")
print(data.head())


print(data.shape)

print(data.dtypes.value_counts())

#Ahora es para calcular cuantas columnas potenciales seran despues de una codificacion OneHotEncoder

mask = data.select_dtypes(include = ["object"])

categorical_cols = mask

print(categorical_cols)

#Determina cuantas columnas extra
num_ohc_cols = (categorical_cols.apply(lambda x: x.nunique())
                .sort_values(ascending=False))


# Evita que codifique si solo hay una categoria unica
small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]

#Elimina las que tengas una categoria
small_num_ohc_cols -= 1

# Son 215 columnas 
# La cantidad extra es muy grande
print(small_num_ohc_cols.sum())



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Copy of the data
data_ohc = data.copy()

# The encoders
le = LabelEncoder()
ohc = OneHotEncoder()

for col in num_ohc_cols.index:
    
    # Integer encode the string categories
    dat = le.fit_transform(data_ohc[col]).astype(int)
    
    # Remove the original column from the dataframe
    data_ohc = data_ohc.drop(col, axis=1)

    # One hot encode the data--this returns a sparse array
    new_dat = ohc.fit_transform(dat.reshape(-1,1))

    # Create unique column names
    n_cols = new_dat.shape[1]
    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]

    # Create the new dataframe
    new_df = pd.DataFrame(new_dat.toarray(), 
                          index=data_ohc.index, 
                          columns=col_names)

    # Append the new data to the dataframe
    data_ohc = pd.concat([data_ohc, new_df], axis=1)

# Column difference is as calculated above
data_ohc.shape[1] - data.shape[1]

print(data.shape[1])

# Remove the string columns from the dataframe
data = data.drop(num_ohc_cols.index, axis=1)

print(data.shape[1])

from sklearn.model_selection import train_test_split

y_col = 'SalePrice'

# Split the data that is not one-hot encoded
feature_cols = [x for x in data.columns if x != y_col]
X_data = data[feature_cols]
y_data = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.3, random_state=42)
# Split the data that is one-hot encoded
feature_cols = [x for x in data_ohc.columns if x != y_col]
X_data_ohc = data_ohc[feature_cols]
y_data_ohc = data_ohc[y_col]

X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, 
                                                    test_size=0.3, random_state=42)
                                                    
# Compare the indices to ensure they are identical
(X_train_ohc.index == X_train.index).all()                                                    

print(X_train)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR = LinearRegression()

# Storage for error values
error_df = list()

# Data that have not been one-hot encoded
LR = LR.fit(X_train, y_train)
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)

error_df.append(pd.Series({'train': mean_squared_error(y_train, y_train_pred),
                           'test' : mean_squared_error(y_test,  y_test_pred)},
                           name='no enc'))

# Data that have been one-hot encoded
LR = LR.fit(X_train_ohc, y_train_ohc)
y_train_ohc_pred = LR.predict(X_train_ohc)
y_test_ohc_pred = LR.predict(X_test_ohc)

print(X_train_ohc)

error_df.append(pd.Series({'train': mean_squared_error(y_train_ohc, y_train_ohc_pred),
                           'test' : mean_squared_error(y_test_ohc,  y_test_ohc_pred)},
                          name='one-hot enc'))

#Muestra los Errores medias cuadraticos de un ejemplo, en este caso el valor del error aumento demaciado al hacer la
#Codiciacion OneHotEncoder, se tenian tantas columnas por separado que la regesion se sobreajusto, provocando aun mas error
#Esto no deberia pasar o no aumentar tan excesivamente 
error_df = pd.concat(error_df, axis=1)
print(error_df)


# Mute the setting wtih a copy warnings
pd.options.mode.chained_assignment = None

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


scalers = {'standard': StandardScaler(),
           'minmax': MinMaxScaler(),
           'maxabs': MaxAbsScaler()}

training_test_sets = {
    'not_encoded': (X_train, y_train, X_test, y_test),
    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)}


# Get the list of float columns, and the float data
# so that we don't scale something we already scaled. 
# We're supposed to scale the original data each time
mask = X_train.select_dtypes(include = ["float"])
float_columns = mask

# initialize model
LR = LinearRegression()

# iterate over all possible combinations and get the errors
errors = {}
for encoding_label, (_X_train, _y_train, _X_test, _y_test) in training_test_sets.items():
    for scaler_label, scaler in scalers.items():
        trainingset = _X_train.copy()  # copy because we dont want to scale this more than once.
        testset = _X_test.copy()
        trainingset = scaler.fit_transform(trainingset.select_dtypes(include = ["float"]))
        testset = scaler.transform(testset.select_dtypes(include = ["float"]))
        LR.fit(trainingset, _y_train)
        predictions = LR.predict(testset)
        key = encoding_label + ' - ' + scaler_label + 'scaling'
        errors[key] = mean_squared_error(_y_test, predictions)


#Lo que se muestra es que no es util hacer mas codifciaciones a los flotantes, no mejora en el caso de solo codificaciones
#a los flotantes, mantiene un mismo error, y en el caso de OneHotEncoder + Codificaciones a los flotantes aumenta aun mas
#el error exageradamente
errors = pd.Series(errors)
print(errors.to_string())
print('-' * 80)
for key, error_val in errors.items():
    print(key, error_val)
    
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

ax = plt.axes()
# we are going to use y_test, y_test_pred
ax.scatter(y_test, y_test_pred, alpha=.5)

ax.set(xlabel='Ground truth', 
       ylabel='Predictions',
       title='Ames, Iowa House Price Predictions vs Truth, using Linear Regression');    
plt.show()



###########################################################################################################################

                                           #VALIDACION CRUZADA
                                          
import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer                                           
                                           
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv')
print(data.head())

print(data.dtypes.value_counts())

print(data.info())

X = data.drop(columns=['price'])
y = data['price'].copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

lr = LinearRegression()

lr.fit(X_train, y_train)

predicted =lr.predict(X_test)

#Esto es R2 tambien para datos de entrenamiento
print(lr.score(X_train,y_train))

print(lr.score(X_test,y_test))

print(r2_score(y_true=y_test, y_pred=predicted))

mse = mean_squared_error(y_true=y_test, y_pred=predicted)
#aplica raiz cuadrada solo para que se vea mejor, menor el numero
rmse = np.sqrt(mse)
print(rmse)

some_data = X.iloc[:3]
some_labels = y.iloc[:3]

print(some_data)
print(some_labels)

print("Predictions:", lr.predict(some_data))

predicted =lr.predict(X_test)
print(predicted)

pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])
pipe.fit(X_train,y_train)

print(pipe.score(X_train,y_train))

print(pipe.score(X_test,y_test))

pipe_1 = Pipeline([('nn',Normalizer() ),('lr', LinearRegression())])
pipe_1.fit(X_train, y_train)

print(pipe_1.score(X_train,y_train))

#Da un valor negativo porque el modelo es complejo y tiene sobreajuste
print(pipe_1.score(X_test,y_test))

pred =pipe_1.predict(X_test)

mse = mean_squared_error(y_true=y_test, y_pred=pred)
rmse = np.sqrt(mse)
print(rmse)


#Una lista con todos los nombres de las columnas excepto la etiqueta
features=list(X)
print(features)

R_2=[]
pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])

#Crea una itineracion con los nombres de las columnas, para entrenar regresiones lineales para cada columna y luego
#aplicar R2 para cada modelo de regesion lineal
for feature in features:
    pipe.fit(X_train[[feature]],y_train)
    R_2.append(pipe.score(X_train[[feature]],y_train))

#plotea una barra con los R2 para todas las columnas, recordando que entre mas alta mejor, la varianza es mas pequeña
#solo para los datos de entrenamiento
plt.bar(features,R_2)
plt.xticks(rotation=90)
plt.ylabel("$R^2$")
plt.show()

best=features[np.argmax(R_2)]
print(best)

pipe.fit(X[[best]],y)

#Ahora realizandolo con los datos de prueba
R_2=[]
for feature in features:

      lr.fit(X_train[[feature]], y_train)
      R_2.append(lr.score(X_test[[feature]],y_test))
      
best=features[np.argmax(R_2)]

plt.bar(features,R_2)
plt.xticks(rotation=90) 
plt.ylabel("")

plt.show() 
best=features[np.argmax(R_2)]
print(best)

###Validacion cruzada

print(X)

N=len(X)
print(N)

lr = LinearRegression()

scores = cross_val_score(lr, X, y, scoring ="r2", cv=3)

#Estas puntuaciones son 1 para cada pliegue de la validacion cruzada
print(scores) 

def display_scores(scores, print_=False):
    
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(scores)

scores = cross_val_score(lr, X ,y, scoring ="neg_mean_squared_error", cv=5)
lr_scores = np.sqrt(-scores)
display_scores(lr_scores)

###Con K-Fold la diferencia es que revuelve los datos

n_splits=2
kf = KFold(n_splits = n_splits)

y = data['price'].copy()
X = data.drop(columns=['price'])
R_2 = np.zeros((n_splits,1))
pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])
n=0
for k,(train_index, test_index) in enumerate(kf.split(X,y)):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
X_train, X_test =X.iloc[train_index],X.iloc[test_index]
    
y_train, y_test=y[train_index],y[test_index]
pipe.fit(X_train,y_train)
n=+1
R_2[k]=pipe.score(X_test, y_test)
print(R_2)
print(R_2.mean())


n_splits=3
kf = KFold(n_splits = n_splits)
y = data['price'].copy()
X = data.drop(columns=['price'])
R_2=np.zeros((n_splits,1))
pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])
n=0
for k,(train_index, test_index) in enumerate(kf.split(X,y)):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    
X_train, X_test =X.iloc[train_index],X.iloc[test_index]
    
y_train, y_test=y[train_index],y[test_index]
pipe.fit(X_train,y_train)
n=+1
R_2[k]=pipe.score(X_test, y_test)
    
print(R_2)    
print(R_2.mean())

n_splits=3
kf = KFold(n_splits = n_splits,shuffle=True)
y = data['price'].copy()
X = data.drop(columns=['price'])
R_2=np.zeros((n_splits,1))
pipe = Pipeline([('ss',StandardScaler() ),('lr', LinearRegression())])
n=0
for k,(train_index, test_index) in enumerate(kf.split(X,y)): 
        print("TRAIN:", train_index)
        print("TEST:", test_index)

X_train, X_test =X.iloc[train_index],X.iloc[test_index]

y_train, y_test=y[train_index],y[test_index]
pipe.fit(X_train,y_train)
n=+1
R_2[k]=pipe.score(X_test, y_test)

print(R_2.mean())


###########################################################################################################################

                                           #VALIDACION CRUZADA EJEMPLO

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from ISLP import load_data
boston_data = load_data('Boston')

print(boston_data.keys())

print(boston_data.head())

X = boston_data.drop('medv', axis=1)
y = boston_data.medv

kf = KFold(shuffle=True, random_state=72018, n_splits=3)

#Muestra como va revolviendo los indices
for train_index, test_index in kf.split(X):
    print("Train index:", train_index[:10], len(train_index))
    print("Test index:",test_index[:10], len(test_index))
    print('')

scores = []
lr = LinearRegression()

print(X)
print(kf.split(X))

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = (X.iloc[train_index, :], 
                                        X.iloc[test_index, :], 
                                        y[train_index], 
                                        y[test_index])
    print(X_train)
    print(X_test)                                        
    
    lr.fit(X_train, y_train)        
    y_pred = lr.predict(X_test)
    score = r2_score(y_test.values, y_pred)    
    scores.append(score)
    
print(scores)

#######################################################################################################################

                                         #Validacion cruzada con K-fold

s = StandardScaler()
lr = LinearRegression()

estimator = Pipeline([("scaler", s),
                      ("regression", lr)])

print(kf)

#Usando cross_val_predict
predictions = cross_val_predict(estimator, X, y, cv=kf)

#Es igual al resultado anterior usando este petodo
print(r2_score(y, predictions))

#Es el resultado anterior, calcula la media
print(np.mean(scores))


########################################################################################################################

                                 #Ajustar hiperparametros con validacion cruzada

#Se generan 10 valores para alfa
alphas = np.geomspace(1e-9, 1e0, num=10)
print(alphas)

scores = []
coefs = []

#Se itineran los alfas usando el pipe, las cross validacion con K-fold, luego se puntua cada prediccion con cada alfa
#se agregan luego a una tupla
for alpha in alphas:
    las = Lasso(alpha=alpha, max_iter=100000)
    
    estimator = Pipeline([
        ("scaler", s),
        ("lasso_regression", las)])

    predictions = cross_val_predict(estimator, X, y, cv = kf)
    
    score = r2_score(y, predictions)
    
    scores.append(score)
    
    
list(zip(alphas,scores))    

#Se muestran los coeficientes de la regresion lasso con alfa 1e-6, son 12
print(Lasso(alpha=1e-6).fit(X, y).coef_)

#Se muestran los coeficientes de la regresion lasso con alfa 1, son 12
print(Lasso(alpha=1.0).fit(X, y).coef_)

#Se grafican los alfas vs R2
plt.figure(figsize=(10,6))
plt.semilogx(alphas, scores, '-o')
plt.xlabel('$\\alpha$')
plt.ylabel('$R^2$')
plt.show()

pf = PolynomialFeatures(degree=2)
scores = []

#Itinera valores de alfa para probarlos en la regresion lasso, el estimador expande el datasets, luego lo escala, luego
#Aplica la regresion lasso
#Despues ocupa la validacion cruzada, con el metodo k-fold para tener predicciones, luego las puntua con R2
alphas = np.geomspace(0.001, 10, 5)
for alpha in alphas:
    las = Lasso(alpha=alpha, max_iter=100000)
    estimator = Pipeline([
        ("make_higher_degree", pf),
        ("scaler", s),
        ("lasso_regression", las)])
        
    predictions = cross_val_predict(estimator, X, y, cv = kf)
    score = r2_score(y, predictions)
    
    scores.append(score)

#Muestra todas las puntuaciones R2    
print(scores)

#Como los alfas tienen valores muy diferentes, los grafica en escala logaritmica y=R2, x=alfas
#En alfa=10 se acerca mucho a 0 en puntuacion R2, para valores menores se mantienen en 0.8
plt.semilogx(alphas, scores)
plt.show()

#Se muestra en el grafico que el mejores alfa es: alfa=0.01 porque es el que tiene mejor puntuacion R2


#Realizamos el modelo con ese parametro sera el modelo con el mejor hiperparametro
#Con este pipe
best_estimator = Pipeline([
                    ("make_higher_degree", PolynomialFeatures(degree=2)),
                    ("scaler", s),
                    ("lasso_regression", Lasso(alpha=0.01, max_iter=10000))])

#Le aplicamos el pipe a los datos
print(best_estimator.fit(X, y))

#Le aplicamos el pipe al estimador y vemos la puntuacion al ajuste es muy alto 0.9
print(best_estimator.score(X, y))
print(X.shape)

#Son los coeficientes de la regresion que es lineal
print(best_estimator.named_steps["lasso_regression"].coef_)

#De las formas tanto normal como expandida con polinomial features entrega el mismo alfa de resultado que 
#da mejor puntuacion R2 con alfa=0.01


###OCUPANDO RIDGE

#itineramos alfas de la misma forma, con expancion polinomica
pf = PolynomialFeatures(degree=2)
scores = []

alphas = np.geomspace(4, 20, 20)
for alpha in alphas:
    ridge = Ridge(alpha=alpha, max_iter=100000)

    estimator = Pipeline([
        ("polynomial_features", pf),
        ("scaler", s),
        ("ridge_regression", ridge)])

    predictions = cross_val_predict(estimator, X, y, cv = kf)
    score = r2_score(y, predictions)
    scores.append(score)
    
plt.plot(alphas, scores)
plt.show()


# Once we have found the hyperparameter (alpha~1e-2=0.01)
# make the model and train it on ALL the data
# Then release it into the wild .....
best_estimator = Pipeline([
                    ("make_higher_degree", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", s),
                    ("lasso_regression", Lasso(alpha=0.01, max_iter=10000))])

best_estimator.fit(X, y)

print(best_estimator.score(X, y))

df_importances = pd.DataFrame(zip(best_estimator.named_steps["make_higher_degree"].get_feature_names_out(),
                 best_estimator.named_steps["lasso_regression"].coef_))

col_names_dict = dict(zip(list(range(len(X.columns.values))), X.columns.values))

print(col_names_dict)

print(df_importances.sort_values(by=1))


###########################################################################################################################

                                             #GridSearchCV

from sklearn.model_selection import GridSearchCV

#Creando un pipe que expanda, escale y ocupe regresion ridge
estimator = Pipeline([("polynomial_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("ridge_regression", Ridge())])

#Simplemnte es un difccionario con parametros que se utilizaran
params = {
    'polynomial_features__degree': [1, 2, 3],
    'ridge_regression__alpha': np.geomspace(4, 20, 20)}

#Ocupamos GridSearchCV usando los parametros anteriores con metodo k-fold
grid = GridSearchCV(estimator, params, cv=kf)

grid.fit(X, y)

#Encuentra automaticamete los hiperametros tanto como la expansion y cual alfa debemos ocupar
#Best_score y best_params son para GridSearchCV
print(grid.best_score_, grid.best_params_)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, 
                                   PolynomialFeatures)
from scipy.stats.mstats import normaltest
from scipy.stats import boxcox
from scipy.special import inv_boxcox


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_data = pd.read_csv(file_name)

#Con regresion lineal
lr = LinearRegression()
y_col = "MEDV"
X = boston_data.drop(y_col, axis=1)
y = boston_data[y_col]

#Expandiendo los datos en 2
pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.3, 
                                                    random_state=72018)

#Escalando los datos
s = StandardScaler()
X_train_s = s.fit_transform(X_train)

#Aplicando la transformacion boxcox
bc_result = boxcox(y_train)
y_train_bc = bc_result[0]
lam = bc_result[1]

#ajustamos el modelo con regresion lineal
lr.fit(X_train_s, y_train_bc)
X_test_s = s.transform(X_test)
y_pred_bc = lr.predict(X_test_s)

#Invertimos la trasnformacion  con lamda calculado anteriormente
y_pred_tran = inv_boxcox(y_pred_bc, lam)
print(r2_score(y_pred_tran,y_test)) #RES 0.848052537981275

#Ocupamos denuevo la regresion lineal sin transformacion boxcox
lr = LinearRegression()
lr.fit(X_train_s,y_train)
lr_pred = lr.predict(X_test_s)
print(r2_score(lr_pred, y_test)) #RES 0.8667029116056716


#y_predict = grid.predict(X)

#calculamos la puntuacion R2 para la grid
#print(r2_score(y, y_predict))

#print(grid.best_estimator_.named_steps['ridge_regression'].coef_)

#print(grid.cv_results_)


########################################################################################################################

                                #MODELOS REGULACION Y GRADIENTE DESCENDIENTE ELEGIR EL MEJOR
                                     
import pandas as pd
import numpy as np


data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/X_Y_Sinusoid_Data.csv")
data.head()

X_real = np.linspace(0, 1.0, 100)
Y_real = np.sin(2 * np.pi * X_real)                                     

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_context('talk')
sns.set_palette('dark')

# Plot of the noisy (sparse)
ax = data.set_index('x')['y'].plot(ls='', marker='o', label='data')
ax.plot(X_real, Y_real, ls='--', marker='', label='real function')

ax.legend()
ax.set(xlabel='x data', ylabel='y data')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Setup the polynomial features
degree = 20
pf = PolynomialFeatures(degree)
lr = LinearRegression()

# Aca se toma una muestra de datos que siguen una distribucion sinusoidal, no es la funcion en si solo datos que se aproximan
X_data = data[['x']]
Y_data = data['y']

# Create the features and fit the model
X_poly = pf.fit_transform(X_data)
lr = lr.fit(X_poly, Y_data)
Y_pred = lr.predict(X_poly)

# Plot the result
plt.plot(X_data, Y_data, marker='o', ls='', label='data', alpha=1)
plt.plot(X_real, Y_real, ls='--', label='real function')
plt.plot(X_data, Y_pred, marker='^', alpha=.5, label='predictions w/ polynomial features')
plt.legend()
ax = plt.gca()
ax.set(xlabel='x data', ylabel='y data')
plt.show()

# Mute the sklearn warning about regularization
import warnings
warnings.filterwarnings('ignore', module='sklearn')

from sklearn.linear_model import Ridge, Lasso

# The ridge regression model
rr = Ridge(alpha=0.001)
rr = rr.fit(X_poly, Y_data)
Y_pred_rr = rr.predict(X_poly)

# The lasso regression model
lassor = Lasso(alpha=0.0001)
lassor = lassor.fit(X_poly, Y_data)
Y_pred_lr = lassor.predict(X_poly)

# The plot of the predicted values
plt.plot(X_data, Y_data, marker='o', ls='', label='data')
plt.plot(X_real, Y_real, ls='--', label='real function')
plt.plot(X_data, Y_pred, label='linear regression', marker='^', alpha=.5)
plt.plot(X_data, Y_pred_rr, label='ridge regression', marker='^', alpha=.5)
plt.plot(X_data, Y_pred_lr, label='lasso regression', marker='^', alpha=.5)

plt.legend()

ax = plt.gca()
ax.set(xlabel='x data', ylabel='y data')
plt.show()

#Esto lo que muestra la calidad de predicciones de diferentes regresiones, donde se muestra tanto lasso como
#ridge son las que mejor representan los datos con sus predicciones
#Esto hace que la regresion se muestre mas adecuada en el grafico, no tiene que ver con la cantidad de datos o
#con otro tipo de datos sino que la prediccion es mejor


# let's look at the absolute value of coefficients for each model

coefficients = pd.DataFrame()
coefficients['linear regression'] = lr.coef_.ravel()
coefficients['ridge regression'] = rr.coef_.ravel()
coefficients['lasso regression'] = lassor.coef_.ravel()
coefficients = coefficients.applymap(abs)

#Confirma la gran diferencia que hay entre los coeficientes, siendo la regression lineal la peor con alta desviacion
#estandar
print(coefficients.describe())  # Huge difference in scale between non-regularized vs regularized regression


#Lo siguiente es para mostrar la gran diferencia que hay con los coeficientes de las regresiones, son muy variables
colors = sns.color_palette()

# Setup the dual y-axes
ax1 = plt.axes()
ax2 = ax1.twinx()

# Plot the linear regression data
ax1.plot(lr.coef_.ravel(), 
         color=colors[0], marker='o', label='linear regression')

# Plot the regularization data sets
ax2.plot(rr.coef_.ravel(), 
         color=colors[1], marker='o', label='ridge regression')

ax2.plot(lassor.coef_.ravel(), 
         color=colors[2], marker='o', label='lasso regression')

# Customize axes scales
ax1.set_ylim(-2e14, 2e14)
ax2.set_ylim(-25, 25)

# Combine the legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set(xlabel='coefficients',ylabel='linear regression')
ax2.set(ylabel='ridge and lasso regression')

ax1.set_xticks(range(len(lr.coef_)))
plt.show()

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/Ames_Housing_Sales.csv")
print(data.head(10))

data =pd.get_dummies(data, drop_first=True)
print(data.columns)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.3, random_state=42)

print(train)
print(test)

# Create a list of float colums to check for skewing
mask = data.select_dtypes(float)
float_cols = mask

skew_limit = 0.75
skew_vals = train.select_dtypes(float).skew()

skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {0}'.format(skew_limit)))

print(skew_cols)

# OPTIONAL: Let's look at what happens to one of these features, when we apply np.log1p visually.

field = "BsmtFinSF1"
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
train[field].hist(ax=ax_before)
train[field].apply(np.log1p).hist(ax=ax_after)
ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field))
plt.show()
# a little bit better

# Mute the setting wtih a copy warnings
pd.options.mode.chained_assignment = None

for col in skew_cols.index.tolist():
    if col == "SalePrice":
        continue
    train[col] = np.log1p(train[col])
    test[col]  = test[col].apply(np.log1p)  # same thing

feature_cols = [x for x in train.columns if x != 'SalePrice']
X_train = train[feature_cols]
y_train = train['SalePrice']

X_test  = test[feature_cols]
y_test  = test['SalePrice']

from sklearn.metrics import mean_squared_error


def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))

from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression().fit(X_train, y_train)

linearRegression_rmse = rmse(y_test, linearRegression.predict(X_test))

print(linearRegression_rmse)

f = plt.figure(figsize=(6,6))
ax = plt.axes()

ax.plot(y_test, linearRegression.predict(X_test), 
         marker='o', ls='', ms=3.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Price', 
       ylabel='Predicted Price', 
       xlim=lim,
       ylim=lim,
       title='Linear Regression Results')
       
plt.show()       


###Ocupando Ridge con validacion cruzada incluida

from sklearn.linear_model import RidgeCV

alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

ridgeCV = RidgeCV(alphas=alphas, 
                  cv=6).fit(X_train, y_train)

ridgeCV_rmse = rmse(y_test, ridgeCV.predict(X_test))

print(ridgeCV.alpha_, ridgeCV_rmse)

from sklearn.linear_model import LassoCV

alphas2 = np.array([1e-5, 5e-5, 0.0001, 0.0005])

lassoCV = LassoCV(alphas=alphas2,
                  max_iter=3000,
                  cv=3).fit(X_train, y_train)

lassoCV_rmse = rmse(y_test, lassoCV.predict(X_test))

#Es mas alto el rmse debido a que muchos coeficientes lo mando a 0 porque alfa es muy pequeño
print(lassoCV.alpha_, lassoCV_rmse)  # Lasso is slower

#Para saber cuantos coeficientes son 0, en este caso 10 son 0
print('Of {} coefficients, {} are non-zero with Lasso.'.format(len(lassoCV.coef_), 
                                                               len(lassoCV.coef_.nonzero()[0])))


###Ahora probando ElasticNet una combinacion de Lasso y Ridge, con los alfas optimos anteiores

from sklearn.linear_model import ElasticNetCV

l1_ratios = np.linspace(0.1, 0.9, 9)

elasticNetCV = ElasticNetCV(alphas=alphas2, 
                            l1_ratio=l1_ratios,
                            max_iter=1000).fit(X_train, y_train)
elasticNetCV_rmse = rmse(y_test, elasticNetCV.predict(X_test))

#Dice cuales son los alfas optimos y da un error intermedio entre Lasso y Ridge anteiores
print(elasticNetCV.alpha_, elasticNetCV.l1_ratio_, elasticNetCV_rmse)

#Mostrando todos los valores
rmse_vals = [linearRegression_rmse, ridgeCV_rmse, lassoCV_rmse, elasticNetCV_rmse]

labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']

rmse_df = pd.Series(rmse_vals, index=labels).to_frame()
rmse_df.rename(columns={0: 'RMSE'}, inplace=1)
print(rmse_df)

#Se muestra que Ridge tuvo el mejor rendimiento en cuanto a error, mientras que la regresion linear tuvo el meyor
#Error por lejos

f = plt.figure(figsize=(6,6))
ax = plt.axes()

labels = ['Ridge', 'Lasso', 'ElasticNet']

models = [ridgeCV, lassoCV, elasticNetCV]

for mod, lab in zip(models, labels):
    ax.plot(y_test, mod.predict(X_test), 
             marker='o', ls='', ms=3.0, label=lab)


leg = plt.legend(frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.0)

#Se puede ver graficamente los puntos predichos por los diferentes modelos, en general entre estos tres modelos tienen
#Predicciones bastante similares
ax.set(xlabel='Actual Price', 
       ylabel='Predicted Price', 
       title='Linear Regression Results')
plt.show()       


###GRADIENTE DESCENDIENTE

#El gradiente al igual que los modelos, es muy sensible al escalamiento, por lo que se debe tener cuidado,
#Ademas una tasa de aprendizaje alta puede hacer que no converja, en cambio una muy pequeña podria tardar demaciado

# Import SGDRegressor and prepare the parameters

from sklearn.linear_model import SGDRegressor

model_parameters_dict = {
    #'Linear': {'penalty': ''},
    'Lasso': {'penalty': 'l2',
           'alpha': lassoCV.alpha_},
    'Ridge': {'penalty': 'l1',
           'alpha': ridgeCV_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elasticNetCV.alpha_,
                   'l1_ratio': elasticNetCV.l1_ratio_}
}

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test))

rmse_df['RMSE-SGD'] = pd.Series(new_rmses)
print(rmse_df)

#El error de con Gradiete es enorme, esto debido a que el gradente diverge, esto sucede por el escalado alto o porque
#La tasa de aprendizaje es alta

# Import SGDRegressor and prepare the parameters

from sklearn.linear_model import SGDRegressor

model_parameters_dict = {
    #'Linear': {'penalty': 'none'},
    'Lasso': {'penalty': 'l2',
           'alpha': lassoCV.alpha_},
    'Ridge': {'penalty': 'l1',
           'alpha': ridgeCV_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elasticNetCV.alpha_,
                   'l1_ratio': elasticNetCV.l1_ratio_}
}

###La tasa de aprendizaje es "eta0", la reducimos bastante
new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments    
    SGD = SGDRegressor(eta0=1e-7, **parameters)
    SGD.fit(X_train, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test))

rmse_df['RMSE-SGD-learningrate'] = pd.Series(new_rmses)
print(rmse_df)

#Los modelos ahora convergene l gradiente y muestras errores bastante similares


#Ahora escalando los datos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#En este caso no se configura la tasa de aprendizaje, pero si fueron escalados
new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train_scaled, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test_scaled))

rmse_df['RMSE-SGD-scaled'] = pd.Series(new_rmses)
print(rmse_df)

#En este caso hay resultados muy diferentes mostrando que ridge, es el que mayor error tiene sin contar el modelo lineal


#Ahora ajustando la tasa de aprendizaje junto con el escalado, notamos que en realidad no ayuda combinarlos con la tasa 
#de aprendizaje, el error es mas grando en la mayoria de los casos, en un caso en particular 1e-2, se mantiene igual el
#error

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments
    SGD = SGDRegressor(eta0=1e-2, **parameters)
    SGD.fit(X_train_scaled, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test_scaled))

rmse_df['RMSE-SGD-scaled'] = pd.Series(new_rmses)
print(rmse_df)


########################################################################################################################

                                         #COMPARACIONES SIMPLES DE REGRESSION


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(72018)


def to_2d(array):
    return array.reshape(array.shape[0], -1)

def plot_exponential_data():
    data = np.exp(np.random.normal(size=1000))
    plt.hist(data)
    plt.show()
    return data
    
def plot_square_normal_data():
    data = np.square(np.random.normal(loc=5, size=1000))
    plt.hist(data)
    plt.show()
    return data
    
from ISLP import load_data
boston_data = load_data('Boston')    

print(boston_data.head(5))
    
y_col = "medv"

X = boston_data.drop(y_col, axis=1)
y = boston_data[y_col]    


#Estandarizamos los datos    
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
X_ss = s.fit_transform(X)   

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

y_col = "medv"

X = boston_data.drop(y_col, axis=1)
y = boston_data[y_col]

#Aplicando el modelo de regresion a datos no escalados OJO
lr.fit(X, y)
print(lr.coef_)



#Probamos otro modelo de regresion lineal con los datos estandarizados son mucho menores que los escalados
lr2 = LinearRegression()
lr2.fit(X_ss, y)
print(lr2.coef_) 

#Hay columnas que tienen coeficientes muchos mas grandes ya sea por arriba o por debajo de 0, son mucho mas 
#significantes para el modelo
print(pd.DataFrame(zip(X.columns, lr2.coef_)).sort_values(by=1))

###Lasso

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False,)
X_pf = pf.fit_transform(X)

X_pf_ss = s.fit_transform(X_pf)

#La regresion lasso me hizo 0 muchos coeficientes, lo que me dejo solo 7 coeficientes
#Donde solo 2 tienen una significancia alta para el modelo
las = Lasso()
las.fit(X_pf_ss, y)
print(las.coef_)

#comparacion de diferentes alfas

#En este caso alfa=0.1 aumenta los valores de los coeficientes, auenta la cantidad de coeficientes que no son 0 a 21
las01 = Lasso(alpha = 0.1)
las01.fit(X_pf_ss, y)
print('sum of coefficients:', abs(las01.coef_).sum() )
print('number of coefficients not equal to 0:', (las01.coef_!=0).sum())

#En esto caso alfa=1 aumenta, pero muy poco los valores, aumenta la cantidad de coeficientes diferentes de 0 solo en 1 a 8
las1 = Lasso(alpha = 1)
las1.fit(X_pf_ss, y)
print('sum of coefficients:',abs(las1.coef_).sum() )
print('number of coefficients not equal to 0:',(las1.coef_!=0).sum())

#El que tuvo mejor rendiminto fue el que tuvo mas coeficientes diferentes de 0
from sklearn.metrics import r2_score
print(r2_score(y,las.predict(X_pf_ss)))
print(r2_score(y,las01.predict(X_pf_ss)))
print(r2_score(y,las1.predict(X_pf_ss)))


#Los datos anteriores no habian sido entrenados y probados, ahora ver resultados con entrenamiento
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.3, 
                                                    random_state=72018)
#Con lasso con alfa no especificado                                                    
X_train_s = s.fit_transform(X_train)
las.fit(X_train_s, y_train)
X_test_s = s.transform(X_test)
y_pred = las.predict(X_test_s)
print(r2_score(y_pred, y_test))                                                    

#Con lasso con alfa 0.01
#Obtiene mucho mejor rendimiento                                               
X_train_s = s.fit_transform(X_train)
las01.fit(X_train_s, y_train)
X_test_s = s.transform(X_test)
y_pred = las01.predict(X_test_s)
print(r2_score(y_pred, y_test))                                                    
    

###Añadiento un lasso con alfa=0.001 y una regresion lineal

las001 = Lasso(alpha = 0.001, max_iter=5000)
X_train_s = s.fit_transform(X_train)
las001.fit(X_train_s, y_train)
X_test_s = s.transform(X_test)
y_pred = las001.predict(X_test_s)

#Calculado R2 con alfa=0.001
#Con alfa 0.001 tuvo el mejor resultado de todos los modelos, aumento mucho los coeficiente diferentes de 0, a 85
#en este caso casi todos los coeficientes
print("r2 score for alpha = 0.001:", r2_score(y_test,y_pred))


###Con regresion lineal

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

#Calculando R2
#Con regresion lineal tuvo un buen resultado
print("r2 score for Linear Regression:", r2_score(y_test,y_pred_lr,))


#Comparando coeficientes
print('Magnitude of Lasso coefficients:', abs(las001.coef_).sum())
print('Number of coeffients not equal to 0 for Lasso:', (las001.coef_!=0).sum())

print('Magnitude of Linear Regression coefficients:', abs(lr.coef_).sum())
print('Number of coeffients not equal to 0 for Linear Regression:', (lr.coef_!=0).sum())


###Comparando Ridge y Lasso

from sklearn.linear_model import Ridge    


# Decreasing regularization and ensuring convergence
r = Ridge(alpha = 0.001)
X_train_s = s.fit_transform(X_train)
r.fit(X_train_s, y_train)
X_test_s = s.transform(X_test)
y_pred_r = r.predict(X_test_s)

# Mostrando los coeficientes no genera ninguno 0 la regression Ridge
print(r.coef_)

#Comparando con el mismo alfa Lasso
print(las001.coef_)

print(np.sum(np.abs(r.coef_)))
print(np.sum(np.abs(las001.coef_)))

print(np.sum(r.coef_ != 0))
print(np.sum(las001.coef_ != 0))

#En las puntuaciones R2 las dos regresiones tienen puntuaciones bastente parecidas
#Es solo un caso en el que alfa es 0.001, pero no son todos los casos
y_pred = r.predict(X_pf_ss)
print(r2_score(y, y_pred))

y_pred = las001.predict(X_pf_ss)
print(r2_score(y, y_pred))

###COMPROBANDO CUANDO ESTAN ESCALADOS

X_train, X_test, y_train, y_test = train_test_split(X_ss, y, test_size=0.3, 
                                                    random_state=72018)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test,y_pred))   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=72018)

s = StandardScaler()
lr_s = LinearRegression()
X_train_s = s.fit_transform(X_train)
lr_s.fit(X_train_s, y_train)
X_test_s = s.transform(X_test)
y_pred_s = lr_s.predict(X_test_s)
print(r2_score(y_test,y_pred_s)) 

#Se comprueba que el escalado en si no afecta casi nada     


########################################################################################################################  

                                            #SOBREAJUSTE Y REGULACION

#Tanto las regresiones Lasso, Ridge y ElasticNet ponen penalizaciones a los coeficientes muy altos, para que el modelo
#Logre converger, es importante poner muchas iteraciones para que la no convergencia se note bastante.

#En el caso que tengamos sobreajuste, podemos ocupar Lasso para reducir los coeficientes, Ridge reduce el impacto de las
#caracteristicas que no son importantes, ElasticNet combina un poco de las dos elimina coeficientes y reduce el impacto
#de las menos importantes. Todo esto es necesario para mejorar las prediciones del modelo.



import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

#Funciones utilies

#Calcula cada coeficiente de R2 dependiendo del modelo de entrada
def get_R2_features(model,test=True): 
    #X: global  
    features=list(X)
    features.remove("three")
    
    R_2_train=[]
    R_2_test=[]

    for feature in features:
        model.fit(X_train[[feature]],y_train)
        
        R_2_test.append(model.score(X_test[[feature]],y_test))
        R_2_train.append(model.score(X_train[[feature]],y_train))
        
    plt.bar(features,R_2_train,label="Train")
    plt.bar(features,R_2_test,label="Test")
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)),str(np.mean(R_2_test))) )
    print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)),str(np.max(R_2_test))) )


#Esta funcion traza los coeficientes de R2 de todas las columnas
def plot_coef(X,model,name=None):
    

    plt.bar(X.columns[2:],abs(model.coef_[2:]))
    plt.xticks(rotation=90)
    plt.ylabel("$coefficients$")
    plt.title(name)
    plt.show()
    print("R^2 on training  data ",model.score(X_train, y_train))
    print("R^2 on testing data ",model.score(X_test,y_test))

#Traza los histogramas de 2 entradas
def  plot_dis(y,yhat):
    
    plt.figure()
    ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
    sns.distplot(yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
    plt.legend()

    plt.title('Actual vs Fitted Values')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv')
print(data.head())

print(data.info())

X = data.drop('price', axis=1)
y = data.price

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])


#Probando regression lineal
lm = LinearRegression()

lm.fit(X_train, y_train)

predicted = lm.predict(X_test)

print("R^2 on training  data ",lm.score(X_train, y_train))
print("R^2 on testing data ",lm.score(X_test,y_test))

plot_dis(y_test,predicted)

plot_coef(X,lm,name="Linear Regression")


#Regression Ridge
rr = Ridge(alpha=0.01)
print(rr)

rr.fit(X_train, y_train)

rr.predict(X_test)

print("R^2 on training  data ",rr.score(X_train, y_train))
print("R^2 on testing data ",rr.score(X_test,y_test))

#Tiene puntuaciones R2 demaciado similares con la regresion lineal

#Los coeficientes son bastante iguales
plot_coef(X,lm,name="Linear Regression")
plot_coef(X,rr,name="Ridge Regression")

#Cambiandole el alfa Ridge las columnas que antes no importaban ahora importan
#Al aumentar la importancia de caracteristicas irrelevantes, se le llama que el modelo se hace mas complejo
#IMPORTANTE!!
rr = Ridge(alpha=1)
rr.fit(X_train, y_train)
plot_coef(X,rr)

#Para elegir el mejor alfa hay que hacer validacion cruzada

#Ofrecemos diferentes aldas y los probamos para ver su puntuacion R2
alphas = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
R_2=[]
coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(abs(ridge.coef_))
    R_2.append(ridge.score(X_test,y_test))

#Graficamos el resultado de los diferentes alfas
ax = plt.gca()
#Grafica los coeficientes que se van generando con cada alfa diferente, a medida que alfa aumenta los coeficiente 
#se ponen mas pequeños
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization (regularization path)")
plt.show()

#Muestra la puntuacion de cada alfa en R2, la mejor puntuacion es 0.1
ax = plt.gca()
ax.plot(alphas, R_2)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("$R^2$")
plt.title("$R^2$ as a function of the regularization")
plt.show() 
 
#Para mostrar el error de los diferentes alfas, dice que el menor error se produce en 0.1
alphas = [0.00001,0.0001,0.001,0.01,0.1,1,10]
MEAN_SQE=[]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    MEAN_SQE.append(mean_squared_error(ridge.predict(X_test),y_test))

ax = plt.gca()
ax.plot(alphas, MEAN_SQE)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.title("$MSE$ as a function of the regularization")
plt.show() 
 
###Con caracteristicas polinomicas que aumente las cantidad de columnas

Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)),('ss',StandardScaler() ), ('model',Ridge(alpha=1))]
pipe = Pipeline(Input)

pipe.fit(X_train, y_train)

#En este caso se reduce la puntuacion de caracterisitcas, al hacer caracteristicas polinomiales
predicted=pipe.predict(X_test)
print(pipe.score(X_test, y_test))    

#Hacemos un diccionarios con posibles hiperparametros para buscarlos en GridSearch
param_grid = {
    "polynomial__degree": [1,2,3,4],
    "model__alpha":[0.0001,0.001,0.01,0.1,1,10]
}

search = GridSearchCV(pipe, param_grid, n_jobs=2)

search.fit(X_train, y_train)

#Con esto entrega un dataframe con los calculos que realizo y las puntuaciones a cada hiperparametro
print(pd.DataFrame(search.cv_results_).head())

#Muestra la mejor puntuacion de los hiperparametros
print("best_score_: ",search.best_score_)

#Muestra los mejores parametros explicitamente, tanto alfa como el grado de polinomio
print("best_params_: ",search.best_params_)

predict = search.predict(X_test)
print(predict) 
    
best=search.best_estimator_
print(best)    
    
predict = best.predict(X_test)
print(predict)    
    
best.score(X_test, y_test)    

best.fit(X,y)

columns=['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower',
       'carlength', 'carwidth', 'citympg']

#Muestra los resultados de las predcciones para las columnas mas importantes
#En general el pronostico es bastante acertado
#Es bueno de ver para saber en que esta fallando el modelo
for column in columns:
    search.fit(X_train[[column]], y_train)
    x=np.linspace(X_test[[column]].min(), X_test[[column]].max(),num=100)
    plt.plot(x,search.predict(x.reshape(-1,1)),label="prediction")
    plt.plot(X_test[column],y_test,'ro',label="y")
    plt.xlabel(column)
    plt.ylabel("y")
    plt.legend()
    plt.show()


###Regresion Lasso

#La principal ventaja de lasso es que hace 0 mucho de los coeficientes, es importante en un modelo con muchas
#Caracteristicas nos ahorra mucha memoria en un modelo
#Esto tambien se puede utilizar para seleccion de caracteristicas
#Aunque demora mas tiempo que las anteriores y puede que no tenga solucion unica

la = Lasso(alpha=0.1)
la.fit(X_train,y_train)

predicted = la.predict(X_test)
print(predicted)

#Lasso tiene una puntuacion bastante buena incluso en un modelo simple
print("R^2 on training  data ",lm.score(X_train, y_train))
print("R^2 on testing data ",lm.score(X_test,y_test))

#En el caso de Ridge hace que columnas que no eran importantes ahora son importantes
plot_coef(X,rr,name="Ridge Regression")

#En el caso de Lasso elimina columnas que no eran importante reduciendo la cantidad de columnas importantes
#dejando una o dos columnas importantes en este caso
plot_coef(X,la,name="Lasso Regression")


#Probando valores de alfa
alphas = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
R_2=[]
coefs = []
for alpha in alphas:
    la=Lasso(alpha=alpha)
    
    la.fit(X_train, y_train)
    coefs.append(abs(la.coef_))
    R_2.append(la.score(X_test,y_test))

#Ploteando los lso valores de alfa vs los coeficientes
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization (regularization path)")
plt.show()

#Ploteando los alfas y su puntuacion R2, en este caso la mejor puntuacion es 10
ax = plt.gca()
ax.plot(alphas, R_2)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("$R^2$")
plt.title("$R^2$ as a function of the regularization")
plt.show()


###Ahora con caracteristicas polinomiales 

Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)),('ss',StandardScaler() ), ('model',Lasso(alpha=1, tol = 0.2))]
pipe = Pipeline(Input)

pipe.fit(X_train, y_train)
pipe.predict(X_test)

#En este caso mejor bastante la puntuacion con caracteristicas polinomicas, escalado y alfa 1
print("R^2 on training  data ",pipe.score(X_train, y_train))
print("R^2 on testing data ",pipe.score(X_test,y_test))



#Ahora para buscar los mejores hiperparametros
param_grid = {
    "polynomial__degree": [ 1, 2,3,4,5],
    "model__alpha":[0.0001,0.001,0.01,0.1,1,10]
}

search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_train, y_train)

best=search.best_estimator_

#Mantuvo la puntuacion
print(best.score(X_test,y_test))   

#Dice que el mejor parametro es grado 3 y el valor 10
print("best_score_: ",search.best_score_)
print("best_params_: ",search.best_params_)  


###ElasticNet

#Combina las dos regresiones anteriores, para ello tiene un coeficiente adicional llamado radio

#En este caso es para alfa = 0.1 y l1_ratio=0.5
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train,y_train)

predicted=enet.predict(X_test)
print(predicted)

print("R^2 on training  data ", enet.score(X_train, y_train))
print("R^2 on testing data ", enet.score(X_test,y_test))   

plot_coef(X,la,name="Lasso Regression")
plot_coef(X,enet,name="Elastic net ")

#Configurando elasticnet alfa=0.01 y l1_ratio=0
enet = ElasticNet(alpha=0.01, l1_ratio=0)
enet.fit(X_train,y_train)
rr = Ridge(alpha=0.01)
rr.fit(X_train,y_train)

plot_coef(X,rr,name="Ridge Regression")

#Graficando elasticnet l1_ratio=0
plot_coef(X,enet,name="Elastic net l1_ratio=0 ")


Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)),('ss',StandardScaler() ), ('model',ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=2000))]
pipe = Pipeline(Input)
pipe.fit(X_train, y_train)
print("R^2 on training  data ",pipe.score(X_train, y_train))
print("R^2 on testing data ",pipe.score(X_test,y_test))


#Encontrando los mejores hiperparametros
param_grid = {
    "polynomial__degree": [ 1, 2,3,4,5],
    "model__alpha":[0.0001,0.001,0.01,0.1,1,10],
    "model__l1_ratio":[0.1,0.25,0.5,0.75,0.9]
}

#Ocupamos grid para encontrar los 3 hiperparametros demora un poco mas encontrarlos
Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)),('ss',StandardScaler() ), ('model',ElasticNet(tol = 0.2))]
pipe = Pipeline(Input)
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_test, y_test)
best=search.best_estimator_

#Con una puntuacion bastante baja
print(best.score(X_test,y_test))

#Encuentra los hiperparametros: alfa=10, l1_ratio=0.25, grado polinomial=2
print("best_score_: ",search.best_score_)
print("best_params_: ",search.best_params_)  


###PCA de los modelos de regresion   

#Escalar primero los datos
scaler = StandardScaler()
X_train[:] = scaler.fit_transform(X_train)

X_train.columns = [f'{c} (scaled)' for c in X_train.columns]

#Lo que hace PCA es reducir la cantidad de columnas a lo que nosotros le digamos, lo hace dejando las columnas
#mas importantes
pca = PCA()
pca.fit(X_train)

print(X_train)

#Aqui aplica la trasnformacion a los datos
X_train_hat = pca.transform(X_train)
print(X_train_hat.shape)


X_train_hat_PCA = pd.DataFrame(columns=[f'Projection  on Component {i+1}' for i in range(len(X_train.columns))], data=X_train_hat)
print(X_train_hat_PCA)

plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise variance and cumulative explained variance")
plt.show()

N = 20
X_train_hat_PCA = X_train_hat_PCA.iloc[:, :N]
print(X_train_hat_PCA)

#Ahora aplicamos un modelo elasticnet con las columnas reducidas
enet = ElasticNet(tol = 0.2, alpha=100, l1_ratio=0.75)
enet.fit(X_train_hat_PCA, y_train)

#Hacemos un pipe con diferentes transformaciones, PCA reduce a 20 columnas
Input=[ ('scaler', StandardScaler()), ('pca', PCA(n_components = N)), ('model', ElasticNet(tol =0.2, alpha=0.1, l1_ratio=0.1, max_iter=100000))]
pipe = Pipeline(Input)
pipe.fit(X_train, y_train)
X_test.columns = [f'{c} (scaled)' for c in X_test.columns]
#print(X_test)
#print(y_test)

#Vemos la puntuacion de nuestro nuevo modelo con columnas reducidas
print("R^2 on training  data ", pipe.score(X_train, y_train))
print("R^2 on testing data ", pipe.score(X_test,y_test))

       
