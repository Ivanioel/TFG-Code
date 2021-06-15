#!/home/ivanioel/miniconda2/envs/tfg-env/bin/python3

import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor

# Lectura del dataset
fullDataset = pd.read_csv('../datasets/Sotavento/data_target_stv_2016_2017.csv', index_col=0, parse_dates=True)
x_col, target_col = fullDataset.columns[:-1], fullDataset.columns[-1]
data, target = fullDataset[x_col], fullDataset[target_col]

# Creación del modelo de regresion final
reg = Ridge(max_iter=10000)
# Pipeline
regr_base = Pipeline(steps=[('std_sc', StandardScaler()),
				            ('reg', reg)])

# Estandarizamos el target también
y_transformer = StandardScaler()
inner_estimator = TransformedTargetRegressor(regressor=regr_base, transformer=y_transformer)

# Parametros a hiperparametrizar
l_alpha = [10.**k for k in range(1, 4)]
param_grid = {
    'regressor__reg__alpha': l_alpha,
}
# Numero de splits para Cross Validation
n_splits = 2
kf = KFold(n_splits, shuffle=False)
# Hiperparametrizacion
cv_estimator = GridSearchCV(estimator=inner_estimator, 
                            param_grid=param_grid, 
                            cv=kf, 
                            scoring='neg_mean_absolute_error', 
                            return_train_score=True, 
                            n_jobs=3, 
                            verbose=1,
                            refit=True)

# Fit datos 
_ = cv_estimator.fit(data.values, target.values)
print(f"Mejor estimador: {cv_estimator.best_estimator_}")
print(f"Mejor puntuacion - mejor estimador: {cv_estimator.best_score_}")
# Guardar datos en fichero
path = '../estimators/'
filename = f'cv_estimator__ridge_no_PCA'
fullPath = path + filename
dump(cv_estimator, fullPath)


