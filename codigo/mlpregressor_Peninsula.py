#!/home/ivanioel/miniconda2/envs/tfg-env/bin/python3

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from joblib import load, dump


# Lectura del dataset
data = load('./df_pen_wind_2016_2017_X_train.pkl')
target = load('./df_pen_wind_2016_2017_Y_train.pkl')

# Creación de PCA sin parámetros para luego hiperparametrizar
pca = PCA()
# Creación del modelo de regresion final

hidden_layer_sizes = (200, 100)
#solver: lbfgs para problemas pequeños, adam para problemas grandes
mlp_m =  MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                      activation='relu', 
                      #solver='lbfgs',
                      solver='adam',
                      tol=1.e-6, 
                      max_iter=5000,
		              n_iter_no_change=100,
		              random_state=1)

# Pipeline
regr_base = Pipeline(steps=[('std_sc', StandardScaler()),
                            ('pca', pca),
				            ('mlp_m', mlp_m)])

# Estandarizamos el target también
y_transformer = StandardScaler()
inner_estimator = TransformedTargetRegressor(regressor=regr_base, transformer=y_transformer)

# Parametros a hiperparametrizar
l_alpha = [10.**k for k in range(-2, 5)]
n_components = list(range(50, 201, 25))

param_grid = {
    'regressor__mlp_m__alpha': l_alpha,
    'regressor__pca__n_components': n_components,
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
                            n_jobs=5, 
                            verbose=1)


_ = cv_estimator.fit(data, target)
print(f"Mejor estimador: {cv_estimator.best_estimator_}")
print(f"Mejor puntuacion - mejor estimador: {cv_estimator.best_score_}")

path = '../estimators/'
filename = f'cv_estimator_Peninsula__mlp_{n_components[0]}_{n_components[-1]}_{n_components[1]-n_components[0]}_200_100n'
fullPath = path + filename
dump(cv_estimator, fullPath)
