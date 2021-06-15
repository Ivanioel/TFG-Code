#!/home/ivanioel/miniconda2/envs/tfg-env/bin/python3

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.compose import TransformedTargetRegressor
from joblib import load, dump


# Lectura del dataset
fullDataset = pd.read_csv('../datasets/Sotavento/data_target_stv_2016_2017.csv', index_col=0, parse_dates=True)

x_col, target_col = fullDataset.columns[:-1], fullDataset.columns[-1]
data, target = fullDataset[x_col], fullDataset[target_col]

# Creación del modelo de regresion final
pls = PLSRegression()

# Pipeline
regr_base = Pipeline(steps=[('std_sc', StandardScaler()),
                            ('pls', pls)])

# Estandarizamos el target también
y_transformer = StandardScaler()
inner_estimator = TransformedTargetRegressor(regressor=regr_base, transformer=y_transformer)

# Parametros a hiperparametrizar
n_components = list(range(1, 21, 1))
param_grid = {
    'regressor__pls__n_components': n_components,
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


_ = cv_estimator.fit(data.values, target.values)
print(f"Mejor estimador: {cv_estimator.best_estimator_}")
print(f"Mejor puntuacion - mejor estimador: {cv_estimator.best_score_}")

path = '../estimators/'
filename = f'cv_estimator__pls_{n_components[0]}_{n_components[-1]}_{n_components[1]-n_components[0]}_full_tg_200_100n'
fullPath = path + filename
dump(cv_estimator, fullPath)
