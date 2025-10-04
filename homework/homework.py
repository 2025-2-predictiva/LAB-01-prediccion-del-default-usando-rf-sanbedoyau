# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
# #
import os
import gzip
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def cleanData(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns=['ID'])
    df = df.dropna()
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x >= 4 else x).astype('category')
    x = df.drop(columns=['default'])
    y = df['default']
    return x, y


def pipeline() -> Pipeline:
    caracteristicas = ['SEX', 'EDUCATION', 'MARRIAGE']
    preprocessor = ColumnTransformer(
        transformers=[('categoria', OneHotEncoder(handle_unknown='ignore'), caracteristicas)],
        remainder='passthrough'
    )
    return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
            ])


def hiperParams(pipeline, x, y):
    params = {
        'classifier__n_estimators': [200],
        'classifier__max_depth': [10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
    }
    gridSearch = GridSearchCV(
        pipeline,
        params,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        refit=True
    )
    return gridSearch.fit(x, y)


def save(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as file:
        pickle.dump(model, file)


def metrics(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metricsTrain = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': float(precision_score(y_train, y_train_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_train, y_train_pred)),
        'recall': float(recall_score(y_train, y_train_pred, zero_division=0)),
        'f1_score': float(f1_score(y_train, y_train_pred, zero_division=0))
    }

    metricsTest = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_test_pred, zero_division=0))
    }

    return metricsTrain, metricsTest


def confMatrix(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_metrics_train = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
        'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
    }

    cm_metrics_test = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
        'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
    }

    return cm_metrics_train, cm_metrics_test


def saveMetrics(metrics_train, metrics_test, cm_metrics_train, cm_metrics_test, file_path='files/output/metrics.json'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    metrics_totales = [metrics_train, metrics_test, cm_metrics_train, cm_metrics_test]
    with open(file_path, 'w') as f:
        for metrica in metrics_totales:
            f.write(json.dumps(metrica) + '\n')


test = pd.read_csv('files/input/test_data.csv.zip', compression='zip')
train = pd.read_csv('files/input/train_data.csv.zip', compression='zip')

x_test, y_test = cleanData(test)
x_train, y_train = cleanData(train)

pipe = pipeline()
opt_model = hiperParams(pipe, x_train, y_train)

save(opt_model)

metrics_train, metrics_test = metrics(opt_model, x_train, y_train, x_test, y_test)
cm_metrics_train, cm_metrics_test = confMatrix(opt_model, x_train, y_train, x_test, y_test)

saveMetrics(metrics_train, metrics_test, cm_metrics_train, cm_metrics_test)