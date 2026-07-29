"""
Module de modelisation DropoutIA.

Contient les fonctions reutilisables pour preparer les donnees a la
modelisation et comparer plusieurs algorithmes de Machine Learning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def preparer_donnees(df, colonne_cible, test_size=0.2, random_state=42):
    X = df.drop(columns=[colonne_cible])
    y = df[colonne_cible]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def obtenir_modeles():
    return {
        "Regression logistique": LogisticRegression(
            class_weight='balanced', max_iter=1000
        ),
        "Arbre de decision": DecisionTreeClassifier(
            class_weight='balanced', max_depth=8, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight='balanced', n_estimators=200, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def comparer_modeles(X_train, X_test, y_train, y_test):
    modeles = obtenir_modeles()
    resultats = []
    modeles_entraines = {}

    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)

        resultats.append({
            "modele": nom,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 3),
            "f1_score": round(f1_score(y_test, y_pred, average='weighted'), 3),
        })
        modeles_entraines[nom] = modele

    tableau = pd.DataFrame(resultats).sort_values("f1_score", ascending=False).reset_index(drop=True)
    return tableau, modeles_entraines


def rapport_detaille(modele, X_test, y_test):
    y_pred = modele.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
