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
from sklearn.utils.class_weight import compute_sample_weight
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


def comparer_modeles(X_train, X_test, y_train, y_test, classe_positive=1):
    modeles = obtenir_modeles()
    resultats = []
    modeles_entraines = {}

    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    for nom, modele in modeles.items():
        if nom == "Gradient Boosting":
            modele.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            modele.fit(X_train, y_train)

        y_pred = modele.predict(X_test)

        resultats.append({
            "modele": nom,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision_globale": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 3),
            "recall_global": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 3),
            "f1_global": round(f1_score(y_test, y_pred, average='weighted'), 3),
            f"recall_classe_{classe_positive}": round(
                recall_score(y_test, y_pred, pos_label=classe_positive, average='binary', zero_division=0)
                if len(set(y_test)) == 2 else
                recall_score(y_test, y_pred, average='macro', zero_division=0), 3
            ),
        })
        modeles_entraines[nom] = modele

    colonne_tri = f"recall_classe_{classe_positive}"
    tableau = pd.DataFrame(resultats).sort_values(colonne_tri, ascending=False).reset_index(drop=True)
    return tableau, modeles_entraines


def rapport_detaille(modele, X_test, y_test):
    y_pred = modele.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
