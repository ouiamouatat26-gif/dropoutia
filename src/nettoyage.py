"""
Module de nettoyage des données DropoutIA.

Ce fichier contient des FONCTIONS réutilisables. On les appelle depuis le
notebook (ou plus tard depuis l'API) plutôt que de réécrire le code partout.
C'est une bonne pratique : un seul endroit à corriger si on trouve un bug.
"""

import pandas as pd


def gerer_absences_manquantes(df, colonne="NbrAbsence"):
    """
    Gère les valeurs manquantes de NbrAbsence.

    IMPORTANT (confirmé par l'encadrant) : une valeur manquante ne veut PAS
    dire 0 absence. Elle veut dire "on ne sait pas". Remplacer par 0 serait
    une erreur qui fausserait le modèle (on ferait croire au modèle qu'un
    élève n'a jamais été absent, alors qu'on n'a simplement pas l'info).

    Stratégie retenue ici : imputation par la médiane + colonne indicatrice.
    - On garde une trace du fait que la valeur était manquante (colonne
      f"{colonne}_manquant" = 1 si manquant, 0 sinon). Le modèle peut ainsi
      apprendre que "donnée manquante" est en soi une information utile.
    - On remplace la valeur manquante par la MÉDIANE des valeurs connues
      (pas la moyenne, qui est plus sensible aux valeurs extrêmes).

    Alternative plus stricte (proposée par l'encadrant) : supprimer les
    lignes avec valeur manquante. Voir la fonction
    `supprimer_lignes_absences_manquantes` ci-dessous si vous préférez
    cette option -- mais elle fait perdre ~50% des données, donc à discuter
    avec votre encadrant avant de choisir.
    """
    df = df.copy()
    df[f"{colonne}_manquant"] = df[colonne].isnull().astype(int)
    mediane = df[colonne].median()
    df[colonne] = df[colonne].fillna(mediane)
    return df


def supprimer_lignes_absences_manquantes(df, colonne="NbrAbsence"):
    """
    Alternative : supprime purement les lignes où la donnée est manquante.
    Plus rigoureux (aucune valeur inventée) mais fait perdre beaucoup de
    données. À utiliser seulement si votre encadrant le demande explicitement.
    """
    return df.dropna(subset=[colonne]).copy()


def calculer_age(df, colonne_date="dateNaisEleve", date_reference="2024-09-01"):
    """
    Transforme une date de naissance en âge (en années).
    Les modèles ML ne comprennent pas les dates telles quelles, il faut
    les transformer en une valeur numérique exploitable.
    """
    df = df.copy()
    df[colonne_date] = pd.to_datetime(df[colonne_date])
    df["age"] = (pd.Timestamp(date_reference) - df[colonne_date]).dt.days // 365
    return df


def encoder_categorielles(df, colonnes):
    """
    Encode des colonnes catégorielles (texte) en nombres, car les modèles
    ML ne travaillent qu'avec des chiffres.
    Exemple : "Masculin"/"Féminin" -> 0/1
    """
    df = df.copy()
    for col in colonnes:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def retirer_identifiants(df, colonnes_a_retirer):
    """
    Retire les colonnes qui sont de simples identifiants (numéro d'élève,
    code établissement, date brute...) : elles n'aident pas à prédire,
    elles créeraient juste du bruit ou une fausse corrélation.
    """
    colonnes_existantes = [c for c in colonnes_a_retirer if c in df.columns]
    return df.drop(columns=colonnes_existantes)
