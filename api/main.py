"""
API DropoutIA -- expose les 2 modeles (niveau de risque + decrochage)
via un endpoint /predict qui accepte un fichier Excel entier.
"""

import sys
import os
import io

import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath("../src"))
import nettoyage

MODELE_DECROCHAGE = joblib.load("../models/modele_decrochage.pkl")
MODELE_RISQUE = joblib.load("../models/modele_risque.pkl")

COLONNES_MODELE_DECROCHAGE = [
    "id_Genre", "nefstat", "NbrAbsence", "NbrRedoublement",
    "MoyenneNotes", "cd_com", "NbrAbsence_manquant", "age"
]
COLONNES_MODELE_RISQUE = [
    "id_Genre", "Nefstat", "NbrAbsence", "NbrRedoublement",
    "MoyenneNotes", "CD_REG", "CD_PROV", "CD_MIL",
    "NbrAbsence_manquant", "age"
]

app = FastAPI(title="DropoutIA API")


def preparer_fichier(df_brut: pd.DataFrame) -> pd.DataFrame:
    df = df_brut.copy()
    df = nettoyage.gerer_absences_manquantes(df, colonne="NbrAbsence")
    df = nettoyage.calculer_age(df, colonne_date="dateNaisEleve")
    df = nettoyage.encoder_categorielles(df, ["nefstat"])
    if "nefstat" in df.columns and "Nefstat" not in df.columns:
        df["Nefstat"] = df["nefstat"]
    return df


@app.post("/predict")
async def predict(fichier: UploadFile = File(...)):
    if not fichier.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Le fichier doit etre un Excel (.xlsx)")

    contenu = await fichier.read()
    try:
        df_brut = pd.read_excel(io.BytesIO(contenu))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier Excel : {e}")

    df_pret = preparer_fichier(df_brut)

    colonnes_manquantes = set(COLONNES_MODELE_DECROCHAGE + COLONNES_MODELE_RISQUE) - set(df_pret.columns)
    if colonnes_manquantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes dans le fichier envoye : {colonnes_manquantes}"
        )

    X_decrochage = df_pret[COLONNES_MODELE_DECROCHAGE]
    X_risque = df_pret[COLONNES_MODELE_RISQUE]

    proba_decrochage = MODELE_DECROCHAGE.predict_proba(X_decrochage)[:, 1]
    niveau_risque_brut = MODELE_RISQUE.predict(X_risque)

    niveau_risque_final = [
        max(int(niveau), 2) if proba > 0.5 else int(niveau)
        for niveau, proba in zip(niveau_risque_brut, proba_decrochage)
    ]

    resultats = []
    for i in range(len(df_pret)):
        resultats.append({
            "codeEleve": str(df_brut.iloc[i].get("codeEleve", f"eleve_{i}")),
            "niveau_risque_predit": niveau_risque_final[i],
            "probabilite_decrochage_pct": round(float(proba_decrochage[i]) * 100, 1),
        })

    return JSONResponse(content={"nombre_eleves": len(resultats), "resultats": resultats})


@app.get("/")
def accueil():
    return {"message": "API DropoutIA active. Voir /docs pour la documentation interactive."}
