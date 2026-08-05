"""
API DropoutIA -- version simplifiee (Option B).
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

COLONNES_MODELE_DECROCHAGE = [
    "id_Genre", "nefstat", "NbrAbsence", "NbrRedoublement",
    "MoyenneNotes", "cd_com", "NbrAbsence_manquant", "age"
]

app = FastAPI(title="DropoutIA API")


def preparer_fichier(df_brut: pd.DataFrame) -> pd.DataFrame:
    df = df_brut.copy()
    df = nettoyage.gerer_absences_manquantes(df, colonne="NbrAbsence")
    df = nettoyage.calculer_age(df, colonne_date="dateNaisEleve")
    df = nettoyage.encoder_categorielles(df, ["nefstat"])
    return df


def deriver_niveau_risque(probabilite_pct: float) -> int:
    if probabilite_pct < 20:
        return 1
    elif probabilite_pct < 50:
        return 2
    elif probabilite_pct < 75:
        return 3
    else:
        return 4


LIBELLES_NIVEAU = {1: "Tres faible", 2: "Faible", 3: "Modere", 4: "Severe"}

RECOMMANDATIONS = {
    1: "Aucune action requise. Suivi de routine.",
    2: "Suivi recommande lors du prochain conseil de classe.",
    3: "Entretien conseille avec l'eleve dans les prochaines semaines.",
    4: "Intervention prioritaire : entretien avec l'eleve et sa famille recommande dans les meilleurs delais.",
}


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

    colonnes_manquantes = set(COLONNES_MODELE_DECROCHAGE) - set(df_pret.columns)
    if colonnes_manquantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes dans le fichier envoye : {colonnes_manquantes}"
        )

    X = df_pret[COLONNES_MODELE_DECROCHAGE]
    proba_decrochage = MODELE_DECROCHAGE.predict_proba(X)[:, 1]

    resultats = []
    for i in range(len(df_pret)):
        proba_pct = round(float(proba_decrochage[i]) * 100, 1)
        niveau = deriver_niveau_risque(proba_pct)
        resultats.append({
            "codeEleve": str(df_brut.iloc[i].get("codeEleve", f"eleve_{i}")),
            "niveau_risque_predit": niveau,
            "niveau_risque_libelle": LIBELLES_NIVEAU[niveau],
            "probabilite_decrochage_pct": proba_pct,
            "recommandation": RECOMMANDATIONS[niveau],
        })

    return JSONResponse(content={"nombre_eleves": len(resultats), "resultats": resultats})


@app.get("/")
def accueil():
    return {"message": "API DropoutIA active (architecture simplifiee). Voir /docs."}
