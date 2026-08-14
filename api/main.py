import io
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
import nettoyage

MODELE_DECROCHAGE = joblib.load(BASE_DIR / "models" / "modele_decrochage.pkl")
# MODELE_RISQUE n'est plus charge : Option B, un seul modele source (probabilite -> seuils)

COLONNES_MODELE_DECROCHAGE = [
    "id_Genre", "nefstat", "NbrAbsence", "NbrRedoublement",
    "MoyenneNotes", "cd_com", "NbrAbsence_manquant", "age"
]

COLONNES_MINIMALES = [
    "id_Genre", "nefstat", "NbrAbsence", "NbrRedoublement",
    "MoyenneNotes", "cd_com", "dateNaisEleve"
]

LIBELLES_NIVEAU = {1: "Tres faible", 2: "Faible", 3: "Modere", 4: "Severe"}

RECOMMANDATIONS = {
    1: "Suivi habituel.",
    2: "Verifier la situation au prochain conseil de classe.",
    3: "Prevoir un entretien avec l'eleve et analyser les absences recentes.",
    4: "Prioriser un entretien avec l'eleve, sa famille et l'equipe pedagogique.",
}

app = FastAPI(title="DropoutIA API")


def convertir_colonnes_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    colonnes_numeriques = [
        "id_Genre", "NbrAbsence", "NbrRedoublement", "MoyenneNotes",
        "cd_com", "CD_REG", "CD_PROV", "CD_MIL"
    ]
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mediane = df[col].median()
            if pd.isna(mediane):
                mediane = 0
            df[col] = df[col].fillna(mediane)
    return df


def preparer_fichier(df_brut: pd.DataFrame) -> pd.DataFrame:
    df = df_brut.copy()
    df = nettoyage.gerer_absences_manquantes(df, colonne="NbrAbsence")
    df = nettoyage.calculer_age(df, colonne_date="dateNaisEleve")
    df = nettoyage.encoder_categorielles(df, ["nefstat"])
    if "nefstat" in df.columns and "Nefstat" not in df.columns:
        df["Nefstat"] = df["nefstat"]
    df = convertir_colonnes_numeriques(df)
    return df


def niveau_depuis_probabilite(probabilite: float) -> int:
    if probabilite < 0.20:
        return 1
    if probabilite < 0.50:
        return 2
    if probabilite < 0.75:
        return 3
    return 4


def appliquer_regle_coherence(niveau_risque: int, probabilite_decrochage: float) -> int:
    # Garde-fou : ne se declenche plus jamais en Option B (niveau deja derive de la probabilite),
    # laisse en place par securite, sans effet.
    if probabilite_decrochage >= 0.50 and niveau_risque == 1:
        return 2
    return niveau_risque


def lire_resultat_observe(valeur):
    try:
        if pd.isna(valeur):
            return None
        return int(valeur)
    except Exception:
        return None


@app.post("/predict")
async def predict(fichier: UploadFile = File(...)):
    if not fichier.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Le fichier doit etre un Excel (.xlsx ou .xls).")

    contenu = await fichier.read()
    try:
        df_brut = pd.read_excel(io.BytesIO(contenu))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier Excel : {e}")

    if df_brut.empty:
        raise HTTPException(status_code=400, detail="Le fichier est vide.")

    colonnes_manquantes_min = [c for c in COLONNES_MINIMALES if c not in df_brut.columns]
    if colonnes_manquantes_min:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes minimales manquantes : {colonnes_manquantes_min}"
        )

    df_pret = preparer_fichier(df_brut)

    colonnes_decrochage_manquantes = [c for c in COLONNES_MODELE_DECROCHAGE if c not in df_pret.columns]
    if colonnes_decrochage_manquantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes pour le modele decrochage : {colonnes_decrochage_manquantes}"
        )

    X_decrochage = df_pret[COLONNES_MODELE_DECROCHAGE]
    proba_decrochage = MODELE_DECROCHAGE.predict_proba(X_decrochage)[:, 1]
    prediction_decrochage = (proba_decrochage >= 0.50).astype(int)

    # Option B : le niveau de risque vient TOUJOURS de la probabilite, jamais d'un second modele.
    niveau_risque_modele = [niveau_depuis_probabilite(float(p)) for p in proba_decrochage]
    source_niveau = "probabilite_decrochage"

    absences_toutes_manquantes = bool(df_brut["NbrAbsence"].isna().all())
    resultats = []
    nb_incoherences_observees = 0

    for i in range(len(df_pret)):
        proba = float(proba_decrochage[i])
        proba_pct = round(proba * 100, 1)
        niveau_modele = int(niveau_risque_modele[i])
        niveau_final = appliquer_regle_coherence(niveau_modele, proba)
        resultat_observe = lire_resultat_observe(df_brut.iloc[i].get("Resultat"))
        pred_binaire = int(prediction_decrochage[i])

        ligne = {
            "index_ligne": i + 1,
            "codeEleve": str(df_brut.iloc[i].get("codeEleve", f"eleve_{i + 1}")),
            "probabilite_decrochage_pct": proba_pct,
            "prediction_decrochage": pred_binaire,
            "prediction_decrochage_libelle": "Risque de decrochage" if pred_binaire == 1 else "Pas de decrochage predit",
            "niveau_risque_predit": niveau_final,
            "niveau_risque_modele": niveau_modele,
            "niveau_risque_libelle": LIBELLES_NIVEAU[niveau_final],
            "source_niveau_risque": source_niveau,
            "regle_coherence_appliquee": niveau_final != niveau_modele,
            "recommandation": RECOMMANDATIONS[niveau_final],
        }

        if resultat_observe is not None:
            if pred_binaire != resultat_observe:
                nb_incoherences_observees += 1

        resultats.append(ligne)

    meta = {
        "nombre_eleves": len(resultats),
        "source_niveau_risque": source_niveau,
        "absences_toutes_manquantes": absences_toutes_manquantes,
        "message_absences": (
            "NbrAbsence est entierement vide : remplacement technique par 0 avec indicateur NbrAbsence_manquant=1."
            if absences_toutes_manquantes else
            "Traitement normal des absences : mediane + indicateur de valeur manquante."
        ),
        "nb_lignes_a_verifier_si_resultat_present": nb_incoherences_observees,
    }

    return JSONResponse(content={"meta": meta, "nombre_eleves": len(resultats), "resultats": resultats})


@app.get("/")
def accueil():
    return {"message": "API DropoutIA active. Voir /docs."}
