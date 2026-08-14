"""
Interface Streamlit DropoutIA - version autonome (sans API separee).
Le modele est charge et utilise directement ici.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import joblib

st.set_page_config(page_title="DropoutIA", page_icon="D", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
import nettoyage

MODELE_DECROCHAGE = joblib.load(BASE_DIR / "models" / "modele_decrochage.pkl")

COLONNES_MODELE_DECROCHAGE = [
    "id_Genre", "nefstat", "NbrAbsence", "NbrRedoublement",
    "MoyenneNotes", "cd_com", "NbrAbsence_manquant", "age"
]

COLONNES_MINIMALES = [
    "codeEleve", "dateNaisEleve", "id_Genre", "nefstat", "NbrAbsence",
    "NbrRedoublement", "MoyenneNotes", "cd_com"
]

LIBELLES_NIVEAU = {1: "Tres faible", 2: "Faible", 3: "Modere", 4: "Severe"}

RECOMMANDATIONS = {
    1: "Suivi habituel.",
    2: "Verifier la situation au prochain conseil de classe.",
    3: "Prevoir un entretien avec l'eleve et analyser les absences recentes.",
    4: "Prioriser un entretien avec l'eleve, sa famille et l'equipe pedagogique.",
}

COULEURS_NIVEAU = {
    1: "#d4edda",
    2: "#fff3cd",
    3: "#ffe0b2",
    4: "#f8d7da",
}


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


def analyser_fichier(df_brut: pd.DataFrame):
    df_pret = preparer_fichier(df_brut)

    colonnes_manquantes = [c for c in COLONNES_MODELE_DECROCHAGE if c not in df_pret.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes pour le modele : {colonnes_manquantes}")

    X = df_pret[COLONNES_MODELE_DECROCHAGE]
    proba_decrochage = MODELE_DECROCHAGE.predict_proba(X)[:, 1]
    prediction_decrochage = (proba_decrochage >= 0.50).astype(int)

    absences_toutes_manquantes = bool(df_brut["NbrAbsence"].isna().all())

    resultats = []
    for i in range(len(df_pret)):
        proba = float(proba_decrochage[i])
        proba_pct = round(proba * 100, 1)
        niveau = niveau_depuis_probabilite(proba)
        pred_binaire = int(prediction_decrochage[i])

        resultats.append({
            "index_ligne": i + 1,
            "codeEleve": str(df_brut.iloc[i].get("codeEleve", f"eleve_{i + 1}")),
            "probabilite_decrochage_pct": proba_pct,
            "prediction_decrochage": pred_binaire,
            "prediction_decrochage_libelle": "Risque de decrochage" if pred_binaire == 1 else "Pas de decrochage predit",
            "niveau_risque_predit": niveau,
            "niveau_risque_libelle": LIBELLES_NIVEAU[niveau],
            "recommandation": RECOMMANDATIONS[niveau],
        })

    meta = {
        "nombre_eleves": len(resultats),
        "absences_toutes_manquantes": absences_toutes_manquantes,
        "message_absences": (
            "NbrAbsence est entierement vide : remplacement technique par 0 avec indicateur NbrAbsence_manquant=1."
            if absences_toutes_manquantes else
            "Traitement normal des absences : mediane + indicateur de valeur manquante."
        ),
    }

    return pd.DataFrame(resultats), meta


def generer_pdf(df: pd.DataFrame, meta: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("DropoutIA - Rapport de prediction", styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Nombre d'eleves analyses : {len(df)}", styles["Normal"]))
    elements.append(Paragraph(f"Traitement des absences : {meta.get('message_absences', '')}", styles["Normal"]))
    elements.append(Paragraph(
        "Les probabilites sont des predictions du modele, elles ne remplacent pas une decision pedagogique.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 10))

    colonnes = [
        "index_ligne", "codeEleve", "probabilite_decrochage_pct",
        "prediction_decrochage_libelle", "niveau_risque_libelle", "recommandation"
    ]

    apercu = df[colonnes].head(200).copy()
    donnees = [colonnes] + apercu.astype(str).values.tolist()
    tableau = Table(donnees, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    tableau.setStyle(TableStyle(style))
    elements.append(tableau)
    if len(df) > 200:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Le PDF affiche les 200 premieres lignes. Le CSV contient toutes les lignes.", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def afficher_colonnes_attendues():
    st.sidebar.header("Fichier attendu")
    st.sidebar.write("Le fichier Excel doit contenir au minimum :")
    for col in COLONNES_MINIMALES:
        st.sidebar.code(col)

    st.sidebar.header("Lecture des resultats")
    st.sidebar.write("Probabilite : sortie du modele de decrochage.")
    st.sidebar.write("Niveau de risque : derive automatiquement de la probabilite (seuils fixes).")

    st.sidebar.header("Cas limite corrige")
    st.sidebar.write(
        "Si NbrAbsence est entierement vide, l'application ne plante plus : "
        "elle remplace techniquement par 0 et conserve NbrAbsence_manquant=1."
    )


afficher_colonnes_attendues()

st.title("DropoutIA - Detection du risque de decrochage")
st.caption("Analyse automatique d'un fichier Excel Massar avec probabilite, niveau de risque et recommandations.")

fichier_uploade = st.file_uploader("Importer le fichier Excel", type=["xlsx", "xls"])

if fichier_uploade is not None:
    st.success(f"Fichier recu : {fichier_uploade.name}")

    if st.button("Lancer l'analyse", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                df_brut = pd.read_excel(io.BytesIO(fichier_uploade.getvalue()))
                df_resultats, meta = analyser_fichier(df_brut)
                st.session_state["df_resultats"] = df_resultats
                st.session_state["meta_resultats"] = meta
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")

if "df_resultats" in st.session_state:
    df_resultats = st.session_state["df_resultats"]
    meta = st.session_state.get("meta_resultats", {})

    st.subheader("Synthese")
    c1, c2, c3 = st.columns(3)
    c1.metric("Eleves analyses", len(df_resultats))
    c2.metric("Risque severe", int((df_resultats["niveau_risque_predit"] == 4).sum()))
    c3.metric("Probabilite moyenne", f"{df_resultats['probabilite_decrochage_pct'].mean():.1f} %")

    if meta.get("absences_toutes_manquantes"):
        st.warning(meta.get("message_absences"))
    else:
        st.info(meta.get("message_absences", "Traitement normal des donnees."))

    st.subheader("Visualisations")
    col_a, col_b = st.columns(2)
    with col_a:
        repartition = df_resultats["niveau_risque_libelle"].value_counts().reindex(
            ["Tres faible", "Faible", "Modere", "Severe"], fill_value=0
        )
        st.bar_chart(repartition)
    with col_b:
        tranches = pd.cut(
            df_resultats["probabilite_decrochage_pct"],
            bins=[0, 20, 40, 60, 80, 100],
            include_lowest=True,
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        )
        st.bar_chart(tranches.value_counts().sort_index())

    st.subheader("Resultats detailles")
    colonnes_affichage = [
        "index_ligne", "codeEleve", "probabilite_decrochage_pct",
        "prediction_decrochage_libelle", "niveau_risque_libelle", "recommandation"
    ]
    df_affichage = df_resultats[colonnes_affichage].copy()
    st.dataframe(df_affichage, use_container_width=True, hide_index=True)

    st.subheader("Exports")
    col_csv, col_pdf = st.columns(2)
    with col_csv:
        csv = df_resultats.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Telecharger les resultats CSV",
            data=csv,
            file_name="resultats_dropoutia_corriges.csv",
            mime="text/csv",
        )
    with col_pdf:
        pdf_bytes = generer_pdf(df_resultats, meta)
        st.download_button(
            "Telecharger le rapport PDF",
            data=pdf_bytes,
            file_name="rapport_dropoutia.pdf",
            mime="application/pdf",
        )