"""
Interface Streamlit DropoutIA.
"""

import io

import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="DropoutIA", page_icon="D", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

COLONNES_MINIMALES = [
    "codeEleve", "dateNaisEleve", "id_Genre", "nefstat", "NbrAbsence",
    "NbrRedoublement", "MoyenneNotes", "cd_com"
]

COULEURS_NIVEAU = {
    1: "#d4edda",
    2: "#fff3cd",
    3: "#ffe0b2",
    4: "#f8d7da",
}


def couleur_niveau(niveau):
    return f"background-color: {COULEURS_NIVEAU.get(int(niveau), '#ffffff')}"


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
    st.sidebar.write("Optionnel pour validation :")
    st.sidebar.code("Resultat")

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
                fichiers = {"fichier": (fichier_uploade.name, fichier_uploade.getvalue())}
                reponse = requests.post(API_URL, files=fichiers, timeout=180)

                if reponse.status_code == 200:
                    donnees = reponse.json()
                    st.session_state["df_resultats"] = pd.DataFrame(donnees["resultats"])
                    st.session_state["meta_resultats"] = donnees.get("meta", {})
                else:
                    detail = reponse.json().get("detail", reponse.text)
                    st.error(f"Erreur de l'API ({reponse.status_code}) : {detail}")

            except requests.exceptions.ConnectionError:
                st.error("API non joignable. Lancez d'abord : uvicorn main:app --reload dans le dossier api.")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")

if "df_resultats" in st.session_state:
    df_resultats = st.session_state["df_resultats"]
    meta = st.session_state.get("meta_resultats", {})

    st.subheader("Synthese")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eleves analyses", len(df_resultats))
    c2.metric("Risque severe", int((df_resultats["niveau_risque_predit"] == 4).sum()))
    c3.metric("Probabilite moyenne", f"{df_resultats['probabilite_decrochage_pct'].mean():.1f} %")
    c4.metric("Lignes a verifier", int(meta.get("nb_lignes_a_verifier_si_resultat_present", 0)))

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
    st.dataframe(
        df_affichage,
        use_container_width=True,
        hide_index=True,
    )

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
