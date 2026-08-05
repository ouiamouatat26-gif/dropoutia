"""
Interface Streamlit DropoutIA -- upload d'un fichier Excel, appel a
l'API, affichage des resultats, visualisations, export CSV et PDF avec
alertes visuelles.
"""

import streamlit as st
import pandas as pd
import requests
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="DropoutIA", page_icon="🎓", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

COULEURS_NIVEAU = {
    1: colors.HexColor("#d4edda"),
    2: colors.HexColor("#fff3cd"),
    3: colors.HexColor("#ffe0b2"),
    4: colors.HexColor("#f8d7da"),
}


def generer_pdf(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("DropoutIA — Rapport de suivi du risque de décrochage", styles["Title"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Nombre d'élèves analysés : {len(df)} — "
        f"Élèves en situation sévère : {(df['niveau_risque_predit'] == 4).sum()}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 10))

    # --- Legende des couleurs ---
    elements.append(Paragraph("Légende des couleurs :", styles["Heading3"]))
    donnees_legende = [["Couleur", "Niveau de risque", "Signification"]]
    legende_texte = {
        1: "Aucune action requise, suivi de routine.",
        2: "A surveiller lors du prochain conseil de classe.",
        3: "Entretien conseille avec l'eleve dans les prochaines semaines.",
        4: "Intervention prioritaire, entretien avec l'eleve et sa famille.",
    }
    for niveau, libelle in [(1, "Tres faible"), (2, "Faible"), (3, "Modere"), (4, "Severe")]:
        donnees_legende.append(["", libelle, legende_texte[niveau]])

    tableau_legende = Table(donnees_legende, colWidths=[2 * cm, 3 * cm, 11.5 * cm])
    style_legende = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, niveau in enumerate([1, 2, 3, 4], start=1):
        style_legende.append(("BACKGROUND", (0, i), (0, i), COULEURS_NIVEAU[niveau]))
    tableau_legende.setStyle(TableStyle(style_legende))
    elements.append(tableau_legende)
    elements.append(Spacer(1, 16))

    # --- Tableau principal : ordre d'origine du fichier importe, non trie ---
    elements.append(Paragraph("Détail par élève (ordre du fichier importé) :", styles["Heading3"]))
    elements.append(Spacer(1, 6))

    donnees_tableau = [["Code élève", "Niveau de risque", "Probabilité", "Recommandation"]]
    for _, ligne in df.iterrows():
        donnees_tableau.append([
            ligne["codeEleve"],
            ligne["niveau_risque_libelle"],
            f"{ligne['probabilite_decrochage_pct']} %",
            Paragraph(ligne["recommandation"], styles["Normal"]),
        ])

    tableau = Table(donnees_tableau, colWidths=[3 * cm, 3 * cm, 2.5 * cm, 8 * cm], repeatRows=1)

    style_tableau = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, (_, ligne) in enumerate(df.iterrows(), start=1):
        couleur = COULEURS_NIVEAU.get(int(ligne["niveau_risque_predit"]), colors.white)
        style_tableau.append(("BACKGROUND", (0, i), (-1, i), couleur))

    tableau.setStyle(TableStyle(style_tableau))
    elements.append(tableau)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


st.title("🎓 DropoutIA — Detection du risque de decrochage")
st.write(
    "Importez un fichier Excel contenant les donnees des eleves "
    "(format Massar) pour obtenir le niveau de risque et la probabilite "
    "de decrochage de chacun."
)

fichier_uploade = st.file_uploader("Choisissez un fichier Excel (.xlsx)", type=["xlsx"])

if fichier_uploade is not None:
    st.success(f"Fichier recu : {fichier_uploade.name}")

    if st.button("Lancer l'analyse"):
        with st.spinner("Analyse en cours..."):
            try:
                fichiers = {"fichier": (fichier_uploade.name, fichier_uploade.getvalue())}
                reponse = requests.post(API_URL, files=fichiers, timeout=120)

                if reponse.status_code == 200:
                    donnees = reponse.json()
                    df_resultats = pd.DataFrame(donnees["resultats"])
                    st.session_state["df_resultats"] = df_resultats
                else:
                    st.error(f"Erreur de l'API ({reponse.status_code}) : {reponse.json().get('detail')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Impossible de contacter l'API. Verifiez qu'elle tourne bien "
                    "sur http://127.0.0.1:8000 (voir le terminal de l'API)."
                )

# On garde les resultats en memoire (session_state) pour qu'ils restent
# affiches meme apres avoir clique sur un bouton de telechargement
# (Streamlit reexecute tout le script a chaque clic, sinon on perdrait
# l'affichage)
if "df_resultats" in st.session_state:
    df_resultats = st.session_state["df_resultats"]

    st.success(f"{len(df_resultats)} eleves analyses avec succes.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre d'eleves", len(df_resultats))
    col2.metric("Eleves en situation severe", int((df_resultats["niveau_risque_predit"] == 4).sum()))
    col3.metric(
        "Probabilite moyenne de decrochage",
        f"{round(df_resultats['probabilite_decrochage_pct'].mean(), 1)} %"
    )

    st.subheader("📊 Visualisations")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Répartition des élèves par niveau de risque**")
        repartition = df_resultats["niveau_risque_libelle"].value_counts().reindex(
            ["Tres faible", "Faible", "Modere", "Severe"]
        )
        st.bar_chart(repartition)

    with col_b:
        st.write("**Distribution de la probabilité de décrochage**")
        tranches = pd.cut(
            df_resultats["probabilite_decrochage_pct"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        )
        st.bar_chart(tranches.value_counts().sort_index())

    st.subheader("Résultats détaillés")
    st.dataframe(df_resultats, use_container_width=True)

    col_csv, col_pdf = st.columns(2)
    with col_csv:
        csv = df_resultats.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Telecharger les resultats (CSV)",
            data=csv,
            file_name="resultats_dropoutia.csv",
            mime="text/csv",
        )
    with col_pdf:
        pdf_bytes = generer_pdf(df_resultats)
        st.download_button(
            label="🚨 Telecharger le rapport (PDF)",
            data=pdf_bytes,
            file_name="rapport_dropoutia.pdf",
            mime="application/pdf",
        )
