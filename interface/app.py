"""
Interface Streamlit DropoutIA -- upload d'un fichier Excel, appel a
l'API, affichage des resultats et export.

Lancement : streamlit run app.py  (depuis le dossier interface/)
L'API doit deja tourner en parallele sur http://127.0.0.1:8000
"""

import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="DropoutIA", page_icon="🎓", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

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

                    st.success(f"{donnees['nombre_eleves']} eleves analyses avec succes.")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Nombre d'eleves", donnees["nombre_eleves"])
                    col2.metric(
                        "Niveau de risque moyen",
                        round(df_resultats["niveau_risque_predit"].mean(), 2)
                    )
                    col3.metric(
                        "Probabilite moyenne de decrochage",
                        f"{round(df_resultats['probabilite_decrochage_pct'].mean(), 1)} %"
                    )

                    st.subheader("Resultats detailles")
                    st.dataframe(df_resultats, use_container_width=True)

                    csv = df_resultats.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Telecharger les resultats (CSV)",
                        data=csv,
                        file_name="resultats_dropoutia.csv",
                        mime="text/csv",
                    )

                else:
                    st.error(f"Erreur de l'API ({reponse.status_code}) : {reponse.json().get('detail')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Impossible de contacter l'API. Verifiez qu'elle tourne bien "
                    "sur http://127.0.0.1:8000 (voir le terminal de l'API)."
                )
