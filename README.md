# DropoutIA

Système de détection précoce du décrochage scolaire par IA.

## Structure du projet

```
dropoutia/
├── data/
│   ├── raw/           # Fichiers Excel originaux (jamais commités sur Git)
│   └── processed/     # Données nettoyées, prêtes pour la modélisation
├── notebooks/          # Notebooks d'exploration, nettoyage, modélisation
├── src/                # Code Python réutilisable (fonctions de nettoyage, etc.)
├── models/             # Modèles entraînés sauvegardés (.pkl)
├── requirements.txt    # Librairies Python nécessaires
└── README.md
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate      # sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Avancement

- [x] Structure du projet créée
- [ ] Exploration des données
- [ ] Nettoyage des données
- [ ] Modèle A : niveau de risque
- [ ] Modèle B : prédiction de décrochage
- [ ] API FastAPI
- [ ] Interface Streamlit
- [ ] Génération de rapport PDF
