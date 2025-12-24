# prc_visualization/dashboard.py

"""
Dashboard interactif (Streamlit/Dash) pour suivre progression R0.

Vues principales:
- Vue d'ensemble: % runs complétés, temps restant estimé
- Matrice Γ×D: heatmap scores agrégés
- Trajectoires: évolution états pour runs sélectionnés
- Analyse tests: distribution PASS/FAIL par test
- Comparaison: overlay plusieurs Γ sur mêmes métriques
"""

import streamlit as st
import plotly.express as px
from database import query_progress, query_scores_matrix

st.title("PRC R0 - Monitoring Exhaustif")

# Vue d'ensemble
st.header("Progression globale")
progress = query_progress(db)
st.metric("Runs complétés", f"{progress['completed']}/{progress['total']}")
st.metric("Temps écoulé", f"{progress['elapsed_hours']:.1f}h")
st.metric("Temps restant estimé", f"{progress['eta_hours']:.1f}h")

# Matrice scores
st.header("Matrice scores Γ×D")
scores_matrix = query_scores_matrix(db)
fig = px.imshow(scores_matrix, 
                labels=dict(x="D_base", y="Γ", color="Score"),
                x=scores_matrix.columns,
                y=scores_matrix.index)
st.plotly_chart(fig)

# Détails par Γ
selected_gamma = st.selectbox("Sélectionner Γ", get_all_gamma_ids())
st.header(f"Détails {selected_gamma}")

gamma_stats = query_gamma_statistics(db, selected_gamma)
col1, col2, col3 = st.columns(3)
col1.metric("Score moyen", f"{gamma_stats['mean_score']:.1f}/20")
col2.metric("Robustesse", f"{gamma_stats['robustness']:.0%}")
col3.metric("Majorité PASS", f"{gamma_stats['majority']:.0%}")

# ... plus de visualisations