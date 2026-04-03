from pathlib import Path

import streamlit as st

from services.frontend.api_client import APIClient
from services.frontend.components import load_css, render_cards


def create_api_client() -> APIClient:
	return APIClient()


def apply_theme() -> None:
	css_path = Path(__file__).with_name("theme.css")
	if css_path.exists():
		st.markdown(f"<style>{load_css(css_path)}</style>", unsafe_allow_html=True)


def render_header() -> None:
	st.markdown(
		"""
		<div class='hero'>
			<h1>Steam Reco Explorer</h1>
			<p>Recherche et recommandations en temps réel via FastAPI + Elasticsearch.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


def search_view(client: APIClient) -> None:
	st.subheader("Recherche")

	col1, col2 = st.columns([3, 1])
	with col1:
		query = st.text_input("Mot-clé", placeholder="ex: action, rpg, puzzle")
	with col2:
		size = st.number_input("Nombre", min_value=1, max_value=100, value=10)

	genres = st.multiselect(
		"Genres (optionnel)",
		["Action", "RPG", "Adventure", "Simulation", "Indie", "Strategy", "Casual"],
	)

	c1, c2 = st.columns(2)
	with c1:
		min_ratio = st.slider("Positive ratio min", 0.0, 1.0, 0.5, 0.05)
	with c2:
		max_price = st.slider("Prix max", 0.0, 80.0, 30.0, 1.0)

	if st.button("Lancer la recherche", use_container_width=True):
		try:
			payload = client.search_games(
				q=query or None,
				genres=genres or None,
				min_positive_ratio=min_ratio,
				max_price=max_price,
				size=int(size),
			)
			items = payload.get("items", [])
			st.markdown(
				f"<div class='kpi-strip'><span class='kpi-pill'>Résultats: {len(items)}</span></div>",
				unsafe_allow_html=True,
			)
			render_cards(st, "Jeux trouvés", items, mode="search")
		except Exception as exc:
			st.error(f"Erreur API /search: {exc}")


def recommend_view(client: APIClient) -> None:
	st.subheader("Recommandations")

	col1, col2 = st.columns(2)
	with col1:
		top_n = st.number_input("Top N", min_value=1, max_value=100, value=10)
	with col2:
		genre = st.selectbox("Genre (optionnel)", ["", "Action", "RPG", "Adventure", "Strategy", "Indie"])

	if st.button("Générer recommandations", use_container_width=True):
		try:
			payload = client.recommend_games(n=int(top_n), genre=genre or None)
			items = payload.get("items", [])
			st.markdown(
				f"<div class='kpi-strip'><span class='kpi-pill'>Suggestions: {len(items)}</span></div>",
				unsafe_allow_html=True,
			)
			render_cards(st, "Top recommandations", items, mode="recommend")
		except Exception as exc:
			st.error(f"Erreur API /recommend: {exc}")


def run() -> None:
	st.set_page_config(page_title="Steam Frontend", page_icon="🎮", layout="wide")
	apply_theme()

	client = create_api_client()

	render_header()
	tab_search, tab_reco = st.tabs(["Recherche", "Recommandations"])

	with tab_search:
		search_view(client)

	with tab_reco:
		recommend_view(client)


if __name__ == "__main__":
	run()

