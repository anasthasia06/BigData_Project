from pathlib import Path
from typing import Dict, List, Optional


def load_css(css_path: Path) -> str:
	return css_path.read_text(encoding="utf-8")


def format_genres(genres: Optional[List[str]]) -> str:
	if not genres:
		return "-"
	return " • ".join([g for g in genres if str(g).strip()])


def game_card_html(game: Dict, mode: str = "search") -> str:
	name = game.get("name", "Unknown")
	appid = game.get("appid", "-")
	genres_text = format_genres(game.get("genres", []))

	if mode == "recommend":
		score = game.get("ranking_score", game.get("score", 0))
		meta = f"score: {float(score):.3f}"
	else:
		score = game.get("score", 0)
		ratio = game.get("positive_ratio")
		price = game.get("price")
		ratio_text = "-" if ratio is None else f"{float(ratio):.2f}"
		price_text = "-" if price is None else f"{float(price):.2f}$"
		meta = f"score: {float(score):.3f} | ratio+: {ratio_text} | prix: {price_text}"

	return f"""
	<div class='game-card'>
	  <div class='game-card-title'>{name}</div>
	  <div class='game-card-sub'>AppID: {appid}</div>
	  <div class='game-card-genres'>{genres_text}</div>
	  <div class='game-card-meta'>{meta}</div>
	</div>
	"""


def render_cards(st, title: str, items: List[Dict], mode: str = "search") -> None:
	st.subheader(title)
	if not items:
		st.info("Aucun résultat.")
		return
	for item in items:
		st.markdown(game_card_html(item, mode=mode), unsafe_allow_html=True)

