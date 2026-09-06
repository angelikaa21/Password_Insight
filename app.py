from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import analyze_password
from src.classifier import CLASS_LABELS, load_metadata, load_model
from src.features import FEATURE_LABELS


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "password_strength_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

STRENGTH_COLORS = ["#EF6A6A", "#F08A5D", "#E9B949", "#43B5A0", "#35A36F"]


st.set_page_config(
    page_title="Password Insight",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --surface-0:#0d1117;
      --surface-1:#131922;
      --surface-2:#19212c;
      --border:#2b3745;
      --text:#eef3f8;
      --muted:#9aa8b7;
      --accent:#6d93f8;
      --accent-soft:#6d93f81a;
    }
    .stApp {background:radial-gradient(circle at 18% -10%, #1c2735 0, #10161e 38%, var(--surface-0) 78%);}
    [data-testid="stHeader"] {background:transparent;}
    .block-container {max-width:1120px; padding-top:2rem; padding-bottom:4rem;}
    h1, h2, h3, p, label, .stMarkdown {color:var(--text);}
    .hero {padding:1.1rem 0 1.55rem;}
    .hero-badge {display:inline-block; padding:.38rem .72rem; border:1px solid #5574bd;
      border-radius:999px; color:#a9bff8; background:var(--accent-soft); font-size:.74rem;
      font-weight:650; letter-spacing:.075em;}
    .hero h1 {font-size:2.85rem; margin:.72rem 0 .5rem; line-height:1.05; letter-spacing:-.025em;}
    .hero p {color:var(--muted); max-width:760px; font-size:1.01rem; line-height:1.65;}
    .summary-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem;
      margin:.35rem 0 1.15rem;}
    .result-card {background:linear-gradient(145deg,var(--surface-2),#161d26); border:1px solid var(--border);
      border-radius:15px; padding:1.05rem 1.15rem; min-height:122px; box-sizing:border-box;
      box-shadow:0 12px 26px #00000020;}
    .result-label {color:var(--muted); text-transform:uppercase; letter-spacing:.075em;
      font-size:.7rem; font-weight:650;}
    .result-value {color:var(--text); font-size:1.5rem; font-weight:720; margin-top:.48rem; line-height:1.2;}
    .result-note {color:#aeb9c5; font-size:.8rem; margin-top:.3rem; line-height:1.35;}
    .privacy {background:#151f2d; border:1px solid #2c4668; border-radius:12px;
      padding:.82rem 1rem; color:#b9cdf2; margin:.5rem 0 1.05rem; line-height:1.5;}
    .privacy strong {color:#d8e4fb; font-size:.72rem; letter-spacing:.07em; margin-right:.45rem;}
    [data-testid="stForm"] {background:var(--surface-1); border:1px solid var(--border);
      border-radius:16px; padding:1.15rem; box-shadow:0 14px 32px #00000018;}
    .score-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem;
      margin:0 0 1.3rem; clear:both;}
    .score-panel {background:linear-gradient(145deg,var(--surface-2),#151c25); border:1px solid var(--border);
      border-radius:17px; padding:1.15rem 1.25rem; display:flex; align-items:center; gap:1.2rem;
      min-height:188px; box-sizing:border-box; overflow:hidden;}
    .score-ring {width:126px; height:126px; border-radius:50%; display:grid; place-items:center;
      flex:0 0 126px; box-shadow:0 10px 24px #00000028;}
    .score-ring-inner {width:96px; height:96px; border-radius:50%; background:#121821;
      display:flex; flex-direction:column; align-items:center; justify-content:center;}
    .score-number {font-size:1.6rem; color:#fff; font-weight:750; line-height:1;}
    .score-scale {font-size:.7rem; color:var(--muted); margin-top:.32rem;}
    .score-copy {min-width:0;}
    .score-copy h4 {color:var(--text); font-size:1.02rem; margin:0 0 .42rem;}
    .score-copy p {color:#aeb9c5; font-size:.84rem; line-height:1.5; margin:0;}
    .recommendation {background:#171e28; border:1px solid #283442; border-left:3px solid #6688de;
      border-radius:8px; padding:.72rem .9rem; color:#dbe3ec; margin:.55rem 0; line-height:1.45;}
    .recommendation-danger {border-left-color:#ef6a6a;}
    button[kind="primary"] {box-shadow:0 7px 20px #4f7de72b; font-weight:650;}
    div[data-baseweb="tab-list"] {gap:1.15rem;}
    button[data-baseweb="tab"] {padding-left:.2rem; padding-right:.2rem;}
    div[data-baseweb="tab-highlight"] {background-color:var(--accent);}
    @media (max-width:850px) {
      .summary-grid, .score-grid {grid-template-columns:1fr;}
    }
    @media (max-width:620px) {
      .score-panel {flex-direction:column; text-align:center;}
      .hero h1 {font-size:2.25rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH)


def score_ring(value: float, number: str, title: str, description: str, color: str) -> str:
    bounded = max(0.0, min(100.0, value))
    return (
        '<div class="score-panel">'
        f'<div class="score-ring" style="background:conic-gradient({color} {bounded:.1f}%, '
        f'#2d3744 {bounded:.1f}% 100%);">'
        '<div class="score-ring-inner">'
        f'<div class="score-number">{number}</div>'
        f'<div class="score-scale">{bounded:.0f}% skali</div>'
        '</div></div>'
        f'<div class="score-copy"><h4>{title}</h4><p>{description}</p></div>'
        '</div>'
    )


def format_feature_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.3f}"


def shap_chart(explanation: list[dict[str, object]], class_label: str) -> go.Figure:
    selected = list(reversed(explanation[:8]))
    values = [float(item["contribution"]) for item in selected]
    labels = [
        f'{item["label"]} = {format_feature_value(float(item["value"]))}'
        for item in selected
    ]
    colors = ["#6D93F8" if value >= 0 else "#E6A45B" for value in values]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    figure.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=20, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DDE6EF"},
        xaxis={
            "title": f'Wpływ na prawdopodobieństwo klasy „{class_label}”',
            "gridcolor": "#2A3542",
        },
        yaxis={"title": "", "automargin": True},
    )
    return figure


def probability_chart(probabilities: dict[int, float]) -> go.Figure:
    labels = [CLASS_LABELS[index] for index in range(5)]
    values = [probabilities[index] * 100 for index in range(5)]
    maximum = max(values) if values else 0
    colors = []
    for color, value in zip(STRENGTH_COLORS, values):
        if value == maximum:
            colors.append(color)
        else:
            red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
            colors.append(f"rgba({red},{green},{blue},0.45)")
    figure = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    figure.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DDE6EF"},
        yaxis={"title": "Prawdopodobieństwo [%]", "range": [0, 100], "gridcolor": "#2A3542"},
        xaxis={"title": ""},
    )
    return figure


metadata = load_metadata(METADATA_PATH)
try:
    model = cached_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.markdown(
    """
    <div class="hero">
      <span class="hero-badge">SYSTEM OCENY SIŁY I RYZYKA HASEŁ</span>
      <h1>Password Insight</h1>
      <p>Wielowymiarowa ocena siły i ryzyka hasła z wykorzystaniem modelu uczenia
      maszynowego, informacji o znanych naruszeniach oraz lokalnych wyjaśnień SHAP.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

analysis_tab, details_tab, about_tab = st.tabs(["Analiza hasła", "Szczegóły", "O systemie"])

with analysis_tab:
    st.markdown(
        '<div class="privacy"><strong>PRYWATNOŚĆ</strong> Hasło nie jest zapisywane ani umieszczane w wynikach. '
        "Do HIBP wysyłany jest wyłącznie pięcioznakowy prefiks jego skrótu SHA-1.</div>",
        unsafe_allow_html=True,
    )
    with st.form("password_form", clear_on_submit=True):
        password = st.text_input(
            "Hasło do analizy",
            type="password",
            placeholder="Wprowadź hasło…",
            help="Analiza rozpocznie się dopiero po użyciu przycisku.",
        )
        include_hibp = st.checkbox("Sprawdź wystąpienie w HIBP", value=True)
        submitted = st.form_submit_button("Przeprowadź analizę", type="primary", width="stretch")

    if submitted:
        if not password:
            st.warning("Wprowadź hasło przed rozpoczęciem analizy.")
        elif len(password) > 256:
            st.warning("System przyjmuje hasła o długości do 256 znaków.")
        else:
            with st.spinner("Analizowanie hasła…"):
                st.session_state["safe_result"] = analyze_password(model, password, include_hibp)

    result = st.session_state.get("safe_result")
    if result:
        strength = result["strength"]
        breach = result["breach"]
        risk = result["risk"]
        strength_color = STRENGTH_COLORS[strength["class_id"]]

        st.subheader("Wynik analizy")
        if breach["status"] == "checked" and breach["found"]:
            breach_value = "Wykryto"
            breach_note = f'{breach["count"]:,} wystąpień'.replace(",", " ")
            breach_color = "#EF6A6A"
        elif breach["status"] == "checked":
            breach_value = "Nie wykryto"
            breach_note = "Brak trafienia nie jest gwarancją bezpieczeństwa"
            breach_color = "#35A36F"
        else:
            breach_value = "Brak wyniku"
            breach_note = breach["message"]
            breach_color = "#E9B949"

        st.markdown(
            f"""
            <div class="summary-grid">
              <div class="result-card"><div class="result-label">Siła hasła</div>
                <div class="result-value" style="color:{strength_color}">{strength["class_label"]}</div>
                <div class="result-note">Klasa {strength["class_id"]}/4</div></div>
              <div class="result-card"><div class="result-label">Poziom ryzyka</div>
                <div class="result-value" style="color:{risk["color"]}">{risk["label"]}</div>
                <div class="result-note">Wskaźnik {risk["score"]:.0f}/100</div></div>
              <div class="result-card"><div class="result-label">Znane naruszenia</div>
                <div class="result-value" style="color:{breach_color}">{breach_value}</div>
                <div class="result-note">{breach_note}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        strength_panel = score_ring(
            strength["class_id"] * 25,
            f'{strength["class_id"]}/4',
            "Siła strukturalna",
            "Wynik modelu wyznaczony na podstawie cech opisujących budowę hasła.",
            strength_color,
        )
        risk_panel = score_ring(
            risk["score"],
            f'{risk["score"]:.0f}',
            "Wskaźnik ryzyka",
            "Łączy ryzyko strukturalne z wynikiem kontroli HIBP.",
            risk["color"],
        )
        st.markdown(
            f'<div class="score-grid">{strength_panel}{risk_panel}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Najważniejsze informacje")
        if breach["status"] == "unavailable":
            st.warning("Ocena ryzyka jest niepełna, ponieważ HIBP było niedostępne.")
        for recommendation in result["recommendations"]:
            extra_class = " recommendation-danger" if "naruszeni" in recommendation.lower() else ""
            st.markdown(
                f'<div class="recommendation{extra_class}">{recommendation}</div>',
                unsafe_allow_html=True,
            )

with details_tab:
    result = st.session_state.get("safe_result")
    if not result:
        st.info("Najpierw przeprowadź analizę hasła.")
    else:
        st.subheader("Rozkład wyniku modelu")
        st.plotly_chart(probability_chart(result["strength"]["probabilities"]), width="stretch")

        st.subheader("Lokalne wyjaśnienie SHAP")
        if result["explanation"]:
            class_label = result["strength"]["class_label"]
            st.plotly_chart(
                shap_chart(result["explanation"], class_label),
                width="stretch",
            )
            st.caption(
                f'Wykres dotyczy klasy „{class_label}”. Wartości niebieskie zwiększają jej '
                "prawdopodobieństwo, a pomarańczowe je zmniejszają względem wartości bazowej "
                "modelu. Wartość ujemna opisuje wpływ konkretnej wartości cechy i nie oznacza, "
                "że dana cecha jest ogólnie niekorzystna."
            )
        else:
            st.warning(result["explanation_error"])

        st.subheader("Cechy strukturalne")
        feature_rows = [
            {"Cecha": FEATURE_LABELS[name], "Wartość": round(value, 4)}
            for name, value in result["strength"]["features"].items()
        ]
        st.dataframe(pd.DataFrame(feature_rows), hide_index=True, width="stretch")

        st.subheader("Składowe ryzyka")
        risk = result["risk"]
        risk_rows = [
            {"Składowa": "Ryzyko strukturalne", "Wartość": risk["structural_component"]},
            {"Składowa": "Ryzyko naruszenia", "Wartość": risk["breach_component"]},
            {"Składowa": "Wynik końcowy", "Wartość": risk["score"]},
        ]
        st.dataframe(pd.DataFrame(risk_rows), hide_index=True, width="stretch")

with about_tab:
    st.subheader("Zakres systemu")
    st.write(
        "System jest środkiem badawczym przygotowanym do porównania metod klasyfikacji "
        "i analizy różnicy między siłą strukturalną a ryzykiem wynikającym z naruszenia."
    )
    st.markdown(
        """
        - **Model klasyfikacyjny:** Random Forest uczony na cechach strukturalnych.
        - **Naruszenia:** HIBP Pwned Passwords i zapytanie zakresowe.
        - **Wyjaśnialność:** lokalne wkłady SHAP dla przewidywanej klasy.
        - **Prywatność:** brak bazy danych, historii analiz i logowania treści haseł.
        """
    )
    st.subheader("Ograniczenia")
    st.write(
        "Ocena modelu zależy od danych treningowych. Brak hasła w HIBP nie potwierdza, "
        "że nigdy nie zostało ono ujawnione. Wynik nie zastępuje MFA, ograniczania liczby "
        "prób ani bezpiecznego przechowywania haseł po stronie usługi."
    )
    model_status = "model badawczy" if metadata.get("purpose") == "RESEARCH_MODEL" else "status nieustalony"
    st.caption(f'Model: {metadata.get("model_type", "brak danych")} · {model_status}')
