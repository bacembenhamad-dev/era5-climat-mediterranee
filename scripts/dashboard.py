"""
dashboard.py — Tableau de bord Dash AVANCE pour le projet ERA5 1980-2024.

Onglets :
    1. Apercu          - vue mono-zone (KPIs + 4 graphes)
    2. Comparaison     - 3 zones superposees (Europe/Tunisie/Mediterranee)
    3. Anomalies       - ecarts a la climatologie 1981-2010 + tests stats
    4. Carte animee    - evolution spatiale annee par annee

Lancement : python scripts/dashboard.py
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback

# Theme issue de Claude Design (era5_theme.py + assets/ERA5_styles.css)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from era5_theme import INDEX_STRING, apply_theme  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed" / "era5_daily.parquet"
DATA_SP = PROJECT_ROOT / "data" / "processed" / "era5_spatial_yearly.parquet"

if not DATA.exists():
    raise SystemExit(f"Donnees absentes : {DATA}\nLancez d'abord preprocessing.py")

df = pd.read_parquet(DATA)
df["date"] = pd.to_datetime(df["date"])
df_sp = pd.read_parquet(DATA_SP) if DATA_SP.exists() else None

YEARS = sorted(df["year"].unique())
ZONES = sorted(df["zone"].unique())
REF_PERIOD = (1981, 2010)  # Reference WMO climatologie standard
HEATWAVE_THRESHOLD = 30   # degC a 12 UTC -> proxy d'une vague de chaleur
HEATWAVE_MIN_DAYS = 3

# --- Caches : agreges precalcules une seule fois au demarrage ---
_VARS = ["t2m", "d2m", "rh", "tp"]
ANNUAL = (df.groupby(["zone", "year"])[_VARS].mean().reset_index())
MONTHLY = (df.groupby(["zone", "year", "month"])[_VARS].mean().reset_index())
SEASONAL_CYCLE = (df.groupby(["zone", "month"])[_VARS].mean().reset_index())

VAR_LABELS = {
    "t2m": "Temperature 2m (degC)",
    "rh":  "Humidite relative (%)",
    "tp":  "Precipitations 12 UTC (mm/h)",
    "d2m": "Point de rosee (degC)",
}
VAR_ICONS = {"t2m": "🌡️", "rh": "💧", "tp": "🌧️", "d2m": "❄️"}

# Palette "Dark Command Center" issue de Claude Design (ERA5 v2)
ZONE_COLORS = {"europe": "#22d3ee",       # cyan — rive nord
               "tunisie": "#fbbf24",      # amber — rive sud
               "mediterranee": "#a78bfa"} # violet — bassin
ZONE_LABELS = {"europe": "Europe (rive nord)",
               "tunisie": "Tunisie (rive sud)",
               "mediterranee": "Méditerranée (bassin)"}

COLORS = {
    "primary":   "#6366f1",   # indigo
    "primary_l": "#818cf8",   # indigo light
    "secondary": "#22d3ee",   # cyan
    "accent":    "#f97316",   # orange
    "danger":    "#ef4444",
    "success":   "#22c55e",
    "purple":    "#a78bfa",
    "blue":      "#3b82f6",
    "bg":        "transparent",
    "card":      "rgba(15,22,40,0.85)",
    "surface":   "rgba(20,30,55,0.6)",
    "text":      "rgba(226,232,255,0.95)",
    "text2":     "rgba(148,163,210,0.85)",
    "muted":     "rgba(100,116,165,0.85)",
    "border":    "rgba(99,102,241,0.18)",
}

# Palette pour Plotly : axes, grille, fond
PLOTLY_DARK_BG = "rgba(0,0,0,0)"
PLOTLY_GRID = "rgba(99,102,241,0.10)"
PLOTLY_AXIS = "rgba(148,163,210,0.85)"
PLOTLY_FONT = "DM Sans, Inter, system-ui, sans-serif"


# =============================================================================
# Statistiques avancees
# =============================================================================

def mann_kendall(x):
    """Test de Mann-Kendall (tendance non-parametrique).
    Retourne (tau, p_value, decision)."""
    x = np.asarray(x)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, "n trop petit"
    tau, p = stats.kendalltau(np.arange(n), x)
    if p < 0.001:
        decision = "tres significative"
    elif p < 0.05:
        decision = "significative"
    elif p < 0.10:
        decision = "marginale"
    else:
        decision = "non significative"
    return tau, p, decision


def sen_slope(x):
    """Pente de Sen (estimateur robuste). Wrapper sur scipy.stats.theilslopes."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    slope, _, _, _ = stats.theilslopes(x, np.arange(len(x)))
    return slope


def detect_heatwaves(daily_t, threshold=30, min_duration=3):
    """Detecte les episodes consecutifs >= threshold sur min_duration jours."""
    above = (daily_t > threshold).astype(int)
    events = []
    current_start, current_len = None, 0
    for i, v in enumerate(above):
        if v:
            if current_start is None:
                current_start = i
            current_len += 1
        else:
            if current_start is not None and current_len >= min_duration:
                events.append((current_start, current_len))
            current_start, current_len = None, 0
    if current_start is not None and current_len >= min_duration:
        events.append((current_start, current_len))
    return events


# =============================================================================
# Application Dash + CSS
# =============================================================================

app = Dash(__name__,
           assets_folder=str(Path(__file__).resolve().parent.parent / "dashboard"))
app.title = "Climat Mediterranee · ERA5"
app.index_string = INDEX_STRING


def kpi_card(title, value, subtitle="", color=None, icon=""):
    color = color or COLORS["primary"]
    glow = _hex_to_rgba_bg(color, 0.12)
    return html.Div(
        className="kpi-card glass-card",
        style={"--accent-c": color, "--accent-c2": color,
               "--accent-glow": glow},
        children=[
            html.Div(className="kpi-head", children=[
                html.Div(title.upper(), className="kpi-label"),
                html.Div(icon, className="kpi-icon",
                         style={"background": glow, "color": color}),
            ]),
            html.Div(value, className="kpi-value"),
            html.Div(subtitle, className="kpi-sub"),
        ])


def _hex_to_rgba_bg(hex_color, alpha=0.10):
    """Convertit un hex en rgba avec alpha pour le fond d'icone."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def graph_card(graph_id, title, subtitle=""):
    return html.Div(
        className="glass-card",
        style={"padding": "20px"},
        children=[
            html.Div(title, className="graph-title"),
            html.Div(subtitle, className="graph-subtitle",
                     style={"marginBottom": "14px"}),
            dcc.Graph(id=graph_id,
                      config={"displayModeBar": "hover",
                              "toImageButtonOptions":
                              {"format": "png", "scale": 2},
                              "displaylogo": False}),
        ])


def section_card(*children, padding="20px", controls=False):
    return html.Section(
        className="glass-card controls" if controls else "glass-card",
        style={"padding": padding, "marginBottom": "16px"},
        children=list(children))


# =============================================================================
# Onglet 1 — Vue d'ensemble mono-zone
# =============================================================================

def tab_overview():
    return html.Div([
        section_card(
            html.Div("Paramètres", className="section-eyebrow"),
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "1fr 1fr 2fr", "gap": "20px",
                       "alignItems": "end"},
                children=[
                    html.Div([
                        html.Label("Variable",
                                   style={"fontSize": "13px",
                                          "fontWeight": 500,
                                          "marginBottom": "6px",
                                          "display": "block"}),
                        dcc.Dropdown(id="ov-var",
                                     options=[{"label": f"{VAR_ICONS[k]}  {v}",
                                               "value": k}
                                              for k, v in VAR_LABELS.items()],
                                     value="t2m", clearable=False),
                    ]),
                    html.Div([
                        html.Label("Zone",
                                   style={"fontSize": "13px",
                                          "fontWeight": 500,
                                          "marginBottom": "6px",
                                          "display": "block"}),
                        dcc.Dropdown(id="ov-zone",
                                     options=[{"label": "📍 " + ZONE_LABELS[z],
                                               "value": z} for z in ZONES],
                                     value="tunisie", clearable=False),
                    ]),
                    html.Div([
                        html.Label(id="ov-range-label",
                                   style={"fontSize": "13px",
                                          "fontWeight": 500,
                                          "marginBottom": "6px",
                                          "display": "block"}),
                        dcc.RangeSlider(id="ov-years",
                                        min=min(YEARS), max=max(YEARS), step=1,
                                        value=[min(YEARS), max(YEARS)],
                                        marks={y: str(y) for y in
                                               range(min(YEARS),
                                                     max(YEARS)+1, 5)},
                                        tooltip={"placement": "bottom"}),
                    ]),
                ]),
            controls=True,
        ),
        html.Div(id="ov-kpis",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "16px", "marginBottom": "20px"}),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                        "marginBottom": "20px"},
                 children=[
                     graph_card("ov-ts", "Tendance annuelle",
                                "Moyenne par année + ajustement OLS"),
                     graph_card("ov-box", "Distribution saisonnière",
                                "Quartiles par saison"),
                 ]),
        graph_card("ov-heat", "Heatmap mensuelle",
                   "Année × mois — détection visuelle des saisons et anomalies"),
        html.Div(style={"marginTop": "20px"}, children=[
            graph_card(
                "ov-heatwaves",
                f"Vagues de chaleur (≥ {HEATWAVE_MIN_DAYS} jours consécutifs "
                f"avec T à 12 UTC > {HEATWAVE_THRESHOLD}°C)",
                "Nombre d'épisodes par an. Le seuil porte sur la valeur "
                "instantanée à 12 UTC, qui sous-estime le maximum diurne "
                "(proxy d'une canicule, pas définition stricte OMM)."),
        ]),
    ])


# =============================================================================
# Onglet 2 — Comparaison 3 zones
# =============================================================================

def tab_compare():
    return html.Div([
        section_card(
            html.Div("Comparaison Europe vs Tunisie vs Méditerranée",
                     className="section-eyebrow"),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "1fr 2fr",
                            "gap": "20px", "alignItems": "end"},
                     children=[
                         html.Div([
                             html.Label("Variable",
                                        style={"fontSize": "13px",
                                               "fontWeight": 500,
                                               "marginBottom": "6px",
                                               "display": "block"}),
                             dcc.Dropdown(id="cmp-var",
                                          options=[
                                              {"label":
                                               f"{VAR_ICONS[k]}  {v}",
                                               "value": k}
                                              for k, v in VAR_LABELS.items()],
                                          value="t2m", clearable=False),
                         ]),
                         html.Div([
                             html.Label(id="cmp-range-label",
                                        style={"fontSize": "13px",
                                               "fontWeight": 500,
                                               "marginBottom": "6px",
                                               "display": "block"}),
                             dcc.RangeSlider(id="cmp-years",
                                             min=min(YEARS), max=max(YEARS),
                                             step=1,
                                             value=[min(YEARS), max(YEARS)],
                                             marks={y: str(y) for y in
                                                    range(min(YEARS),
                                                          max(YEARS)+1, 5)},
                                             tooltip={"placement": "bottom"}),
                         ]),
                     ]),
            controls=True,
        ),
        html.Div(id="cmp-stats",
                 style={"marginBottom": "20px"}),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                        "marginBottom": "20px"},
                 children=[
                     graph_card("cmp-ts",
                                "Évolution comparée (3 zones)",
                                "Moyennes annuelles + tendances OLS"),
                     graph_card("cmp-violin",
                                "Distribution par zone",
                                "Violin plots — densité de probabilité"),
                 ]),
        graph_card("cmp-seasonal",
                   "Cycle saisonnier comparé",
                   "Moyenne par mois calendaire pour chaque zone"),
    ])


# =============================================================================
# Onglet 3 — Anomalies vs 1981-2010
# =============================================================================

def tab_anomalies():
    return html.Div([
        section_card(
            html.Div("Anomalies vs climatologie 1981-2010 (référence WMO)",
                     className="section-eyebrow"),
            html.Div(
                "L'anomalie est l'écart entre la valeur d'une année et la "
                "moyenne de la période de référence 1981-2010 (standard "
                "Organisation Météorologique Mondiale). Elle permet de comparer "
                "les zones sur une base commune malgré leurs climats différents.",
                style={"fontSize": "13px", "color": COLORS["muted"],
                       "marginBottom": "16px", "lineHeight": "1.5"}),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
                     children=[
                         html.Div([
                             html.Label("Variable",
                                        style={"fontSize": "13px",
                                               "fontWeight": 500,
                                               "marginBottom": "6px",
                                               "display": "block"}),
                             dcc.Dropdown(id="anom-var",
                                          options=[
                                              {"label":
                                               f"{VAR_ICONS[k]}  {v}",
                                               "value": k}
                                              for k, v in VAR_LABELS.items()],
                                          value="t2m", clearable=False),
                         ]),
                         html.Div([
                             html.Label("Zone",
                                        style={"fontSize": "13px",
                                               "fontWeight": 500,
                                               "marginBottom": "6px",
                                               "display": "block"}),
                             dcc.Dropdown(id="anom-zone",
                                          options=[
                                              {"label":
                                               "📍 " + ZONE_LABELS[z],
                                               "value": z} for z in ZONES],
                                          value="tunisie", clearable=False),
                         ]),
                     ]),
            controls=True,
        ),
        html.Div(id="anom-stats",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "16px", "marginBottom": "20px"}),
        graph_card("anom-bars",
                   "Anomalies annuelles (vs 1981-2010)",
                   "Barres rouges = au-dessus de la normale, "
                   "bleues = en-dessous"),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                        "marginTop": "20px"},
                 children=[
                     graph_card("anom-decadal",
                                "Comparaison décennale",
                                "Moyennes par décennie"),
                     graph_card("anom-distrib",
                                "Distribution par décennie",
                                "Glissement de la distribution dans le temps"),
                 ]),
    ])


# =============================================================================
# Onglet 4 — Carte spatiale animée
# =============================================================================

def tab_map():
    return html.Div([
        section_card(
            html.Div("Évolution spatiale animée",
                     className="section-eyebrow"),
            html.Div(
                "Carte de la moyenne annuelle de la variable sélectionnée. "
                "Utilisez le bouton ▶️ pour animer, ou le slider pour explorer "
                "année par année. Données : grille ERA5 0.25° × 0.25°.",
                style={"fontSize": "13px", "color": COLORS["muted"],
                       "marginBottom": "16px", "lineHeight": "1.5"}),
            html.Div([
                html.Label("Variable",
                           style={"fontSize": "13px", "fontWeight": 500,
                                  "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(id="map-var",
                             options=[{"label": f"{VAR_ICONS[k]}  {v}",
                                       "value": k}
                                      for k, v in VAR_LABELS.items()],
                             value="t2m", clearable=False,
                             style={"maxWidth": "400px"}),
            ]),
            controls=True,
        ),
        graph_card("map-anim",
                   "Carte ERA5 animée — moyenne annuelle",
                   "28°N – 46°N / -10°E – 40°E  ·  appuyez sur ▶️ pour lancer"),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                        "marginTop": "20px"},
                 children=[
                     graph_card("map-diff",
                                "Différence : 2015-2024 vs 1980-1989",
                                "Réchauffement spatial sur 4 décennies"),
                     graph_card("map-mean",
                                "Climatologie 1980-2024",
                                "Moyenne sur toute la période"),
                 ]),
    ])


# =============================================================================
# Layout principal
# =============================================================================

# Le header et le footer sont injectes par INDEX_STRING (era5_theme.py).
# L'app Dash mounte uniquement entre les deux : tabs + tab-content.
app.layout = html.Div(
    children=[
        dcc.Tabs(
            id="tabs", value="ov",
            className="custom-tabs",
            children=[
                dcc.Tab(label="▦  Aperçu", value="ov",
                        className="custom-tab",
                        selected_className="custom-tab--selected"),
                dcc.Tab(label="↕  Comparaison 3 zones", value="cmp",
                        className="custom-tab",
                        selected_className="custom-tab--selected"),
                dcc.Tab(label="≋  Anomalies & tendances", value="anom",
                        className="custom-tab",
                        selected_className="custom-tab--selected"),
                dcc.Tab(label="◎  Carte animée", value="map",
                        className="custom-tab",
                        selected_className="custom-tab--selected"),
            ]),
        html.Div(
            id="tab-content",
            className="chart-container",
            style={"maxWidth": "1500px", "margin": "0 auto",
                   "padding": "28px 40px 48px"}),
    ])


# =============================================================================
# Routage des onglets
# =============================================================================

@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"ov": tab_overview(), "cmp": tab_compare(),
            "anom": tab_anomalies(), "map": tab_map()}.get(tab, tab_overview())


# =============================================================================
# Onglet 1 — callback
# =============================================================================

@callback(
    [Output("ov-kpis", "children"), Output("ov-ts", "figure"),
     Output("ov-box", "figure"), Output("ov-heat", "figure"),
     Output("ov-heatwaves", "figure"),
     Output("ov-range-label", "children")],
    [Input("ov-var", "value"), Input("ov-zone", "value"),
     Input("ov-years", "value")],
)
def cb_overview(var, zone, year_range):
    sub = df[(df["zone"] == zone) &
             (df["year"] >= year_range[0]) &
             (df["year"] <= year_range[1])].copy()
    label = VAR_LABELS[var]
    annual = (ANNUAL[(ANNUAL.zone == zone) &
                     (ANNUAL.year.between(*year_range))]
              .set_index("year")[var])

    # Stats
    if len(annual) > 1:
        slope_ols = np.polyfit(annual.index, annual.values, 1)[0]
        slope_sen = sen_slope(annual.values)
        tau, p_mk, decision = mann_kendall(annual.values)
        trend_str = f"{slope_ols*10:+.2f}/déc."
        mk_color = (COLORS["danger"] if p_mk < 0.05 and slope_ols > 0
                    else COLORS["primary"] if p_mk < 0.05
                    else COLORS["muted"])
        mk_subtitle = f"Sen: {slope_sen*10:+.2f}/déc · MK: {decision}"
    else:
        trend_str = "n/a"; mk_subtitle = ""; mk_color = COLORS["muted"]

    # Detection des episodes canicule (seuil 12 UTC, proxy de Tmax journalier)
    daily_t = sub.sort_values("date")["t2m"].values
    events = (detect_heatwaves(daily_t, HEATWAVE_THRESHOLD, HEATWAVE_MIN_DAYS)
              if "t2m" in sub else [])
    n_events = len(events)
    n_hot_days = int(sum(length for _, length in events))

    kpis = [
        kpi_card("Période", f"{year_range[0]}-{year_range[1]}",
                 f"{year_range[1]-year_range[0]+1} années · {len(sub):,} jours",
                 COLORS["primary"], "📅"),
        kpi_card(f"{label[:25]} — moyenne", f"{sub[var].mean():.2f}",
                 f"min {sub[var].min():.1f} · max {sub[var].max():.1f}",
                 COLORS["secondary"], VAR_ICONS[var]),
        kpi_card("Tendance (OLS)", trend_str, mk_subtitle, mk_color, "📈"),
        kpi_card(f"Épisodes ≥{HEATWAVE_MIN_DAYS}j > {HEATWAVE_THRESHOLD}°C",
                 f"{n_events}",
                 f"{n_hot_days} jours cumulés · seuil au sample 12 UTC",
                 COLORS["danger"], "🔥"),
    ]

    # Time series
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=annual.index, y=annual.values,
                                mode="lines+markers",
                                line=dict(color=COLORS["primary"], width=2.5),
                                marker=dict(size=6),
                                name=label,
                                hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>"))
    if len(annual) > 1:
        z = np.polyfit(annual.index, annual.values, 1)
        fig_ts.add_trace(go.Scatter(x=annual.index,
                                    y=z[0]*annual.index+z[1],
                                    mode="lines",
                                    name=f"OLS: {z[0]*10:+.2f}/déc",
                                    line=dict(dash="dash",
                                              color=COLORS["danger"], width=2)))
    fig_ts.update_layout(xaxis_title="Année", yaxis_title=label,
                         margin=dict(l=20, r=20, t=20, b=40), height=360,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                     xanchor="right", x=1,
                                     bgcolor="rgba(0,0,0,0)"),
                         hovermode="x unified")

    # Boxplot
    season_colors = {"DJF": "#3B82F6", "MAM": "#10B981",
                     "JJA": "#F59E0B", "SON": "#EF4444"}
    fig_box = px.box(sub, x="season", y=var, color="season",
                     category_orders={"season": ["DJF", "MAM", "JJA", "SON"]},
                     color_discrete_map=season_colors, points=False)
    fig_box.update_layout(showlegend=False,
                          xaxis_title="Saison", yaxis_title=label,
                          margin=dict(l=20, r=20, t=20, b=40), height=360)

    # Heatmap (utilise le cache MONTHLY)
    monthly = MONTHLY[(MONTHLY.zone == zone) &
                      (MONTHLY.year.between(*year_range))]
    pivot = monthly.pivot(index="year", columns="month", values=var)
    cs = "RdYlBu_r" if var in ["t2m", "d2m"] else \
         ("Blues" if var == "tp" else "YlGnBu")
    fig_heat = px.imshow(pivot,
                         labels=dict(x="Mois", y="Année", color=label),
                         aspect="auto", color_continuous_scale=cs)
    fig_heat.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                           margin=dict(l=20, r=20, t=20, b=40), height=420,
                           xaxis=dict(tickmode="array",
                                      tickvals=list(range(1, 13)),
                                      ticktext=["Jan","Fév","Mar","Avr","Mai",
                                                "Juin","Juil","Aoû","Sep",
                                                "Oct","Nov","Déc"]))

    # Vagues de chaleur par annee (utilise detect_heatwaves)
    sub_sorted = sub.sort_values("date")
    hw_per_year = []
    for y, g in sub_sorted.groupby("year"):
        ev = detect_heatwaves(g["t2m"].values, HEATWAVE_THRESHOLD,
                              HEATWAVE_MIN_DAYS)
        hw_per_year.append((y, len(ev), sum(length for _, length in ev)))
    hw_df = pd.DataFrame(hw_per_year,
                         columns=["year", "n_episodes", "n_days"])
    fig_hw = go.Figure()
    fig_hw.add_trace(go.Bar(x=hw_df["year"], y=hw_df["n_episodes"],
                            marker_color=COLORS["danger"], opacity=0.85,
                            name="Épisodes",
                            customdata=hw_df["n_days"],
                            hovertemplate=("<b>%{x}</b><br>"
                                           "Épisodes : %{y}<br>"
                                           "Jours cumulés : %{customdata}"
                                           "<extra></extra>")))
    if len(hw_df) > 1:
        z_hw = np.polyfit(hw_df["year"], hw_df["n_episodes"], 1)
        fig_hw.add_trace(go.Scatter(x=hw_df["year"],
                                    y=z_hw[0]*hw_df["year"]+z_hw[1],
                                    mode="lines", name=f"OLS: {z_hw[0]*10:+.2f}/déc",
                                    line=dict(dash="dash",
                                              color=COLORS["text"], width=2)))
    fig_hw.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                         xaxis_title="Année",
                         yaxis_title="Nombre d'épisodes",
                         margin=dict(l=20, r=20, t=20, b=40), height=340,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                     xanchor="right", x=1,
                                     bgcolor="rgba(0,0,0,0)"),
                         hovermode="x unified")

    range_label = (f"Plage d'années  ·  {year_range[0]} → {year_range[1]}  "
                   f"({year_range[1]-year_range[0]+1} ans)")
    for _f in (fig_ts, fig_box, fig_heat, fig_hw):
        apply_theme(_f)
    return kpis, fig_ts, fig_box, fig_heat, fig_hw, range_label


# =============================================================================
# Onglet 2 — Comparaison 3 zones
# =============================================================================

@callback(
    [Output("cmp-stats", "children"), Output("cmp-ts", "figure"),
     Output("cmp-violin", "figure"), Output("cmp-seasonal", "figure"),
     Output("cmp-range-label", "children")],
    [Input("cmp-var", "value"), Input("cmp-years", "value")],
)
def cb_compare(var, year_range):
    sub = df[(df["year"] >= year_range[0]) &
             (df["year"] <= year_range[1])].copy()
    label = VAR_LABELS[var]
    annual_cache = ANNUAL[ANNUAL.year.between(*year_range)]

    # Tableau de stats par zone (Mann-Kendall + Sen)
    rows = []
    for zone in ["europe", "tunisie", "mediterranee"]:
        zsub = sub[sub.zone == zone]
        annual = annual_cache[annual_cache.zone == zone].set_index("year")[var]
        if len(annual) > 3:
            slope = np.polyfit(annual.index, annual.values, 1)[0] * 10
            sen = sen_slope(annual.values) * 10
            tau, p, decision = mann_kendall(annual.values)
            badge_class = ("b-danger" if p < 0.001
                           else "b-warning" if p < 0.05
                           else "b-muted")
        else:
            slope = sen = tau = p = np.nan
            decision = "n/a"; badge_class = "b-muted"
        rows.append(html.Tr([
            html.Td(html.Span(ZONE_LABELS[zone],
                              style={"fontWeight": 600,
                                     "color": ZONE_COLORS[zone]})),
            html.Td(f"{zsub[var].mean():.2f}"),
            html.Td(f"{slope:+.3f}"),
            html.Td(f"{sen:+.3f}"),
            html.Td(f"{tau:+.3f}"),
            html.Td(f"{p:.4f}"),
            html.Td(html.Span(decision, className=f"badge {badge_class}")),
        ]))
    stats_table = section_card(
        html.Div("Tableau statistique — tendances par zone",
                 className="section-eyebrow"),
        html.Table(className="stat-table", children=[
            html.Thead(html.Tr([html.Th(c) for c in
                ["Zone", "Moyenne", "OLS /déc", "Sen /déc",
                 "Mann-Kendall τ", "p-value", "Significativité"]])),
            html.Tbody(rows),
        ]))

    # Time series superposees
    fig_ts = go.Figure()
    for zone in ["europe", "tunisie", "mediterranee"]:
        annual = (annual_cache[annual_cache.zone == zone]
                  .set_index("year")[var])
        fig_ts.add_trace(go.Scatter(x=annual.index, y=annual.values,
                                    mode="lines+markers",
                                    name=ZONE_LABELS[zone],
                                    line=dict(color=ZONE_COLORS[zone],
                                              width=2.5),
                                    marker=dict(size=5)))
        if len(annual) > 1:
            z = np.polyfit(annual.index, annual.values, 1)
            fig_ts.add_trace(go.Scatter(x=annual.index,
                                        y=z[0]*annual.index+z[1],
                                        mode="lines", showlegend=False,
                                        line=dict(dash="dash",
                                                  color=ZONE_COLORS[zone],
                                                  width=1.5),
                                        opacity=0.6, hoverinfo="skip"))
    fig_ts.update_layout(xaxis_title="Année", yaxis_title=label,
                         margin=dict(l=20, r=20, t=20, b=40), height=400,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                     xanchor="right", x=1,
                                     bgcolor="rgba(0,0,0,0)"),
                         hovermode="x unified")

    # Violin
    fig_v = go.Figure()
    for zone in ["europe", "tunisie", "mediterranee"]:
        zsub = sub[sub.zone == zone]
        fig_v.add_trace(go.Violin(y=zsub[var], name=ZONE_LABELS[zone],
                                  box_visible=True, meanline_visible=True,
                                  fillcolor=ZONE_COLORS[zone],
                                  line_color=ZONE_COLORS[zone],
                                  opacity=0.7))
    fig_v.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                        xaxis_title="", yaxis_title=label,
                        margin=dict(l=20, r=20, t=20, b=40), height=400,
                        showlegend=False)

    # Cycle saisonnier (mois calendaire) — cache pre-calcule
    fig_s = go.Figure()
    for zone in ["europe", "tunisie", "mediterranee"]:
        m = (SEASONAL_CYCLE[SEASONAL_CYCLE.zone == zone]
             .set_index("month")[var])
        fig_s.add_trace(go.Scatter(x=m.index, y=m.values,
                                   mode="lines+markers",
                                   name=ZONE_LABELS[zone],
                                   line=dict(color=ZONE_COLORS[zone],
                                             width=3),
                                   marker=dict(size=8)))
    fig_s.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                        xaxis=dict(tickmode="array",
                                   tickvals=list(range(1, 13)),
                                   ticktext=["Jan","Fév","Mar","Avr","Mai",
                                             "Juin","Juil","Aoû","Sep",
                                             "Oct","Nov","Déc"]),
                        yaxis_title=label,
                        margin=dict(l=20, r=20, t=20, b=40), height=380,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1,
                                    bgcolor="rgba(0,0,0,0)"),
                        hovermode="x unified")

    range_label = (f"Plage d'années  ·  {year_range[0]} → {year_range[1]}  "
                   f"({year_range[1]-year_range[0]+1} ans)")
    for _f in (fig_ts, fig_v, fig_s):
        apply_theme(_f)
    return stats_table, fig_ts, fig_v, fig_s, range_label


# =============================================================================
# Onglet 3 — Anomalies
# =============================================================================

@callback(
    [Output("anom-stats", "children"), Output("anom-bars", "figure"),
     Output("anom-decadal", "figure"), Output("anom-distrib", "figure")],
    [Input("anom-var", "value"), Input("anom-zone", "value")],
)
def cb_anomalies(var, zone):
    sub = df[df.zone == zone].copy()
    label = VAR_LABELS[var]

    annual = ANNUAL[ANNUAL.zone == zone].set_index("year")[var]
    ref = annual.loc[REF_PERIOD[0]:REF_PERIOD[1]].mean()
    anom = annual - ref

    # KPIs anomalies
    last10 = anom.loc[2015:2024].mean()
    first10 = anom.loc[1980:1989].mean()
    n_above = int((anom > 0).sum())
    n_record = int((annual > annual.loc[REF_PERIOD[0]:REF_PERIOD[1]].max()).sum())

    kpis = [
        kpi_card("Référence WMO", f"{ref:.2f}",
                 f"moyenne 1981-2010 · {ZONE_LABELS[zone]}",
                 COLORS["primary"], "📏"),
        kpi_card("Anomalie 2015-2024", f"{last10:+.2f}",
                 f"vs ref. (1ère décennie : {first10:+.2f})",
                 COLORS["danger"] if last10 > 0 else COLORS["primary"], "🌡️"),
        kpi_card("Années > normale",
                 f"{n_above}/{len(anom)}",
                 f"{100*n_above/len(anom):.0f}% au-dessus",
                 COLORS["accent"], "📊"),
        kpi_card("Années record",
                 f"{n_record}",
                 "années dépassant le max 1981-2010",
                 COLORS["danger"], "⚠️"),
    ]

    # Bar chart anomalies
    colors = ["#DC2626" if v > 0 else "#3B82F6" for v in anom.values]
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(x=anom.index, y=anom.values,
                              marker_color=colors,
                              name="Anomalie",
                              hovertemplate="<b>%{x}</b><br>%{y:+.2f}<extra></extra>"))
    fig_bars.add_hline(y=0, line=dict(color=PLOTLY_AXIS, width=1))
    # Encadre la periode de reference
    fig_bars.add_vrect(x0=REF_PERIOD[0]-0.5, x1=REF_PERIOD[1]+0.5,
                       fillcolor="gray", opacity=0.10,
                       annotation_text="Référence 1981-2010",
                       annotation_position="top left")
    fig_bars.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                           xaxis_title="Année",
                           yaxis_title=f"Anomalie {label}",
                           margin=dict(l=20, r=20, t=20, b=40), height=400,
                           showlegend=False)

    # Decadal comparison
    sub["decade"] = (sub["year"] // 10) * 10
    decadal = sub.groupby("decade")[var].mean().reset_index()
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Bar(x=[f"{d}s" for d in decadal["decade"]],
                             y=decadal[var],
                             marker_color=COLORS["primary"],
                             text=[f"{v:.2f}" for v in decadal[var]],
                             textposition="outside",
                             hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>"))
    fig_dec.add_hline(y=ref, line=dict(color=COLORS["danger"], dash="dash"),
                      annotation_text=f"Réf. 1981-2010 = {ref:.2f}",
                      annotation_position="top left")
    fig_dec.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                          xaxis_title="Décennie", yaxis_title=label,
                          margin=dict(l=20, r=20, t=20, b=40), height=380,
                          showlegend=False)

    # Distribution par decennie
    fig_dist = go.Figure()
    decades = sorted(sub["decade"].unique())
    palette = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
    for i, d in enumerate(decades):
        dsub = sub[sub.decade == d]
        fig_dist.add_trace(go.Violin(y=dsub[var], name=f"{d}s",
                                     box_visible=True,
                                     meanline_visible=True,
                                     fillcolor=palette[i % len(palette)],
                                     line_color=palette[i % len(palette)],
                                     opacity=0.6))
    fig_dist.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                           xaxis_title="Décennie", yaxis_title=label,
                           margin=dict(l=20, r=20, t=20, b=40), height=380,
                           showlegend=False)

    for _f in (fig_bars, fig_dec, fig_dist):
        apply_theme(_f)
    return kpis, fig_bars, fig_dec, fig_dist


# =============================================================================
# Onglet 4 — Carte animee
# =============================================================================

@callback(
    [Output("map-anim", "figure"), Output("map-diff", "figure"),
     Output("map-mean", "figure")],
    [Input("map-var", "value")],
)
def cb_map(var):
    label = VAR_LABELS[var]
    cs = "RdYlBu_r" if var in ["t2m", "d2m"] else \
         ("Blues" if var == "tp" else "YlGnBu")

    if df_sp is None:
        empty = go.Figure()
        empty.add_annotation(text="Données spatiales manquantes",
                             showarrow=False)
        return empty, empty, empty

    # Carte animee : 1 frame par annee (echantillonne tous les 5 ans + 2024)
    sample_years = sorted(set(list(range(1980, 2025, 5)) + [2024]))
    sp = df_sp[df_sp["year"].isin(sample_years)]
    vmin, vmax = float(sp[var].quantile(0.02)), float(sp[var].quantile(0.98))

    frames = []
    for y in sample_years:
        g = (df_sp[df_sp.year == y]
             .pivot(index="latitude", columns="longitude", values=var)
             .sort_index(ascending=False))
        frames.append(go.Frame(
            data=[go.Heatmap(z=g.values,
                             x=g.columns.values.astype(float),
                             y=g.index.values.astype(float),
                             colorscale=cs, zmin=vmin, zmax=vmax,
                             colorbar=dict(title=label))],
            name=str(y), layout=go.Layout(title=f"Année {y}")))

    g0 = (df_sp[df_sp.year == sample_years[0]]
          .pivot(index="latitude", columns="longitude", values=var)
          .sort_index(ascending=False))
    fig_anim = go.Figure(
        data=[go.Heatmap(z=g0.values,
                         x=g0.columns.values.astype(float),
                         y=g0.index.values.astype(float),
                         colorscale=cs, zmin=vmin, zmax=vmax,
                         colorbar=dict(title=label))],
        frames=frames)
    fig_anim.update_layout(
        template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
        title=f"Année {sample_years[0]}",
        xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=40, t=60, b=80), height=520,
        updatemenus=[dict(type="buttons", showactive=False,
                          x=0.05, y=-0.15,
                          xanchor="left",
                          buttons=[
                              dict(label="▶️  Animer",
                                   method="animate",
                                   args=[None, {"frame": {"duration": 600},
                                                "fromcurrent": True}]),
                              dict(label="⏸  Pause",
                                   method="animate",
                                   args=[[None],
                                         {"frame": {"duration": 0},
                                          "mode": "immediate"}]),
                          ])],
        sliders=[dict(active=0, x=0.15, len=0.80, y=-0.10,
                      currentvalue=dict(prefix="Année : ",
                                        font=dict(size=14)),
                      steps=[dict(method="animate",
                                  args=[[str(y)],
                                        {"frame": {"duration": 0,
                                                   "redraw": True},
                                         "mode": "immediate"}],
                                  label=str(y)) for y in sample_years])])

    # Difference 2015-2024 vs 1980-1989
    early = (df_sp[df_sp.year.between(1980, 1989)]
             .groupby(["latitude", "longitude"])[var].mean()
             .reset_index()
             .pivot(index="latitude", columns="longitude", values=var)
             .sort_index(ascending=False))
    late = (df_sp[df_sp.year.between(2015, 2024)]
            .groupby(["latitude", "longitude"])[var].mean()
            .reset_index()
            .pivot(index="latitude", columns="longitude", values=var)
            .sort_index(ascending=False))
    diff = late - early
    abs_max = float(max(abs(diff.values.min()), abs(diff.values.max())))
    fig_diff = go.Figure(go.Heatmap(
        z=diff.values,
        x=diff.columns.values.astype(float),
        y=diff.index.values.astype(float),
        colorscale="RdBu_r",
        zmid=0, zmin=-abs_max, zmax=abs_max,
        colorbar=dict(title=f"Δ {label}")))
    fig_diff.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                           xaxis_title="Longitude (°E)",
                           yaxis_title="Latitude (°N)",
                           margin=dict(l=20, r=20, t=20, b=40), height=400)

    # Climatologie 1980-2024
    mean_grid = (df_sp.groupby(["latitude", "longitude"])[var].mean()
                 .reset_index()
                 .pivot(index="latitude", columns="longitude", values=var)
                 .sort_index(ascending=False))
    fig_mean = go.Figure(go.Heatmap(
        z=mean_grid.values,
        x=mean_grid.columns.values.astype(float),
        y=mean_grid.index.values.astype(float),
        colorscale=cs,
        colorbar=dict(title=label)))
    fig_mean.update_layout(template="plotly_dark", paper_bgcolor=PLOTLY_DARK_BG, plot_bgcolor=PLOTLY_DARK_BG, font=dict(family=PLOTLY_FONT, color=PLOTLY_AXIS, size=11), xaxis_gridcolor=PLOTLY_GRID, xaxis_zeroline=False, yaxis_gridcolor=PLOTLY_GRID, yaxis_zeroline=False,
                           xaxis_title="Longitude (°E)",
                           yaxis_title="Latitude (°N)",
                           margin=dict(l=20, r=20, t=20, b=40), height=400)

    for _f in (fig_anim, fig_diff, fig_mean):
        apply_theme(_f)
    return fig_anim, fig_diff, fig_mean


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dashboard ERA5 AVANCE  —  http://127.0.0.1:8050")
    print("Onglets : Apercu  ·  Comparaison  ·  Anomalies  ·  Carte animee")
    print("Ctrl+C pour arreter.")
    print("=" * 60)
    app.run(debug=False, port=8050)
