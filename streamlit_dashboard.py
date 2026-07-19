"""
====================================================================================
 🏨 StayPredict — AI-Powered Hotel Revenue & Cancellation Management Dashboard
====================================================================================
Bu Streamlit tətbiqi StayPredict.ipynb notebook-unda qurulan tam iş axınını
(data cleaning → feature engineering → XGBoost modeli → SHAP izahlılığı →
revenue analytics → what-if simulator → tövsiyə mühərriki → overbooking
simulyasiyası) interaktiv, professional bir dashboard-a çevirir.

İşə salmaq üçün:
    streamlit run streamlit_dashboard.py

Tələb olunan fayl:
    hotel_bookings_cleaned.csv — bu skriptlə eyni qovluqda olmalı, ya da tətbiq
    daxilində yükləmə paneli vasitəsilə yüklənməlidir.
====================================================================================
"""

import os
import warnings

import groq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import xgboost as xgb
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ====================================================================================
# 0. SƏHİFƏ TƏNZİMLƏMƏLƏRİ VƏ STİL
# ====================================================================================
st.set_page_config(
    page_title="StayPredict | Hotel Revenue & Cancellation AI",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2563eb"
DANGER = "#e74c3c"
WARNING = "#f39c12"
SUCCESS = "#2ecc71"
MUTED = "#64748b"

CUSTOM_CSS = f"""
<style>
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; }}
    div[data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetricLabel"] {{ font-weight: 600; color: {MUTED}; }}
    h1, h2, h3 {{ font-weight: 700; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    .risk-high {{ color: {DANGER}; font-weight: 700; }}
    .risk-medium {{ color: {WARNING}; font-weight: 700; }}
    .risk-low {{ color: {SUCCESS}; font-weight: 700; }}
    footer {{ visibility: hidden; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SEASON_MAP = {
    "December": "Winter", "January": "Winter", "February": "Winter",
    "March": "Spring", "April": "Spring", "May": "Spring",
    "June": "Summer", "July": "Summer", "August": "Summer",
    "September": "Autumn", "October": "Autumn", "November": "Autumn",
}
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
ID_LIKE_COLS = ["agent", "company"]
DEFAULT_DATA_PATH = "hotel_bookings.csv"


# ====================================================================================
# 1. DATA YÜKLƏMƏ VƏ TƏMİZLƏMƏ (cache olunur)
# ====================================================================================
@st.cache_data(show_spinner="Data yüklənir və təmizlənir...")
def load_and_clean_data(file_bytes_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_bytes_or_path)

    # --- Tip düzəlişləri ---
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
    for col in ["children", "agent", "company"]:
        df[col] = df[col].fillna(0).astype(int)

    # --- Məntiqsiz sətirləri təmizləmə: qonaqsız rezervasiyalar ---
    guests_zero = (df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0)
    df = df[~guests_zero].reset_index(drop=True)

    # --- Feature Engineering (notebook ilə eyni məntiq) ---
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["revenue"] = df["adr"] * df["total_nights"]
    df["season"] = df["arrival_date_month"].astype(str).map(SEASON_MAP)
    df["previous_total"] = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["is_domestic"] = (df["country"] == "PRT").astype(int)
    df["room_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

    # --- total_nights = 0 olan sətirləri təmizləmə ---
    df = df[df["total_nights"] > 0].reset_index(drop=True)

    # --- ADR-də mənfi/ekstrem outlier-ləri sadə şəkildə kəsmə (dashboard stabilliyi üçün) ---
    df = df[(df["adr"] >= 0) & (df["adr"] < df["adr"].quantile(0.999))].reset_index(drop=True)

    return df


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor, numeric_features, categorical_features


# ====================================================================================
# 2. MODEL TRAINING (cache olunur — tətbiq açıldıqda yalnız 1 dəfə işləyir)
# ====================================================================================
@st.cache_resource(show_spinner="Modellər öyrədilir (Logistic Regression, Random Forest, XGBoost)...")
def train_models(df: pd.DataFrame):
    df_model = df.drop(columns=LEAKAGE_COLS + ID_LIKE_COLS)
    X = df_model.drop(columns=["is_canceled"])
    y = df_model["is_canceled"]

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, n_jobs=-1)),
        ]),
    }

    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "pipeline": pipe,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            },
        }

    best_name = max(results, key=lambda n: results[n]["metrics"]["ROC-AUC"])
    champion = results[best_name]["pipeline"]

    # --- Optimal threshold (ən yüksək F1-ə görə) ---
    y_proba_champ = results[best_name]["y_proba"]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_champ)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(f1_scores) > 1 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    return {
        "results": results,
        "best_name": best_name,
        "champion": champion,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_threshold": best_threshold,
        "precisions": precisions, "recalls": recalls, "thresholds": thresholds,
    }


@st.cache_resource(show_spinner="SHAP izahlılıq dəyərləri hesablanır...")
def compute_shap(_champion_pipeline, X_test: pd.DataFrame, sample_size: int = 400):
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    preprocessor = _champion_pipeline.named_steps["preprocessor"]
    classifier = _champion_pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    # Ağac əsaslı modellər (XGBoost, Random Forest) üçün sürətli TreeExplainer,
    # digər modellər (məs. Logistic Regression) üçün model-aqnostik Explainer istifadə olunur.
    tree_based = isinstance(classifier, (xgb.XGBClassifier, RandomForestClassifier))

    if tree_based:
        explainer = shap.TreeExplainer(classifier)
        raw_values = explainer.shap_values(X_transformed)
        if isinstance(raw_values, list):  # bəzi versiyalarda [class0, class1] siyahısı qaytarılır
            raw_values = raw_values[1]
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:  # (n_samples, n_features, n_classes)
            raw_values = raw_values[:, :, 1]
    else:
        background = shap.sample(X_transformed, min(100, X_transformed.shape[0]), random_state=42)
        explainer = shap.Explainer(classifier.predict_proba, background)
        explanation = explainer(X_transformed)
        raw_values = explanation.values
        if raw_values.ndim == 3:  # (n_samples, n_features, n_classes)
            raw_values = raw_values[:, :, 1]

    return {
        "sample": sample.reset_index(drop=True),
        "X_transformed": X_transformed,
        "feature_names": feature_names,
        "explainer": explainer,
        "shap_values": raw_values,
    }


def get_groq_client():
    api_key = st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return groq.Client(api_key=api_key)
    except Exception:
        return None


def explain_risk_with_ai(client, row_shap, feature_names, probability, model="llama-3.3-70b-versatile"):
    """Notebook-dakı `explain_risk_with_ai` funksiyasının dashboard versiyası:
    seçilmiş rezervasiya üçün SHAP dəyərlərini götürür və menecer dilində izah alır."""
    top_idx = np.argsort(np.abs(row_shap))[::-1][:5]
    top_factors = []
    for idx in top_idx:
        direction = "riski ARTIRIR" if row_shap[idx] > 0 else "riski AZALDIR"
        top_factors.append(f"- {feature_names[idx]} → {direction} (SHAP təsiri: {row_shap[idx]:+.3f})")
    factors_text = "\n".join(top_factors)

    prompt = f"""Sən bir otel gəlir menecerinə kömək edən AI köməkçisisən.
Machine Learning modeli bu rezervasiya üçün ləğv ehtimalını {probability*100:.1f}% hesablayıb.
Bu ehtimala ən çox təsir edən amillər (SHAP analizindən):
{factors_text}

Tapşırıq: 3-4 cümlə ilə, SADƏ və PRAKTİK dildə (texniki terminlər olmadan) bu rezervasiyanın
niyə riskli (və ya risksiz) olduğunu izah et. Sonda 1 konkret tövsiyə ver.
Azərbaycan dilində, professional tonda cavab ver."""

    response = client.chat.completions.create(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def build_dataset_summary(df: pd.DataFrame) -> str:
    return f"""
- Ümumi rezervasiya sayı: {df.shape[0]:,}
- Ləğv nisbəti: {df['is_canceled'].mean()*100:.1f}%
- Otel növləri: {', '.join(df['hotel'].unique())}
- Orta lead time: {df['lead_time'].mean():.0f} gün
- Orta ADR: ${df['adr'].mean():.2f}
- Ən çox rast gəlinən ölkələr: {', '.join(df['country'].value_counts().head(5).index)}
- Market seqmentləri: {', '.join(df['market_segment'].unique())}
- Depozit tipləri: {', '.join(df['deposit_type'].unique())}
- Ümumi potensial gəlir: ${df['revenue'].sum():,.0f}
- Ləğvlər səbəbiylə itki: ${df.loc[df['is_canceled']==1, 'revenue'].sum():,.0f}
""".strip()


def ask_dataset_question(client, question: str, dataset_summary: str, model="llama-3.3-70b-versatile") -> str:
    prompt = f"""Sən otel rezervasiya datasetini analiz edən AI köməkçisisən.
Dataset xülasəsi:
{dataset_summary}

İstifadəçinin sualı: "{question}"

Tapşırıq: Yuxarıdakı xülasəyə əsaslanaraq sualı QISA (maksimum 3 cümlə), dəqiq və
Azərbaycan dilində cavabla. Əgər sual xülasədə olmayan məlumat tələb edirsə, bunu qeyd et."""

    response = client.chat.completions.create(
        model=model, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def recommend_action(risk_probability: float) -> str:
    if risk_probability > 0.80:
        return "🔴 Depozit Tələb Et — Yüksək Risk"
    elif risk_probability >= 0.50:
        return "🟡 Xatırlatma Göndər — Orta Risk"
    else:
        return "🟢 Tədbir Lazım Deyil — Aşağı Risk"


def risk_css_class(p: float) -> str:
    if p > 0.80:
        return "risk-high"
    elif p >= 0.50:
        return "risk-medium"
    return "risk-low"


def simulate_overbooking(overbooking_pct, capacity, cancel_probs, walk_cost, empty_cost, n_sim=1500, seed=42):
    rng = np.random.default_rng(seed)
    accepted_bookings = int(capacity * (1 + overbooking_pct))
    total_costs = np.empty(n_sim)

    for i in range(n_sim):
        sampled_probs = rng.choice(cancel_probs, size=accepted_bookings, replace=True)
        cancellations = rng.binomial(1, sampled_probs)
        arrivals = accepted_bookings - cancellations.sum()
        if arrivals > capacity:
            cost = (arrivals - capacity) * walk_cost
        else:
            cost = (capacity - arrivals) * empty_cost
        total_costs[i] = cost

    return float(total_costs.mean()), float(total_costs.std())


# ====================================================================================
# 3. VERİLƏNLƏRİ YÜKLƏMƏ AXINI
# ====================================================================================
def get_dataframe() -> pd.DataFrame | None:
    st.sidebar.markdown("### 🤖 AI Assistant (Groq)")
    st.session_state["groq_api_key"] = st.sidebar.text_input(
        "Groq API Key", type="password",
        value=st.session_state.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")),
        help="AI Risk Assistant və Dataset Q&A funksiyaları üçün lazımdır.",
    )

    st.sidebar.markdown("### 📂 Data Mənbəyi")
    uploaded = st.sidebar.file_uploader("hotel_bookings.csv yüklə", type=["csv"])

    if uploaded is not None:
        return load_and_clean_data(uploaded)

    if os.path.exists(DEFAULT_DATA_PATH):
        st.sidebar.caption(f"✅ Lokal fayl tapıldı: `{DEFAULT_DATA_PATH}`")
        return load_and_clean_data(DEFAULT_DATA_PATH)

    return None


# ====================================================================================
# 4. TƏTBİQ BAŞLIĞI
# ====================================================================================
st.title("🏨 StayPredict")
st.caption("AI-Powered Hotel Revenue & Cancellation Management System")

df = get_dataframe()

if df is None:
    st.warning(
        "⚠️ Data tapılmadı. Davam etmək üçün soldakı paneldən **hotel_bookings_cleaned.csv** "
        "faylını yükləyin, ya da faylı bu skriptlə eyni qovluğa qoyun."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Naviqasiya")
page = st.sidebar.radio(
    "Bölmə seçin",
    [
        "🏠 Ümumi Baxış",
        "🔍 Kəşfiyyat Analizi (EDA)",
        "🤖 Model Performansı",
        "🧠 İzahlılıq (SHAP)",
        "💰 Gəlir Analitikası",
        "🎛️ What-If Simulyator",
        "📋 Tövsiyə Sistemi",
        "🎲 Overbooking Simulyasiyası",
    ],
    label_visibility="collapsed",
)

with st.spinner("Modellər hazırlanır..."):
    training = train_models(df)

results = training["results"]
best_name = training["best_name"]
champion = training["champion"]
X_test = training["X_test"]
y_test = training["y_test"]
y_proba_champ = results[best_name]["y_proba"]
best_threshold = training["best_threshold"]

# ====================================================================================
# SƏHİFƏ 1 — ÜMUMİ BAXIŞ
# ====================================================================================
if page == "🏠 Ümumi Baxış":
    total_revenue = df["revenue"].sum()
    lost_revenue = df.loc[df["is_canceled"] == 1, "revenue"].sum()
    realized_revenue = df.loc[df["is_canceled"] == 0, "revenue"].sum()
    cancellation_rate = df["is_canceled"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ümumi Rezervasiya", f"{df.shape[0]:,}")
    c2.metric("Ləğv Nisbəti", f"{cancellation_rate:.1f}%")
    c3.metric("Ümumi Potensial Gəlir", f"${total_revenue:,.0f}")
    c4.metric("Ləğvlər səbəbiylə İtki", f"${lost_revenue:,.0f}",
              delta=f"-{(lost_revenue/total_revenue)*100:.1f}% gəlirdən", delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("🏆 Qalib Model")
        st.markdown(
            f"**Model:** `{best_name}`  \n"
            f"**ROC-AUC:** `{results[best_name]['metrics']['ROC-AUC']:.4f}`  \n"
            f"**Recall:** `{results[best_name]['metrics']['Recall']:.4f}`  \n"
            f"**Optimal Threshold:** `{best_threshold:.2f}`"
        )
        metrics_df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
        st.dataframe(metrics_df.style.format("{:.4f}").highlight_max(axis=0, color="#d1fae5"),
                     width="stretch")

    with col2:
        st.subheader("📉 Ləğv Paylanması")
        cancel_counts = df["is_canceled"].map({0: "Ləğv Olunmayıb", 1: "Ləğv Olunub"}).value_counts()
        fig = px.pie(values=cancel_counts.values, names=cancel_counts.index,
                     color=cancel_counts.index,
                     color_discrete_map={"Ləğv Olunmayıb": SUCCESS, "Ləğv Olunub": DANGER}, hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("🗓️ Rezervasiya Trendi (Ay üzrə)")
    month_order = ["January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"]
    trend = df.groupby("arrival_date_month").size().reindex(month_order)
    fig = px.line(x=trend.index, y=trend.values, markers=True,
                  labels={"x": "Ay", "y": "Rezervasiya sayı"})
    fig.update_traces(line_color=PRIMARY)
    st.plotly_chart(fig, width="stretch")


# ====================================================================================
# SƏHİFƏ 2 — EDA
# ====================================================================================
elif page == "🔍 Kəşfiyyat Analizi (EDA)":
    tab1, tab2, tab3 = st.tabs(["📊 Kateqoriyalar üzrə", "📈 Ədədi Dəyişənlər", "🔗 Korrelyasiya"])

    with tab1:

        st.markdown("### 🌍 Ölkələr üzrə Analiz")
        st.caption("Rezervasiyaların hansı ölkələrdən gəldiyini, ləğv nisbətini və gəlir töhfəsini araşdırın.")

        # ---------------------------------------
        # Country -> ISO3 Conversion
        # ---------------------------------------
        def country_to_iso3(country):
            if pd.isna(country):
                return None

            if PYCOUNTRY_AVAILABLE:
                try:
                    return pycountry.countries.lookup(country).alpha_3
                except Exception:
                    pass

            try:
                special = {
                    "PRT": "PRT",
                    "GBR": "GBR",
                    "USA": "USA",
                    "FRA": "FRA",
                    "ESP": "ESP",
                    "DEU": "DEU",
                    "ITA": "ITA",
                    "BRA": "BRA",
                    "IRL": "IRL",
                    "BEL": "BEL",
                    "NLD": "NLD",
                    "CHE": "CHE",
                    "AUT": "AUT",
                    "CHN": "CHN",
                    "RUS": "RUS",
                    "CAN": "CAN",
                    "AUS": "AUS"
                }

                if country in special:
                    return special[country]

                # Dataset ölkə kodları artıq çox vaxt ISO3 formatındadır (məs. "PRT")
                if isinstance(country, str) and len(country) == 3 and country.isalpha():
                    return country.upper()

                return None
            except Exception:
                return None

        country_df = df.copy()
        country_df["country_iso3"] = country_df["country"].apply(country_to_iso3)

        country_df = country_df.dropna(subset=["country_iso3"])

        country_stats = (
            country_df.groupby("country_iso3")
            .agg(
                rezervasiya=("is_canceled", "count"),
                legv_nisbeti=("is_canceled", "mean"),
                gelir=("revenue", "sum")
            )
            .reset_index()
        )

        country_stats["legv_nisbeti"] *= 100

        map_metric = st.radio(
            "Xəritə üçün göstərici seçin:",
            ["Rezervasiya sayı", "Ləğv nisbəti (%)", "Gəlir ($)"],
            horizontal=True
        )

        metric_col = {
            "Rezervasiya sayı": "rezervasiya",
            "Ləğv nisbəti (%)": "legv_nisbeti",
            "Gəlir ($)": "gelir"
        }[map_metric]

        fig = px.choropleth(
            country_stats,
            locations="country_iso3",
            color=metric_col,
            hover_name="country_iso3",
            color_continuous_scale="Blues" if map_metric != "Ləğv nisbəti (%)" else "OrRd",
            projection="natural earth",
            labels={metric_col: map_metric}
        )

        fig.update_geos(showframe=False, showcoastlines=True)

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        col1, col2 = st.columns(2)
        with col1:
            rate = df.groupby("hotel")["is_canceled"].mean().mul(100).reset_index()
            fig = px.bar(rate, x="hotel", y="is_canceled", color="hotel",
                         labels={"is_canceled": "Ləğv nisbəti (%)", "hotel": "Otel növü"},
                         title="Otel Növünə görə Ləğv Nisbəti")
            st.plotly_chart(fig, width="stretch")

            rate = df.groupby("deposit_type")["is_canceled"].mean().mul(100).reset_index()
            fig = px.bar(rate, x="deposit_type", y="is_canceled", color="deposit_type",
                         labels={"is_canceled": "Ləğv nisbəti (%)", "deposit_type": "Depozit tipi"},
                         title="Depozit Tipinə görə Ləğv Nisbəti")
            st.plotly_chart(fig, width="stretch")

        with col2:
            rate = df.groupby("market_segment")["is_canceled"].mean().mul(100).sort_values().reset_index()
            fig = px.bar(rate, x="is_canceled", y="market_segment", orientation="h",
                         labels={"is_canceled": "Ləğv nisbəti (%)", "market_segment": "Market seqmenti"},
                         title="Market Seqmentinə görə Ləğv Nisbəti")
            st.plotly_chart(fig, width="stretch")

            top_countries = df["country"].value_counts().head(10).reset_index()
            top_countries.columns = ["country", "count"]
            fig = px.bar(top_countries, x="count", y="country", orientation="h",
                         title="Top 10 Ölkə üzrə Rezervasiya Sayı")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            lead_df = df[["lead_time"]].copy()
            lead_df["Ləğv olunub?"] = df["is_canceled"].map({0: "Xeyr", 1: "Bəli"})
            fig = px.histogram(lead_df, x="lead_time", color="Ləğv olunub?",
                                nbins=60, barmode="overlay", opacity=0.65,
                                title="Lead Time Paylanması (Ləğv statusuna görə)")
            st.plotly_chart(fig, width="stretch")
        with col2:
            adr_capped = df[df["adr"] < df["adr"].quantile(0.99)].copy()
            adr_capped["Ləğv olunub?"] = adr_capped["is_canceled"].map({0: "Xeyr", 1: "Bəli"})
            fig = px.histogram(adr_capped, x="adr", color="Ləğv olunub?",
                                nbins=60, barmode="overlay", opacity=0.65,
                                title="ADR (Gündəlik Orta Qiymət) Paylanması")
            st.plotly_chart(fig, width="stretch")

    with tab3:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Ədədi Dəyişənlər üzrə Korrelyasiya Xəritəsi", aspect="auto")
        st.plotly_chart(fig, width="stretch")


# ====================================================================================
# SƏHİFƏ 3 — MODEL PERFORMANSI
# ====================================================================================
elif page == "🤖 Model Performansı":
    st.subheader("Model Müqayisəsi")
    metrics_df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Model"})
    metrics_melt = metrics_df.melt(id_vars="Model", var_name="Metrika", value_name="Dəyər")
    fig = px.bar(metrics_melt, x="Metrika", y="Dəyər", color="Model", barmode="group",
                 title="Bütün Modellərin Metrika Müqayisəsi")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(metrics_df.set_index("Model").style.format("{:.4f}").highlight_max(axis=0, color="#d1fae5"),
                 width="stretch")

    st.markdown("---")
    selected_model = st.selectbox("Ətraflı baxılacaq model:", list(results.keys()),
                                   index=list(results.keys()).index(best_name))
    sel = results[selected_model]

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, sel["y_pred"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Proqnoz", y="Real", color="Say"),
                         x=["Ləğv olunmayıb", "Ləğv olunub"], y=["Ləğv olunmayıb", "Ləğv olunub"],
                         title=f"{selected_model} — Confusion Matrix")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fpr, tpr, _ = roc_curve(y_test, sel["y_proba"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=selected_model, line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", dash="dash")))
        fig.update_layout(title=f"{selected_model} — ROC Curve (AUC={sel['metrics']['ROC-AUC']:.3f})",
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width="stretch")

    if selected_model == best_name:
        st.markdown("---")
        st.subheader("🎯 Threshold Optimizasiyası (Qalib Model üçün)")
        precisions, recalls = training["precisions"], training["recalls"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines", line=dict(color=PRIMARY, width=3),
                                  name="Precision-Recall əyrisi"))
        st.plotly_chart(fig, width="stretch")

        threshold_slider = st.slider("Qərar Threshold-u", 0.05, 0.95, float(round(best_threshold, 2)), 0.01)
        y_pred_at_t = (y_proba_champ >= threshold_slider).astype(int)
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("Precision", f"{precision_score(y_test, y_pred_at_t):.3f}")
        tcol2.metric("Recall", f"{recall_score(y_test, y_pred_at_t):.3f}")
        tcol3.metric("F1", f"{f1_score(y_test, y_pred_at_t):.3f}")


# ====================================================================================
# SƏHİFƏ 4 — SHAP İZAHLILIQ
# ====================================================================================
elif page == "🧠 İzahlılıq (SHAP)":
    st.subheader(f"Explainable AI — {best_name} Modelinin Qərarları")
    shap_data = compute_shap(champion, X_test, sample_size=400)

    tab1, tab2 = st.tabs(["🌍 Qlobal Əhəmiyyət", "🔎 Fərdi Rezervasiya İzahı"])

    with tab1:
        importance = np.abs(shap_data["shap_values"]).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature": shap_data["feature_names"], "importance": importance
        }).sort_values("importance", ascending=False).head(20)
        fig = px.bar(imp_df.sort_values("importance"), x="importance", y="feature", orientation="h",
                     title="SHAP — Ən Vacib 20 Dəyişən (orta |SHAP| dəyəri)", color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")
        st.caption("Bu qrafik modelin ləğv proqnozu verərkən ən çox hansı dəyişənlərə əsaslandığını göstərir.")

    with tab2:
        idx = st.slider("Rezervasiya nümunəsi seçin (test set-dən)", 0, len(shap_data["sample"]) - 1, 0)
        row_shap = shap_data["shap_values"][idx]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:10]

        proba = champion.predict_proba(shap_data["sample"].iloc[[idx]])[:, 1][0]
        st.markdown(f"**Ləğv ehtimalı:** <span class='{risk_css_class(proba)}'>{proba*100:.1f}%</span> — "
                    f"{recommend_action(proba)}", unsafe_allow_html=True)

        local_df = pd.DataFrame({
            "feature": [shap_data["feature_names"][i] for i in top_idx],
            "shap_value": [row_shap[i] for i in top_idx],
        }).sort_values("shap_value")
        local_df["direction"] = np.where(local_df["shap_value"] > 0, "Riski artırır", "Riski azaldır")
        fig = px.bar(local_df, x="shap_value", y="feature", orientation="h", color="direction",
                     color_discrete_map={"Riski artırır": DANGER, "Riski azaldır": SUCCESS},
                     title="Local Explanation — Bu Rezervasiya üçün Ən Təsirli 10 Amil")
        st.plotly_chart(fig, width="stretch")


# ====================================================================================
# SƏHİFƏ 5 — GƏLİR ANALİTİKASI
# ====================================================================================
elif page == "💰 Gəlir Analitikası":
    total_revenue = df["revenue"].sum()
    lost_revenue = df.loc[df["is_canceled"] == 1, "revenue"].sum()
    realized_revenue = df.loc[df["is_canceled"] == 0, "revenue"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Ümumi Potensial Gəlir", f"${total_revenue:,.0f}")
    c2.metric("Reallaşan Gəlir", f"${realized_revenue:,.0f}")
    c3.metric("Ləğvlər səbəbiylə İtki", f"${lost_revenue:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        rev_hotel = df.groupby("hotel").apply(
            lambda g: pd.Series({
                "Ümumi Gəlir": g["revenue"].sum(),
                "İtirilən Gəlir": g.loc[g["is_canceled"] == 1, "revenue"].sum(),
            })
        ).reset_index()
        fig = px.bar(rev_hotel.melt(id_vars="hotel", var_name="Tip", value_name="Gəlir"),
                     x="hotel", y="Gəlir", color="Tip", barmode="group",
                     color_discrete_map={"Ümumi Gəlir": PRIMARY, "İtirilən Gəlir": DANGER},
                     title="Otel Növünə görə Ümumi və İtirilən Gəlir")
        st.plotly_chart(fig, width="stretch")

        rev_segment = df.groupby("market_segment")["revenue"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(rev_segment, x="revenue", y="market_segment", orientation="h",
                     title="Market Seqmentinə görə Ümumi Gəlir", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, width="stretch")

    with col2:
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        rev_season = df.groupby("season")["revenue"].sum().reindex(season_order).reset_index()
        fig = px.bar(rev_season, x="season", y="revenue", title="Fəsilə görə Ümumi Gəlir",
                     color="season", color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig, width="stretch")

        # --- Recoverable Revenue: qalib model əsasında ---
        df_test_proba = X_test.copy()
        df_test_proba["cancel_probability"] = y_proba_champ
        df_test_proba["actual_canceled"] = y_test.values
        df_test_proba["revenue"] = df.loc[X_test.index, "revenue"].values

        high_risk_not_canceled = df_test_proba[
            (df_test_proba["cancel_probability"] >= 0.5) & (df_test_proba["actual_canceled"] == 0)
        ]
        recoverable_revenue = high_risk_not_canceled["revenue"].sum()

        st.markdown("#### 💡 Recoverable Revenue")
        st.metric("Qorunma Potensialı Olan Gəlir (test set)", f"${recoverable_revenue:,.0f}",
                   help="Model 'yüksək riskli' kimi qiymətləndirdiyi, amma faktiki olaraq ləğv "
                        "olunmamış rezervasiyaların gəliri. Erkən müdaxilə ilə qorunma şansı var.")
        st.caption(f"Bu, {len(high_risk_not_canceled)} rezervasiyaya aiddir.")


# ====================================================================================
# SƏHİFƏ 6 — WHAT-IF SIMULATOR
# ====================================================================================
elif page == "🎛️ What-If Simulyator":
    st.subheader("Fərdi Rezervasiya üçün Ləğv Riskini Simulyasiya Et")
    st.caption("Parametrləri dəyişərək modelin proqnozunun necə dəyişdiyini canlı izləyin.")

    numeric_features = training["numeric_features"]
    categorical_features = training["categorical_features"]
    base_row = X_test.iloc[[0]].copy()

    # --- Ən vacib dəyişənlər üçün istifadəçi input-ları ---
    col1, col2, col3 = st.columns(3)
    with col1:
        lead_time = st.slider("Lead Time (gün)", 0, 500, int(base_row["lead_time"].iloc[0]))
        adr = st.slider("ADR ($)", 0.0, 500.0, float(base_row["adr"].iloc[0]))
        total_nights = st.slider("Ümumi Gecə Sayı", 1, 30, int(base_row["total_nights"].iloc[0]))
    with col2:
        deposit_type = st.selectbox("Depozit Tipi", df["deposit_type"].unique(),
                                     index=list(df["deposit_type"].unique()).index(base_row["deposit_type"].iloc[0])
                                     if base_row["deposit_type"].iloc[0] in df["deposit_type"].unique() else 0)
        customer_type = st.selectbox("Müştəri Tipi", df["customer_type"].unique())
        market_segment = st.selectbox("Market Seqmenti", df["market_segment"].unique())
    with col3:
        previous_cancellations = st.slider("Əvvəlki Ləğv Sayı", 0, 10, int(base_row["previous_cancellations"].iloc[0]))
        is_domestic = st.selectbox("Yerli Müştəridir?", ["Xeyr", "Bəli"])
        room_mismatch = st.selectbox("Otaq Dəyişikliyi Olub?", ["Xeyr", "Bəli"])

    scenario = base_row.copy()
    scenario["lead_time"] = lead_time
    scenario["adr"] = adr
    scenario["total_nights"] = total_nights
    scenario["deposit_type"] = deposit_type
    scenario["customer_type"] = customer_type
    scenario["market_segment"] = market_segment
    scenario["previous_cancellations"] = previous_cancellations
    scenario["previous_total"] = previous_cancellations + scenario["previous_bookings_not_canceled"].iloc[0]
    scenario["is_domestic"] = 1 if is_domestic == "Bəli" else 0
    scenario["room_mismatch"] = 1 if room_mismatch == "Bəli" else 0
    scenario["revenue"] = adr * total_nights

    proba = champion.predict_proba(scenario)[:, 1][0]

    st.markdown("---")
    gcol1, gcol2 = st.columns([1, 1.4])
    with gcol1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Ləğv Ehtimalı (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PRIMARY},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": best_threshold * 100},
            },
        ))
        st.plotly_chart(fig, width="stretch")
    with gcol2:
        st.markdown("### 📋 Tövsiyə")
        rec = recommend_action(proba)
        st.markdown(f"<h3 class='{risk_css_class(proba)}'>{rec}</h3>", unsafe_allow_html=True)
        st.write(f"Bu ssenari üçün proqnozlaşdırılan gəlir: **${scenario['revenue'].iloc[0]:,.0f}**")
        st.caption(f"Qərar threshold-u: {best_threshold:.2f} (qalib model: {best_name})")

    # ---- 🤖 AI Risk Assistant (Groq) ----
    st.markdown("---")
    st.subheader("🤖 AI Risk Assistant (Groq)")
    groq_client = get_groq_client()

    if not groq_client:
        st.info("AI izahı üçün soldakı paneldə **Groq API Key** daxil edin.")
    else:
        if st.button("🧠 Bu ssenari üçün AI izahı al"):
            with st.spinner("AI izah hazırlanır..."):
                try:
                    shap_bundle = compute_shap(champion, X_test, sample_size=400)
                    preprocessor = champion.named_steps["preprocessor"]
                    classifier = champion.named_steps["classifier"]
                    scenario_transformed = preprocessor.transform(scenario)

                    if isinstance(classifier, (xgb.XGBClassifier, RandomForestClassifier)):
                        row_shap = shap_bundle["explainer"].shap_values(scenario_transformed)
                        if isinstance(row_shap, list):
                            row_shap = row_shap[1]
                        elif isinstance(row_shap, np.ndarray) and row_shap.ndim == 3:
                            row_shap = row_shap[:, :, 1]
                        row_shap = row_shap[0]
                    else:
                        exp = shap_bundle["explainer"](scenario_transformed)
                        vals = exp.values
                        row_shap = vals[0, :, 1] if vals.ndim == 3 else vals[0]

                    ai_text = explain_risk_with_ai(
                        groq_client, row_shap, shap_bundle["feature_names"], proba
                    )
                    st.success(ai_text)
                except Exception as e:
                    st.error(f"AI izahı alınarkən xəta baş verdi: {e}")

    # ---- 💬 Dataset Q&A (Groq) ----
    st.markdown("---")
    st.subheader("💬 Dataset Haqqında Sual Ver")
    question = st.text_input("Sualınızı yazın (məs: 'Hansı ölkədən ən çox rezervasiya var?')")
    if st.button("Cavabla") and question:
        if not groq_client:
            st.info("Sual vermək üçün soldakı paneldə **Groq API Key** daxil edin.")
        else:
            with st.spinner("Cavab hazırlanır..."):
                try:
                    summary = build_dataset_summary(df)
                    answer = ask_dataset_question(groq_client, question, summary)
                    st.success(answer)
                except Exception as e:
                    st.error(f"Cavab alınarkən xəta baş verdi: {e}")


# ====================================================================================
# SƏHİFƏ 7 — TÖVSİYƏ SİSTEMİ
# ====================================================================================
elif page == "📋 Tövsiyə Sistemi":
    st.subheader("Riskli Rezervasiyalar üçün Tövsiyə Mühərriki")

    recommendations = pd.DataFrame({"cancel_probability": y_proba_champ}, index=X_test.index)
    recommendations["recommendation"] = recommendations["cancel_probability"].apply(recommend_action)
    recommendations["revenue"] = df.loc[X_test.index, "revenue"].values
    recommendations["hotel"] = df.loc[X_test.index, "hotel"].values
    recommendations["lead_time"] = df.loc[X_test.index, "lead_time"].values
    recommendations["deposit_type"] = df.loc[X_test.index, "deposit_type"].values
    recommendations["actual_canceled"] = y_test.values

    rec_counts = recommendations["recommendation"].value_counts()
    c1, c2, c3 = st.columns(3)
    for col, key, color in zip(
        [c1, c2, c3],
        ["🔴 Depozit Tələb Et — Yüksək Risk", "🟡 Xatırlatma Göndər — Orta Risk", "🟢 Tədbir Lazım Deyil — Aşağı Risk"],
        [DANGER, WARNING, SUCCESS],
    ):
        col.metric(key.split("—")[0].strip(), int(rec_counts.get(key, 0)))

    fig = px.bar(rec_counts, x=rec_counts.values, y=rec_counts.index, orientation="h",
                 color=rec_counts.index,
                 color_discrete_map={
                     "🔴 Depozit Tələb Et — Yüksək Risk": DANGER,
                     "🟡 Xatırlatma Göndər — Orta Risk": WARNING,
                     "🟢 Tədbir Lazım Deyil — Aşağı Risk": SUCCESS,
                 },
                 title="Tövsiyə Paylanması (Test Set üzrə)", labels={"x": "Rezervasiya sayı", "y": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("🔝 Ən Riskli Rezervasiyalar")
    min_prob = st.slider("Minimum ləğv ehtimalı filtri", 0.0, 1.0, 0.5, 0.05)
    top_n = st.number_input("Göstərilən sətir sayı", 5, 100, 15)

    filtered = recommendations[recommendations["cancel_probability"] >= min_prob].sort_values(
        "cancel_probability", ascending=False
    ).head(int(top_n))

    st.dataframe(
        filtered[["hotel", "lead_time", "deposit_type", "revenue", "cancel_probability", "recommendation"]]
        .style.format({"cancel_probability": "{:.1%}", "revenue": "${:,.0f}"})
        .background_gradient(subset=["cancel_probability"], cmap="Reds"),
        width="stretch",
    )
    st.caption("`actual_canceled` sütunu göstərilmir — bu, real dünyada məlum olmayan gələcək məlumatdır; "
               "burada yalnız modelin tövsiyə keyfiyyətini qiymətləndirmək üçün arxa planda istifadə olunur.")


# ====================================================================================
# SƏHİFƏ 8 — OVERBOOKING SİMULYASİYASI
# ====================================================================================
elif page == "🎲 Overbooking Simulyasiyası":
    st.subheader("Monte Carlo Overbooking Simulyasiyası")
    st.caption("Model tərəfindən proqnozlaşdırılan ləğv ehtimalları fonduna əsaslanaraq, "
               "otaq tutumunu artıq satmağın (overbooking) optimal səviyyəsini tapır.")

    col1, col2, col3, col4 = st.columns(4)
    room_capacity = col1.number_input("Otaq Tutumu", 10, 1000, 100, 10)
    walk_multiplier = col2.slider("Walk Cost Əmsalı (× ADR)", 1.0, 6.0, 3.0, 0.5)
    max_overbooking = col3.slider("Maksimum Overbooking (%)", 5, 50, 30, 5)
    n_sim = col4.select_slider("Simulyasiya sayı", options=[200, 500, 1000, 1500, 2500], value=1000)

    avg_adr = df["adr"].mean()
    walk_cost = avg_adr * walk_multiplier
    empty_cost = avg_adr

    m1, m2, m3 = st.columns(3)
    m1.metric("Orta ADR", f"${avg_adr:.2f}")
    m2.metric("Walk Cost / qonaq", f"${walk_cost:.2f}")
    m3.metric("Boş Otaq İtkisi / otaq", f"${empty_cost:.2f}")

    if st.button("▶️ Simulyasiyanı işə sal", type="primary"):
        overbooking_levels = np.arange(0, max_overbooking / 100 + 0.001, 0.02)
        progress = st.progress(0, text="Simulyasiya işləyir...")
        rows = []
        for i, level in enumerate(overbooking_levels):
            mean_cost, std_cost = simulate_overbooking(
                level, room_capacity, y_proba_champ, walk_cost, empty_cost, n_sim=n_sim
            )
            rows.append({"overbooking_pct": level * 100, "avg_cost": mean_cost, "std_cost": std_cost})
            progress.progress((i + 1) / len(overbooking_levels), text="Simulyasiya işləyir...")
        progress.empty()

        sim_df = pd.DataFrame(rows)
        optimal = sim_df.loc[sim_df["avg_cost"].idxmin()]

        st.success(
            f"🏆 Optimal Overbooking Faizi: **{optimal['overbooking_pct']:.0f}%** — "
            f"gözlənilən orta gündəlik itki: **${optimal['avg_cost']:.2f}**"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_df["overbooking_pct"], y=sim_df["avg_cost"], mode="lines+markers",
            line=dict(color="#8e44ad", width=3), name="Gözlənilən itki",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([sim_df["overbooking_pct"], sim_df["overbooking_pct"][::-1]]),
            y=pd.concat([sim_df["avg_cost"] + sim_df["std_cost"], (sim_df["avg_cost"] - sim_df["std_cost"])[::-1]]),
            fill="toself", fillcolor="rgba(142,68,173,0.15)", line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_vline(x=optimal["overbooking_pct"], line_dash="dash", line_color=DANGER,
                      annotation_text=f"Optimal: {optimal['overbooking_pct']:.0f}%")
        fig.update_layout(
            title="Overbooking Faizinə görə Gözlənilən Gündəlik İtki",
            xaxis_title="Overbooking Faizi (%)", yaxis_title="Gözlənilən Orta İtki ($)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Sıfır overbooking-də itki əsasən boş otaqlardan, yüksək overbooking-də isə "
                   "'walk' etmə xərclərindən qaynaqlanır. Optimal nöqtə bu ikisinin balansıdır.")
    else:
        st.info("Parametrləri seçib **'Simulyasiyanı işə sal'** düyməsinə basın.")


# ====================================================================================
# ALT MƏTN
# ====================================================================================
st.sidebar.markdown("---")
st.sidebar.caption("StayPredict · AI-Powered Hotel Revenue & Cancellation Management System")
