# 🏨 StayPredict
### AI-Powered Hotel Revenue and Cancellation Management System

---

## Bu paketdə nə var?

| Fayl | Təsvir |
|---|---|
| `StayPredict.ipynb` | Əsas notebook — 20 mərhələnin hamısı + 🤖 AI Risk Assistant (bonus), ətraflı şərhlərlə |
| `streamlit_dashboard.py` | Bonus — Mərhələ 20 üçün interaktiv Streamlit dashboard (Plotly, dünya xəritəsi, What-If Simulator + AI Risk Assistant daxil) |
| `requirements.txt` | Layihə üçün lazımi bütün Python kitabxanaları |
| `.streamlit/config.toml` | Dashboard üçün açıq (light) tema tənzimləməsi |
| `README.md` | Bu fayl — quraşdırma təlimatı |

---

## Necə işə salmaq olar?

### 1. Google Colab-da (tövsiyə olunur)
1. `StayPredict.ipynb` faylını [Google Colab](https://colab.research.google.com)-a yükləyin.
2. `hotel_bookings.csv` faylını Colab-a yükləyin (sol paneldəki Files → Upload).
3. Hüceyrələri sırayla işə salın (`Runtime → Run all`).

### 2. Lokal Jupyter-də
```bash
pip install -r requirements.txt
jupyter notebook StayPredict.ipynb
```

### 3. Dashboard-u işə salmaq
`hotel_bookings.csv`, `streamlit_dashboard.py` və `.streamlit/config.toml` eyni qovluqda olmalıdır.
```bash
streamlit run streamlit_dashboard.py
```

---

## Notebook Strukturu

Notebook planı 20 mərhələni izləyir: Business Understanding → Data Loading → Cleaning → EDA → Feature Engineering → Feature Selection → Train/Test Split → Preprocessing Pipeline → Baseline Models → Model Comparison → Champion Model → Hyperparameter Tuning → Cross Validation → Threshold Optimization → SHAP → Revenue Analytics → What-If Simulator → Recommendation Engine → **Time Series Forecasting & Overbooking Simulation** → Dashboard.


### Diqqət tələb edən nöqtələr

1. **Mərhələ 19 (Time Series Forecasting)** `statsmodels` kitabxanasını tələb edir (`pip install statsmodels`) — Colab-da adətən artıq qurulu olur.
2. **🤖 AI Risk Assistant** (Mərhələ 15-in bonusu, həm notebook-da, həm dashboard-da) **Anthropic API açarı** tələb edir ([console.anthropic.com](https://console.anthropic.com)-dan pulsuz əldə edilə bilər). Açarsız işlətsəniz, sadəcə həmin bonus hissə işləməyəcək — qalan hər şey normal davam edir.

---

## 📊 Dataset

| Dataset | Mənbə | İstifadə |
|---|---|---|
| Hotel Booking Demand Dataset | Kaggle (Antonio, Almeida & Nunes, 2019) | Cancellation Prediction, Revenue Analytics, Time Series Forecasting, Overbooking Simulation |

### Live Demo

https://staypredict-z3bhfqsjwxrsbgenl2unrr.streamlit.app


