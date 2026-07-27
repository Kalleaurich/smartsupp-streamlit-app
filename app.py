"""SmartSupp — Streamlit-App für Supplement-Empfehlung & Fortschritts-Tracking."""

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="SmartSupp",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS für perfekten Kontrast im Light- und Dark-Mode
st.markdown(
    """
    <style>
    /* Wenn die App im Light Mode ist, erzwinge lesbare Textfarben und Hintergründe */
    [data-theme="light"] {
        background-color: #f8f9fa !important;
        color: #1f2937 !important;
    }
    [data-theme="light"] .stMarkdown, [data-theme="light"] p, [data-theme="light"] h1, [data-theme="light"] h2, [data-theme="light"] h3, [data-theme="light"] span {
        color: #1f2937 !important;
        font-weight: 500;
    }
    /* Sorge dafür, dass st.container(border=True) Kacheln im Light Mode weiß sind und einen grauen Rand haben */
    [data-theme="light"] [data-testid="stMetricContainer"], [data-theme="light"] .st-emotion-cache-12w0qpk {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        padding: 15px !important;
        border-radius: 8px !important;
    }
    /* Im Dark Mode lassen wir die Standardfarben komplett unangetastet */
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #262b36;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .ss-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
    .ss-subtitle { color: #9aa4b2; margin-top: 0; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Lädt Modelle und Metadaten. Gibt (clf, reg, meta, error) zurück."""
    try:
        clf = joblib.load("model_supplement_classifier.pkl")
        reg = joblib.load("model_progress_regressor.pkl")
        meta = joblib.load("smartsupp_model_metadata.pkl")
        return clf, reg, meta, None
    except FileNotFoundError as e:
        return None, None, None, f"Datei nicht gefunden: {e.filename}"
    except Exception as e:
        return None, None, None, f"Fehler beim Laden der Modelle: {e}"


# ---------------------------------------------------------------------------
# ML-Modellsteuerung — echtes Live-Training von SVM / Decision Tree / Random
# Forest auf dem rohen Trainingsdatensatz, exakt dieselben Features und
# derselbe Train/Test-Split (random_state=42, 20% Testdaten) wie im Notebook.
# Es werden ausschließlich echte, gerade eben trainierte Modelle verwendet —
# keine simulierten oder fest hinterlegten Genauigkeitswerte.
# ---------------------------------------------------------------------------
CLF_CATEGORICAL_FEATURES = [
    "Gender", "Age", "Height(cm)", "Fitness_Level", "Weekly_Training", "Training_Type", "Diet_Type"
]
CLF_NUMERIC_FEATURES = ["Weight(kg)", "Body_Fat(%)"]
CLF_TARGET = "Supplement"
RAW_SUPPLEMENT_DATA_PATH = "fitness_supplements_dataset.csv"
RANDOM_STATE = 42

ALGO_DESCRIPTIONS = {
    "Random Forest (Ensemble)": (
        "Ensemble aus vielen Entscheidungsbäumen — robust und meist gute Güte, "
        "aber als Ganzes weniger transparent als ein Einzelbaum."
    ),
    "Support Vector Machine (SVM)": (
        "Sucht die optimale Trennlinie (Hyperebene) zwischen den Klassen — kann hohe "
        "Güte erreichen, ist aber praktisch eine \"Black Box\" ohne einfache Erklärbarkeit."
    ),
    "Decision Tree (Einzelbaum)": (
        "Ein einzelner, nachvollziehbarer Entscheidungsbaum — maximale Transparenz "
        "(jede Entscheidung lässt sich als Regel lesen), dafür anfälliger für Overfitting."
    ),
}


@st.cache_data
def load_raw_supplement_data() -> pd.DataFrame:
    """Lädt den rohen Supplement-Datensatz für das Live-Training der Vergleichsmodelle."""
    df = pd.read_csv(RAW_SUPPLEMENT_DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource(show_spinner="🔄 Trainiere Modell live auf dem echten Datensatz...")
def train_classifier_live(
    algorithm: str,
    n_estimators: int = 300,
    dt_max_depth: int = 5,
    svm_kernel: str = "linear",
    svm_c: float = 1.0,
):
    """Trainiert live einen Klassifikator (RF / SVM / Decision Tree) auf dem echten
    fitness_supplements_dataset.csv, mit derselben Preprocessing-Pipeline
    (ColumnTransformer: OneHotEncoder + SimpleImputer(median) + StandardScaler) wie im
    Notebook. Gibt die trainierte Pipeline sowie die echte Test-Accuracy auf 20%
    ungesehenen Daten zurück. Ergebnisse werden pro Hyperparameter-Kombination gecacht,
    damit wiederholte Slider-Werte nicht erneut trainiert werden müssen."""
    df = load_raw_supplement_data()
    X = df[CLF_CATEGORICAL_FEATURES + CLF_NUMERIC_FEATURES]
    y = df[CLF_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CLF_CATEGORICAL_FEATURES),
        ("num", numeric_pipeline, CLF_NUMERIC_FEATURES),
    ])

    if algorithm == "rf":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=RANDOM_STATE)
    elif algorithm == "svm":
        model = SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=RANDOM_STATE)
    elif algorithm == "dt":
        model = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unbekannter Algorithmus: {algorithm}")

    pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, test_accuracy


clf_model, reg_model, metadata, load_error = load_artifacts()

st.markdown('<p class="ss-title">💊 SmartSupp</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="ss-subtitle">Supplement-Empfehlung &amp; Fortschritts-Tracking auf Basis deiner Trainingsdaten</p>',
    unsafe_allow_html=True,
)

if load_error:
    st.error(
        f"⚠️ SmartSupp konnte nicht vollständig geladen werden.\n\n{load_error}\n\n"
        "Stelle sicher, dass sich `model_supplement_classifier.pkl`, "
        "`model_progress_regressor.pkl` und `smartsupp_model_metadata.pkl` "
        "im selben Ordner wie `app.py` befinden."
    )
    st.stop()

# `supplement_stats` existiert nur, wenn die Notebook-Ergänzung zum Speichern der
# historischen Ø-Werte bereits ausgeführt wurde. Die App bleibt auch ohne diese
# Metadaten funktionsfähig (Fallback: reine Modell-Wahrscheinlichkeit).
supplement_stats = metadata.get("supplement_stats", {})

# Rein optische Zuordnung: ein thematisches Emoji pro Supplement fürs Dashboard-Design.
SUPPLEMENT_EMOJIS = {
    "Whey_Protein": "🥛",
    "Creatine": "💪",
    "BCAA": "🧬",
    "Fish_Oil": "🐟",
    "Multivitamin": "🌿",
    "Pre_Workout": "⚡",
    "Energy_Drink": "🥤",
    "L_Carnitine": "🔥",
}
RANK_MEDALS = ["🥇", "🥈", "🥉"]

# Verschönerte Anzeige-Namen für die Top-3-Kacheln — wissenschaftlich klingende,
# präsentationstaugliche Begriffe statt der rohen Original-Encoding-Werte.
SUPPLEMENT_DISPLAY_NAMES = {
    "Whey_Protein": "Proteinpulver",
    "Pre_Workout": "Pre-Workout-Booster",
    "Creatine": "Kreatin",
    "BCAA": "BCAAs (Verzweigtkettige Aminosäuren)",
    "Fish_Oil": "Omega-3-Fettsäuren (Fischöl)",
    "L_Carnitine": "L-Carnitin (Fatburner-Matrix)",
}

# Kurze, allgemeinverständliche Erklärungen — keine individuelle Dosierungsempfehlung,
# sondern Grundwissen zur Einordnung der Modell-Empfehlung.
SUPPLEMENT_INFO = {
    "Creatine": "Kraftsteigerung & schnellere ATP-Synthese in den Muskelzellen – der am besten "
                "erforschte Wirkstoff für Kraft- und Muskelaufbau.",
    "Whey_Protein": "Schnell verfügbares Molkenprotein – unterstützt Muskelaufbau und "
                    "Regeneration nach dem Training.",
    "BCAA": "Verzweigtkettige Aminosäuren – bei ausreichender Gesamt-Proteinzufuhr meist ohne "
            "zusätzlichen Mehrwert.",
    "Pre_Workout": "Koffein- und Stimulanzien-Mix – kurzfristiger Energie- und Fokus-Boost vor "
                   "dem Training.",
    "Energy_Drink": "Koffeinhaltiges Getränk – ähnlicher Effekt wie Pre-Workout, oft mit "
                     "deutlich mehr Zucker.",
    "Multivitamin": "Deckt Mikronährstofflücken ab – wirkt eher auf die allgemeine Gesundheit "
                     "als direkt auf Kraft oder Muskelaufbau.",
    "Fish_Oil": "Omega-3-Fettsäuren – unterstützen Herz-Kreislauf-Gesundheit sowie Regeneration "
                "und Entzündungshemmung.",
    "L_Carnitine": "Transportiert Fettsäuren in die Mitochondrien – möglicher Nutzen für den "
                   "Fettstoffwechsel, die Studienlage ist aber uneindeutig.",
}

# Allgemeine Tagesdosis-Richtwerte aus der Sportnahrungs-Literatur (min, max, Kurzbegründung).
# Bewusst nur für Supplements mit einigermaßen etabliertem Konsens-Bereich gepflegt — für
# Pre-Workout/Energy-Drink (stark produktabhängiger Koffeingehalt) und Multivitamin
# (herstellerabhängige Tablettendosierung) wäre eine feste Gramm-Range irreführend.
DOSE_RANGES_G = {
    "Creatine": (3, 5, "gängige Erhaltungsdosis in der Sportwissenschaft"),
    "Fish_Oil": (1, 3, "übliche kombinierte EPA/DHA-Tagesmenge"),
    "L_Carnitine": (1, 3, "häufig untersuchter Tagesdosis-Bereich in Studien"),
}

# Empfohlener Protein-Zielbereich für Kraftsport-Aktive (g Protein je kg Körpergewicht/Tag).
PROTEIN_PER_KG_RANGE = (1.6, 2.2)

# Anzeige-Texte (Deutsch) → Original-Encoding des Datensatzes (wird ans Modell übergeben).
# So bleibt die UI-Sprache durchgängig Deutsch, ohne die vom Modell erwarteten Kategoriewerte zu ändern.
GENDER_OPTIONS = {"Männlich": "Male", "Weiblich": "Female"}
FITNESS_LEVEL_OPTIONS = {"Anfänger": "Beginner", "Fortgeschritten": "Intermediate", "Experte": "Advanced"}
WEEKLY_TRAINING_OPTIONS = {
    "Weniger als 3": "Less_than_3",
    "3–5 Mal": "3-5_times",
    "Mehr als 5": "More_than_5",
}
TRAINING_TYPE_OPTIONS = {
    "HIIT": "HIIT",
    "Krafttraining": "Strength_Training",
    "Ausdauertraining": "Endurance",
    "Beweglichkeit": "Flexibility",
}
DIET_TYPE_OPTIONS = {"Normal": "Regular", "Vegan": "Vegan", "Low Carb": "Low_Carb", "Proteinreich": "High_Protein"}


def age_to_category(age: float) -> str:
    """Bildet den numerischen Alter-Slider-Wert auf die vom Klassifikator erwarteten
    Trainings-Kategorien ab (Original-Encoding im Datensatz: 'Under_25', '25-35', 'Over_35')."""
    if age < 25:
        return "Under_25"
    if age <= 35:
        return "25-35"
    return "Over_35"


def height_to_category(height_cm: float) -> str:
    """Bildet den numerischen Körpergrößen-Slider-Wert auf die vom Klassifikator erwarteten
    Trainings-Kategorien ab (Original-Encoding im Datensatz: '150-160', ..., 'Above_190')."""
    if height_cm <= 160:
        return "150-160"
    if height_cm <= 170:
        return "160-170"
    if height_cm <= 180:
        return "170-180"
    if height_cm <= 190:
        return "180-190"
    return "Above_190"

# ---------------------------------------------------------------------------
# Sidebar — Nutzerprofil für die Live-Klassifikation in Tab 1
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧍 Dein Nutzerprofil")
    st.caption("Wird live für die Supplement-Empfehlung in Tab 1 verwendet.")
    st.divider()

    gender = GENDER_OPTIONS[st.selectbox("Geschlecht", list(GENDER_OPTIONS))]
    age = st.slider("Alter", min_value=14, max_value=80, value=21)
    height_cm = st.slider("Körpergröße (in cm)", min_value=140, max_value=220, value=175)
    fitness_level = FITNESS_LEVEL_OPTIONS[st.selectbox("Fitness-Level", list(FITNESS_LEVEL_OPTIONS))]

    st.divider()

    weekly_training = WEEKLY_TRAINING_OPTIONS[
        st.selectbox("Trainingshäufigkeit / Woche", list(WEEKLY_TRAINING_OPTIONS))
    ]
    training_type_label = st.selectbox("Trainingsart", list(TRAINING_TYPE_OPTIONS))
    training_type = TRAINING_TYPE_OPTIONS[training_type_label]
    diet_type = DIET_TYPE_OPTIONS[st.selectbox("Ernährungsform", list(DIET_TYPE_OPTIONS))]

    st.divider()

    weight_kg = st.slider("Gewicht (kg)", min_value=40, max_value=150, value=75)
    body_fat_pct = st.slider("Körperfettanteil (%)", min_value=5, max_value=45, value=20)

    st.divider()

    with st.expander("⚙️ ML-Modellsteuerung & Hyperparameter"):
        st.caption(
            "Vergleiche Random Forest, SVM und Decision Tree live auf dem echten "
            "Trainingsdatensatz. Jede Auswahl trainiert das jeweilige Modell in Echtzeit "
            "neu und zeigt die tatsächliche Test-Genauigkeit auf 20% ungesehenen Daten — "
            "keine simulierten Werte."
        )

        algorithm_label = st.selectbox(
            "Algorithmus",
            ["Random Forest (Ensemble)", "Support Vector Machine (SVM)", "Decision Tree (Einzelbaum)"],
        )
        st.caption(ALGO_DESCRIPTIONS[algorithm_label])

        if algorithm_label == "Random Forest (Ensemble)":
            n_estimators = st.slider("Anzahl der Bäume (n_estimators)", 10, 500, 300, step=10)
            active_clf_model, active_clf_accuracy = train_classifier_live("rf", n_estimators=n_estimators)
        elif algorithm_label == "Support Vector Machine (SVM)":
            kernel_label = st.selectbox("Kernel-Typ", ["Linear", "RBF"])
            svm_c = st.slider("Regularisierung (C)", 0.1, 5.0, 1.0, step=0.1)
            active_clf_model, active_clf_accuracy = train_classifier_live(
                "svm", svm_kernel=kernel_label.lower(), svm_c=svm_c
            )
        else:
            dt_max_depth = st.slider("Max. Tiefe (max_depth)", 1, 20, 5)
            active_clf_model, active_clf_accuracy = train_classifier_live("dt", dt_max_depth=dt_max_depth)

        clf_baseline = 1 / len(active_clf_model.classes_)
        st.metric("Test-Genauigkeit (live gemessen)", f"{active_clf_accuracy * 100:.1f}%")
        st.caption(
            f"Zufalls-Baseline bei {len(active_clf_model.classes_)} Klassen: "
            f"{clf_baseline * 100:.1f}%. Diese Werte steuern live die Empfehlungen unten."
        )

tab1, tab2 = st.tabs(["🧪 Supplement-Empfehlung", "📈 Fortschritts-Tracking"])

# ---------------------------------------------------------------------------
# Tab 1 — Live-Klassifikation: personalisierte Liste + Geld-Spar-Warnsystem
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### 🎯 Deine persönlichen Top-Empfehlungen")
    st.caption("Aktualisiert sich live, sobald du dein Profil in der Sidebar änderst.")

    try:
        input_row = {
            "Gender": gender,
            "Age": age_to_category(age),
            "Height(cm)": height_to_category(height_cm),
            "Fitness_Level": fitness_level,
            "Weekly_Training": weekly_training,
            "Training_Type": training_type,
            "Diet_Type": diet_type,
            "Weight(kg)": weight_kg,
            "Body_Fat(%)": body_fat_pct,
        }
        input_df = pd.DataFrame([input_row])[metadata["clf_features"]]

        proba = active_clf_model.predict_proba(input_df)[0]
        classes = active_clf_model.classes_
        n_classes = len(classes)
        baseline_proba = 1 / n_classes  # Zufalls-Niveau bei n_classes gleich wahrscheinlichen Optionen

        results = pd.DataFrame({"Supplement": classes, "Wahrscheinlichkeit": proba})
        results["Anzeige_Name"] = results["Supplement"].map(
            lambda s: SUPPLEMENT_DISPLAY_NAMES.get(s, s.replace("_", " "))
        )
        results["Perf_Schnitt"] = results["Supplement"].map(
            lambda s: supplement_stats.get(s, {}).get("mean_performance")
        )
        results = results.sort_values("Wahrscheinlichkeit", ascending=False).reset_index(drop=True)

        # --- Personalisierte Empfehlungsliste (Top 3) als Dashboard-Kacheln ---
        top3 = results.head(3).reset_index(drop=True)
        dash_cols = st.columns(len(top3))
        for i, col in enumerate(dash_cols):
            row = top3.iloc[i]
            pct = row["Wahrscheinlichkeit"] * 100
            emoji = SUPPLEMENT_EMOJIS.get(row["Supplement"], "💊")
            medal = RANK_MEDALS[i] if i < len(RANK_MEDALS) else "▫️"
            explanation = SUPPLEMENT_INFO.get(row["Supplement"], "Keine Detailinfo hinterlegt.")

            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='text-align:center; font-size:1.05rem; font-weight:600;'>"
                        f"{medal} {emoji} {row['Anzeige_Name']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Erfolgschance", f"{pct:.0f}%")
                    if pd.notna(row["Perf_Schnitt"]):
                        st.caption(f"Ø Leistungssteigerung: +{row['Perf_Schnitt']:.1f}%")
                    else:
                        st.caption("Modell-Prognose basiert auf deinem individuellen Profil.")
                    st.markdown(
                        f"<div style='font-size:0.85rem; color:#c9d1d9; margin-top:0.4rem; "
                        f"line-height:1.35;'>{explanation}</div>",
                        unsafe_allow_html=True,
                    )

                    # --- Dynamische, wissenschaftliche Mengenangabe je Supplement ---
                    dosage_msg = None
                    if row["Supplement"] == "Creatine":
                        dosage_msg = (
                            "💡 Empfohlene Dosierung: Täglich 3-5g durchgehend mit ausreichend "
                            "Wasser einnehmen."
                        )
                    elif row["Supplement"] == "Whey_Protein":
                        dosage_msg = "💡 Empfohlene Dosierung: Ca. 30-40g pro Portion."
                    elif row["Supplement"] == "Pre_Workout":
                        dosage_msg = (
                            "💡 Empfohlene Dosierung: 1 Portion ca. 20-30 Minuten vor dem Training. "
                            "Nicht zu spät am Abend einnehmen!"
                        )
                    elif row["Supplement"] == "Fish_Oil":
                        dosage_msg = (
                            "💡 Empfohlene Dosierung: Täglich 1-3g. Unterstützt die Herzgesundheit, "
                            "Gelenke und wirkt entzündungshemmend. Der Zusatz 'Deep_Sea' im "
                            "Datensatz ist reines Marketing."
                        )
                    elif row["Supplement"] == "L_Carnitine":
                        dosage_msg = (
                            "💡 Empfohlene Dosierung: 1-3g vor dem Training. Wird in der "
                            "Fitness-Szene oft für die Fettverbrennung beworben. Studien zeigen "
                            "jedoch: Bei ausreichender Zufuhr über die Nahrung hat es für gesunde "
                            "Sportler kaum einen messbaren Effekt."
                        )

                    if dosage_msg:
                        st.markdown(
                            f"<div style='font-size:0.8rem; color:#58a6ff; margin-top:0.5rem; "
                            f"line-height:1.35;'>{dosage_msg}</div>",
                            unsafe_allow_html=True,
                        )

                    # --- BCAA-Sonderregel: unabhängig vom Rang immer die Geld-Spar-Warnung ---
                    if row["Supplement"] == "BCAA":
                        with st.container(border=True):
                            st.markdown(
                                "<div style='opacity:0.7; font-size:0.85rem; line-height:1.4;'>"
                                "💸 Geld-Spar-Warnung: Obwohl das Modell BCAAs für dein Profil "
                                "vorschlägt, zeigt die aktuelle Studienlage: Wenn du deinen "
                                "täglichen Proteinbedarf deckst, sind BCAAs reine "
                                "Geldverschwendung. Investiere das Geld lieber in hochwertiges "
                                "Proteinpulver!</div>",
                                unsafe_allow_html=True,
                            )

        st.divider()

        # --- Geld-Spar-Warnsystem ---
        # Kombiniert zwei Signale, um nicht allein auf die (nur ~26% genaue)
        # Modell-Confidence zu vertrauen:
        #   1) Wahrscheinlichkeit unter Zufalls-Baseline für DIESES Profil
        #   2) historisch unterdurchschnittliche Wirkung über ALLE Nutzer hinweg
        if supplement_stats:
            perf_values = pd.Series(
                {s: v.get("mean_performance") for s, v in supplement_stats.items()}
            )
            median_perf = perf_values.median()
            money_waste = results[
                (results["Wahrscheinlichkeit"] < baseline_proba)
                & (results["Perf_Schnitt"].notna())
                & (results["Perf_Schnitt"] < median_perf)
            ]
        else:
            # Fallback ohne historische Statistik: nur sehr niedrige Modellwahrscheinlichkeit
            money_waste = results[results["Wahrscheinlichkeit"] < baseline_proba / 2]

        if not money_waste.empty:
            st.markdown("#### 💸 Geld-Spar-Warnsystem")
            st.caption(
                "Kombiniert die Modell-Wahrscheinlichkeit für dein Profil mit historischen "
                "Durchschnittswerten aus den Trainingsdaten."
            )
            for _, row in money_waste.iterrows():
                msg = (
                    f"**{row['Anzeige_Name']}** passt laut Modell nur zu "
                    f"{row['Wahrscheinlichkeit']*100:.0f}% zu deinem Profil"
                )
                if pd.notna(row["Perf_Schnitt"]):
                    msg += f" und zeigte historisch nur +{row['Perf_Schnitt']:.1f}% Leistungssteigerung"
                msg += " — für dich vermutlich Geldverschwendung."
                with st.container(border=True):
                    st.markdown(f"💸 {msg}")
        elif not supplement_stats:
            st.info(
                "💡 Für ein vollständiges Geld-Spar-Warnsystem inkl. historischer Wirksamkeit "
                "fehlt noch `supplement_stats` in deinen Metadaten. Ergänze im Notebook die kurze "
                "Zusatzzelle nach dem Modell-Speichern und lade die App neu."
            )

        # --- Zusammenfassende KI-Coach-Box ---
        if len(top3) >= 2:
            # Hängt die weiter oben definierte Mengenangabe an den Anzeige-Namen an,
            # damit die Coach-Box dieselbe Dosierung wie die Kacheln zeigt.
            def _mit_dosierung(row: pd.Series) -> str:
                name = row["Anzeige_Name"]
                if row["Supplement"] == "Whey_Protein":
                    return f"{name} (ca. 30-40g)"
                if row["Supplement"] == "Creatine":
                    low, high, _ = DOSE_RANGES_G["Creatine"]
                    return f"{name} ({low}-{high}g)"
                return name

            platz1 = _mit_dosierung(top3.iloc[0])
            platz2 = _mit_dosierung(top3.iloc[1])
            protein_low = weight_kg * PROTEIN_PER_KG_RANGE[0]
            protein_high = weight_kg * PROTEIN_PER_KG_RANGE[1]
            fazit_text = (
                f"Basierend auf deinen Zielen ({training_type_label}) und deinem Profil "
                f"empfiehlt die KI als optimalen Core-Stack die Kombination aus "
                f"{platz1} und {platz2}. Diese Kombination deckt deine Performance- "
                "und Regenerationsbedürfnisse statistisch am besten ab. Achte darauf, diese "
                "Basis-Supplements priorisiert einzunehmen, bevor du Geld für weitere "
                "Produkte ausgibst."
                "<br><br>"
                f"Bei deinem Körpergewicht von {weight_kg} kg liegt dein täglicher "
                f"Protein-Zielbereich bei rund {protein_low:.0f}–{protein_high:.0f} g "
                "(1,6–2,2 g/kg Körpergewicht, üblicher Richtwert für Kraftsport)."
            )
            st.divider()
            with st.container():
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 25px; border-radius: 10px; text-align: center; border: 1px solid #e5e7eb; margin-top: 20px;">
                    <h3 style="color: #1f2937; margin-top: 0; margin-bottom: 12px; font-family: sans-serif;">📋 Deine SmartSupp-Empfehlung:</h3>
                    <p style="color: #1f2937; font-size: 16px; line-height: 1.6; margin: 0; font-family: sans-serif;">{fazit_text}</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        with st.expander("Alle Wahrscheinlichkeiten im Detail ansehen"):
            st.bar_chart(results.set_index("Anzeige_Name")[["Wahrscheinlichkeit"]])

    except Exception as e:
        st.error(f"⚠️ Bei der Vorhersage ist ein Fehler aufgetreten: {e}")

# ---------------------------------------------------------------------------
# Tab 2 — Fortschritts-Regression: Soll-Ist-Vergleich + Multi-Supplement-Tracking
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Fortschritts-Tracking")
    st.caption(
        "Wöchentlicher Check-in: SmartSupp vergleicht dein tatsächliches Gewicht (Ist) mit der "
        "Modell-Prognose auf Basis deines Verhaltens (Soll) und leitet daraus Empfehlungen ab."
    )

    # Kleiner Helfer, damit Tab 2 überall dieselben schönen Anzeige-Namen wie Tab 1 nutzt,
    # während im Hintergrund weiterhin mit den rohen Modell-Encoding-Werten gerechnet wird.
    def supp_display(s: str) -> str:
        return SUPPLEMENT_DISPLAY_NAMES.get(s, s.replace("_", " "))

    st.markdown("##### 💊 Aktuell genutzte Supplements")
    current_supplements = st.multiselect(
        "Welche Supplements nutzt du aktuell?",
        options=sorted(active_clf_model.classes_),
        default=["Whey_Protein"],
        format_func=supp_display,
    )

    dosages = {}
    if current_supplements:
        dosage_cols = st.columns(len(current_supplements))
        for i, supp in enumerate(current_supplements):
            with dosage_cols[i]:
                dosages[supp] = st.number_input(
                    f"Dosis {supp_display(supp)} (g/Tag)",
                    min_value=0, max_value=100, value=5, step=1,
                    key=f"dose_{supp}",
                )
                # Sportwissenschaftlicher Dosierungs-Hinweis speziell für Whey Protein,
                # sobald der Nutzer eine (positive) Gramm-Menge unter 25g einträgt.
                if supp == "Whey_Protein" and 0 < dosages[supp] < 25:
                    st.caption(
                        f"💡 Dosierungs-Tipp für Whey: Deine eingestellte Portion von "
                        f"{dosages[supp]}g ist recht gering. Um die Muskelproteinsynthese "
                        "optimal zu stimulieren, empfiehlt die Literatur eine Portionsgröße "
                        "von ca. 30-40g Proteinpulver."
                    )
    else:
        st.caption("Wähle mindestens ein Supplement aus, um die Dosierung anzugeben.")

    st.divider()

    st.caption(
        f"ℹ️ Alter ({age} Jahre) und Geschlecht werden automatisch aus deinem Profil in der "
        "Sidebar übernommen."
    )
    # Außerhalb des Forms platziert, damit die Slider beim Klicken sofort (ohne Submit)
    # ein-/ausgeblendet werden — Widgets innerhalb eines st.form lösen erst beim Submit
    # einen Rerun aus.
    no_precise_tracking = st.checkbox(
        "Ich tracke meine Ernährung aktuell nicht präzise",
        help=(
            "Wenn aktiviert, rechnet SmartSupp im Hintergrund mit soliden "
            "Standard-Mittelwerten (2500 kcal, 1.5g Protein pro kg Körpergewicht) statt mit "
            "deinen Slider-Eingaben."
        ),
    )

    with st.form("reg_form"):
        col1, col2 = st.columns(2)

        with col1:
            bmi = st.slider("BMI", min_value=15.0, max_value=45.0, value=24.0, step=0.1)
            experience_level = st.select_slider(
                "Trainingserfahrung (1 = Anfänger, 3 = Fortgeschritten)",
                options=[1, 2, 3],
                value=2,
            )
            workout_duration = st.slider(
                "Trainingsdauer (Minuten)", min_value=10, max_value=150, value=65, step=5
            )
            steps_walked = st.slider(
                "Schritte pro Tag", min_value=1000, max_value=20000, value=8300, step=100
            )

        with col2:
            if no_precise_tracking:
                # Solide Standard-Mittelwerte, damit der Regressor auch ohne präzises
                # Ernährungstracking stabil weiterrechnet.
                calories_intake = 2500
                protein_intake = round(1.5 * weight_kg)
                st.caption(
                    "🔒 Kalorien- & Proteinzufuhr ausgeblendet — SmartSupp rechnet mit "
                    f"Standard-Mittelwerten: **{calories_intake} kcal/Tag** und "
                    f"**{protein_intake} g Protein/Tag** (1.5g/kg bei {weight_kg}kg "
                    "Körpergewicht laut Sidebar-Profil)."
                )
            else:
                calories_intake = st.slider(
                    "Kalorienzufuhr (kcal/Tag)", min_value=1200, max_value=4500, value=2700, step=50
                )
                protein_intake = st.slider(
                    "Proteinzufuhr (g/Tag)", min_value=30, max_value=250, value=115, step=5
                )

        actual_weight = st.slider(
            "Dein tatsächliches Gewicht heute (Ist-Wert, kg)",
            min_value=40.0, max_value=150.0, value=75.0, step=0.1,
        )

        st.divider()

        # Diese beiden Werte fließen NICHT ins ML-Modell ein (sie sind kein Bestandteil der
        # Trainingsdaten), sondern werden ausschließlich für regelbasierte Text-Heuristiken
        # direkt nach der Berechnung genutzt.
        muskelkater = st.slider(
            "Muskelkater-Level (1 = keiner, 10 = sehr stark)", min_value=1, max_value=10, value=4
        )
        schlafqualitaet = st.slider(
            "Schlafqualität (1 = sehr schlecht, 10 = ausgezeichnet)", min_value=1, max_value=10, value=7
        )

        submitted_reg = st.form_submit_button("Fortschritt berechnen 📊", use_container_width=True)

    if submitted_reg:
        try:
            input_row = {
                "Gender": gender,
                "Calories_Intake": calories_intake,
                "Protein_Intake_g": protein_intake,
                "Workout_Duration_min": workout_duration,
                "Steps_Walked": steps_walked,
                "Age": age,
                "BMI": bmi,
                "Experience_Level": experience_level,
            }
            input_df = pd.DataFrame([input_row])[metadata["reg_features"]]
            predicted_weight = reg_model.predict(input_df)[0]  # "Soll"-Gewicht laut Modell
            deviation = actual_weight - predicted_weight

            st.divider()
            st.markdown("##### 📊 Soll-Ist-Vergleich")

            with st.container(border=True):
                col_a, col_b = st.columns(2)
                col_a.metric(
                    "Soll-Gewicht (Modell-Prognose)",
                    f"{predicted_weight:.1f} kg",
                    help=(
                        "Das Soll-Gewicht ist der Wert, den unser Machine-Learning-Regressionsmodell "
                        "basierend auf deinen wöchentlichen Verhaltensdaten (Kalorien, Schritte, "
                        "Training) prognostiziert. Weicht dein echtes Ist-Gewicht stark davon ab, "
                        "deutet das auf eine Stagnation (Plateau) oder einen unerwarteten Fortschritt hin."
                    ),
                )
                col_b.metric(
                    "Ist-Gewicht (dein Eintrag)",
                    f"{actual_weight:.1f} kg",
                    delta=f"{deviation:+.1f} kg",
                )
                st.caption(
                    "ℹ️ Das **Soll-Gewicht** ist die Modell-Prognose auf Basis deines Verhaltens diese "
                    "Woche (Kalorien, Schritte, Training). Eine große Abweichung zum **Ist-Gewicht** "
                    "kann auf ein Plateau oder einen unerwarteten Fortschritt hindeuten."
                )

            # Verlauf in der Session merken (Soll-Ist-Vergleich & Supplement-Nutzungsdauer über Zeit).
            # Hinweis: Dies ist In-Memory und setzt sich beim Neustart der App zurück —
            # für eine produktive Mehrnutzer-Version wäre eine kleine Datenbank/Datei nötig.
            last_actual_weight = st.session_state.get("last_actual_weight")
            weight_trend = actual_weight - last_actual_weight if last_actual_weight is not None else None

            supplement_weeks_prev = st.session_state.get("supplement_weeks", {})
            supplement_weeks_new = {
                supp: supplement_weeks_prev.get(supp, 0) + 1 for supp in current_supplements
            }

            st.divider()
            st.markdown("##### 💊 Dosierungs-Check")
            st.caption(
                "Allgemeine Richtwerte aus der Sportnahrungs-Literatur — keine individuelle "
                "Ernährungs- oder medizinische Beratung."
            )

            fixed_range_feedback_given = False

            # --- Feste Gramm-Ranges für Supplements mit etabliertem Konsens-Bereich ---
            for supp, (low, high, rationale) in DOSE_RANGES_G.items():
                if supp in dosages:
                    dose = dosages[supp]
                    name = supp_display(supp)
                    fixed_range_feedback_given = True
                    if dose < low:
                        st.info(
                            f"💊 **{name}-Dosis:** Mit {dose} g/Tag liegst du unter dem üblichen "
                            f"Bereich von {low}–{high} g ({rationale}). Eine leichte Erhöhung könnte "
                            "die Wirkung verbessern."
                        )
                    elif dose > high:
                        st.info(
                            f"💊 **{name}-Dosis:** Mit {dose} g/Tag liegst du über dem üblichen "
                            f"Bereich von {low}–{high} g ({rationale}). Mehr ist hier meist nicht "
                            "wirksamer — der Überschuss wird in der Regel einfach ausgeschieden."
                        )
                    else:
                        st.success(
                            f"💊 **{name}-Dosis:** Deine {dose} g/Tag liegen im üblichen Bereich "
                            f"von {low}–{high} g."
                        )

            # --- Proteinzufuhr im Verhältnis zum Körpergewicht (unabhängig von der Whey-Einzeldosis) ---
            low_ratio, high_ratio = PROTEIN_PER_KG_RANGE
            if actual_weight > 0:
                protein_per_kg = protein_intake / actual_weight
                if protein_per_kg < low_ratio:
                    st.info(
                        f"🥩 **Protein-Ziel:** Deine Gesamt-Proteinzufuhr entspricht "
                        f"{protein_per_kg:.2f} g/kg Körpergewicht — unter dem für Kraftsport oft "
                        f"empfohlenen Bereich von {low_ratio}–{high_ratio} g/kg. Mehr Protein "
                        "(z. B. über Whey) könnte den Muskelaufbau zusätzlich unterstützen."
                    )
                elif protein_per_kg > high_ratio:
                    st.info(
                        f"🥩 **Protein-Ziel:** Deine Gesamt-Proteinzufuhr entspricht "
                        f"{protein_per_kg:.2f} g/kg Körpergewicht — über dem üblichen Zielbereich "
                        f"von {low_ratio}–{high_ratio} g/kg. Zusätzliches Protein bringt meist "
                        "keinen weiteren Vorteil mehr — hier ließe sich eventuell Geld sparen."
                    )
                else:
                    st.success(
                        f"🥩 **Protein-Ziel:** Deine Gesamt-Proteinzufuhr von "
                        f"{protein_per_kg:.2f} g/kg Körpergewicht liegt im empfohlenen Bereich "
                        f"({low_ratio}–{high_ratio} g/kg)."
                    )

            if not fixed_range_feedback_given:
                st.caption(
                    "Für die aktuell ausgewählten Supplements liegt kein fester Gramm-Richtwert vor "
                    "(z. B. bei Whey Protein, Pre-Workout, Energy Drink oder Multivitamin) — hierfür "
                    "gilt stattdessen die Protein-Ziel-Einschätzung oben bzw. die Herstellerangabe."
                )

            st.divider()
            st.markdown("##### 🧭 Handlungsempfehlung")

            supp_list_text = ", ".join(supp_display(s) for s in current_supplements) or "deinen aktuellen Supplements"

            if abs(deviation) <= 0.3:
                st.success(
                    "Dein tatsächliches Gewicht liegt nah an der Modellprognose — "
                    "du bist auf Kurs mit deinem aktuellen Training & deinen Supplements."
                )
            elif deviation > 0.3 and weight_trend is not None and abs(weight_trend) < 0.2:
                st.warning(
                    "Dein Gewicht stagniert trotz konstantem Training — möglicher **Kraftplateau**-Hinweis. "
                    f"Erwäge eine leichte Anpassung von Dosis oder Timing bei **{supp_list_text}**, "
                    "z. B. die Einnahme direkt im Anschluss ans Training zu verschieben."
                )
            elif deviation > 0.3:
                st.info(
                    "**Dein Gewicht liegt über der Modell-Prognose.**\n\n"
                    "- **Falls Muskelaufbau dein primäres Ziel ist:** Hervorragender Fortschritt! "
                    "Das Modell signalisiert eine positive Energiebilanz. Achte weiterhin auf eine "
                    "progressive Überlastung im Training und halte deine Proteinzufuhr konstant "
                    "hoch, um die neu gewonnene Masse optimal in Muskelgewebe umzuwandeln.\n\n"
                    "- **Falls Fettabbau dein primäres Ziel ist:** Dann lohnt sich ein kritischer "
                    f"Blick auf deine aktuelle Kalorienzufuhr ({calories_intake} kcal) und dein "
                    f"Aktivitätslevel ({steps_walked} Schritte). Das Modell empfiehlt, die "
                    "täglichen Kalorien testweise um 10–15 % zu senken oder die Alltagsbewegung "
                    "um 2.000 Schritte zu erhöhen, um die Fettverbrennung wieder zu stimulieren."
                )
            else:
                st.success(
                    "Dein Gewicht liegt unter der Modell-Prognose — dein aktueller Ansatz scheint zu wirken."
                )

            for supp in current_supplements:
                if supp in ("Pre_Workout", "Energy_Drink") and supplement_weeks_new[supp] >= 6:
                    st.warning(
                        f"Du nutzt **{supp_display(supp)}** bereits seit "
                        f"{supplement_weeks_new[supp]} Wochen ohne Pause. Um einer Koffein-Gewöhnung "
                        "vorzubeugen, ist eine kurze Einnahmepause (z. B. 1 Woche) sinnvoll."
                    )

            # --- Reine Text-Heuristiken auf Basis von Muskelkater & Schlafqualität ---
            # Beide Werte sind keine Modell-Features, sondern zusätzlicher Kontext für
            # regelbasierte Timing-/Regenerations-Tipps.
            if muskelkater > 7:
                st.info(
                    "**Timing-Optimierung:** Dein Muskelkater ist sehr hoch. Verschiebe die Einnahme "
                    "von Proteinen/Aminosäuren auf direkt nach dem Training in Kombination mit "
                    "schnellen Kohlenhydraten."
                )

            if schlafqualitaet < 5:
                st.warning(
                    "**Schlaf-Optimierung:** Deine Schlafqualität ist niedrig. Achte darauf, "
                    "Pre-Workout-Booster oder Koffein mindestens 6 Stunden vor dem Schlafen zu meiden."
                )

            st.caption(
                "Hinweis: Die Handlungsempfehlung ist eine regelbasierte Ergänzung auf Basis der "
                "Modell-Prognose (Soll-Wert) und deines Verlaufs — keine medizinische oder "
                "ernährungswissenschaftliche Beratung."
            )

            st.session_state["last_actual_weight"] = actual_weight
            st.session_state["supplement_weeks"] = supplement_weeks_new
            if dosages:
                st.session_state["last_dosages"] = dosages

        except Exception as e:
            st.error(f"⚠️ Bei der Vorhersage ist ein Fehler aufgetreten: {e}")
