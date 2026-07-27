[README.md](https://github.com/user-attachments/files/30404169/README.md)
# K — Knowledge Transfer: Deployment

## Live-Anwendung
Die trainierten Modelle wurden als interaktive Streamlit-Anwendung bereitgestellt:
👉 **[Link zur Live-App hier einfügen]**

## Technischer Hinweis zur Ordnerstruktur
Die produktiven Dateien für das Deployment liegen bewusst im **Hauptverzeichnis** des
Repositories, nicht in diesem Ordner:

- `app.py` — Streamlit-Frontend
- `model_supplement_classifier.pkl`, `model_progress_regressor.pkl`,
  `smartsupp_model_metadata.pkl` — serialisierte, trainierte Pipelines (Joblib)
- `requirements.txt` — Python-Abhängigkeiten

Grund: Streamlit Community Cloud erwartet die Hauptdatei (`app.py`) sowie
`requirements.txt` standardmäßig im Repository-Root, und `app.py` lädt die `.pkl`-Dateien
über relative Pfade aus genau diesem Verzeichnis. Eine Verschiebung in Unterordner würde
das Deployment ohne Anpassung der Ladepfade brechen — daher wurde hier bewusst die
funktionierende, produktive Struktur beibehalten.

## Architektur
`ColumnTransformer`-Pipelines (OneHotEncoder + SimpleImputer + StandardScaler) wurden
zusammen mit den trainierten Modellen via Joblib serialisiert, damit Vorverarbeitung und
Vorhersage bei jeder neuen Nutzereingabe identisch zum Training ablaufen — ohne
Diskrepanz zwischen Trainings- und Inferenzzeit.
