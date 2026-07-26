# 💊 SmartSupp – KI-gestützter Supplement-Optimierer & Tracker

SmartSupp ist eine interaktive, datenbasierte Web-Applikation (entwickelt mit Streamlit), die Machine-Learning-Vorhersagen mit einem sportwissenschaftlichen Regelwerk (Expertensystem) kombiniert. Die App schützt Fitness-Einsteiger aktiv vor Marketing-getriebenen Fehlkäufen und bietet Fortgeschrittenen ein präzises, wöchentliches Fortschritts-Tracking.

## 🔗 Live-Anwendung
Die lauffähige Web-Applikation ist unter folgendem Link direkt im Browser erreichbar:
👉 **http://localhost:8501/**

---

## 🏗️ System-Architektur (Hybrid-Ansatz)
Reine Machine-Learning-Modelle neigen in der Praxis dazu, statistische Verzerrungen (Biases) aus historischen Daten unreflektiert weiterzugeben. SmartSupp löst dieses Problem durch eine **Hybrid-Architektur**:
1. **Machine Learning Core**: Liefert die datenbasierte Kern-Prognose und berechnet Erfolgswahrscheinlichkeiten auf Basis gelernter Muster (SVM, Decision Tree, Random Forest).
2. **Regelbasiertes Expertensystem (Heuristiken)**: Fängt ungenaue Rohdaten-Begriffe ab, berechnet dynamische, individuelle Dosierungen basierend auf biometrischen Echtzeitdaten (g/kg Körpergewicht) und steuert das integrierte **Geld-Spar-Warnsystem**.

---

## 📊 Repository-Struktur (nach dem QUA³CK-Prozessmodell)
Das Repository ist streng nach den Phasen des wissenschaftlichen QUA³CK-Modells gegliedert:

*   **📂 Phase_Q (Question):** Dokumentation der Forschungsfrage, Zielgruppen-Definition und Definition der operativen KPIs.
*   **📂 Phase_U (Understanding):** Explorative Datenanalyse (EDA), Analyse des Klassenungleichgewichts, Imputations-Strategie für Missing Values und Data Leakage Prävention.
*   **📂 Phase_A3 (Analysis & Algorithms):** Quellcode des Modell-Trainings. Enthält die Scikit-Learn-Pipeline-Architektur und das Training von Random Forest, SVM und Entscheidungsbäumen inklusive Hyperparameter-Tuning.
*   **📂 Phase_C (Conclude & Compare):** Mathematische Validierung mittels 5-Fold Cross-Validation, Analyse des R²-Wertes und Dokumentation des Algorithmen-Vergleichs.
*   **📂 Phase_K (Knowledge Transfer):** Der produktive Quellcode der Streamlit-Anwendung (`app.py`), die serialisierten Modelle (`*.pkl`) und Design-Assets.

---

## 💻 UI-Features & Live-Dashboard
*   **Globale Sidebar:** Zentrale Steuerung des biometrischen Nutzerprofils mit sofortiger reaktiver Inferenz auf dem Hauptbildschirm.
*   **⚙️ Interaktive Modellsteuerung:** Integriertes Experten-Panel in der Sidebar. Erlaubt den Live-Wechsel der Algorithmen und die Echtzeit-Visualisierung von Overfitting durch manuelle Hyperparameter-Anpassung.
*   **Geld-Spar-Warnsystem:** Clean zentriertes Kachel-Design, das evidenzbasiert vor unwissenschaftlichen Fehlkäufen (z. B. BCAAs) warnt.
*   **Flexibles Tracking:** Optionale Ausblendung des Makro-Trackings per Checkbox; das System rechnet im Hintergrund stabil mit sportwissenschaftlichen Mittelwerten weiter.

---

## 🛠️ Lokale Installation & Start

1. **Repository klonen und in den Ordner wechseln**:
   ```bash
   cd "Big Data Präsi"
   ```
2. **Abhängigkeiten installieren**:
   ```bash
   pip install streamlit scikit-learn joblib pandas numpy python-pptx
   ```
3. **Streamlit-App fehlerfrei starten**:
   ```bash
   python3 -W ignore -m streamlit run app.py
   ```
