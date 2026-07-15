Erstelle in diesem Ordner eine neue Datei namens `README.md` mit folgendem Inhalt für meinen Professor:

# 💊 SmartSupp – KI-gestützter Supplement-Optimierer

SmartSupp ist eine interaktive, datenbasierte Web-Applikation (entwickelt mit Streamlit), die Machine-Learning-Vorhersagen mit einem sportwissenschaftlichen Regelwerk (Expertensystem) kombiniert. Die App schützt Fitness-Einsteiger aktiv vor Marketing-getriebenen Fehlkäufen und bietet Fortgeschrittenen ein präzises, wöchentliches Fortschritts-Tracking.

## 🏗️ System-Architektur (Hybrid-Ansatz)
Reine Machine-Learning-Modelle neigen in der Praxis dazu, statistische Verzerrungen (Biases) aus historischen Daten unreflektiert weiterzugeben. SmartSupp löst dieses Problem durch eine **Hybrid-Architektur**:
1. **Machine Learning Core**: Liefert die datenbasierte Kern-Prognose und berechnet Erfolgswahrscheinlichkeiten auf Basis gelernter Muster.
2. **Regelbasiertes Expertensystem (Heuristiken)**: Fängt ungenaue Rohdaten-Begriffe ab, berechnet dynamische, individuelle Dosierungen basierend auf biometrischen Echtzeitdaten (z. B. g/kg Körpergewicht) und steuert das integrierte **Geld-Spar-Warnsystem** (z. B. zur Aufklärung über den wissenschaftlich umstrittenen Nutzen von BCAAs).

## 📊 Multi-Datensatz-Architektur & ML-Modelle
Um eine saubere mathematische Trennung zu gewährleisten und die Vorgabe mehrerer Datenquellen zu erfüllen, nutzt das Projekt zwei voneinander unabhängige Datensätze und ML-Modelle:

*   **Tab 1: Der Einstiegs-Rechner (Klassifikation)**
    *   **Modell**: `model_supplement_classifier.pkl` (Random Forest Classifier)
    *   **Datenbasis**: Historische Supplement-Konsumdaten, demografische Profile und subjektive Zufriedenheitswerte.
    *   **Funktion**: Berechnet die Live-Wahrscheinlichkeit für das optimale Supplement-Paket angepasst an das Nutzerprofil in der Sidebar.
*   **Tab 2: Der Fortschritts-Tracker (Regression)**
    *   **Modell**: `model_progress_regressor.pkl` (Random Forest Regressor)
    *   **Datenbasis**: Wöchentliche biometrische Trainingsdaten (Gewichtsverlauf, Kalorienzufuhr, Kraftwerte, Ermüdungsparameter).
    *   **Funktion**: Führt einen präzisen Soll-Ist-Vergleich des Körpergewichts durch und prognostiziert Plateaus.

## 💻 UI-Features & Aufbau
*   **Globale Sidebar**: Zentrale Steuerung des detaillierten Nutzerprofils (Alter, Größe, Fitness-Level etc.) mit direkter Echtzeit-Synchronisation im Hauptbereich.
*   **Geld-Spar-Warnsystem**: Filtert Produkte mit schlechter Evidenz oder geringer Modell-Zustimmung aktiv heraus und schützt den Verbraucher vor Fehlkäufen.
*   **Flexibles Tracking**: Ermöglicht einen optionalen Ernährungs-Check-in (Slider werden bei ungenauem Tracking ausgeblendet; das Modell rechnet stabil mit wissenschaftlichen Standard-Mittelwerten weiter).
*   **Dosierungs- & Ermüdungs-Checks**: Dynamische Infoboxen analysieren Muskelkater, Schlafqualität und gemeldete Gramm-Mengen, um direkt anwendbare Tipps zur Einnahme und zum Timing zu geben.

## 🛠️ Installation & Lokaler Start
Um die Applikation lokal auf Ihrem System auszuführen, müssen die trainierten `.pkl`-Modelle im Hauptverzeichnis liegen.

1. **Repository klonen oder in den Projektordner wechseln**:
   ```bash
   cd "Big Data Präsi"
   ```
2. **Abhängigkeiten installieren**:
   ```bash
   pip install streamlit scikit-learn joblib pandas numpy
   ```
3. **Streamlit-App fehlerfrei starten**:
   ```bash
   python3 -W ignore -m streamlit run app.py
   ```
