[README.md](https://github.com/user-attachments/files/30404088/README.md)
# Q — Question: Geschäftsfrage & Zielsetzung

## Problem
Der Supplement-Markt ist stark marketinggetrieben. Nutzer:innen treffen Kaufentscheidungen
oft ohne individuelle Grundlage (z. B. BCAA-Hype trotz schwacher Evidenz), Dosierungsempfehlungen
sind meist nicht auf Alter, Gewicht oder Trainingsstand zugeschnitten, und es fehlt ein
Feedback-Loop zwischen Einnahme und tatsächlichem Trainingsfortschritt.

## Zielgruppe
Freizeitsportler:innen, die eine datenbasierte Unterstützung bei der Supplement-Wahl und
beim Tracking ihres Trainingsfortschritts suchen.

## Leitfrage
Kann eine Multi-Datensatz-Architektur aus Klassifikation (Supplement-Empfehlung) und
Regression (Fortschritts-Prognose) genutzt werden, um Nutzer:innen datenbasiert zu beraten
und zu begleiten — kombiniert mit einem regelbasierten Sicherheitsnetz, das die Grenzen
des Modells auffängt?

## Erfolgsmetriken (KPIs)
- Klassifikation: Test-Accuracy deutlich über der Zufalls-Baseline (12,5 % bei 8 Klassen)
- Regression: transparente, ehrlich kommunizierte Modellgüte (R²) als Entscheidungsgrundlage
- Deployment: öffentlich erreichbare, funktionsfähige Streamlit-App mit < 500 ms Antwortzeit

## Deliverables
- Jupyter Notebook mit vollständiger QUA³CK-Pipeline (→ `03_A3_Modellierung/`)
- Trainierte, deploybare Modelle (→ Hauptverzeichnis, für Streamlit Cloud)
- Interaktive Streamlit-Anwendung (Link siehe Haupt-README.md)
