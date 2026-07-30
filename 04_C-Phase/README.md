[README.md](https://github.com/user-attachments/files/30404131/README.md)
# C — Conclude & Compare: Evaluation & Modellgüte

## Klassifikator (Supplement-Empfehlung)
- **Test-Accuracy: 28,5 %** auf ungesehenen Testdaten (8 Supplement-Klassen)
- **Zufalls-Baseline: 12,5 %** — das Modell liegt also mehr als doppelt so hoch wie reines Raten
- Alle drei getesteten Algorithmen (Random Forest, SVM, Decision Tree) liegen nah beieinander
  (28,5 % / 28,6 % / 28,8 %) — ein starkes Indiz dafür, dass nicht die Algorithmus-Wahl das
  Problem ist, sondern die Datenqualität selbst (verrauschte, historische Konsumdaten)

## Regressor (Fortschritts-Prognose)
- **R² niedrig bis negativ** auf Validierungsdaten
- Kein Implementierungsfehler, sondern eine ehrliche Erkenntnis: tägliche Gewichtsschwankungen
  enthalten viel stochastisches Rauschen, das aus Kalorien/Schritten allein nicht vollständig
  erklärbar ist

## Konsequenz: Hybrid-Architektur
Weil beide Modelle für sich genommen nicht zuverlässig genug sind, um blind vertraut zu werden,
wird die ML-Prognose in der App mit einem regelbasierten Expertensystem kombiniert
(Geld-Spar-Warnsystem, BCAA-Sonderregel, Dosierungs-Richtwerte). Details dazu direkt in der
Streamlit-App (Tab 1) sowie in der Projekt-Präsentation.

## Quellcode der Evaluation
Siehe `03_A3_Modellierung/` — Abschnitt "C — Conclude & Compare" im Notebook.
