# Klärung zu Rückfragen aus der Präsentation

Bei der Präsentation kamen zwei berechtigte Rückfragen auf, die ich hier
ausführlich beantworte.

## 1. Warum liegt Decision Tree (28,8 %) minimal über Random Forest (28,5 %)?

Das ist statistisches Rauschen, kein reproduzierbarer Vorteil des Decision Tree.

Der Testdatensatz umfasst rund 758 Beispiele (20 % von 3.788 Zeilen). Der
Unterschied von 0,3 Prozentpunkten entspricht rechnerisch etwa **zwei**
zusätzlich richtig klassifizierten Beispielen – das liegt innerhalb der
normalen Schwankung eines einzelnen Train-Test-Splits, nicht außerhalb.

Der eigentlich relevante Befund ist ein anderer: Random Forest (28,5 %), SVM
(28,6 %) und Decision Tree (28,8 %) liegen trotz völlig unterschiedlicher
Funktionsweise (Ensemble, Hyperebene, Einzelbaum) alle in einem Band von nur
0,3 Prozentpunkten. Wäre die Algorithmus-Wahl der limitierende Faktor, müssten
die Werte deutlich weiter auseinanderliegen. Da sie es nicht tun, liegt die
Grenze in der Datenqualität – nicht im gewählten Modell.

## 2. Wie kann R² negativ werden?

Anders als Accuracy (0–100 %) ist R² nach unten nicht begrenzt.

R² vergleicht das Modell mit der einfachsten denkbaren Alternative: „sage
immer den Trainings-Mittelwert voraus“. Ist das Modell **schlechter** als
dieser naive Tipp, wird der Wert negativ – es gibt keine Untergrenze.

Bei SmartSupp liegt der Mittelwert über eine 5-fache stratifizierte
Kreuzvalidierung bei **R² ≈ −0,12** (Werte je Fold: −0,15, −0,09, −0,26,
+0,05, −0,14). Das ist kein Ausreißer eines einzelnen Splits, sondern über
fünf unabhängige Aufteilungen hinweg konsistent reproduzierbar. Ursache sind
vermutlich Tag-zu-Tag-Gewichtsschwankungen (Wassereinlagerung, Tagesform),
die in den vorhandenen Merkmalen (Kalorien, Schritte, Training) schlicht
nicht erfasst sind.

## Gemeinsames Fazit

Beide Beobachtungen bestätigen dieselbe Kernaussage der Präsentation: Nicht
die Modellwahl ist der limitierende Faktor, sondern die zugrunde liegende
Datenqualität – historische Konsumdaten ohne klare kausale Struktur zur
Zielvariable.
