# 🎨 KI & Schatten Hackathon - Luftbild Klassifikation

## 📋 Projektbeschreibung

Automatische Erkennung dominanter Objekte auf Luftbildern aus Bonn mithilfe von MobileNet und TensorFlow.js.

## 🛠️ Tech-Stack

- **Node.js** - Runtime Environment
- **TensorFlow.js** - Machine Learning Framework
- **MobileNet** - Vortrainiertes Klassifikationsmodell
- **Canvas** - Bildverarbeitung

## 🚀 Installation & Ausführung

### 1. Dependencies installieren
```bash
npm install
```

### 2. Projekt starten
```bash
npm start
```

## 📁 Projektstruktur

```
ki-und-schatten-hackathon/
├── images/
│   └── rohdaten/           # Luftbilder (50 JPG-Dateien)
├── index.js                # Hauptskript
├── package.json           # Dependencies
├── ergebnisse.csv         # Generierte Ergebnisse
└── README.md              # Diese Datei
```

## 🎯 Funktionsweise

1. **Bildladen**: Lädt alle JPG-Dateien aus `./images/rohdaten/`
2. **Skalierung**: Konvertiert Bilder auf 224×224 Pixel (MobileNet-Eingabeformat)
3. **Klassifikation**: Nutzt vortrainiertes MobileNet für Objekterkennung
4. **Export**: Speichert Ergebnisse als `ergebnisse.csv`

## 📊 Ausgabe

### CSV-Format
```csv
Bildname,Label,Wahrscheinlichkeit(%)
1_70501950.jpg,"residential area",87.45
2_70501950.jpg,"parking lot",92.18
...
```

### Konsolen-Output
- Live-Klassifikation jedes Bildes
- Fortschrittsanzeige
- Zusammenfassung der häufigsten Labels
- Durchschnittliche Konfidenz

## 🎯 Zielgruppe

- **Bürger**: Schnelle Kategorisierung von Stadtluftbildern
- **Entwickler**: Basis für weitere KI-Anwendungen
- **Stadtplanung**: Automatische Flächenanalyse

## ⚡ Besonderheiten

- **Kein Training nötig**: Nutzt vortrainiertes MobileNet
- **Inference-basiert**: Ideal für Hackathon-Zeitbudget
- **Memory-optimiert**: Automatische Tensor-Freigabe
- **Fehlerbehandlung**: Robuste Verarbeitung auch bei defekten Bildern

## 🔧 Technische Details

- **Eingabeformat**: 224×224×3 RGB
- **Modell**: MobileNet v1 (ImageNet-trainiert)
- **Batch-Verarbeitung**: Sequenziell für Memory-Effizienz
- **Canvas-Rendering**: Für präzise Bildskalierung

---

*Erstellt für den KI & Schatten Hackathon 2024* 