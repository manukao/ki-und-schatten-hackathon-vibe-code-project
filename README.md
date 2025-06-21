# 🎨 KI & Schatten Hackathon - Luftbild-Klassifikation mit Ultra-Detail-Analyse

## 📋 Projektübersicht

Dieses Projekt entwickelte sich von einer einfachen Luftbild-Klassifikation zu einem hochdetaillierten Analyse-System für Luftbilder aus Bonn. Mithilfe von **MobileNet** und **TensorFlow.js** können wir automatisch dominante Objekte erkennen und mit verschiedenen Visualisierungsansätzen darstellen.

## 🚀 Projektentwicklung - Unsere Reise

### Phase 1: Grundlegende Klassifikation (`index.js`)
- **Ziel**: Einfache Objekterkennung auf 50 Luftbildern
- **Ansatz**: Ein Label pro Bild mit MobileNet
- **Ergebnis**: CSV mit Bildname, Label und Konfidenz
- **Erkenntnisse**: 29.8% durchschnittliche Konfidenz, diverse Labels (Solarpanels, Berge, Puzzles)

### Phase 2: Fokussierte Kategorisierung (`annotate_simple.js`)
- **Ziel**: Konzentration auf relevante Kategorien
- **Innovation**: Mapping von MobileNet-Labels zu 3 Hauptkategorien:
  - 🌳 **Trees/Vegetation** (Bäume, Vegetation)
  - ⚡ **Solar/Technology** (Solarpanels, Technologie)
  - 🏢 **Buildings/Infrastructure** (Gebäude, Infrastruktur)
- **Verbesserung**: Top-5 Predictions statt nur Top-1
- **Ergebnis**: 94% high-confidence Klassifikationen

### Phase 3: Multi-Kategorie Detektion (`annotate_multi.js`)
- **Ziel**: Simultane Erkennung aller 3 Kategorien pro Bild
- **Ansatz**: 4-Quadranten-Grid-System
- **Visualisierung**: Farbkodierte Overlays mit Legende
- **Statistiken**: 91.7% Multi-Kategorie-Erfolg, 100% Vegetation-Erkennung

### Phase 4: Hochauflösende Analyse (`annotate_detail.js`)
- **Ziel**: Präzisere Objektlokalisierung
- **Innovation**: 4×4 Grid = 16 Mini-Quadranten pro Bild
- **Features**: 
  - Regionsbasierte Analyse (R1C1 bis R4C4)
  - Mini-Icons für platzsparende Annotation
  - Grid-Linien für räumliche Orientierung
- **Ergebnis**: 95.3% aktive Regionen, 87.5% Vollständigkeit

### Phase 5: Ultra-Detail-System (`annotate_ultra.js`)
- **Ziel**: Maximale Detailtiefe für Bäume und Gebäude
- **Revolution**: 8×8 Grid = 64 Ultra-Mini-Quadranten
- **Fokus**: Nur 2 Kategorien (Trees vs Buildings)
- **Technologie**: 
  - 1% Confidence-Threshold mit Keyword-Matching
  - Weighted Scoring: `probability × (1 + keywords × 0.3)`
  - Dominanz-Analyse pro Bild
- **Performance**: 58.1% aktive Regionen, 100% Erkennungsrate

### Phase 6: Skalierung auf alle Bilder (`annotate_ultra_alle.js`)
- **Ziel**: Vollständige Datensatz-Analyse
- **Herausforderung**: 50 Bilder × 64 Regionen = 3.200 Analysen
- **Optimierung**: Progress-Tracking, geschätzte 100 Minuten Laufzeit
- **Output**: Separate Ordner-Struktur für Vollanalyse

## 🛠️ Tech-Stack

- **Node.js** - Runtime Environment
- **TensorFlow.js** v3.21.0 - Machine Learning Framework
- **MobileNet** v2.1.0 - Vortrainiertes Klassifikationsmodell (~16MB)
- **Canvas** - Hochpräzise Bildverarbeitung
- **ImageNet** - 1.000 Kategorien Trainingsbasis

## 🚀 Installation & Ausführung

### 1. Dependencies installieren
```bash
npm install --legacy-peer-deps
```

### 2. Verschiedene Analyse-Modi ausführen

#### Basis-Klassifikation
```bash
node index.js
```

#### Fokussierte 3-Kategorie-Analyse
```bash
node annotate_simple.js
```

#### Multi-Kategorie mit Visualisierung
```bash
node annotate_multi.js
```

#### Hochauflösende 4×4 Grid-Analyse
```bash
node annotate_detail.js
```

#### Ultra-Detail 8×8 Grid-System
```bash
node annotate_ultra.js
```

#### Vollständige Datensatz-Analyse
```bash
node annotate_ultra_alle.js
```

## 📁 Projektstruktur

```
ki-und-schatten-hackathon/
├── images/
│   └── rohdaten/                    # 50 Luftbilder (JPG)
├── 
├── 🎯 HAUPTSKRIPTE:
├── index.js                         # Basis-Klassifikation
├── annotate_simple.js               # 3-Kategorie-System
├── annotate_multi.js                # Multi-Kategorie + Visualisierung
├── annotate_detail.js               # 4×4 Grid-System
├── annotate_ultra.js                # 8×8 Ultra-Detail-System
├── annotate_ultra_alle.js           # Vollanalyse aller Bilder
├── 
├── 📊 GENERIERTE AUSGABEN:
├── simple_multi_bilder/             # 3-Kategorie Visualisierungen
├── detail_multi_bilder/             # 4×4 Grid Visualisierungen  
├── ultra_detail_bilder/             # 8×8 Grid Visualisierungen
├── ultra_detail_bilder_alle_50/     # Vollanalyse-Visualisierungen
├── *.csv                           # Verschiedene Ergebnis-CSVs
├── 
├── 🔧 KONFIGURATION:
├── package.json                     # Dependencies
├── .gitignore                      # Git-Ausschlüsse
└── README.md                       # Diese Dokumentation
```

## 🎯 Analyse-Modi im Detail

### 1. **Basis-Modus** (`index.js`)
- **Ein Label pro Bild**
- **Einfache CSV-Ausgabe**
- **Schnell für Überblick**

### 2. **Smart-Kategorisierung** (`annotate_simple.js`)
- **3 Hauptkategorien**
- **Top-5 Predictions**
- **Intelligentes Label-Mapping**

### 3. **Multi-Kategorie** (`annotate_multi.js`)
- **4-Quadranten-System**
- **Simultane Kategorie-Erkennung**
- **Farbkodierte Visualisierung**

### 4. **Hochauflösung** (`annotate_detail.js`)
- **16 Mini-Regionen**
- **Präzise Lokalisierung**
- **Grid-basierte Koordinaten**

### 5. **Ultra-Detail** (`annotate_ultra.js`)
- **64 Ultra-Mini-Regionen**
- **Nur Trees vs Buildings**
- **Dominanz-Analyse**
- **Maximale Detailtiefe**

## 📊 Beispiel-Ergebnisse

### CSV-Format (Ultra-Detail)
```csv
Bild,Aktive_Regionen,Baum_Regionen,Gebäude_Regionen,Dominante_Kategorie,Confidence_Avg
1_70501950.jpg,42,28,14,Trees/Vegetation,0.234
2_70501950.jpg,38,18,20,Buildings/Concrete,0.198
```

### Visualisierung-Features
- **Farbkodierung**: Grün (Bäume), Rot (Gebäude), Blau (Solar)
- **Mini-Icons**: 🌳 🏢 ⚡ für platzsparende Darstellung
- **Grid-Linien**: Räumliche Orientierung
- **Legende**: Erklärung aller Symbole
- **Statistiken**: Zusammenfassung pro Bild

## 🔬 Technische Innovationen

### **Intelligentes Label-Mapping**
```javascript
const categoryKeywords = {
  'Trees/Vegetation': ['tree', 'forest', 'park', 'garden', 'lawn'],
  'Buildings/Infrastructure': ['building', 'house', 'roof', 'street'],
  'Solar/Technology': ['solar', 'panel', 'dish', 'antenna']
};
```

### **Weighted Scoring-System**
```javascript
const score = prediction.probability * (1 + matchingKeywords.length * 0.3);
```

### **Ultra-Hochauflösende Grid-Analyse**
- **8×8 = 64 Regionen** pro Bild
- **Region-Größe**: ~28×28 Pixel bei 224×224 Input
- **Skalierung**: Jede Region wird auf 224×224 skaliert
- **Batch-Processing**: Sequenziell für Memory-Effizienz

## 📈 Leistungsmetriken

| Modus | Regionen/Bild | Kategorien | Erkennungsrate | Laufzeit |
|-------|---------------|------------|----------------|----------|
| Basis | 1 | Alle | 100% | ~2 Min |
| Smart | 1 | 3 | 94% | ~3 Min |
| Multi | 4 | 3 | 91.7% | ~8 Min |
| Detail | 16 | 3 | 95.3% | ~25 Min |
| Ultra | 64 | 2 | 100% | ~100 Min |

## 🎯 Anwendungsfälle

### **Stadtplanung**
- Automatische Flächennutzungsanalyse
- Grünflächenverteilung
- Bebauungsdichte-Mapping

### **Umweltmonitoring**
- Vegetationsanteil-Tracking
- Versiegelungsgrad-Analyse
- Klimawandel-Auswirkungen

### **Forschung & Entwicklung**
- KI-Modell-Benchmarking
- Grid-basierte Analyse-Methoden
- Luftbild-Klassifikations-Pipeline

## ⚡ Besonderheiten

- **Kein Training erforderlich**: Nutzt vortrainiertes MobileNet
- **Skalierbare Auflösung**: Von 1 bis 64 Regionen pro Bild
- **Memory-optimiert**: Automatische Tensor-Bereinigung
- **Robuste Fehlerbehandlung**: Funktioniert auch bei problematischen Bildern
- **Modularer Aufbau**: Verschiedene Skripte für verschiedene Anwendungen
- **Visualisierung**: Automatische Generierung annotierter Bilder

## 🔧 Entwicklungshinweise

### **Git-Repository bereinigt**
- `node_modules/` entfernt (wird via `npm install` regeneriert)
- Generierte Bilder-Ordner in `.gitignore`
- Nur Quellcode und Rohdaten im Repository

### **Performance-Optimierungen**
- Sequenzielle Verarbeitung verhindert Memory-Overflow
- Canvas-basierte Skalierung für beste Qualität
- Automatische Garbage Collection zwischen Bildern

---

*Entwickelt für den KI & Schatten Hackathon 2024*  
*Von einfacher Klassifikation zu Ultra-Detail-Analyse* 🚀 