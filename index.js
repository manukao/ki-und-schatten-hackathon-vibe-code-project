// Polyfill für fetch in Node.js
global.fetch = require('node-fetch');

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

class LuftbildKlassifikation {
    constructor() {
        this.model = null;
        this.results = [];
        
        // Mapping von MobileNet-Labels zu unseren Zielkategorien
        this.categoryMapping = {
            // Bäume und Vegetation
            'trees': ['broccoli', 'artichoke', 'cauliflower', 'corn', 'acorn', 'leaf'],
            'vegetation': ['maze, labyrinth', 'alp', 'cliff, drop, drop-off', 'lakeside, lakeshore'],
            
            // Solaranlagen und technische Strukturen
            'solar': ['solar dish, solar collector, solar furnace', 'spotlight, spot'],
            'technical': ['radar, microwave radar', 'satellite dish', 'radio telescope'],
            
            // Gebäude und urbane Strukturen
            'buildings': ['castle', 'palace', 'monastery', 'church, church building'],
            'infrastructure': ['viaduct', 'suspension bridge', 'steel arch bridge', 'container ship, containership, container vessel'],
            'urban': ['parking lot', 'residential area', 'street sign', 'tile roof'],
            
            // Geometrische Muster (oft urbane Strukturen von oben)
            'geometric': ['jigsaw puzzle', 'envelope', 'maze, labyrinth', 'honeycomb']
        };
    }

    async init() {
        console.log('🚀 Lade MobileNet Modell...');
        this.model = await mobilenet.load();
        console.log('✅ MobileNet erfolgreich geladen!');
    }

    // Ordnet MobileNet-Labels unseren Zielkategorien zu
    kategorisiereLabel(label) {
        for (const [category, labels] of Object.entries(this.categoryMapping)) {
            if (labels.some(l => label.toLowerCase().includes(l.toLowerCase()))) {
                switch(category) {
                    case 'trees':
                    case 'vegetation':
                        return { category: 'Bäume/Vegetation', confidence: 'hoch' };
                    case 'solar':
                    case 'technical':
                        return { category: 'Solaranlagen/Technik', confidence: 'hoch' };
                    case 'buildings':
                    case 'infrastructure':
                    case 'urban':
                        return { category: 'Gebäude/Infrastruktur', confidence: 'hoch' };
                    case 'geometric':
                        return { category: 'Gebäude/Infrastruktur', confidence: 'mittel' };
                    default:
                        return { category: 'Unbekannt', confidence: 'niedrig' };
                }
            }
        }
        
        // Fallback: Versuche basierend auf häufigen Luftbild-Mustern zu kategorisieren
        if (label.includes('volcano') || label.includes('mountain') || label.includes('hill')) {
            return { category: 'Bäume/Vegetation', confidence: 'mittel' };
        }
        if (label.includes('roof') || label.includes('building') || label.includes('house')) {
            return { category: 'Gebäude/Infrastruktur', confidence: 'hoch' };
        }
        
        return { category: 'Unbekannt', confidence: 'niedrig' };
    }

    async ladeUndSkaliereBild(bildpfad) {
        try {
            // Bild laden
            const img = await loadImage(bildpfad);
            
            // Canvas erstellen und auf 224x224 skalieren (MobileNet-Anforderung)
            const canvas = createCanvas(224, 224);
            const ctx = canvas.getContext('2d');
            
            // Bild auf Canvas zeichnen (automatische Skalierung)
            ctx.drawImage(img, 0, 0, 224, 224);
            
            // Canvas zu Tensor konvertieren
            const imageData = ctx.getImageData(0, 0, 224, 224);
            // Erstelle einen 3D-Tensor aus den Pixeldaten
            const tensor = tf.tidy(() => {
                const pixels = tf.browser.fromPixels(canvas, 3);
                return pixels.expandDims(0); // Batch-Dimension hinzufügen
            });
            
            return tensor;
        } catch (error) {
            console.error(`❌ Fehler beim Laden des Bildes ${bildpfad}:`, error.message);
            return null;
        }
    }

    async klassifiziereBild(bildpfad, bildname) {
        console.log(`🔍 Klassifiziere: ${bildname}`);
        
        const tensor = await this.ladeUndSkaliereBild(bildpfad);
        if (!tensor) return;

        try {
            // Klassifikation durchführen (Top 5 Predictions für bessere Analyse)
            const predictions = await this.model.classify(tensor, 5);
            
            // Alle Predictions kategorisieren
            const kategorisiert = predictions.map(pred => ({
                ...pred,
                kategorisierung: this.kategorisiereLabel(pred.className)
            }));

            // Beste Kategorie für Bäume, Solar, Gebäude finden
            const relevante = kategorisiert.filter(pred => 
                pred.kategorisierung.category !== 'Unbekannt' && pred.probability > 0.05
            );

            let finalPrediction;
            let finalCategory;

            if (relevante.length > 0) {
                // Höchste Wahrscheinlichkeit unter den relevanten Kategorien
                finalPrediction = relevante[0];
                finalCategory = finalPrediction.kategorisierung.category;
            } else {
                // Fallback zur besten Prediction
                finalPrediction = predictions[0];
                finalCategory = this.kategorisiereLabel(finalPrediction.className).category;
            }
            
            // Ergebnis speichern
            this.results.push({
                bildname: bildname,
                originalLabel: finalPrediction.className,
                kategorie: finalCategory,
                wahrscheinlichkeit: (finalPrediction.probability * 100).toFixed(2),
                confidence: relevante.length > 0 ? 'hoch' : 'niedrig'
            });

            console.log(`   → ${finalCategory} | ${finalPrediction.className} (${(finalPrediction.probability * 100).toFixed(1)}%)`);
            
        } catch (error) {
            console.error(`❌ Klassifikationsfehler für ${bildname}:`, error.message);
        } finally {
            // Tensor freigeben um Memory Leaks zu vermeiden
            tensor.dispose();
        }
    }

    async verarbeiteAlleBilder() {
        const bilderOrdner = path.join(__dirname, 'images', 'rohdaten');
        
        try {
            // Alle JPG-Dateien im Ordner finden
            const dateien = fs.readdirSync(bilderOrdner)
                .filter(datei => datei.toLowerCase().endsWith('.jpg'))
                .sort(); // Sortierung für konsistente Reihenfolge
            
            console.log(`📁 Gefunden: ${dateien.length} Bilder in ${bilderOrdner}`);
            console.log('🎯 Starte Klassifikation...\n');

            // Jedes Bild verarbeiten
            for (let i = 0; i < dateien.length; i++) {
                const dateiname = dateien[i];
                const bildpfad = path.join(bilderOrdner, dateiname);
                
                console.log(`[${i + 1}/${dateien.length}]`);
                await this.klassifiziereBild(bildpfad, dateiname);
            }
            
        } catch (error) {
            console.error('❌ Fehler beim Lesen des Bilderordners:', error.message);
        }
    }

    speichereErgebnisse() {
        if (this.results.length === 0) {
            console.log('⚠️ Keine Ergebnisse zum Speichern vorhanden');
            return;
        }

        // CSV Header für fokussierte Kategorien
        let csvContent = 'Bildname,Kategorie,Original_Label,Wahrscheinlichkeit(%),Confidence\n';
        
        // Daten hinzufügen
        this.results.forEach(result => {
            csvContent += `${result.bildname},"${result.kategorie}","${result.originalLabel}",${result.wahrscheinlichkeit},${result.confidence}\n`;
        });

        // CSV Datei schreiben
        const csvDatei = path.join(__dirname, 'ergebnisse_fokussiert.csv');
        fs.writeFileSync(csvDatei, csvContent, 'utf8');
        
        console.log(`\n✅ Ergebnisse gespeichert in: ${csvDatei}`);
        console.log(`📊 Insgesamt ${this.results.length} Bilder klassifiziert`);
    }

    zeigeZusammenfassung() {
        console.log('\n📈 FOKUSSIERTE ANALYSE - BÄUME, SOLAR & GEBÄUDE:');
        console.log('=================================================');
        
        // Kategorien zählen
        const kategorienCounts = {};
        this.results.forEach(result => {
            const kategorie = result.kategorie;
            kategorienCounts[kategorie] = (kategorienCounts[kategorie] || 0) + 1;
        });

        // Kategorien-Verteilung anzeigen
        console.log('🎯 Erkannte Kategorien:');
        Object.entries(kategorienCounts)
            .sort(([,a], [,b]) => b - a)
            .forEach(([kategorie, count]) => {
                const percentage = ((count / this.results.length) * 100).toFixed(1);
                let icon = '📊';
                if (kategorie.includes('Bäume')) icon = '🌳';
                else if (kategorie.includes('Solar')) icon = '☀️';
                else if (kategorie.includes('Gebäude')) icon = '🏢';
                else if (kategorie.includes('Unbekannt')) icon = '❓';
                
                console.log(`${icon} ${kategorie}: ${count}x (${percentage}%)`);
            });

        // High-Confidence Ergebnisse
        const highConfidence = this.results.filter(r => r.confidence === 'hoch');
        console.log(`\n✅ Zuverlässige Erkennungen: ${highConfidence.length}/${this.results.length}`);

        // Spezifische Insights
        const bäume = this.results.filter(r => r.kategorie.includes('Bäume'));
        const solar = this.results.filter(r => r.kategorie.includes('Solar'));
        const gebäude = this.results.filter(r => r.kategorie.includes('Gebäude'));

        console.log('\n🔍 Detailanalyse:');
        console.log(`🌳 Bäume/Vegetation: ${bäume.length} Bilder`);
        console.log(`☀️ Solaranlagen/Technik: ${solar.length} Bilder`);
        console.log(`🏢 Gebäude/Infrastruktur: ${gebäude.length} Bilder`);

        // Durchschnittliche Konfidenz
        const avgConfidence = this.results.reduce((sum, result) => 
            sum + parseFloat(result.wahrscheinlichkeit), 0) / this.results.length;
        
        console.log(`\n🎯 Durchschnittliche Konfidenz: ${avgConfidence.toFixed(1)}%`);
        console.log(`📊 Tipp: Ergebnisse mit "hoch" Confidence sind zuverlässiger für Luftbildanalyse`);
    }
}

// Hauptfunktion
async function main() {
    console.log('🎨 KI & Schatten Hackathon - Luftbild Klassifikation');
    console.log('=====================================================\n');
    
    const classifier = new LuftbildKlassifikation();
    
    try {
        // Modell initialisieren
        await classifier.init();
        
        // Alle Bilder verarbeiten
        await classifier.verarbeiteAlleBilder();
        
        // Ergebnisse speichern
        classifier.speichereErgebnisse();
        
        // Zusammenfassung anzeigen
        classifier.zeigeZusammenfassung();
        
        console.log('\n🎉 Klassifikation erfolgreich abgeschlossen!');
        
    } catch (error) {
        console.error('💥 Fataler Fehler:', error.message);
        process.exit(1);
    }
}

// Programm starten
if (require.main === module) {
    main().catch(console.error);
}

module.exports = LuftbildKlassifikation; 