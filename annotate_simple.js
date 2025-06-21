// Polyfill für fetch in Node.js
global.fetch = require('node-fetch');

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

class VereinfachteMultiKategorieAnalyse {
    constructor() {
        this.mobilenetModel = null;
        this.results = [];
        
        // Hauptkategorien mit spezifischen Farben und erweiterten Keywords
        this.hauptkategorien = {
            'Bäume/Vegetation': {
                color: '#228B22',  // Forest Green
                icon: '🌳',
                keywords: [
                    'broccoli', 'artichoke', 'cauliflower', 'corn', 'acorn', 'leaf',
                    'maze, labyrinth', 'alp', 'cliff, drop, drop-off', 'lakeside, lakeshore',
                    'volcano', 'mountain', 'hill', 'forest', 'tree', 'plant', 'grass',
                    'jungle', 'field', 'garden', 'park', 'nature'
                ]
            },
            'Solaranlagen/Technik': {
                color: '#FFD700',  // Gold
                icon: '☀️',
                keywords: [
                    'solar dish, solar collector, solar furnace', 'spotlight, spot',
                    'radar, microwave radar', 'satellite dish', 'radio telescope',
                    'antenna', 'panel', 'dish', 'solar', 'technical', 'machinery',
                    'equipment', 'installation', 'technology'
                ]
            },
            'Gebäude/Infrastruktur': {
                color: '#DC143C',  // Crimson
                icon: '🏢',
                keywords: [
                    'castle', 'palace', 'monastery', 'church, church building',
                    'viaduct', 'suspension bridge', 'steel arch bridge',
                    'container ship, containership, container vessel',
                    'parking lot', 'residential area', 'street sign', 'tile roof',
                    'jigsaw puzzle', 'envelope', 'honeycomb', 'roof', 'building',
                    'house', 'structure', 'architecture', 'urban', 'city',
                    'construction', 'infrastructure', 'bridge', 'road'
                ]
            }
        };
    }

    async init() {
        console.log('🚀 Lade vereinfachtes Multi-Kategorie-System...');
        console.log('   📊 MobileNet für intelligente Kategorisierung...');
        this.mobilenetModel = await mobilenet.load();
        console.log('✅ MobileNet erfolgreich geladen!');
    }

    // Analysiert Bildregionen für verschiedene Kategorien
    async analysiereRegionen(img) {
        const regionen = [];
        const originalWidth = img.width;
        const originalHeight = img.height;
        
        // Teile Bild in 4 Quadranten für bessere Performance
        const regionWidth = Math.floor(originalWidth / 2);
        const regionHeight = Math.floor(originalHeight / 2);
        
        const quadranten = [
            { name: 'Oben-Links', x: 0, y: 0 },
            { name: 'Oben-Rechts', x: regionWidth, y: 0 },
            { name: 'Unten-Links', x: 0, y: regionHeight },
            { name: 'Unten-Rechts', x: regionWidth, y: regionHeight }
        ];
        
        for (let i = 0; i < quadranten.length; i++) {
            const quad = quadranten[i];
            const width = (i === 1 || i === 3) ? originalWidth - quad.x : regionWidth;
            const height = (i === 2 || i === 3) ? originalHeight - quad.y : regionHeight;
            
            // Region-Canvas erstellen
            const regionCanvas = createCanvas(224, 224);
            const regionCtx = regionCanvas.getContext('2d');
            
            // Bildausschnitt in Region kopieren
            regionCtx.drawImage(img, quad.x, quad.y, width, height, 0, 0, 224, 224);
            
            // Region analysieren
            const tensor = tf.tidy(() => {
                const pixels = tf.browser.fromPixels(regionCanvas, 3);
                return pixels.expandDims(0);
            });
            
            const predictions = await this.mobilenetModel.classify(tensor, 10);
            tensor.dispose();
            
            // Kategorien für diese Region ermitteln
            const kategorien = this.kategorisiereRegion(predictions);
            
            regionen.push({
                name: quad.name,
                x: quad.x, 
                y: quad.y, 
                width: width, 
                height: height,
                kategorien: kategorien,
                topPredictions: predictions.slice(0, 3),
                confidence: predictions.length > 0 ? predictions[0].probability : 0
            });
        }
        
        return regionen;
    }

    // Zusätzliche Gesamtbild-Analyse
    async analysiereGesamtbild(img) {
        // Gesamtbild analysieren
        const canvas = createCanvas(224, 224);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 224, 224);
        
        const tensor = tf.tidy(() => {
            const pixels = tf.browser.fromPixels(canvas, 3);
            return pixels.expandDims(0);
        });
        
        const predictions = await this.mobilenetModel.classify(tensor, 15);
        tensor.dispose();
        
        return this.kategorisiereRegion(predictions);
    }

    kategorisiereRegion(predictions) {
        const erkannteKategorien = new Set();
        const kategorieScores = {};
        
        predictions.forEach(pred => {
            const label = pred.className.toLowerCase();
            
            // Prüfe jede Hauptkategorie
            for (const [kategorieName, kategorieData] of Object.entries(this.hauptkategorien)) {
                const matchingKeywords = kategorieData.keywords.filter(keyword => 
                    label.includes(keyword.toLowerCase())
                );
                
                if (matchingKeywords.length > 0 && pred.probability > 0.03) {
                    erkannteKategorien.add(kategorieName);
                    
                    // Scoring für bessere Kategoriezuordnung
                    if (!kategorieScores[kategorieName]) {
                        kategorieScores[kategorieName] = 0;
                    }
                    kategorieScores[kategorieName] += pred.probability * matchingKeywords.length;
                }
            }
        });
        
        return {
            kategorien: Array.from(erkannteKategorien),
            scores: kategorieScores
        };
    }

    async analysiereUndAnnotiereBild(bildpfad, bildname) {
        console.log(`🔍 Multi-Kategorie-Analyse: ${bildname}`);
        
        try {
            // Bild laden
            const img = await loadImage(bildpfad);
            
            // 1. Region-basierte Analyse (4 Quadranten)
            const regionen = await this.analysiereRegionen(img);
            
            // 2. Gesamtbild-Analyse
            const gesamtAnalyse = await this.analysiereGesamtbild(img);
            
            // 3. Alle erkannten Kategorien sammeln
            const alleKategorien = new Set();
            const kategorieRegionen = {};
            
            // Kategorien aus Regionen sammeln
            regionen.forEach(region => {
                region.kategorien.kategorien.forEach(kat => {
                    alleKategorien.add(kat);
                    if (!kategorieRegionen[kat]) {
                        kategorieRegionen[kat] = [];
                    }
                    kategorieRegionen[kat].push(region.name);
                });
            });
            
            // Kategorien aus Gesamtanalyse hinzufügen
            gesamtAnalyse.kategorien.forEach(kat => alleKategorien.add(kat));
            
            // 4. Annotiertes Bild erstellen
            const annotatedPath = await this.erstelleAnnotiertesBild(
                img, 
                regionen, 
                Array.from(alleKategorien),
                kategorieRegionen,
                bildname
            );
            
            // 5. Ergebnis speichern
            this.results.push({
                bildname: bildname,
                annotatedPath: annotatedPath,
                erkannteKategorien: Array.from(alleKategorien),
                kategorieRegionen: kategorieRegionen,
                regionenAnalyse: regionen.map(r => ({
                    region: r.name,
                    kategorien: r.kategorien.kategorien,
                    confidence: r.confidence,
                    topPrediction: r.topPredictions[0]?.className || 'Unbekannt'
                })),
                gesamtAnalyse: gesamtAnalyse
            });

            console.log(`   → ${alleKategorien.size} Hauptkategorien erkannt: ${Array.from(alleKategorien).join(', ')}`);
            if (Object.keys(kategorieRegionen).length > 0) {
                console.log(`   → Regionen-Verteilung: ${Object.entries(kategorieRegionen).map(([kat, regionen]) => 
                    `${this.hauptkategorien[kat]?.icon}${kat.split('/')[0]}(${regionen.length})`
                ).join(', ')}`);
            }
            
        } catch (error) {
            console.error(`❌ Analysefehler für ${bildname}:`, error.message);
        }
    }

    async erstelleAnnotiertesBild(originalImg, regionen, kategorien, kategorieRegionen, bildname) {
        const canvas = createCanvas(originalImg.width, originalImg.height);
        const ctx = canvas.getContext('2d');
        
        // Originalbild zeichnen
        ctx.drawImage(originalImg, 0, 0);
        
        // 1. Region-Overlays zeichnen
        regionen.forEach(region => {
            if (region.kategorien.kategorien.length > 0) {
                // Für jede erkannte Kategorie in der Region
                region.kategorien.kategorien.forEach((kategorie, index) => {
                    const kategorieData = this.hauptkategorien[kategorie];
                    if (kategorieData) {
                        // Semi-transparente Überlagerung
                        ctx.fillStyle = kategorieData.color + '30'; // 30 = ~20% opacity
                        ctx.fillRect(region.x, region.y, region.width, region.height);
                        
                        // Rahmen um Region
                        ctx.strokeStyle = kategorieData.color;
                        ctx.lineWidth = 4;
                        ctx.strokeRect(region.x, region.y, region.width, region.height);
                        
                        // Kategorie-Label in der Region
                        const labelY = region.y + 30 + (index * 25);
                        const labelText = `${kategorieData.icon} ${kategorie.split('/')[0]}`;
                        
                        // Hintergrund für Label
                        ctx.font = 'bold 14px Arial';
                        const textWidth = ctx.measureText(labelText).width;
                        ctx.fillStyle = kategorieData.color;
                        ctx.fillRect(region.x + 10, labelY - 18, textWidth + 10, 22);
                        
                        // Label-Text
                        ctx.fillStyle = 'white';
                        ctx.fillText(labelText, region.x + 15, labelY - 2);
                    }
                });
                
                // Region-Name
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(region.x + 5, region.y + 5, 120, 20);
                ctx.fillStyle = 'white';
                ctx.font = 'bold 12px Arial';
                ctx.fillText(region.name, region.x + 10, region.y + 18);
            }
        });

        // 2. Legende erstellen
        if (kategorien.length > 0) {
            const legendeY = 20;
            let legendeX = originalImg.width - 280;
            
            // Hintergrund für Legende
            ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
            ctx.fillRect(legendeX - 10, legendeY - 10, 270, (kategorien.length * 30) + 50);
            
            // Legende-Titel
            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('🎯 Erkannte Kategorien:', legendeX, legendeY + 15);
            
            // Kategorien in Legende mit Regionen-Info
            kategorien.forEach((kategorie, index) => {
                const kategorieData = this.hauptkategorien[kategorie];
                if (kategorieData) {
                    const y = legendeY + 50 + (index * 30);
                    
                    // Farbbox
                    ctx.fillStyle = kategorieData.color;
                    ctx.fillRect(legendeX, y - 18, 18, 18);
                    
                    // Kategorie-Text
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px Arial';
                    ctx.fillText(`${kategorieData.icon} ${kategorie}`, legendeX + 25, y - 5);
                    
                    // Regionen-Info
                    if (kategorieRegionen[kategorie]) {
                        ctx.font = '11px Arial';
                        ctx.fillStyle = '#CCCCCC';
                        ctx.fillText(`Regionen: ${kategorieRegionen[kategorie].join(', ')}`, legendeX + 25, y + 8);
                    }
                }
            });
        }

        // 3. Statistik-Header
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(10, 10, 350, 40);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`🎯 Multi-Kategorie: ${kategorien.length} Kategorien erkannt`, 15, 30);
        ctx.font = '12px Arial';
        ctx.fillText(`Region-basierte Analyse mit 4 Quadranten`, 15, 45);

        // Annotiertes Bild speichern
        const outputDir = path.join(__dirname, 'simple_multi_bilder');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }

        const outputPath = path.join(outputDir, `simple_multi_${bildname.replace('.jpg', '.png')}`);
        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(outputPath, buffer);

        return outputPath;
    }

    async verarbeiteAlleBilder() {
        const bilderOrdner = path.join(__dirname, 'images', 'rohdaten');
        
        try {
            const dateien = fs.readdirSync(bilderOrdner)
                .filter(datei => datei.toLowerCase().endsWith('.jpg'))
                .sort()
                .slice(0, 12); // Erste 12 Bilder
            
            console.log(`📁 Vereinfachte Multi-Kategorie-Analyse für ${dateien.length} Bilder`);
            console.log('🎯 Region-basierte Erkennung: Bäume, Solar & Gebäude...\n');

            for (let i = 0; i < dateien.length; i++) {
                const dateiname = dateien[i];
                const bildpfad = path.join(bilderOrdner, dateiname);
                
                console.log(`[${i + 1}/${dateien.length}]`);
                await this.analysiereUndAnnotiereBild(bildpfad, dateiname);
            }
            
        } catch (error) {
            console.error('❌ Fehler bei der vereinfachten Multi-Kategorie-Analyse:', error.message);
        }
    }

    speichereErgebnisse() {
        if (this.results.length === 0) {
            console.log('⚠️ Keine Ergebnisse zum Speichern vorhanden');
            return;
        }

        // CSV mit Multi-Kategorie-Daten
        let csvContent = 'Bildname,Erkannte_Kategorien,Kategorien_Anzahl,Bäume_Vegetation,Solar_Technik,Gebäude_Infrastruktur,Regionen_Details,Annotiert_Pfad\n';
        
        this.results.forEach(result => {
            const hasBäume = result.erkannteKategorien.includes('Bäume/Vegetation') ? 'JA' : 'NEIN';
            const hasSolar = result.erkannteKategorien.includes('Solaranlagen/Technik') ? 'JA' : 'NEIN';
            const hasGebäude = result.erkannteKategorien.includes('Gebäude/Infrastruktur') ? 'JA' : 'NEIN';
            
            const regionenDetails = Object.entries(result.kategorieRegionen).map(([kat, regionen]) => 
                `${kat}:[${regionen.join(',')}]`
            ).join('; ');
            
            csvContent += `${result.bildname},"${result.erkannteKategorien.join('; ')}",${result.erkannteKategorien.length},${hasBäume},${hasSolar},${hasGebäude},"${regionenDetails}",${result.annotatedPath}\n`;
        });

        const csvDatei = path.join(__dirname, 'simple_multi_ergebnisse.csv');
        fs.writeFileSync(csvDatei, csvContent, 'utf8');
        
        console.log(`\n✅ Multi-Kategorie-Ergebnisse gespeichert in: ${csvDatei}`);
        console.log(`🖼️ Annotierte Bilder gespeichert in: ./simple_multi_bilder/`);
    }

    zeigeZusammenfassung() {
        console.log('\n📈 VEREINFACHTE MULTI-KATEGORIE-ANALYSE - ZUSAMMENFASSUNG:');
        console.log('============================================================');
        
        // Kategorie-Statistiken
        const bäumeCount = this.results.filter(r => r.erkannteKategorien.includes('Bäume/Vegetation')).length;
        const solarCount = this.results.filter(r => r.erkannteKategorien.includes('Solaranlagen/Technik')).length;
        const gebäudeCount = this.results.filter(r => r.erkannteKategorien.includes('Gebäude/Infrastruktur')).length;
        const multiKategorieCount = this.results.filter(r => r.erkannteKategorien.length > 1).length;
        const alleKategorienCount = this.results.filter(r => r.erkannteKategorien.length === 3).length;

        console.log('🎯 Kategorie-Verteilung:');
        console.log(`🌳 Bäume/Vegetation erkannt: ${bäumeCount}/${this.results.length} Bilder (${(bäumeCount/this.results.length*100).toFixed(1)}%)`);
        console.log(`☀️ Solaranlagen/Technik erkannt: ${solarCount}/${this.results.length} Bilder (${(solarCount/this.results.length*100).toFixed(1)}%)`);
        console.log(`🏢 Gebäude/Infrastruktur erkannt: ${gebäudeCount}/${this.results.length} Bilder (${(gebäudeCount/this.results.length*100).toFixed(1)}%)`);
        
        console.log('\n🎭 Multi-Kategorie-Erfolg:');
        console.log(`📊 Bilder mit mehreren Kategorien: ${multiKategorieCount}/${this.results.length} (${(multiKategorieCount/this.results.length*100).toFixed(1)}%)`);
        console.log(`🎯 Bilder mit allen 3 Hauptkategorien: ${alleKategorienCount}/${this.results.length} (${(alleKategorienCount/this.results.length*100).toFixed(1)}%)`);

        // Region-basierte Statistiken
        const regionStats = {};
        this.results.forEach(result => {
            result.regionenAnalyse.forEach(region => {
                if (!regionStats[region.region]) {
                    regionStats[region.region] = { kategorienGesamt: 0, bilder: 0 };
                }
                regionStats[region.region].kategorienGesamt += region.kategorien.length;
                if (region.kategorien.length > 0) {
                    regionStats[region.region].bilder++;
                }
            });
        });

        console.log('\n🗂️ Region-basierte Analyse:');
        Object.entries(regionStats).forEach(([region, stats]) => {
            console.log(`   ${region}: ${stats.bilder}/${this.results.length} Bilder mit Kategorien`);
        });

        // Detailierte Aufschlüsselung der erfolgreichsten Bilder
        const erfolgreicheBilder = this.results.filter(r => r.erkannteKategorien.length >= 2).slice(0, 5);
        if (erfolgreicheBilder.length > 0) {
            console.log('\n🏆 Top Multi-Kategorie-Erkennungen:');
            erfolgreicheBilder.forEach(result => {
                console.log(`   ${result.bildname}: [${result.erkannteKategorien.map(k => 
                    this.hauptkategorien[k]?.icon + k.split('/')[0]
                ).join(' + ')}]`);
            });
        }

        console.log('\n🎨 Visualisierung-Features:');
        console.log('   🗂️ 4-Quadranten-Analyse (Oben-Links, Oben-Rechts, Unten-Links, Unten-Rechts)');
        console.log('   🎨 Farbkodierte Überlagerungen pro Kategorie');
        console.log('   🏷️ Detaillierte Legende mit Regionen-Zuordnung');
        console.log('   📊 Erweiterte Keyword-Erkennung für bessere Kategorisierung');
        console.log('   🎯 Kombinierte Region- und Gesamtbild-Analyse');
        
        console.log(`\n📁 Alle annotierten Bilder: ./simple_multi_bilder/`);
    }
}

// Hauptfunktion
async function main() {
    console.log('🎨 KI & Schatten Hackathon - Vereinfachte Multi-Kategorie-Analyse');
    console.log('==================================================================');
    console.log('🎯 Region-basierte Erkennung: Bäume, Solar & Gebäude');
    console.log('🔧 Optimiert für vorhandene Dependencies');
    console.log('================================================================\n');
    
    const analyzer = new VereinfachteMultiKategorieAnalyse();
    
    try {
        await analyzer.init();
        await analyzer.verarbeiteAlleBilder();
        analyzer.speichereErgebnisse();
        analyzer.zeigeZusammenfassung();
        
        console.log('\n🎉 Vereinfachte Multi-Kategorie-Analyse erfolgreich abgeschlossen!');
        
    } catch (error) {
        console.error('💥 Fataler Fehler:', error.message);
        process.exit(1);
    }
}

// Programm starten
if (require.main === module) {
    main().catch(console.error);
}

module.exports = VereinfachteMultiKategorieAnalyse; 