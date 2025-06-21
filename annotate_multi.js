// Polyfill für fetch in Node.js
global.fetch = require('node-fetch');

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

class MultiKategorieAnalyse {
    constructor() {
        this.mobilenetModel = null;
        this.cocoSsdModel = null;
        this.results = [];
        
        // Hauptkategorien mit spezifischen Farben
        this.hauptkategorien = {
            'Bäume/Vegetation': {
                color: '#228B22',  // Forest Green
                icon: '🌳',
                keywords: ['broccoli', 'artichoke', 'cauliflower', 'corn', 'acorn', 'leaf', 'maze, labyrinth', 'alp', 'cliff, drop, drop-off', 'lakeside, lakeshore', 'volcano', 'mountain', 'hill']
            },
            'Solaranlagen/Technik': {
                color: '#FFD700',  // Gold
                icon: '☀️',
                keywords: ['solar dish, solar collector, solar furnace', 'spotlight, spot', 'radar, microwave radar', 'satellite dish', 'radio telescope']
            },
            'Gebäude/Infrastruktur': {
                color: '#DC143C',  // Crimson
                icon: '🏢',
                keywords: ['castle', 'palace', 'monastery', 'church, church building', 'viaduct', 'suspension bridge', 'steel arch bridge', 'container ship, containership, container vessel', 'parking lot', 'residential area', 'street sign', 'tile roof', 'jigsaw puzzle', 'envelope', 'honeycomb', 'roof', 'building', 'house']
            }
        };

        // COCO-SSD Objekte für zusätzliche Erkennung
        this.cocoKategorien = {
            'person': { kategorie: 'Menschen', color: '#FF6B6B' },
            'car': { kategorie: 'Fahrzeuge', color: '#4ECDC4' },
            'truck': { kategorie: 'Fahrzeuge', color: '#4ECDC4' },
            'bus': { kategorie: 'Fahrzeuge', color: '#4ECDC4' },
            'motorcycle': { kategorie: 'Fahrzeuge', color: '#4ECDC4' },
            'bicycle': { kategorie: 'Fahrzeuge', color: '#4ECDC4' },
            'airplane': { kategorie: 'Flugzeuge', color: '#9B59B6' },
            'boat': { kategorie: 'Boote', color: '#3498DB' }
        };
    }

    async init() {
        console.log('🚀 Lade Multi-Kategorie-Analysesystem...');
        console.log('   📊 MobileNet für Kategorisierung...');
        this.mobilenetModel = await mobilenet.load();
        
        console.log('   🎯 COCO-SSD für Objekterkennung...');
        this.cocoSsdModel = await cocoSsd.load();
        
        console.log('✅ Alle Modelle erfolgreich geladen!');
    }

    // Analysiert Bildregionen für verschiedene Kategorien
    async analysiereRegionen(img) {
        const regionen = [];
        const originalWidth = img.width;
        const originalHeight = img.height;
        
        // Teile Bild in 9 Regionen (3x3 Grid)
        const regionWidth = Math.floor(originalWidth / 3);
        const regionHeight = Math.floor(originalHeight / 3);
        
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                const x = col * regionWidth;
                const y = row * regionHeight;
                const width = (col === 2) ? originalWidth - x : regionWidth;
                const height = (row === 2) ? originalHeight - y : regionHeight;
                
                // Region-Canvas erstellen
                const regionCanvas = createCanvas(224, 224);
                const regionCtx = regionCanvas.getContext('2d');
                
                // Bildausschnitt in Region kopieren
                regionCtx.drawImage(img, x, y, width, height, 0, 0, 224, 224);
                
                // Region analysieren
                const tensor = tf.tidy(() => {
                    const pixels = tf.browser.fromPixels(regionCanvas, 3);
                    return pixels.expandDims(0);
                });
                
                const predictions = await this.mobilenetModel.classify(tensor, 5);
                tensor.dispose();
                
                // Beste Kategorie für diese Region finden
                const kategorien = this.kategorisiereRegion(predictions);
                
                regionen.push({
                    x, y, width, height,
                    row, col,
                    kategorien: kategorien,
                    predictions: predictions
                });
            }
        }
        
        return regionen;
    }

    kategorisiereRegion(predictions) {
        const erkannteKategorien = new Set();
        
        predictions.forEach(pred => {
            const label = pred.className.toLowerCase();
            
            // Prüfe jede Hauptkategorie
            for (const [kategorieName, kategorieData] of Object.entries(this.hauptkategorien)) {
                if (kategorieData.keywords.some(keyword => 
                    label.includes(keyword.toLowerCase()) && pred.probability > 0.05
                )) {
                    erkannteKategorien.add(kategorieName);
                }
            }
        });
        
        return Array.from(erkannteKategorien);
    }

    async analysiereUndAnnotiereBild(bildpfad, bildname) {
        console.log(`🔍 Multi-Kategorie-Analyse: ${bildname}`);
        
        try {
            // Bild laden
            const img = await loadImage(bildpfad);
            
            // 1. Region-basierte Analyse
            const regionen = await this.analysiereRegionen(img);
            
            // 2. COCO-SSD Objekterkennung
            const objektDetektionen = await this.cocoSsdModel.detect(img);
            const relevanteObjekte = objektDetektionen.filter(d => 
                d.score > 0.4 && this.cocoKategorien[d.class]
            );
            
            // 3. Alle erkannten Kategorien sammeln
            const alleKategorien = new Set();
            regionen.forEach(region => {
                region.kategorien.forEach(kat => alleKategorien.add(kat));
            });
            
            // 4. Annotiertes Bild erstellen
            const annotatedPath = await this.erstelleMultiKategorienBild(
                img, 
                regionen, 
                relevanteObjekte, 
                Array.from(alleKategorien),
                bildname
            );
            
            // 5. Ergebnis speichern
            this.results.push({
                bildname: bildname,
                annotatedPath: annotatedPath,
                erkannteKategorien: Array.from(alleKategorien),
                regionenAnalyse: regionen.map(r => ({
                    position: `${r.row},${r.col}`,
                    kategorien: r.kategorien,
                    topPrediction: r.predictions[0]
                })),
                objektDetektionen: relevanteObjekte.length,
                detektierteObjekte: relevanteObjekte.map(o => ({
                    objekt: o.class,
                    confidence: (o.score * 100).toFixed(1),
                    kategorie: this.cocoKategorien[o.class]?.kategorie
                }))
            });

            console.log(`   → ${alleKategorien.size} Hauptkategorien erkannt: ${Array.from(alleKategorien).join(', ')}`);
            console.log(`   → ${relevanteObjekte.length} Objekte detektiert`);
            
        } catch (error) {
            console.error(`❌ Analysefehler für ${bildname}:`, error.message);
        }
    }

    async erstelleMultiKategorienBild(originalImg, regionen, objekte, kategorien, bildname) {
        const canvas = createCanvas(originalImg.width, originalImg.height);
        const ctx = canvas.getContext('2d');
        
        // Originalbild zeichnen
        ctx.drawImage(originalImg, 0, 0);
        
        // 1. Regionen-Overlays zeichnen
        regionen.forEach(region => {
            if (region.kategorien.length > 0) {
                // Für jede erkannte Kategorie in der Region
                region.kategorien.forEach((kategorie, index) => {
                    const kategorieData = this.hauptkategorien[kategorie];
                    if (kategorieData) {
                        // Semi-transparente Überlagerung
                        ctx.fillStyle = kategorieData.color + '40'; // 40 = 25% opacity
                        ctx.fillRect(region.x, region.y, region.width, region.height);
                        
                        // Rahmen um Region
                        ctx.strokeStyle = kategorieData.color;
                        ctx.lineWidth = 3;
                        ctx.strokeRect(region.x, region.y, region.width, region.height);
                        
                        // Kategorie-Label
                        const labelY = region.y + 25 + (index * 25);
                        ctx.fillStyle = kategorieData.color;
                        ctx.fillRect(region.x + 5, labelY - 20, 150, 22);
                        ctx.fillStyle = 'white';
                        ctx.font = 'bold 14px Arial';
                        ctx.fillText(`${kategorieData.icon} ${kategorie}`, region.x + 8, labelY - 5);
                    }
                });
            }
        });

        // 2. COCO-SSD Objekte zeichnen
        objekte.forEach((objekt, index) => {
            const [x, y, width, height] = objekt.bbox;
            const cocoData = this.cocoKategorien[objekt.class];
            
            if (cocoData) {
                // Bounding Box
                ctx.strokeStyle = cocoData.color;
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, width, height);
                
                // Objekt-Label
                const labelText = `${objekt.class} ${(objekt.score * 100).toFixed(0)}%`;
                ctx.font = 'bold 12px Arial';
                const textWidth = ctx.measureText(labelText).width;
                
                ctx.fillStyle = cocoData.color;
                ctx.fillRect(x, y - 18, textWidth + 6, 18);
                ctx.fillStyle = 'white';
                ctx.fillText(labelText, x + 3, y - 5);
            }
        });

        // 3. Legende erstellen
        const legendeY = 20;
        let legendeX = originalImg.width - 250;
        
        // Hintergrund für Legende
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(legendeX - 10, legendeY - 10, 240, (kategorien.length * 25) + 40);
        
        // Legende-Titel
        ctx.fillStyle = 'white';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('Erkannte Kategorien:', legendeX, legendeY + 15);
        
        // Kategorien in Legende
        kategorien.forEach((kategorie, index) => {
            const kategorieData = this.hauptkategorien[kategorie];
            if (kategorieData) {
                const y = legendeY + 40 + (index * 25);
                
                // Farbbox
                ctx.fillStyle = kategorieData.color;
                ctx.fillRect(legendeX, y - 15, 15, 15);
                
                // Text
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.fillText(`${kategorieData.icon} ${kategorie}`, legendeX + 20, y - 5);
            }
        });

        // 4. Statistik-Header
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(10, 10, 300, 35);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`🎯 ${kategorien.length} Kategorien | ${objekte.length} Objekte`, 15, 32);

        // Annotiertes Bild speichern
        const outputDir = path.join(__dirname, 'multi_annotierte_bilder');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }

        const outputPath = path.join(outputDir, `multi_${bildname.replace('.jpg', '.png')}`);
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
                .slice(0, 10); // Erste 10 Bilder für Demo
            
            console.log(`📁 Multi-Kategorie-Analyse für ${dateien.length} Bilder`);
            console.log('🎯 Starte simultane Erkennung aller Hauptkategorien...\n');

            for (let i = 0; i < dateien.length; i++) {
                const dateiname = dateien[i];
                const bildpfad = path.join(bilderOrdner, dateiname);
                
                console.log(`[${i + 1}/${dateien.length}]`);
                await this.analysiereUndAnnotiereBild(bildpfad, dateiname);
            }
            
        } catch (error) {
            console.error('❌ Fehler bei der Multi-Kategorie-Analyse:', error.message);
        }
    }

    speichereErgebnisse() {
        if (this.results.length === 0) {
            console.log('⚠️ Keine Ergebnisse zum Speichern vorhanden');
            return;
        }

        // Erweiterte CSV mit Multi-Kategorie-Daten
        let csvContent = 'Bildname,Erkannte_Kategorien,Kategorien_Anzahl,Bäume_Vegetation,Solar_Technik,Gebäude_Infrastruktur,Zusatz_Objekte,Annotiert_Pfad\n';
        
        this.results.forEach(result => {
            const hasBäume = result.erkannteKategorien.includes('Bäume/Vegetation') ? 'JA' : 'NEIN';
            const hasSolar = result.erkannteKategorien.includes('Solaranlagen/Technik') ? 'JA' : 'NEIN';
            const hasGebäude = result.erkannteKategorien.includes('Gebäude/Infrastruktur') ? 'JA' : 'NEIN';
            const objektListe = result.detektierteObjekte.map(o => o.objekt).join(';');
            
            csvContent += `${result.bildname},"${result.erkannteKategorien.join('; ')}",${result.erkannteKategorien.length},${hasBäume},${hasSolar},${hasGebäude},"${objektListe}",${result.annotatedPath}\n`;
        });

        const csvDatei = path.join(__dirname, 'multi_kategorie_ergebnisse.csv');
        fs.writeFileSync(csvDatei, csvContent, 'utf8');
        
        console.log(`\n✅ Multi-Kategorie-Ergebnisse gespeichert in: ${csvDatei}`);
        console.log(`🖼️ Annotierte Bilder gespeichert in: ./multi_annotierte_bilder/`);
    }

    zeigeZusammenfassung() {
        console.log('\n📈 MULTI-KATEGORIE-ANALYSE - ZUSAMMENFASSUNG:');
        console.log('===============================================');
        
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
        
        console.log('\n🎭 Multi-Kategorie-Analyse:');
        console.log(`📊 Bilder mit mehreren Kategorien: ${multiKategorieCount}/${this.results.length} (${(multiKategorieCount/this.results.length*100).toFixed(1)}%)`);
        console.log(`🎯 Bilder mit allen 3 Hauptkategorien: ${alleKategorienCount}/${this.results.length} (${(alleKategorienCount/this.results.length*100).toFixed(1)}%)`);

        // Objekt-Detektionen
        const gesamtObjekte = this.results.reduce((sum, r) => sum + r.objektDetektionen, 0);
        console.log(`\n🔍 Zusätzliche Objekterkennung: ${gesamtObjekte} Objekte total`);

        // Detailierte Aufschlüsselung
        console.log('\n📋 Detailierte Bildanalyse:');
        this.results.forEach(result => {
            console.log(`   ${result.bildname}: [${result.erkannteKategorien.map(k => 
                this.hauptkategorien[k]?.icon + k.split('/')[0]
            ).join(' + ')}] + ${result.objektDetektionen} Objekte`);
        });

        console.log('\n🎨 Visualisierung-Features:');
        console.log('   🗂️ Region-basierte Analyse (3x3 Grid)');
        console.log('   🎨 Farbkodierte Überlagerungen pro Kategorie');
        console.log('   🏷️ Legende mit allen erkannten Kategorien');
        console.log('   📦 Zusätzliche COCO-SSD Objekterkennung');
        console.log('   📊 Statistik-Header pro Bild');
        
        console.log(`\n📁 Alle multi-annotierten Bilder: ./multi_annotierte_bilder/`);
    }
}

// Hauptfunktion
async function main() {
    console.log('🎨 KI & Schatten Hackathon - Multi-Kategorie-Analyse');
    console.log('====================================================');
    console.log('🎯 Simultane Erkennung: Bäume, Solar & Gebäude');
    console.log('==================================================\n');
    
    const analyzer = new MultiKategorieAnalyse();
    
    try {
        await analyzer.init();
        await analyzer.verarbeiteAlleBilder();
        analyzer.speichereErgebnisse();
        analyzer.zeigeZusammenfassung();
        
        console.log('\n🎉 Multi-Kategorie-Analyse erfolgreich abgeschlossen!');
        
    } catch (error) {
        console.error('💥 Fataler Fehler:', error.message);
        process.exit(1);
    }
}

// Programm starten
if (require.main === module) {
    main().catch(console.error);
}

module.exports = MultiKategorieAnalyse; 