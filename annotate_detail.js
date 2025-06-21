// Polyfill für fetch in Node.js
global.fetch = require('node-fetch');

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

class DetailMultiKategorieAnalyse {
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
                    'jungle', 'field', 'garden', 'park', 'nature', 'coral reef'
                ]
            },
            'Solaranlagen/Technik': {
                color: '#FFD700',  // Gold
                icon: '☀️',
                keywords: [
                    'solar dish, solar collector, solar furnace', 'spotlight, spot',
                    'radar, microwave radar', 'satellite dish', 'radio telescope',
                    'antenna', 'panel', 'dish', 'solar', 'technical', 'machinery',
                    'equipment', 'installation', 'technology', 'crane'
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
                    'construction', 'infrastructure', 'bridge', 'road', 'birdhouse'
                ]
            }
        };
    }

    async init() {
        console.log('🚀 Lade Detail-Multi-Kategorie-System...');
        console.log('   📊 MobileNet für hochauflösende Kategorisierung...');
        console.log('   🔍 4x4 Grid (16 Mini-Quadranten) für präzise Analyse...');
        this.mobilenetModel = await mobilenet.load();
        console.log('✅ MobileNet erfolgreich geladen!');
    }

    // Analysiert Bildregionen mit 4x4 Grid (16 Mini-Quadranten)
    async analysiereDetailRegionen(img) {
        const regionen = [];
        const originalWidth = img.width;
        const originalHeight = img.height;
        
        // 4x4 Grid = 16 Mini-Quadranten
        const gridSize = 4;
        const regionWidth = Math.floor(originalWidth / gridSize);
        const regionHeight = Math.floor(originalHeight / gridSize);
        
        console.log(`   🔍 Analysiere ${gridSize}x${gridSize} = ${gridSize*gridSize} Mini-Regionen...`);
        
        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const x = col * regionWidth;
                const y = row * regionHeight;
                const width = (col === gridSize - 1) ? originalWidth - x : regionWidth;
                const height = (row === gridSize - 1) ? originalHeight - y : regionHeight;
                
                // Region-Name für bessere Identifikation
                const regionName = `R${row + 1}C${col + 1}`;
                const positionName = this.getPositionName(row, col, gridSize);
                
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
                
                const predictions = await this.mobilenetModel.classify(tensor, 8);
                tensor.dispose();
                
                // Kategorien für diese Region ermitteln
                const kategorienAnalyse = this.kategorisiereRegion(predictions);
                
                regionen.push({
                    name: regionName,
                    positionName: positionName,
                    x: x, 
                    y: y, 
                    width: width, 
                    height: height,
                    row: row,
                    col: col,
                    kategorien: kategorienAnalyse.kategorien,
                    scores: kategorienAnalyse.scores,
                    topPredictions: predictions.slice(0, 2),
                    confidence: predictions.length > 0 ? predictions[0].probability : 0
                });
            }
        }
        
        return regionen;
    }

    getPositionName(row, col, gridSize) {
        const verticalPos = row === 0 ? 'Oben' : 
                          row === gridSize - 1 ? 'Unten' : 
                          row < gridSize / 2 ? 'Oben-Mitte' : 'Unten-Mitte';
        
        const horizontalPos = col === 0 ? 'Links' : 
                             col === gridSize - 1 ? 'Rechts' : 
                             col < gridSize / 2 ? 'Links-Mitte' : 'Rechts-Mitte';
        
        return `${verticalPos}-${horizontalPos}`;
    }

    kategorisiereRegion(predictions) {
        const erkannteKategorien = new Set();
        const kategorieScores = {};
        
        predictions.forEach(pred => {
            const label = pred.className.toLowerCase();
            
            // Prüfe jede Hauptkategorie mit höherer Sensitivität
            for (const [kategorieName, kategorieData] of Object.entries(this.hauptkategorien)) {
                const matchingKeywords = kategorieData.keywords.filter(keyword => 
                    label.includes(keyword.toLowerCase())
                );
                
                if (matchingKeywords.length > 0 && pred.probability > 0.02) { // Niedrigere Schwelle für feinere Erkennung
                    erkannteKategorien.add(kategorieName);
                    
                    // Verbesserte Scoring-Methode
                    if (!kategorieScores[kategorieName]) {
                        kategorieScores[kategorieName] = 0;
                    }
                    // Bonus für multiple Keyword-Matches
                    kategorieScores[kategorieName] += pred.probability * (1 + matchingKeywords.length * 0.2);
                }
            }
        });
        
        return {
            kategorien: Array.from(erkannteKategorien),
            scores: kategorieScores
        };
    }

    async analysiereUndAnnotiereBild(bildpfad, bildname) {
        console.log(`🔍 Detail-Multi-Kategorie-Analyse: ${bildname}`);
        
        try {
            // Bild laden
            const img = await loadImage(bildpfad);
            
            // 1. Detail-Region-Analyse (4x4 Grid = 16 Regionen)
            const regionen = await this.analysiereDetailRegionen(img);
            
            // 2. Kategorien aggregieren und analysieren
            const kategorieStatistik = this.analysiereKategorieVerteilung(regionen);
            
            // 3. Annotiertes Bild erstellen
            const annotatedPath = await this.erstelleDetailAnnotiertesBild(
                img, 
                regionen, 
                kategorieStatistik,
                bildname
            );
            
            // 4. Ergebnis speichern
            this.results.push({
                bildname: bildname,
                annotatedPath: annotatedPath,
                erkannteKategorien: kategorieStatistik.alleKategorien,
                kategorieStatistik: kategorieStatistik,
                regionenAnalyse: regionen.map(r => ({
                    region: r.name,
                    position: r.positionName,
                    kategorien: r.kategorien,
                    confidence: r.confidence,
                    topPrediction: r.topPredictions[0]?.className || 'Unbekannt'
                })),
                detailStatistiken: {
                    regionenMitKategorien: regionen.filter(r => r.kategorien.length > 0).length,
                    regionenGesamt: regionen.length,
                    durchschnittlicheKategorienProRegion: regionen.reduce((sum, r) => sum + r.kategorien.length, 0) / regionen.length
                }
            });

            console.log(`   → ${kategorieStatistik.alleKategorien.length} Hauptkategorien in ${kategorieStatistik.aktivenRegionen} von 16 Regionen`);
            console.log(`   → Detail-Verteilung: ${Object.entries(kategorieStatistik.kategorieRegionenCount).map(([kat, count]) => 
                `${this.hauptkategorien[kat]?.icon}${kat.split('/')[0]}(${count})`
            ).join(', ')}`);
            
        } catch (error) {
            console.error(`❌ Detail-Analysefehler für ${bildname}:`, error.message);
        }
    }

    analysiereKategorieVerteilung(regionen) {
        const alleKategorien = new Set();
        const kategorieRegionen = {};
        const kategorieRegionenCount = {};
        const kategorieScores = {};
        
        regionen.forEach(region => {
            region.kategorien.forEach(kat => {
                alleKategorien.add(kat);
                
                // Regionen pro Kategorie sammeln
                if (!kategorieRegionen[kat]) {
                    kategorieRegionen[kat] = [];
                    kategorieRegionenCount[kat] = 0;
                }
                kategorieRegionen[kat].push(`${region.name}(${region.positionName})`);
                kategorieRegionenCount[kat]++;
                
                // Scores aggregieren
                if (!kategorieScores[kat]) {
                    kategorieScores[kat] = 0;
                }
                kategorieScores[kat] += region.scores[kat] || 0;
            });
        });
        
        return {
            alleKategorien: Array.from(alleKategorien),
            kategorieRegionen: kategorieRegionen,
            kategorieRegionenCount: kategorieRegionenCount,
            kategorieScores: kategorieScores,
            aktivenRegionen: regionen.filter(r => r.kategorien.length > 0).length
        };
    }

    async erstelleDetailAnnotiertesBild(originalImg, regionen, kategorieStats, bildname) {
        const canvas = createCanvas(originalImg.width, originalImg.height);
        const ctx = canvas.getContext('2d');
        
        // Originalbild zeichnen
        ctx.drawImage(originalImg, 0, 0);
        
        // 1. Feinere Region-Overlays zeichnen
        regionen.forEach(region => {
            if (region.kategorien.length > 0) {
                // Für jede erkannte Kategorie in der Mini-Region
                region.kategorien.forEach((kategorie, index) => {
                    const kategorieData = this.hauptkategorien[kategorie];
                    if (kategorieData) {
                        // Semi-transparente Überlagerung - schwächer für feineres Grid
                        ctx.fillStyle = kategorieData.color + '25'; // 25 = ~15% opacity
                        ctx.fillRect(region.x, region.y, region.width, region.height);
                        
                        // Dünnerer Rahmen für feineres Grid
                        ctx.strokeStyle = kategorieData.color;
                        ctx.lineWidth = 2;
                        ctx.strokeRect(region.x, region.y, region.width, region.height);
                        
                        // Mini-Icon in der Region statt Text (platzsparender)
                        if (index === 0) { // Nur für stärkste Kategorie
                            const iconSize = Math.min(region.width, region.height) * 0.3;
                            ctx.font = `${iconSize}px Arial`;
                            ctx.fillStyle = kategorieData.color;
                            const iconX = region.x + region.width/2 - iconSize/4;
                            const iconY = region.y + region.height/2 + iconSize/4;
                            ctx.fillText(kategorieData.icon, iconX, iconY);
                        }
                    }
                });
                
                // Mini-Region-ID in der Ecke
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.fillRect(region.x + 2, region.y + 2, 25, 15);
                ctx.fillStyle = 'white';
                ctx.font = 'bold 9px Arial';
                ctx.fillText(region.name, region.x + 4, region.y + 12);
            }
        });

        // 2. Grid-Linien für bessere Orientierung
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        
        // Vertikale Linien
        for (let i = 1; i < 4; i++) {
            const x = (originalImg.width / 4) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, originalImg.height);
            ctx.stroke();
        }
        
        // Horizontale Linien
        for (let i = 1; i < 4; i++) {
            const y = (originalImg.height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(originalImg.width, y);
            ctx.stroke();
        }

        // 3. Erweiterte Legende mit Regionen-Anzahl
        if (kategorieStats.alleKategorien.length > 0) {
            const legendeY = 20;
            let legendeX = originalImg.width - 320;
            
            // Hintergrund für erweiterte Legende
            ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
            ctx.fillRect(legendeX - 10, legendeY - 10, 310, (kategorieStats.alleKategorien.length * 35) + 60);
            
            // Legende-Titel
            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('🎯 Detail-Kategorien (4x4 Grid):', legendeX, legendeY + 15);
            
            // Kategorien in Legende mit detaillierter Info
            kategorieStats.alleKategorien.forEach((kategorie, index) => {
                const kategorieData = this.hauptkategorien[kategorie];
                if (kategorieData) {
                    const y = legendeY + 55 + (index * 35);
                    const regionCount = kategorieStats.kategorieRegionenCount[kategorie];
                    const percentage = ((regionCount / 16) * 100).toFixed(1);
                    
                    // Farbbox
                    ctx.fillStyle = kategorieData.color;
                    ctx.fillRect(legendeX, y - 20, 20, 20);
                    
                    // Kategorie-Text
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px Arial';
                    ctx.fillText(`${kategorieData.icon} ${kategorie}`, legendeX + 28, y - 8);
                    
                    // Detail-Info
                    ctx.font = '11px Arial';
                    ctx.fillStyle = '#CCCCCC';
                    ctx.fillText(`${regionCount}/16 Regionen (${percentage}%)`, legendeX + 28, y + 5);
                }
            });
        }

        // 4. Erweiterte Statistik-Header
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(10, 10, 380, 50);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`🔍 Detail-Grid: ${kategorieStats.alleKategorien.length} Kategorien`, 15, 30);
        ctx.font = '12px Arial';
        ctx.fillText(`${kategorieStats.aktivenRegionen}/16 aktive Regionen | 4x4 Hochauflösungs-Analyse`, 15, 45);

        // Annotiertes Bild speichern
        const outputDir = path.join(__dirname, 'detail_multi_bilder');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }

        const outputPath = path.join(outputDir, `detail_${bildname.replace('.jpg', '.png')}`);
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
                .slice(0, 8); // Weniger Bilder wegen intensiverer Analyse
            
            console.log(`📁 Detail-Multi-Kategorie-Analyse für ${dateien.length} Bilder`);
            console.log('🔍 Hochauflösende 4x4 Grid-Analyse: Bäume, Solar & Gebäude...\n');

            for (let i = 0; i < dateien.length; i++) {
                const dateiname = dateien[i];
                const bildpfad = path.join(bilderOrdner, dateiname);
                
                console.log(`[${i + 1}/${dateien.length}]`);
                await this.analysiereUndAnnotiereBild(bildpfad, dateiname);
            }
            
        } catch (error) {
            console.error('❌ Fehler bei der Detail-Multi-Kategorie-Analyse:', error.message);
        }
    }

    speichereErgebnisse() {
        if (this.results.length === 0) {
            console.log('⚠️ Keine Ergebnisse zum Speichern vorhanden');
            return;
        }

        // Erweiterte CSV mit Detail-Daten
        let csvContent = 'Bildname,Erkannte_Kategorien,Kategorien_Anzahl,Aktive_Regionen,Bäume_Regionen,Solar_Regionen,Gebäude_Regionen,Durchschnitt_Kategorien_Pro_Region,Annotiert_Pfad\n';
        
        this.results.forEach(result => {
            const bäumeRegionen = result.kategorieStatistik.kategorieRegionenCount['Bäume/Vegetation'] || 0;
            const solarRegionen = result.kategorieStatistik.kategorieRegionenCount['Solaranlagen/Technik'] || 0;
            const gebäudeRegionen = result.kategorieStatistik.kategorieRegionenCount['Gebäude/Infrastruktur'] || 0;
            
            csvContent += `${result.bildname},"${result.erkannteKategorien.join('; ')}",${result.erkannteKategorien.length},${result.detailStatistiken.regionenMitKategorien},${bäumeRegionen},${solarRegionen},${gebäudeRegionen},${result.detailStatistiken.durchschnittlicheKategorienProRegion.toFixed(2)},${result.annotatedPath}\n`;
        });

        const csvDatei = path.join(__dirname, 'detail_multi_ergebnisse.csv');
        fs.writeFileSync(csvDatei, csvContent, 'utf8');
        
        console.log(`\n✅ Detail-Multi-Kategorie-Ergebnisse gespeichert in: ${csvDatei}`);
        console.log(`🖼️ Annotierte Bilder gespeichert in: ./detail_multi_bilder/`);
    }

    zeigeZusammenfassung() {
        console.log('\n📈 DETAIL-MULTI-KATEGORIE-ANALYSE - ZUSAMMENFASSUNG:');
        console.log('=====================================================');
        
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

        // Detail-Statistiken
        const durchschnittAktiveRegionen = this.results.reduce((sum, r) => sum + r.detailStatistiken.regionenMitKategorien, 0) / this.results.length;
        const durchschnittKategorienProRegion = this.results.reduce((sum, r) => sum + r.detailStatistiken.durchschnittlicheKategorienProRegion, 0) / this.results.length;
        
        console.log('\n🔍 Detail-Grid-Analyse (4x4 = 16 Regionen):');
        console.log(`📏 Durchschnittlich aktive Regionen: ${durchschnittAktiveRegionen.toFixed(1)}/16 (${(durchschnittAktiveRegionen/16*100).toFixed(1)}%)`);
        console.log(`📊 Durchschnittliche Kategorien pro Region: ${durchschnittKategorienProRegion.toFixed(2)}`);

        // Regionen-Verteilungs-Statistiken
        const regionenStats = {};
        this.results.forEach(result => {
            Object.entries(result.kategorieStatistik.kategorieRegionenCount).forEach(([kat, count]) => {
                if (!regionenStats[kat]) {
                    regionenStats[kat] = { total: 0, max: 0, bilder: 0 };
                }
                regionenStats[kat].total += count;
                regionenStats[kat].max = Math.max(regionenStats[kat].max, count);
                regionenStats[kat].bilder++;
            });
        });

        console.log('\n🗺️ Regionen-Verteilung pro Kategorie:');
        Object.entries(regionenStats).forEach(([kategorie, stats]) => {
            const durchschnitt = stats.total / stats.bilder;
            console.log(`   ${this.hauptkategorien[kategorie]?.icon} ${kategorie}: ⌀${durchschnitt.toFixed(1)} Regionen/Bild (Max: ${stats.max})`);
        });

        // Top-Performer
        const topPerformer = this.results
            .sort((a, b) => b.detailStatistiken.regionenMitKategorien - a.detailStatistiken.regionenMitKategorien)
            .slice(0, 3);

        if (topPerformer.length > 0) {
            console.log('\n🏆 Top Detail-Erkennungen:');
            topPerformer.forEach((result, index) => {
                console.log(`   ${index + 1}. ${result.bildname}: ${result.detailStatistiken.regionenMitKategorien}/16 aktive Regionen`);
                console.log(`      [${result.erkannteKategorien.map(k => this.hauptkategorien[k]?.icon + k.split('/')[0]).join(' + ')}]`);
            });
        }

        console.log('\n🎨 Hochauflösungs-Features:');
        console.log('   🔍 4x4 Grid = 16 Mini-Quadranten pro Bild');
        console.log('   📍 Präzise Positionierung (R1C1-R4C4)');
        console.log('   🎨 Feinabgestimmte farbkodierte Überlagerungen');
        console.log('   📊 Region-spezifische Kategoriezählung');
        console.log('   🗺️ Grid-Linien für bessere räumliche Orientierung');
        console.log('   💫 Mini-Icons in Regionen für platzsparende Annotation');
        
        console.log(`\n📁 Alle detail-annotierten Bilder: ./detail_multi_bilder/`);
    }
}

// Hauptfunktion
async function main() {
    console.log('🎨 KI & Schatten Hackathon - Detail-Multi-Kategorie-Analyse');
    console.log('=============================================================');
    console.log('🔍 Hochauflösende 4x4 Grid-Analyse: Bäume, Solar & Gebäude');
    console.log('🎯 16 Mini-Quadranten für präzise Objektlokalisierung');
    console.log('===========================================================\n');
    
    const analyzer = new DetailMultiKategorieAnalyse();
    
    try {
        await analyzer.init();
        await analyzer.verarbeiteAlleBilder();
        analyzer.speichereErgebnisse();
        analyzer.zeigeZusammenfassung();
        
        console.log('\n🎉 Detail-Multi-Kategorie-Analyse erfolgreich abgeschlossen!');
        
    } catch (error) {
        console.error('💥 Fataler Fehler:', error.message);
        process.exit(1);
    }
}

// Programm starten
if (require.main === module) {
    main().catch(console.error);
}

module.exports = DetailMultiKategorieAnalyse; 