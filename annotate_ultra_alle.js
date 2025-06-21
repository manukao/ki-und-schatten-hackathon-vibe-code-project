// Polyfill für fetch in Node.js
global.fetch = require('node-fetch');

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

class UltraDetailKategorieAnalyse {
    constructor() {
        this.mobilenetModel = null;
        this.results = [];
        
        // Nur 2 Hauptkategorien: Bäume/Vegetation und Gebäude/Beton
        this.hauptkategorien = {
            'Bäume/Vegetation': {
                color: '#228B22',  // Forest Green
                icon: '🌳',
                keywords: [
                    'broccoli', 'artichoke', 'cauliflower', 'corn', 'acorn', 'leaf',
                    'maze, labyrinth', 'alp', 'cliff, drop, drop-off', 'lakeside, lakeshore',
                    'volcano', 'mountain', 'hill', 'forest', 'tree', 'plant', 'grass',
                    'jungle', 'field', 'garden', 'park', 'nature', 'coral reef',
                    'moss', 'fern', 'bush', 'shrub', 'meadow', 'woodland', 'grove'
                ]
            },
            'Gebäude/Beton': {
                color: '#DC143C',  // Crimson
                icon: '🏢',
                keywords: [
                    'castle', 'palace', 'monastery', 'church, church building',
                    'viaduct', 'suspension bridge', 'steel arch bridge',
                    'container ship, containership, container vessel',
                    'parking lot', 'residential area', 'street sign', 'tile roof',
                    'jigsaw puzzle', 'envelope', 'honeycomb', 'roof', 'building',
                    'house', 'structure', 'architecture', 'urban', 'city',
                    'construction', 'infrastructure', 'bridge', 'road', 'birdhouse',
                    'concrete', 'asphalt', 'pavement', 'sidewalk', 'wall', 'tower',
                    'skyscraper', 'factory', 'warehouse', 'stadium', 'airport'
                ]
            }
        };
    }

    async init() {
        console.log('🚀 Lade Ultra-Detail-Kategorie-System...');
        console.log('   📊 MobileNet für ultra-hochauflösende Kategorisierung...');
        console.log('   🔍 8x8 Grid (64 Mini-Quadranten) für extremste Präzision...');
        console.log('   🎯 Fokus: Bäume/Vegetation vs. Gebäude/Beton...');
        console.log('   ⚠️  WARNUNG: 50 Bilder × 64 Regionen = 3.200 Analysen!');
        this.mobilenetModel = await mobilenet.load();
        console.log('✅ MobileNet erfolgreich geladen!');
    }

    // Ultra-feine Bildanalyse mit 8x8 Grid (64 Mini-Quadranten)
    async analysiereUltraDetailRegionen(img) {
        const regionen = [];
        const originalWidth = img.width;
        const originalHeight = img.height;
        
        // 8x8 Grid = 64 Mini-Quadranten
        const gridSize = 8;
        const regionWidth = Math.floor(originalWidth / gridSize);
        const regionHeight = Math.floor(originalHeight / gridSize);
        
        console.log(`   🔍 Analysiere ${gridSize}x${gridSize} = ${gridSize*gridSize} Ultra-Mini-Regionen...`);
        
        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const x = col * regionWidth;
                const y = row * regionHeight;
                const width = (col === gridSize - 1) ? originalWidth - x : regionWidth;
                const height = (row === gridSize - 1) ? originalHeight - y : regionHeight;
                
                // Region-Nomenklatur für 8x8
                const regionName = `R${(row + 1).toString().padStart(2, '0')}C${(col + 1).toString().padStart(2, '0')}`;
                
                // Ultra-Mini-Region-Canvas erstellen
                const regionCanvas = createCanvas(224, 224);
                const regionCtx = regionCanvas.getContext('2d');
                
                // Bildausschnitt in Region kopieren
                regionCtx.drawImage(img, x, y, width, height, 0, 0, 224, 224);
                
                // Region analysieren
                const tensor = tf.tidy(() => {
                    const pixels = tf.browser.fromPixels(regionCanvas, 3);
                    return pixels.expandDims(0);
                });
                
                const predictions = await this.mobilenetModel.classify(tensor, 10);
                tensor.dispose();
                
                // Kategorien für diese Ultra-Mini-Region ermitteln
                const kategorienAnalyse = this.kategorisiereUltraRegion(predictions);
                
                regionen.push({
                    name: regionName,
                    x: x, 
                    y: y, 
                    width: width, 
                    height: height,
                    row: row,
                    col: col,
                    kategorien: kategorienAnalyse.kategorien,
                    scores: kategorienAnalyse.scores,
                    dominantKategorie: kategorienAnalyse.dominantKategorie,
                    topPredictions: predictions.slice(0, 3),
                    confidence: predictions.length > 0 ? predictions[0].probability : 0
                });
            }
        }
        
        return regionen;
    }

    kategorisiereUltraRegion(predictions) {
        const erkannteKategorien = new Set();
        const kategorieScores = {};
        let dominantKategorie = null;
        let maxScore = 0;
        
        predictions.forEach(pred => {
            const label = pred.className.toLowerCase();
            
            // Prüfe nur die 2 Hauptkategorien
            for (const [kategorieName, kategorieData] of Object.entries(this.hauptkategorien)) {
                const matchingKeywords = kategorieData.keywords.filter(keyword => 
                    label.includes(keyword.toLowerCase())
                );
                
                if (matchingKeywords.length > 0 && pred.probability > 0.01) {
                    erkannteKategorien.add(kategorieName);
                    
                    if (!kategorieScores[kategorieName]) {
                        kategorieScores[kategorieName] = 0;
                    }
                    
                    // Gewichteter Score
                    const score = pred.probability * (1 + matchingKeywords.length * 0.3);
                    kategorieScores[kategorieName] += score;
                    
                    // Dominante Kategorie bestimmen
                    if (score > maxScore) {
                        maxScore = score;
                        dominantKategorie = kategorieName;
                    }
                }
            }
        });
        
        return {
            kategorien: Array.from(erkannteKategorien),
            scores: kategorieScores,
            dominantKategorie: dominantKategorie
        };
    }

    async analysiereUndAnnotiereBild(bildpfad, bildname) {
        console.log(`🔍 Ultra-Detail-Analyse: ${bildname}`);
        
        try {
            // Bild laden
            const img = await loadImage(bildpfad);
            
            // 1. Ultra-Detail-Region-Analyse (8x8 Grid = 64 Regionen)
            const regionen = await this.analysiereUltraDetailRegionen(img);
            
            // 2. Ultra-Kategorien-Statistiken
            const ultraStats = this.analysiereKategorieVerteilung(regionen);
            
            // 3. Dominanz-Analyse
            const dominanzAnalyse = this.analysiereDominanz(regionen);
            
            // 4. Annotiertes Ultra-Bild erstellen
            const annotatedPath = await this.erstelleUltraAnnotiertesBild(
                img, 
                regionen, 
                ultraStats,
                dominanzAnalyse,
                bildname
            );
            
            // 5. Ergebnisse speichern
            this.results.push({
                bildname: bildname,
                annotatedPath: annotatedPath,
                erkannteKategorien: ultraStats.alleKategorien,
                ultraStatistik: ultraStats,
                dominanzAnalyse: dominanzAnalyse,
                ultraStatistiken: {
                    regionenMitKategorien: regionen.filter(r => r.kategorien.length > 0).length,
                    regionenGesamt: regionen.length,
                    bäumeRegionen: regionen.filter(r => r.kategorien.includes('Bäume/Vegetation')).length,
                    gebäudeRegionen: regionen.filter(r => r.kategorien.includes('Gebäude/Beton')).length,
                    dominanteRegionenBäume: regionen.filter(r => r.dominantKategorie === 'Bäume/Vegetation').length,
                    dominanteRegionenGebäude: regionen.filter(r => r.dominantKategorie === 'Gebäude/Beton').length
                }
            });

            console.log(`   → ${ultraStats.alleKategorien.length} Kategorien in ${ultraStats.aktivenRegionen}/64 Regionen`);
            console.log(`   → Dominanz: ${dominanzAnalyse.dominantKategorie} (${dominanzAnalyse.dominanzProzent}%)`);
            
        } catch (error) {
            console.error(`❌ Ultra-Analysefehler für ${bildname}:`, error.message);
        }
    }

    analysiereKategorieVerteilung(regionen) {
        const alleKategorien = new Set();
        const kategorieRegionenCount = {};
        
        regionen.forEach(region => {
            region.kategorien.forEach(kat => {
                alleKategorien.add(kat);
                
                if (!kategorieRegionenCount[kat]) {
                    kategorieRegionenCount[kat] = 0;
                }
                kategorieRegionenCount[kat]++;
            });
        });
        
        return {
            alleKategorien: Array.from(alleKategorien),
            kategorieRegionenCount: kategorieRegionenCount,
            aktivenRegionen: regionen.filter(r => r.kategorien.length > 0).length
        };
    }

    analysiereDominanz(regionen) {
        const dominanzCount = {};
        let dominantKategorie = null;
        let maxCount = 0;
        
        regionen.forEach(region => {
            if (region.dominantKategorie) {
                if (!dominanzCount[region.dominantKategorie]) {
                    dominanzCount[region.dominantKategorie] = 0;
                }
                dominanzCount[region.dominantKategorie]++;
                
                if (dominanzCount[region.dominantKategorie] > maxCount) {
                    maxCount = dominanzCount[region.dominantKategorie];
                    dominantKategorie = region.dominantKategorie;
                }
            }
        });
        
        const dominanzProzent = dominantKategorie ? 
            ((dominanzCount[dominantKategorie] / regionen.length) * 100).toFixed(1) : 0;
        
        return {
            dominantKategorie: dominantKategorie,
            dominanzProzent: dominanzProzent,
            dominanzCount: dominanzCount
        };
    }

    async erstelleUltraAnnotiertesBild(originalImg, regionen, ultraStats, dominanz, bildname) {
        const canvas = createCanvas(originalImg.width, originalImg.height);
        const ctx = canvas.getContext('2d');
        
        // Originalbild zeichnen
        ctx.drawImage(originalImg, 0, 0);
        
        // Ultra-feine Region-Overlays zeichnen (8x8 = 64 Regionen)
        regionen.forEach(region => {
            if (region.kategorien.length > 0) {
                const kategorieData = this.hauptkategorien[region.dominantKategorie || region.kategorien[0]];
                if (kategorieData) {
                    // Sehr schwache Überlagerung für 64 Regionen
                    ctx.fillStyle = kategorieData.color + '15';
                    ctx.fillRect(region.x, region.y, region.width, region.height);
                    
                    // Sehr dünner Rahmen
                    ctx.strokeStyle = kategorieData.color + '60';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(region.x, region.y, region.width, region.height);
                    
                    // Mikro-Icon
                    if (region.width > 15 && region.height > 15) {
                        const iconSize = Math.min(region.width, region.height) * 0.4;
                        if (iconSize >= 8) {
                            ctx.font = `${iconSize}px Arial`;
                            ctx.fillStyle = kategorieData.color + 'AA';
                            const iconX = region.x + region.width/2 - iconSize/4;
                            const iconY = region.y + region.height/2 + iconSize/4;
                            ctx.fillText(kategorieData.icon, iconX, iconY);
                        }
                    }
                }
            }
        });

        // Ultra-feine Grid-Linien (8x8)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 0.5;
        
        // Vertikale Linien
        for (let i = 1; i < 8; i++) {
            const x = (originalImg.width / 8) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, originalImg.height);
            ctx.stroke();
        }
        
        // Horizontale Linien
        for (let i = 1; i < 8; i++) {
            const y = (originalImg.height / 8) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(originalImg.width, y);
            ctx.stroke();
        }

        // Legende
        if (ultraStats.alleKategorien.length > 0) {
            const legendeY = 20;
            let legendeX = originalImg.width - 320;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
            ctx.fillRect(legendeX - 10, legendeY - 10, 310, 100);
            
            // Legende-Titel
            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('🔍 Ultra-Detail (8x8 = 64 Regionen):', legendeX, legendeY + 15);
            
            // Dominanz-Info
            if (dominanz.dominantKategorie) {
                const dominantData = this.hauptkategorien[dominanz.dominantKategorie];
                ctx.font = 'bold 14px Arial';
                ctx.fillStyle = dominantData.color;
                ctx.fillText(`👑 ${dominantData.icon} ${dominanz.dominantKategorie}`, legendeX, legendeY + 40);
                ctx.fillStyle = '#CCCCCC';
                ctx.font = '12px Arial';
                ctx.fillText(`${dominanz.dominanzProzent}% Dominanz`, legendeX, legendeY + 55);
            }
            
            // Kategorien-Details
            ultraStats.alleKategorien.forEach((kategorie, index) => {
                const kategorieData = this.hauptkategorien[kategorie];
                if (kategorieData) {
                    const y = legendeY + 75 + (index * 15);
                    const regionCount = ultraStats.kategorieRegionenCount[kategorie];
                    const percentage = ((regionCount / 64) * 100).toFixed(1);
                    
                    // Farbbox
                    ctx.fillStyle = kategorieData.color;
                    ctx.fillRect(legendeX, y - 10, 12, 12);
                    
                    // Kategorie-Text
                    ctx.fillStyle = 'white';
                    ctx.font = '11px Arial';
                    ctx.fillText(`${kategorieData.icon} ${kategorie}: ${regionCount}/64 (${percentage}%)`, legendeX + 18, y - 2);
                }
            });
        }

        // Ultra-Statistik-Header
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(10, 10, 400, 50);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(`🔍 Ultra-Grid: ${ultraStats.alleKategorien.length} Kategorien (Bäume vs. Gebäude)`, 15, 30);
        ctx.font = '12px Arial';
        ctx.fillText(`${ultraStats.aktivenRegionen}/64 aktive Regionen | 8x8 Ultra-Hochauflösung`, 15, 45);

        // Annotiertes Bild speichern
        const outputDir = path.join(__dirname, 'ultra_detail_bilder_alle');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }

        const outputPath = path.join(outputDir, `ultra_${bildname.replace('.jpg', '.png')}`);
        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(outputPath, buffer);

        return outputPath;
    }

    async verarbeiteAlleBilder() {
        const bilderOrdner = path.join(__dirname, 'images', 'rohdaten');
        
        try {
            const dateien = fs.readdirSync(bilderOrdner)
                .filter(datei => datei.toLowerCase().endsWith('.jpg'))
                .sort(); // ALLE 50 BILDER - keine Begrenzung!
            
            console.log(`📁 Ultra-Detail-Analyse für ALLE ${dateien.length} Bilder`);
            console.log('🔍 Ultra-hochauflösende 8x8 Grid-Analyse: Bäume vs. Gebäude/Beton...');
            console.log(`⚡ Geschätzte Dauer: ~${Math.ceil(dateien.length * 2)} Minuten\n`);

            for (let i = 0; i < dateien.length; i++) {
                const dateiname = dateien[i];
                const bildpfad = path.join(bilderOrdner, dateiname);
                
                console.log(`[${i + 1}/${dateien.length}] (${((i+1)/dateien.length*100).toFixed(1)}%)`);
                await this.analysiereUndAnnotiereBild(bildpfad, dateiname);
            }
            
        } catch (error) {
            console.error('❌ Fehler bei der Ultra-Detail-Analyse:', error.message);
        }
    }

    speichereErgebnisse() {
        if (this.results.length === 0) {
            console.log('⚠️ Keine Ergebnisse zum Speichern vorhanden');
            return;
        }

        // Ultra-detaillierte CSV für ALLE Bilder
        let csvContent = 'Bildname,Dominante_Kategorie,Dominanz_Prozent,Aktive_Regionen,Bäume_Regionen,Gebäude_Regionen,Bäume_Dominant,Gebäude_Dominant,Annotiert_Pfad\n';
        
        this.results.forEach(result => {
            csvContent += `${result.bildname},"${result.dominanzAnalyse.dominantKategorie || 'Keine'}",${result.dominanzAnalyse.dominanzProzent},${result.ultraStatistiken.regionenMitKategorien},${result.ultraStatistiken.bäumeRegionen},${result.ultraStatistiken.gebäudeRegionen},${result.ultraStatistiken.dominanteRegionenBäume},${result.ultraStatistiken.dominanteRegionenGebäude},${result.annotatedPath}\n`;
        });

        const csvDatei = path.join(__dirname, 'ultra_detail_ergebnisse_alle_50.csv');
        fs.writeFileSync(csvDatei, csvContent, 'utf8');
        
        console.log(`\n✅ Ultra-Detail-Ergebnisse für ALLE ${this.results.length} Bilder gespeichert in: ${csvDatei}`);
        console.log(`🖼️ Annotierte Bilder gespeichert in: ./ultra_detail_bilder_alle/`);
    }

    zeigeZusammenfassung() {
        console.log('\n📈 ULTRA-DETAIL-ANALYSE - VOLLSTÄNDIGE ZUSAMMENFASSUNG:');
        console.log('========================================================');
        
        // 2-Kategorien-Statistiken
        const bäumeCount = this.results.filter(r => r.erkannteKategorien.includes('Bäume/Vegetation')).length;
        const gebäudeCount = this.results.filter(r => r.erkannteKategorien.includes('Gebäude/Beton')).length;
        const beideKategorienCount = this.results.filter(r => r.erkannteKategorien.length === 2).length;

        console.log('🎯 2-Kategorien-Fokus (Alle 50 Bilder):');
        console.log(`🌳 Bäume/Vegetation erkannt: ${bäumeCount}/${this.results.length} Bilder (${(bäumeCount/this.results.length*100).toFixed(1)}%)`);
        console.log(`🏢 Gebäude/Beton erkannt: ${gebäudeCount}/${this.results.length} Bilder (${(gebäudeCount/this.results.length*100).toFixed(1)}%)`);
        console.log(`🎭 Beide Kategorien: ${beideKategorienCount}/${this.results.length} Bilder (${(beideKategorienCount/this.results.length*100).toFixed(1)}%)`);

        // Ultra-Detail-Statistiken
        const durchschnittAktiveRegionen = this.results.reduce((sum, r) => sum + r.ultraStatistiken.regionenMitKategorien, 0) / this.results.length;
        const durchschnittBäumeRegionen = this.results.reduce((sum, r) => sum + r.ultraStatistiken.bäumeRegionen, 0) / this.results.length;
        const durchschnittGebäudeRegionen = this.results.reduce((sum, r) => sum + r.ultraStatistiken.gebäudeRegionen, 0) / this.results.length;
        
        console.log('\n🔍 Ultra-Grid-Analyse (8x8 = 64 Regionen):');
        console.log(`📏 Durchschnittlich aktive Regionen: ${durchschnittAktiveRegionen.toFixed(1)}/64 (${(durchschnittAktiveRegionen/64*100).toFixed(1)}%)`);
        console.log(`🌳 Durchschnittliche Bäume-Regionen: ${durchschnittBäumeRegionen.toFixed(1)}/64`);
        console.log(`🏢 Durchschnittliche Gebäude-Regionen: ${durchschnittGebäudeRegionen.toFixed(1)}/64`);

        // Dominanz-Analyse
        const dominanzStats = {};
        this.results.forEach(result => {
            const dominant = result.dominanzAnalyse.dominantKategorie;
            if (dominant) {
                dominanzStats[dominant] = (dominanzStats[dominant] || 0) + 1;
            }
        });

        console.log('\n👑 Dominanz-Analyse (Alle 50 Bilder):');
        Object.entries(dominanzStats).forEach(([kategorie, count]) => {
            const percentage = (count / this.results.length * 100).toFixed(1);
            const icon = this.hauptkategorien[kategorie]?.icon || '❓';
            console.log(`   ${icon} ${kategorie}: ${count}/${this.results.length} Bilder (${percentage}%)`);
        });

        console.log('\n🎨 Ultra-Hochauflösungs-Features:');
        console.log('   🔍 8x8 Grid = 64 Ultra-Mini-Quadranten pro Bild');
        console.log('   🎯 2-Kategorien-Fokus: Bäume/Vegetation vs. Gebäude/Beton');
        console.log('   👑 Dominanz-Analyse für Bildcharakterisierung');
        console.log('   💫 Mikro-Icons mit adaptiver Größe');
        console.log(`   📊 Gesamtanalysen: ${this.results.length * 64} Ultra-Mini-Regionen`);
        
        console.log(`\n📁 Alle ${this.results.length} ultra-detail-annotierten Bilder: ./ultra_detail_bilder_alle/`);
    }
}

// Hauptfunktion
async function main() {
    console.log('🎨 KI & Schatten Hackathon - Ultra-Detail-Analyse ALLE 50 BILDER');
    console.log('==================================================================');
    console.log('🔍 Ultra-hochauflösende 8x8 Grid-Analyse');
    console.log('🎯 2-Kategorien-Fokus: Bäume/Vegetation vs. Gebäude/Beton');
    console.log('⚡ 64 Ultra-Mini-Quadranten für extremste Präzision');
    console.log('🚀 ALLE 50 BILDER werden verarbeitet!');
    console.log('==================================================================\n');
    
    const analyzer = new UltraDetailKategorieAnalyse();
    
    try {
        await analyzer.init();
        await analyzer.verarbeiteAlleBilder();
        analyzer.speichereErgebnisse();
        analyzer.zeigeZusammenfassung();
        
        console.log('\n🎉 Ultra-Detail-Analyse für ALLE 50 Bilder erfolgreich abgeschlossen!');
        
    } catch (error) {
        console.error('💥 Fataler Fehler:', error.message);
        process.exit(1);
    }
}

// Programm starten
if (require.main === module) {
    main().catch(console.error);
}

module.exports = UltraDetailKategorieAnalyse; 