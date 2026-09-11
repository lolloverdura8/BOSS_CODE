# Piano d'attacco — Dati sintetici OR 4.1

**Progetto B.O.S.S. · Obiettivo Realizzativo 4.1 · §2.4**

Versione 1.0 — 9 settembre 2026

*Prezzi RunPod Community Cloud rilevati il 9 settembre 2026. Importi in dollari USA, IVA esclusa.*

---

## 1. Il piano in mezza pagina

**Costo totale atteso: $15,59.** Tetto con imprevisti: $19,60.

Il preventivo del consulente proponeva tre scenari: $11,55 (PoC economica), $24,65 (PoC completa), $55,45 (PoC estesa). Questo piano **costa meno della PoC completa** e in più comprende 90 clip di produzione, che nel preventivo non c'erano: quei tre scenari compravano solo test.

L'architettura in una frase:

> **CARLA fornisce la geometria esatta di ciò che sa rappresentare. RunPod genera da testo ciò che CARLA non ha. Il PC annota tutto.**

Il lavoro si divide in tre fasi:

| Fase | Dove | Durata | Costo |
|---|---|---|---|
| **0 — Preparazione** | PC locale | ~2 settimane | **$0** |
| **1 — Generazione** | RunPod, 4 sessioni | ~3 giorni | **$15,59** |
| **2 — Annotazione ed export** | PC locale | ~1 settimana | **$0** |

La Fase 0 non è un preambolo: contiene un test che, se va bene, **cancella del tutto la Fase 1**.

---

## 2. Le due strade, e perché sono due

Le 14 classi B.O.S.S. si dividono in due gruppi che hanno bisogno di cose diverse.

### Ramo A — Sim2Real (11 classi)

Pedone, automobile, bicicletta, motociclo, semaforo, palo della luce, attraversamento pedonale, gradino, ramo sporgente, insegna bassa, ostacolo sospeso generico.

**Tecnicamente.** Queste classi hanno un asset in CARLA, quindi la geometria della scena è nota in modo esatto: profondità densa per pixel, maschere di istanza, posa della camera. Il modello generativo riceve depth e contorni come segnale di controllo e ridipinge solo l'aspetto, lasciando la geometria dov'è. Le etichette non vengono dal generatore, vengono dal simulatore.

**In parole semplici.** CARLA fa lo stampo, il modello ci dipinge sopra. Lo stampo ti dice esattamente dove sta ogni oggetto e a che distanza, e quella parte non può sbagliare perché non è una stima: è il simulatore che ha disegnato la scena.

**Dove gira:** in locale, gratis. Il modello (Wan2.1-VACE-1.3B) ha 1,3 miliardi di parametri e sta comodo nei tuoi 16 GB.

### Ramo B — Generazione libera (3 classi)

Monopattino, veicolo elettrico silenzioso, buca nell'asfalto.

**Tecnicamente.** Non esiste un asset CARLA per queste classi e non è possibile surrogarle: con un modello control-driven la sagoma la impone il segnale di controllo, quindi ridipingere una bicicletta come monopattino lascia la sagoma della bicicletta. L'oggetto va inventato dal generatore a partire dal testo, e le annotazioni devono uscire da SAM 3 e Depth Anything 3 applicati ai frame generati.

**In parole semplici.** Qui non c'è stampo. Scrivi «un monopattino parcheggiato su un marciapiede» e il modello se lo inventa. Il vantaggio è che i pixel vengono realistici da soli; il problema è che nessuno ti dice dove sta il monopattino nell'immagine, quindi lo deve trovare SAM 3.

**Dove gira:** su RunPod, perché il modello ad alta qualità (Wan2.2-T2V-A14B) ha 27 miliardi di parametri e non entra nei 16 GB.

### La simmetria da tenere a mente

| | Ramo A | Ramo B |
|---|---|---|
| Etichette | **esatte** (dal simulatore) | stimate (da SAM 3) |
| Realismo dei pixel | da ottenere ridipingendo | **già buono** |
| Rischio principale | il divario col mondo reale | l'annotazione sbagliata |
| Dove gira | locale, $0 | RunPod |

I due rami hanno difetti opposti, ed è il motivo per cui servono entrambi.

---

## 3. Il calcolo che dimensiona tutta la spesa

La voce più cara del piano è la produzione del ramo B, e dipende da un solo numero: quante clip servono. Il calcolo, con ogni fattore dichiarato in modo che se ne possa discutere uno alla volta:

| Fattore | Valore | Da dove viene |
|---|---|---|
| Classi del ramo B | 3 | monopattino, veicolo silenzioso, buca |
| Istanze annotate per classe | 150 | D4.1.3 §5.1 chiede 100–200 |
| **Istanze necessarie** | **450** | 3 × 150 |
| Frame generati per clip | 81 | nativo Wan 2.2 a 16 fps = 5,06 s |
| Passo temporale | ÷ 4 | regola tua, documento B.1–B.4 §2.1 |
| Frame indipendenti per clip | 20 | 81 ÷ 4 |
| Oggetto effettivamente in campo | × 0,80 | stima |
| Gate di qualità | × 0,60 | utilizzabilità dichiarata nel piano |
| SAM 3 individua l'oggetto | × 0,55 | **misurato**: 54,5 % sulla bicicletta |
| **Istanze utili per clip** | **5,3** | |
| **Clip necessarie** | **84 → si pianificano 90** | 450 ÷ 5,3 |

**In parole semplici.** Una clip da 5 secondi sembra darti 81 fotogrammi, ma fotogrammi consecutivi sono quasi identici e per un detector valgono uno solo. Tenendone uno ogni quattro ne restano 20; di quelli, alcuni non inquadrano l'oggetto, altri li scarta il controllo qualità, e su altri ancora SAM 3 non trova niente. Alla fine ogni clip ti regala cinque annotazioni buone. Per averne 450 servono novanta clip.

**Il fattore più fragile è l'ultimo**, il 55 % di SAM 3, ed è quello che la Sessione 3 misura davvero. Se sale al 70 %, le clip scendono a 66. Se scende al 40 %, salgono a 115.

---

# FASE 0 — Preparazione locale

**Dove:** tutto sul tuo PC. **Costo: $0.** **Durata: circa due settimane.**

Questa fase esiste per due motivi. Il primo è che ogni minuto sul pod è fatturato, quindi il codice va scritto e collaudato prima. Il secondo è che il passo 0.2 può rendere inutile l'intera Fase 1.

---

## Passo 0.1 — Alzare la memoria di WSL2

**Dove:** locale · **Costo:** $0 · **Tempo:** 5 minuti

**Tecnicamente.** Il tuo PC ha 32 GB di RAM fisica ma la macchina virtuale Linux ne vede circa 15, e negli sweep del 28 agosto la memoria host disponibile scendeva a 6,9 GB con l'occupazione del processo a 8,3 GB. Poiché lo scaricamento del text encoder su CPU (`enable_model_cpu_offload`) usa RAM host, è quello il vincolo stretto, non la VRAM.

**In parole semplici.** Il PC ha 32 GB di memoria ma Linux ne può usare solo la metà. Si cambia con un file di tre righe.

Crea `C:\Users\pc\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB
```

Poi da PowerShell: `wsl --shutdown` e riavvia la distribuzione.

**Perché è il primo passo:** senza questo, i due test successivi potrebbero fallire per un motivo che non c'entra col modello.

---

## Passo 0.2 — Il test che può cancellare RunPod

**Dove:** locale · **Costo:** $0 · **Tempo:** mezza giornata

**Tecnicamente.** Prima di assumere che il ramo B richieda il modello da 27 miliardi di parametri, va verificato se il Wan2.2-TI2V-5B — che gira già in locale, misurato a 15,72 GB su 73 frame a 704×1280 — produca clip su cui SAM 3 raggiunga un tasso di individuazione accettabile in modalità testuale. Il criterio non è visivo.

**In parole semplici.** Prova col modello che hai già in casa. Se il monopattino viene abbastanza chiaro che SAM 3 lo riconosce, non ti serve noleggiare niente.

Come si fa:

1. Genera 5 clip con `wan2_2/smoke_test.py` modificando il prompt, per esempio: *"POV shot from a person walking on a city sidewalk, an electric scooter parked ahead on the right, daylight, sharp focus, realistic urban environment"*.
2. Estrai un fotogramma ogni quattro.
3. Passali a SAM 3 con la descrizione testuale `"electric scooter"`, riusando `sam3_1/eval_on_carla.py`.
4. **Conta in quanti fotogrammi lo trova.**

**Il criterio di successo:** sopra il **54,5 %**, cioè quanto SAM 3 ha fatto sulla bicicletta reale di CARLA nelle verifiche B.1–B.4. Sopra quella soglia, il modello locale basta e la Fase 1 si riduce a niente. Sotto, serve il modello grande e la Fase 1 va fatta.

**Ripeti lo stesso test per la buca** (`"pothole"`) e per il veicolo silenzioso. La buca è la più a rischio: è sul piano stradale, poco contrastata, e potrebbe comportarsi molto peggio dei veicoli.

---

## Passo 0.3 — Ricalibrare il controllo qualità

**Dove:** locale · **Costo:** $0 · **Tempo:** 1 ora

**Tecnicamente.** Le soglie di `qa_export/quality_gate.py` sono calibrate a 720×480 (nitidezza minima 370, SSIM massima 0,95). La varianza del Laplaciano dipende dalla risoluzione — il file stesso documenta 4.736 a 1008×756 contro 2.109 a 720×480 sugli stessi frame. Generando a 1280×704 le soglie non sono più valide.

**In parole semplici.** Il controllo automatico che scarta le clip venute male è tarato su una risoluzione diversa da quella che userai. Se non lo ritari, scarta e tiene a caso, e non te lo dice.

Rilancia `quality_gate.py --calibrate` sui frame CARLA riportati a 1280×704, e annota le nuove soglie.

**Aggiungi anche un controllo di sicurezza:** il runner deve confrontare la risoluzione di generazione con quella di calibrazione e **rifiutarsi di partire** se non coincidono.

---

## Passo 0.4 — Scrivere `build_jobs.py`

**Dove:** locale · **Costo:** $0 · **Tempo:** mezza giornata

**Tecnicamente.** Trasforma il fabbisogno di copertura in un manifest JSONL, una riga per clip: identificativo stabile, classe B.O.S.S., prompt positivo e negativo, seed, numero di frame, risoluzione, numero di step, guidance. Il manifest è il contratto fra il locale e il pod.

**In parole semplici.** Un file di testo con dentro la lista della spesa: una riga per ogni video da generare, con scritto cosa deve contenere. Il pod legge quella lista e basta.

L'identificativo deve essere **stabile e derivato dal contenuto** (per esempio le prime cifre dell'hash di classe + prompt + seed), perché è la chiave su cui funziona la ripresa.

---

## Passo 0.5 — Scrivere `run_batch.py` e collaudarlo senza GPU

**Dove:** locale · **Costo:** $0 · **Tempo:** 2 giorni

È il pezzo più importante del piano. Cinque requisiti, in ordine di importanza economica.

**1. Riprendibilità.** All'avvio guarda cosa è già stato prodotto e salta quei job. Su Community Cloud l'istanza può essere revocata senza preavviso, e un runner che riparte da zero è il modo più diretto di trasformare un budget da $15 in uno da $150.

**2. Scrittura atomica.** Il video prima su un file temporaneo, poi rinominato; **solo dopo** si scrive il file che lo marca come completato. Se il pod muore a metà scrittura, il job risulta da rifare invece di risultare fatto con dentro un file troncato.

**3. Cattura del segnale di terminazione.** RunPod manda un SIGTERM prima di revocare un'istanza. Intercettandolo si chiude pulito il job in corso.

**4. Gate in linea.** `evaluate_clip()` è già scritta, e il suo commento dice già che è la funzione che `run_batch.py` importerà sul pod. Gira su CPU mentre la GPU produce la clip successiva, quindi non costa nulla, e ti dà il tasso di scarto in tempo reale.

**5. Contatore dei costi.** Ogni dieci clip stampa: quante fatte, secondi per clip, tasso di scarto, dollari maturati e proiezione a fine lotto.

**In parole semplici.** Il programma che sul pod genera i video uno dopo l'altro. La cosa più importante che deve saper fare è **riprendere da dove si era interrotto**: se il pod ti viene tolto a metà lavoro, alla riaccensione deve saltare quello che ha già fatto invece di ricominciare.

**Come si collauda senza GPU.** Scrivi un «generatore finto» che non carica nessun modello: dorme due secondi e scrive un mp4 di rumore. Poi:

1. Lancia il runner sul manifest completo col generatore finto.
2. **Ammazzalo a metà lotto.**
3. Rilancialo e verifica che riprenda dal punto giusto.

Se questo passo funziona, il rischio economico della Fase 1 è chiuso. Puoi rifarlo anche dentro un container Docker con la stessa immagine PyTorch che userai sul pod, così provi anche l'ambiente.

---

## Passo 0.6 — Scrivere `sync.py`

**Dove:** locale · **Costo:** $0 · **Tempo:** 2 ore

**Tecnicamente.** In questa architettura verso il pod non viaggia niente di pesante: solo il manifest, che è testo. In discesa vengono le clip, circa 1 MB l'una. Serve un `rsync` periodico che tiri giù la cartella degli output ogni pochi minuti.

**In parole semplici.** Uno script che ogni cinque minuti scarica sul tuo PC i video appena prodotti. Se il pod ti viene tolto mentre dormi, perdi al massimo l'ultima clip.

---

## Passo 0.7 — Provare il ramo A in locale

**Dove:** locale · **Costo:** $0 · **Tempo:** 1 giorno

**Tecnicamente.** Wan2.1-VACE-1.3B (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`) è supportato in `diffusers` tramite `WanVACEPipeline` e accetta condizionamento su profondità, posa e contorni. Con 1,3 miliardi di parametri sta nei 16 GB: il modello base dichiara 8,19 GB di VRAM.

**In parole semplici.** Il modello che ridipinge le scene CARLA gira sul tuo PC. Provalo su una delle sessioni che hai già registrato e cronometra quanto ci mette per clip.

**Perché conta:** questo passo produce **gratis** uno dei due numeri che oggi mancano al piano dei costi, cioè i secondi per clip del ramo A. E se funziona, la produzione delle ~280 clip del ramo A resta in casa a costo zero.

**Nota sull'hardware:** la tua RTX 5070 Ti (Blackwell, ~44 TFLOPS, 896 GB/s) è **più veloce di una A6000** (Ampere, 38,7 TFLOPS, 768 GB/s). Noleggiare la A6000 ti compra memoria, non velocità. Per qualunque modello che entri nei 16 GB, il locale è sia gratis sia più rapido.

---

### Fine Fase 0 — il bivio

| Esito del passo 0.2 | Cosa succede |
|---|---|
| SAM 3 sopra il 54,5 % col modello 5B | **La Fase 1 si annulla.** Tutto in locale, costo totale $0 |
| SAM 3 sotto il 54,5 % | Si procede con le quattro sessioni |

---

# FASE 1 — Le quattro sessioni RunPod

**Costo totale: $15,59.** Durata: circa tre giorni di calendario.

Regole valide per tutte le sessioni:

- **Community Cloud**, non Secure: costa il 40 % in meno, e l'istanza revocabile è accettabile solo perché il runner è riprendibile.
- **Terminate, non Stop.** Un pod fermato continua a fatturare il proprio disco a $0,20/GB al mese. Con i dati sul volume, terminare non perde nulla.
- **Container disk da 100 GB** in ogni sessione che carica il modello grande, perché il checkpoint deve pur atterrare da qualche parte. Costa circa un centesimo all'ora.
- **Volume di rete da 15 GB**, non da 100: ci vanno solo lo stato di ripresa e le clip in transito. Il checkpoint **non si conserva**, si riscarica.

---

## Sessione 1 — Imparare il ciclo

| | |
|---|---|
| **Hardware** | La GPU più economica disponibile (RTX 3090 24 GB, ~$0,22/h) |
| **Durata** | 1 ora |
| **Costo** | **$0,22** |
| **Modello caricato** | nessuno |

**Tecnicamente.** Si prova il ciclo completo di vita del pod su hardware irrilevante: creazione, aggancio del volume di rete, accesso SSH, `runpodctl`, terminazione. Si verifica che il volume sia effettivamente montato su `/workspace` e che sopravviva alla terminazione del pod.

**In parole semplici.** Un giro di prova per imparare i comandi, su una scheda che non costa niente. Serve a non imparare a $0,69 l'ora.

**Riduci ulteriormente questa voce:** le parti che riguardano il container (comandi, mount, esecuzione del runner) le puoi provare **gratis in locale con Docker**, usando la stessa immagine PyTorch. Sul pod resta solo il flusso della console RunPod, che sono dieci minuti.

**Cosa ti porti a casa:** una procedura scritta di accensione e spegnimento, con i comandi esatti.

---

## Sessione 2 — Scaricare i modelli e misurare la banda

| | |
|---|---|
| **Hardware** | La GPU più economica disponibile (~$0,22/h) |
| **Durata** | 2 ore |
| **Costo** | **$0,44** |
| **Modelli caricati** | Wan2.2-T2V-A14B (64 GB) e Wan2.2-TI2V-5B (~10 GB) |

**Tecnicamente.** Il download è puro trasferimento e non usa la GPU: va eseguito sulla scheda meno cara disponibile nella region del volume. In questa sessione si installa anche l'ambiente (`diffusers` allineato alla 0.40.0 usata in locale, per confrontabilità con lo sweep di agosto) e si cronometra la velocità di download da Hugging Face.

**In parole semplici.** Scarichi i due modelli e prepari l'ambiente. Non serve una scheda potente: stai solo scaricando file.

**La misura che conta:** cronometra **quanto ci mette a scaricare i 64 GB**. A 400 MB/s sono tre minuti; a 50 MB/s sono venti. Da quel numero dipende una decisione:

| Velocità misurata | Decisione |
|---|---|
| Sopra ~200 MB/s | Non conservare il checkpoint. Riscaricarlo ogni sessione costa centesimi |
| Sotto ~100 MB/s | Conviene un volume più grande che lo tenga (+$4,50 al mese) |

**Cosa ti porti a casa:** l'ambiente pronto in uno script (`setup_pod.sh`) che nelle sessioni successive lo ricostruisce in un colpo solo.

---

## Sessione 3 — Il test che decide il modello

| | |
|---|---|
| **Hardware** | RTX A6000 48 GB — $0,33/h |
| **Durata** | 6 ore |
| **Costo** | **$1,98** |
| **Modelli** | entrambi, confrontati |

**Tecnicamente.** Si generano gli stessi prompt con gli stessi seed, stessi step e stessa risoluzione su Wan2.2-TI2V-5B e Wan2.2-T2V-A14B, sulle tre classi del ramo B. Si misura, scaricando le clip e annotandole in locale, il tasso di individuazione di SAM 3 in modalità testuale e lo IoU mediano. Si registra anche il tempo per clip di ciascun modello, che dimensiona la Sessione 4.

**In parole semplici.** Generi gli stessi video con il modello piccolo e con quello grande, cambiando **solo** il modello. Poi guardi quale dei due produce monopattini che SAM 3 riesce a trovare. Vince quello, non quello più bello.

**Perché la A6000 e non altro:** non serve la potenza, servono i 48 GB. Con quelli non hai dubbi di capienza mentre stai misurando altro; se il test fallisse per memoria non sapresti se è colpa del modello o della scheda.

**Parametri obbligatori, identici sui due modelli:**

| | |
|---|---|
| Risoluzione | **1280×704** (nativo Wan 2.2, orizzontale) |
| Frame | **81** a 16 fps = 5,06 s |
| Step | 50 |
| Seed | fisso e registrato |

> **Non generare sopra il nativo.** Spingere a 1080p porta il modello fuori distribuzione e peggiora il risultato. 1280×704 sono già 2,9 volte i pixel dei 640×480 del modello edge, che è quello che serve: **generi alto, annoti alto, ridimensioni per addestrare**.

**Come si spendono le 6 ore:** circa 4 sul modello grande (8–12 clip) e 2 sul piccolo (15–25 clip). Tre o quattro clip per classe per modello bastano: ognuna dà ~20 fotogrammi indipendenti, quindi il campione per classe è di una sessantina di immagini.

**Cosa ti porti a casa:** il modello vincitore, il tasso di individuazione per classe, i secondi per clip, e la risposta alla domanda se l'A14B stia nei 32 GB di una 5090 quantizzato a 8 bit.

---

## Sessione 4 — Produzione

| | |
|---|---|
| **Hardware** | RTX 5090 32 GB — $0,69/h *(oppure A6000 se il modello non ci sta)* |
| **Durata** | 15 ore *(oppure 37,5 sull'A6000)* |
| **Costo** | **$10,35** *(oppure $12,38)* |
| **Modello** | il vincitore della Sessione 3 |
| **Prodotto** | **90 clip** |

**Tecnicamente.** Si esegue `run_batch.py` sul manifest completo, con il gate in linea e la sincronizzazione periodica verso il PC. La scelta della scheda dipende dall'esito della Sessione 3: la 5090 ha FP8 nativo, che l'A6000 (Ampere) non ha, e chiude lo stesso lavoro in circa un terzo del tempo a un costo totale inferiore.

**In parole semplici.** Qui si produce sul serio: novanta video. Lanci il programma e lo lasci andare, mentre a casa un altro script ti scarica i risultati man mano.

**Il confronto fra le due schede:**

| | RTX 5090 32 GB | RTX A6000 48 GB |
|---|---|---|
| Prezzo | $0,69/h | $0,33/h |
| FP8 nativo | **sì** | no |
| Minuti per clip (stima) | ~10 | ~25 |
| Ore per 90 clip | **15** | 37,5 |
| **Costo totale** | **$10,35** | $12,38 |

La 5090 **costa meno e finisce in un terzo del tempo**. L'unico dubbio è la capienza, ed è esattamente ciò che la Sessione 3 verifica.

**Cosa ti porti a casa:** 90 clip sul PC, più il log con tasso di scarto e tempi.

---

## Riepilogo Fase 1

| Voce | Hardware | $/h | Ore | Costo |
|---|---|---|---|---|
| Sessione 1 — apprendimento | RTX 3090 24 GB | 0,22 | 1 | $0,22 |
| Sessione 2 — download e ambiente | RTX 3090 24 GB | 0,22 | 2 | $0,44 |
| Sessione 3 — test comparativo | RTX A6000 48 GB | 0,33 | 6 | $1,98 |
| Sessione 4 — produzione 90 clip | RTX 5090 32 GB | 0,69 | 15 | $10,35 |
| Volume di rete 15 GB × 2 mesi | — | 0,07/GB | — | $2,10 |
| Container disk e riscaricamenti | — | — | — | ~$0,50 |
| | | | **Totale** | **$15,59** |

**Tetto con imprevisti: $19,60**, che comprende una seconda sessione di test ($1,98) se il primo confronto non fosse conclusivo, e la Sessione 4 sull'A6000 anziché sulla 5090.

---

# FASE 2 — Annotazione, controllo, export

**Dove:** tutto sul tuo PC. **Costo: $0.**

Questa fase non compare nel preventivo del consulente ed è la più lunga delle tre.

---

## Passo 2.1 — Produzione del ramo A

**Dove:** locale · **Costo:** $0 · **Tempo:** 3–4 notti di macchina

**Tecnicamente.** Si esegue `run_batch.py` con il generatore Wan2.1-VACE-1.3B, alimentato dai segnali di controllo estratti dalle sessioni CARLA. Le ~280 clip del ramo A a 8–10 minuti l'una sono 37–47 ore di GPU locale.

**In parole semplici.** Le clip delle classi che CARLA sa fare le generi a casa, di notte, gratis. Il PC è occupato, ma non paghi niente.

---

## Passo 2.2 — Annotazione automatica

**Dove:** locale · **Costo:** $0

**Tecnicamente.** SAM 3 produce le maschere (picco misurato 4,35 GB), Depth Anything 3 taglia Base le mappe di profondità (1,49 GB), entrambi ampiamente entro i 16,3 GB. Per il ramo A la profondità di riferimento resta quella di CARLA; per il ramo B c'è solo la stima di DA3.

**In parole semplici.** Due programmi che guardano ogni fotogramma e dicono «qui c'è un monopattino, ed è a tre metri». Girano sul tuo PC senza problemi: lo hai già misurato.

**Una cosa da sapere sui risultati**, dal tuo documento B.1–B.4:

| Tipo di oggetto | Maschera | Profondità |
|---|---|---|
| Poggiati a terra (gradino, bici, monopattino) | buona | **buona** (δ 0,88–0,99) |
| Sospesi (rami, insegne) | **ottima** (IoU 0,94–0,99) | **inutilizzabile** (δ 0,00) |
| Sottili (pali, cavi, semafori) | **scarsa** (IoU 0,10–0,12) | — |

Per gli ostacoli sospesi la profondità la dà CARLA, non DA3. Per i sottili la maschera va corretta a mano.

---

## Passo 2.3 — Controllo qualità strutturale

**Dove:** locale · **Costo:** $0

**Tecnicamente.** Per ogni clip si calcolano δ₁ e RMSE relativo fra la profondità della sorgente e quella del generato, lo IoU delle maschere di classe, e la varianza frame-a-frame dello IoU come indice di stabilità temporale. Soglie: δ₁ ≥ 0,85, IoU ≥ 0,70, σ ≤ 0,10.

**In parole semplici.** Un controllo che verifica se il video generato ha rispettato la geometria dell'originale. Vale per il ramo A; per il ramo B non c'è un originale con cui confrontarsi, quindi lì il controllo è il tasso di individuazione di SAM 3.

---

## Passo 2.4 — Correzione umana ed export

**Dove:** locale · **Costo:** $0 in denaro, **5–12 ore-uomo**

**Tecnicamente.** Export in COCO e COCO-Video, manifest con `is_synthetic` e SHA-256, commit su lakeFS.

**In parole semplici.** Metti tutto nel formato che serve all'addestramento, con scritto quali immagini sono sintetiche.

> **Attenzione, questa è la voce più cara di tutto l'OR 4.1.** Il D4.1.3 chiede 800–1.500 fotogrammi annotati e verificati. Con pre-annotazione automatica e sola correzione umana sono **5–12 ore-uomo**; da zero sarebbero 15–50. Sono due ordini di grandezza sopra i $15 di GPU, e oggi non compaiono in nessun documento di progetto. **Vanno pianificate esplicitamente**, e sono il vero motivo per cui la pre-annotazione automatica non è un optional.

---

# Confronto col preventivo del consulente

| | Scenario | Cosa comprava | Costo |
|---|---|---|---|
| Consulente | PoC economica | Test 5B + Fun-Control + Open-Sora | $11,55 |
| Consulente | PoC completa | + CogVideoX 5B remoto + Wan2.2 A14B | $24,65 |
| Consulente | PoC estesa | + Fun-Control A14B + Open-Sora 2.0 | $55,45 |
| **Questo piano** | | **Test + 90 clip di produzione + storage** | **$15,59** |

**Le differenze, in ordine di importanza:**

**1. Qui c'è la produzione.** I tre scenari del consulente comprano solo test: zero clip utilizzabili nel dataset. Questo piano ne produce novanta.

**2. Le ore sono dimensionate, non forfettarie.** Il preventivo assegna a ogni test un blocco uniforme di 10 ore dichiarate «simulate», e i suoi totali (35, 55 e 75 GPU-h) sono multipli di quel valore. Qui le ore vengono da clip × tempo per clip.

**3. Lo storage esiste.** Il preventivo non lo contempla, ed è una voce reale.

**4. Due voci sono state tolte perché non decidono nulla.** Open-Sora 2.0 su H100 ($19,90) e il test Wan2.2 A14B non condizionato ($10,90) generano video **non condizionati**: non producono una coppia sorgente-generato su cui calcolare δ₁ e IoU, quindi non entrano nel criterio di accettazione. *(La riga A14B rientra però nel ramo B, dove la generazione non condizionata è esattamente la funzione richiesta.)*

**5. Una riga del preventivo non ha riscontro.** «Wan2.2-Fun-Control 5B, checkpoint ~23 GB» non esiste: la serie Wan2.2-Fun ha tre varianti, tutte A14B da 64 GB. Il control-driven a bassa taglia esiste su Wan2.1, ed è la strada del ramo A.

**6. Il preventivo sottostima la workstation.** Indica CogVideoX-2b a «~12,5 GB» quando lo sweep misura 7,44 GB, e dà Wan2.2-TI2V-5B come non eseguibile in locale quando arriva a 73 frame. Entrambi gli errori spingono verso il noleggio.

---

# Le cinque trappole

**1. Le soglie del gate dipendono dalla risoluzione.** Calibrate a 720×480; a 1280×704 non valgono più, e il gate sbaglia in silenzio. Passo 0.3.

**2. Il container disk è effimero.** Quanto non sta sul volume sparisce alla terminazione. Lo stato di ripresa va sul volume, non sul disco del container.

**3. Il volume è legato a una region.** Verifica la disponibilità effettiva della scheda **prima** di creare il volume, altrimenti ti ritrovi con lo storage dove la GPU non è mai libera.

**4. «Stop» non è «Terminate».** Un pod fermato continua a fatturare il disco a $0,20/GB al mese.

**5. Nel confronto della Sessione 3, un solo parametro alla volta.** Se cambi risoluzione insieme al modello, stai confrontando la risoluzione. È la trappola che il tuo stesso `wan2_1/smoke_test.py` documenta in testa al file.

---

# Le tre decisioni ancora aperte

**1. Le classi del ramo B.** Il piano assume monopattino, veicolo elettrico silenzioso e buca. Se se ne aggiunge una quarta, le clip passano da 90 a 120 e la Sessione 4 da $10,35 a $13,80.

**2. Il numero di istanze per classe.** Il piano usa 150, che è il centro dell'intervallo 100–200 del D4.1.3. A 100 le clip scendono a 60; a 200 salgono a 113.

**3. Chi esegue le 5–12 ore di annotazione umana**, e su quale budget. È la voce dominante dell'intero OR 4.1 e non compare in nessun documento di progetto.

---

# Riepilogo finale

| Fase | Dove | Costo | Durata |
|---|---|---|---|
| **0 — Preparazione** | PC locale | **$0** | ~2 settimane |
| **1 — Generazione ramo B** | RunPod, 4 sessioni | **$15,59** | ~3 giorni |
| **2 — Ramo A, annotazione, export** | PC locale | **$0** + 5–12 ore-uomo | ~1 settimana |
| | | **$15,59** | **~4 settimane** |

**E ricorda il bivio:** se il passo 0.2 va bene, la Fase 1 si annulla e il totale diventa **$0**.

---

*Fonti dei prezzi: listino RunPod Community Cloud, 9 settembre 2026. Misure locali: sweep VRAM del 28 agosto 2026 e documento «Verifiche locali degli annotatori automatici B.1–B.4», v1.0 dell'8 settembre 2026. Modelli verificati sull'organizzazione ufficiale Wan-AI di Hugging Face.*
