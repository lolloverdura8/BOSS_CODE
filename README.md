# BOSS_CODE

Codice del sottoprogetto **OR4** (riconoscimento ostacoli e calcolo distanze via Edge AI)
del progetto B.O.S.S.: dispositivo wearable di assistenza alla mobilità per non vedenti.

Questo documento descrive **come è organizzata la repository**: cosa c'è in ogni cartella,
a cosa serve, e cosa ci finirà. Non è un documento di progetto — piano di lavoro,
decisioni e risultati misurati vivono nei documenti OR4.1 / OR4.2 / OR4.3, fuori da qui.

## Struttura

```
BOSS_CODE/
├── synthetic_data_pipeline/     generazione e annotazione di dati sintetici
│   ├── CARLA/                   reference set di ground truth da simulatore
│   ├── cogvideox/               generatore video CogVideoX-2b
│   ├── wan2_1/                  generatore video Wan2.1-T2V-1.3B
│   ├── wan2_2/                  generatore video Wan2.2-TI2V-5B
│   ├── sam3_1/                  segmentazione con SAM 3.1
│   ├── depth_anything3/         stima di profondità con Depth Anything 3
│   ├── cloud/                   orchestrazione della generazione su GPU noleggiata
│   ├── envs/                    definizioni di ambiente riproducibili
│   └── qa_export/               controllo qualità ed export del dataset
└── object_detection/            addestramento e valutazione dei modelli OR4.2 / OR4.3
```

## synthetic_data_pipeline/

La pipeline che produce dati di addestramento sintetici per le classi di ostacolo che
non è possibile o sicuro riprendere dal vero. Il flusso è: **sorgente geometrica →
generazione video → annotazione → export**.

### CARLA/

Costruzione del **reference set di ground truth** con il simulatore CARLA 0.9.16: un
pedone virtuale cammina portando quattro sensori allineati all'altezza del torace, e la
sessione registrata fornisce depth densa, segmentazione e posa esatte *per costruzione*.
È la sorgente geometrica della pipeline e il riferimento metrico contro cui si misurano
gli annotatori.

| File | Cosa fa |
|---|---|
| `boss_classes.py` | Le 14 classi B.O.S.S. e la corrispondenza con i tag semantici di CARLA. Solo dati, nessun codice |
| `scenarios.py` | Registro degli scenari di acquisizione: mappa, meteo, traffico, ostacoli da piazzare |
| `carla_capture.py` | Registra una sessione. Richiede il server CARLA in esecuzione |
| `carla_export.py` | Trasforma una sessione in annotazioni: classe, box 2D e distanza metrica per istanza. Non tocca il server |
| `requirements.txt` | Pin dell'ambiente |
| `outputs/` | Sessioni registrate ed export (ignorato da git) |

Il server CARLA non sta nel repo: è un pacchetto da ~20 GB installato separatamente, a
cui gli script si connettono via rete.

### cogvideox/ · wan2_1/ · wan2_2/

Tre **generatori video candidati** per la stessa funzione: produrre clip a partire dal
reference set. Hanno la stessa struttura a due script.

| File | Cosa fa |
|---|---|
| `smoke_test.py` | Una singola generazione completa, per verificare l'ambiente end-to-end e produrre un video ispezionabile |
| `vram_benchmark.py` | Sweep di profilazione: memoria e tempo al variare del numero di frame, per stabilire il limite della macchina. Scrive un CSV in `outputs/` |

I due script di ciascuna cartella condividono risoluzione, prompt, seed e parametri di
generazione, così il video dello smoke test è la controparte visiva della riga
corrispondente del CSV.

- **`cogvideox/`** — CogVideoX-2b, 480×720. Gira sotto WSL2/Ubuntu, non su Windows
  nativo.
- **`wan2_1/`** — Wan2.1-T2V-1.3B, 832×480.
- **`wan2_2/`** — Wan2.2-TI2V-5B, 704×1280. Il modello eccede la VRAM disponibile su
  questa workstation: la cartella è conservata come record riproducibile e come punto di
  partenza per un test su hardware adeguato. Pesi (~35 GB nella cache HuggingFace) e venv
  (~4,7 GB) sono ancora sul disco e vanno rimossi quando servirà spazio.

### sam3_1/ · depth_anything3/

I due **annotatori** che etichettano i frame generati: maschere di segmentazione e mappe
di profondità. Entrambe le cartelle contengono uno `smoke_test.py`, una cartella
`test_images/` e il repository upstream del modello, clonato e non versionato qui
(`sam3_1/sam3/`, `depth_anything3/Depth-Anything-3/`).

Lavorano su singola immagine, quindi girano in locale senza bisogno di GPU noleggiata.

### cloud/

**Vuota.** Ospiterà l'orchestrazione della generazione su GPU noleggiata: costruzione
del manifest dei job, runner riprendibile lato pod, sincronizzazione dei file in
entrambe le direzioni.

### envs/

**Vuota.** Ospiterà le definizioni di ambiente riproducibili. Oggi ogni cartella ha il
proprio `.venv` isolato — i modelli hanno vincoli di versione incompatibili fra loro — e
la ricetta di ciascun ambiente è descritta a parole invece che in un file.

### qa_export/

**Vuota.** Ospiterà il controllo qualità strutturale dei clip generati e l'export del
dataset nei formati di consegna, con il manifest e il versionamento.

## object_detection/

**Vuota.** Ospiterà il codice di OR4.2 e OR4.3: addestramento dei modelli multi-task
(object detection e stima di profondità con backbone condiviso), valutazione e
ottimizzazione per il deployment su edge.

## Convenzioni

- **Un `.venv` per cartella di modello**, mai condiviso: i vincoli di versione dei
  diversi modelli sono incompatibili fra loro. I venv non sono versionati.
- **Ogni script ha i propri parametri in testa al file**, come costanti maiuscole con il
  commento del perché di quel valore. Non ci sono file di configurazione esterni, salvo
  `scenarios.py` in `CARLA/`.
- **`outputs/` non è mai versionata**: contiene video, sessioni e CSV, cioè risultati
  rigenerabili e pesanti.
- **I repository upstream dei modelli sono clonati, non copiati nel repo**, e sono
  esclusi da git.
