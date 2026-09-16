# BOSS_CODE

Codice del sottoprogetto **OR4** (riconoscimento ostacoli e calcolo distanze via Edge AI)
del progetto B.O.S.S.: dispositivo wearable di assistenza alla mobilità per non vedenti.

Questo documento descrive **come è organizzata la repository**: cosa c'è in ogni cartella,
a cosa serve, e cosa ci finirà. Non è un documento di progetto — piano di lavoro,
decisioni e risultati misurati vivono nei documenti OR4.1 / OR4.2 / OR4.3, fuori da qui.

## La regola che decide cosa sta qui

> **Su Git va ciò che gira sul pod RunPod, più il verbale delle fasi chiuse.
> Sul disco locale resta ciò che serve solo a provare questa workstation.**

Non tutto il codice del progetto è versionato, ed è voluto. Gli `smoke_test.py` e i
`vram_benchmark.py` provano che un ambiente **locale** funziona: non producono un
risultato, e sul pod la stessa funzione la fa `cloud/runner.py --limit 1`. Stesso
discorso per `wan2_2/night_run.py`, il watchdog dei run notturni, tarato su una 5070 Ti
desktop. Sono in `.gitignore`, restano sul disco, e **la storia di git li conserva**:
`git log --diff-filter=D --name-only` li elenca, `git show <commit>:<path>` li recupera.

Conseguenza da conoscere: un clone fresco su una macchina nuova non ricostruisce
l'ambiente locale. Ricostruisce quello del pod, che è lo scopo.

## Struttura

```
BOSS_CODE/
└── synthetic_data_pipeline/     generazione e annotazione di dati sintetici
    ├── CARLA/                   reference set di ground truth da simulatore
    ├── cloud/                   tutto ciò che gira su RunPod
    ├── envs/                    ambienti riproducibili, un requirements per venv
    ├── sam3_1/                  SAM 3: valutazione contro la ground truth CARLA
    ├── depth_anything3/         Depth Anything 3: idem
    ├── qa_export/               (vuota) export del dataset nei formati di consegna
    ├── cogvideox/               (fuori da git) CogVideoX-2b, superata
    ├── wan2_1/                  (fuori da git) Wan2.1-T2V-1.3B, superata
    └── wan2_2/                  (fuori da git, salvo generate_test_clips.py)
```

Il flusso è: **sorgente geometrica → generazione video → annotazione → export**.

## Come circolano codice e dati

Il codice si scrive qui, su Windows, in VS Code. Il pod RunPod è usa e getta. Tre
canali distinti, e nessuno passa per il PC in mezzo:

| Cosa | Come | Direzione |
|---|---|---|
| **Codice** (~100 KB) | git, via GitHub | PC ↔ pod, bidirezionale |
| **Pesi dei modelli** (~240 GB) | `hf download`, con `HF_HOME` sul network volume | HuggingFace → pod |
| **Dati generati** (~15-20 GB) | repo dataset HuggingFace privato | pod → HF → PC |

Sul pod il repo è un clone normale su `/workspace`, cioè sul network volume: sopravvive
al Terminate, e i pod successivi fanno solo `git pull`.

## synthetic_data_pipeline/

### CARLA/

Costruzione del **reference set di ground truth** con il simulatore CARLA 0.9.16: un
pedone virtuale cammina portando quattro sensori allineati, e la sessione registrata
fornisce depth densa, segmentazione e posa esatte *per costruzione*. È la sorgente
geometrica della pipeline e il riferimento metrico contro cui si misurano gli
annotatori. **Fase chiusa** (Gate A raggiunto): gira solo in locale, richiede il server
CARLA, e sul pod non arriva mai.

| File | Cosa fa |
|---|---|
| `boss_classes.py` | Le 14 classi B.O.S.S. e la corrispondenza con i tag semantici di CARLA. Solo dati, zero import: lo importa anche `cloud/` |
| `scenarios.py` | Registro degli scenari di acquisizione: mappa, meteo, traffico, ostacoli da piazzare |
| `carla_capture.py` | Registra una sessione. Richiede il server CARLA in esecuzione |
| `carla_export.py` | Trasforma una sessione in annotazioni: classe, box 2D e distanza metrica per istanza |
| `carla_gt.py` | Lettore unico della ground truth, condiviso dai tre venv. Dipende solo da cv2 e numpy |
| `build_eval_set.py` | Seleziona i frame del reference set e scrive `eval_set.json` |
| `eval_set.json` | **Il reference set: 58 frame su 3 sessioni.** Unico dato versionato del repo |
| `requirements.txt` | Pin dell'ambiente CARLA |

Questi script restano versionati perché sono la **provenienza di `eval_set.json`**: un
dato senza il metodo che l'ha prodotto non vale.

Il server CARLA non sta nel repo: è un pacchetto da ~20 GB installato separatamente.

### cloud/

**Tutto ciò che gira sul pod.** Autocontenuta: nessun import fuori da qui, salvo
`CARLA/boss_classes.py` che è solo dati.

| File | Cosa fa |
|---|---|
| `adapters/` | Un modulo per generatore video. Stesso contratto `SPEC` / `load()` / `generate()` / `unload()`, così il resto della pipeline non sa quale modello ha prodotto i frame |
| `runner.py` | Generatore a lotti: riprendibile, scrittura atomica, cattura `SIGTERM`, gate in linea, contatore costi |
| `prompts.py` | I prompt di generazione per classe, più il negative prompt |
| `build_jobs.py` | Prodotto cartesiano modelli × prompt × seed → `jobs.jsonl` |
| `annotate_sam.py` | SAM 3 sui frame generati: tasso di individuazione per frame, maschere, overlay |
| `annotate_da3.py` | Depth Anything 3 sulle stesse maschere: coerenza di profondità. Dipende dal CSV di `annotate_sam.py`, va lanciato dopo |
| `gpu.py` | Telemetria `nvidia-smi` e guardie di memoria, con le soglie come argomenti invece che cablate su una scheda |
| `quality_gate.py` | Controllo qualità su CPU (nitidezza, SSIM). Le soglie sono calibrate a 720×480 e non valgono ad altre risoluzioni: il runner ne registra i valori grezzi, non filtra |
| `compare_models.py` | Unisce manifest e CSV: tabella modello × metriche |
| `sync.py` | Push/pull incrementale verso il dataset HuggingFace privato |
| `setup_pod.sh` | Crea i venv dai requirements, punta `HF_HOME` al network volume, clona i repo upstream ai commit pinnati |

### envs/

Un `requirements-*.txt` per ambiente, rilevati dai venv reali il 15/09/2026.

| File | Per cosa | Vincolo che lo tiene separato |
|---|---|---|
| `requirements-gen.txt` | generatori: Wan2.2-5B, CogVideoX1.5, HunyuanVideo-1.5 | `numpy==2.4.6` |
| `requirements-a14b.txt` | Wan2.2-T2V-A14B | vuole `diffusers` da branch `main` |
| `requirements-ann.txt` | SAM 3 + Depth Anything 3 | `numpy<2`, dichiarato da entrambi i pacchetti upstream |

**`numpy` è la ragione dei venv separati**, non una convenzione: i generatori vogliono
la 2.4.6, gli annotatori la 1.26.4. In locale SAM 3 e DA3 stanno in due venv distinti,
ma sul pod possono condividerne uno: confrontati i due `pip freeze`, i 70 pacchetti in
comune hanno versioni identiche e zero conflitti.

### sam3_1/ · depth_anything3/

I due **annotatori**, misurati contro la ground truth CARLA nelle verifiche B.1–B.4.
Qui resta solo `eval_on_carla.py` per ciascuno: ha prodotto numeri che i documenti di
progetto citano ancora, quindi è verbale di una fase chiusa. Gli script che annotano i
frame *generati* sono passati in `cloud/`.

Ogni cartella contiene anche il repository upstream del modello, clonato e non
versionato (`sam3_1/sam3/`, `depth_anything3/Depth-Anything-3/`), e una `test_images/`
esclusa da git.

**Nota su SAM 3 e non SAM 3.1**: nonostante il nome della cartella, per l'inferenza su
immagine singola va usato il checkpoint `sam3`. Il 3.1 esiste solo come modello video
multiplex, e `build_sam3_image_model` lo accetta senza errore lasciando pesi casuali su
un livello FPN. Il controllo di copertura delle chiavi all'avvio è ciò che trasforma
quel caso in un errore invece che in numeri sbagliati: non va rimosso.

### qa_export/

**Vuota.** Ospiterà l'export del dataset nei formati di consegna (COCO / COCO-Video),
con il manifest e il versionamento. Il controllo qualità che stava qui è passato in
`cloud/quality_gate.py`, perché gira sul pod dentro il runner.

### cogvideox/ · wan2_1/ · wan2_2/

Fuori da git per la regola in testa a questo documento. Restano sul disco come verbale
di misure fatte: lo sweep VRAM del 28/08 che fissa il punto operativo di Wan2.2-5B a 73
frame, e gli smoke test che hanno provato i tre ambienti. `wan2_2/generate_test_clips.py`
è ancora versionato finché `cloud/adapters/wan22_5b.py` non avrà dimostrato di
riprodurre la stessa clip.

## object_detection/

Non ancora creata. Ospiterà il codice di OR4.2 e OR4.3: addestramento dei modelli
multi-task (object detection e stima di profondità con backbone condiviso), valutazione
e ottimizzazione per il deployment su edge.

## Convenzioni

- **Un `.venv` per ambiente, mai condiviso**: i vincoli di versione dei diversi modelli
  sono incompatibili fra loro. I venv non sono versionati, le loro ricette sì (`envs/`).
- **Ogni script ha i propri parametri in testa al file**, come costanti maiuscole con il
  commento del perché di quel valore. Le eccezioni sono dichiarate: `scenarios.py` in
  `CARLA/`, e le soglie hardware in `cloud/`, che sono argomenti perché il pod cambia
  scheda fra una sessione e l'altra.
- **`outputs/` non è mai versionata**: contiene video, sessioni e CSV, cioè risultati
  rigenerabili e pesanti.
- **I repository upstream dei modelli sono clonati, non copiati nel repo**, e sono
  esclusi da git. I commit che servono sono registrati in `envs/requirements-ann.txt`.
