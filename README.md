# BOSS_CODE

Codice del sottoprogetto **OR4** del progetto B.O.S.S. (riconoscimento ostacoli e calcolo
delle distanze con Edge AI). B.O.S.S. è un dispositivo indossabile che aiuta le persone
non vedenti a muoversi.

Questo documento descrive **come è organizzata la repository**: cosa c'è in ogni cartella,
a cosa serve e dove gira. Il piano di lavoro, le decisioni e i risultati misurati stanno
nei documenti OR4.1, OR4.2 e OR4.3, fuori da qui.

## La regola che decide cosa sta qui

> **Su Git va ciò che gira sul pod RunPod, più il verbale delle fasi chiuse.
> Sul disco locale resta ciò che serve solo a provare questa workstation.**

Non tutto il codice del progetto è versionato, ed è voluto:
- gli `smoke_test.py` e i `vram_benchmark.py` provano che un ambiente **locale** funziona,
  ma non producono un risultato. Sul pod la stessa funzione la fa `cloud/runner.py --limit 1`;
- sono in `.gitignore` e **la storia di git li conserva**:
  `git log --diff-filter=D --name-only` li elenca, `git show <commit>:<path>` li recupera.

Una conseguenza da conoscere: un clone fresco su una macchina nuova non ricostruisce
l'ambiente locale. Ricostruisce quello del pod, ed è proprio lo scopo.

## Struttura

```
BOSS_CODE/
├── README.md
├── ruff.toml                    configurazione del linter, unica per tutto il repo
└── synthetic_data_pipeline/     generazione e annotazione di dati sintetici
    ├── CARLA/                   ground truth da simulatore e pacchetti per il Sim2Real
    ├── cloud/                   tutto ciò che gira su RunPod
    ├── envs/                    ricette dei venv del pod, un file per ambiente
    ├── sam3_1/                  SAM 3: valutazione contro la ground truth CARLA
    ├── depth_anything3/         Depth Anything 3: idem
    ├── cogvideox/               (fuori da git) smoke test e benchmark di agosto
    ├── qa_export/               (fuori da git) export del dataset, ancora da scrivere
    └── bakeoff_clip/            (fuori da git) clip scaricate dal pod e loro valutazioni
```

La pipeline ha due rami che finiscono nello stesso punto:

- **text-to-video**: prompt → generatore → annotazione (SAM 3 + DA3) → valutazione;
- **Sim2Real**: sessione CARLA → pacchetto di controlli → modello video-to-video →
  clip con la ground truth già allineata.

## Dove gira cosa

| Cosa | Dove | Perché |
|---|---|---|
| Scrittura del codice, test senza GPU | Windows, VS Code | — |
| SAM 3 sulle clip scaricate | Windows, venv `sam3_1/.venv` | entra nei 16 GB della 5070 Ti |
| Depth Anything 3 | WSL2 (`~/or4-ann`) o pod | su Windows Smart App Control blocca `pycolmap._core` |
| Generatori video, Cosmos, Wan Fun | pod RunPod | servono 48-80 GB di VRAM |
| Server CARLA | pod RunPod | sulla workstation scalda troppo la CPU (05/10/2026) |

Sul pod si lavora solo a mano: i comandi li lancia l'utente, uno alla volta.

## Come circolano codice e dati

| Cosa | Come | Direzione |
|---|---|---|
| **Codice** | git, via GitHub (ramo `LorenzoV`) | PC → GitHub → pod |
| **Pesi dei modelli** | `hf download`, con `HF_HOME=/workspace/hf` sul network volume | HuggingFace → pod |
| **Clip e risultati** | `tar` + `scp` verso `bakeoff_clip/` (c'è anche `cloud/sync.py`, per un dataset HF privato) | pod → PC |

Sul pod il repo è un clone normale in `/workspace/BOSS_CODE`, quindi sul network volume:
sopravvive al Terminate, e i pod successivi fanno solo `git pull`. Sul volume stanno anche
i venv (`/workspace/envs/`), i clone dei repo upstream e i checkpoint.

## synthetic_data_pipeline/

### CARLA/

Un pedone virtuale cammina nel simulatore CARLA 0.9.16 portando quattro sensori
allineati. Ogni sessione registrata dà depth densa, segmentazione e posa esatte *per
costruzione*. Serve a due cose:

1. **reference set di ground truth**, contro cui si misurano gli annotatori (fase A,
   chiusa);
2. **sorgente del ramo Sim2Real**: le scene con gli ostacoli di B.O.S.S. diventano
   pacchetti di controlli per Cosmos-Transfer2.5 e Wan2.2-Fun.

| File | Cosa fa |
|---|---|
| `boss_classes.py` | Le 14 classi B.O.S.S. e la corrispondenza con i tag semantici di CARLA. Solo dati, zero import: lo importa anche `cloud/` |
| `scenarios.py` | Registro degli scenari: mappa, meteo, traffico, ostacoli. Lo scenario finisce testualmente in `session.json` |
| `scene_geometry.py` | Geometria pura del piazzamento degli attori, senza import di carla |
| `scene_actors.py` | Costruzione degli attori: prop, sostegni, ancoraggio agli alberi della mappa, il mezzo che arriva incontro |
| `carla_capture.py` | Registra una sessione. Richiede il server CARLA in esecuzione |
| `acceptance.py` | Controlli A1-A9 su una sessione: intercettano errori che non sollevano eccezioni |
| `preview_session.py` | Anteprima di una sessione (mp4 con le box e foglio di 12 frame), da guardare prima di usarla |
| `carla_export.py` | Trasforma una sessione in annotazioni: classe, box 2D e distanza metrica per istanza |
| `carla_gt.py` | Lettore unico della ground truth. Dipende solo da cv2 e numpy |
| `build_eval_set.py` | Seleziona i frame del reference set e scrive `eval_set.json` |
| `eval_set.json` | **Il reference set: 58 frame su 3 sessioni.** Unico dato versionato del repo |
| `prepare_controls.py` | Da una sessione al pacchetto per un modello video-to-video: controlli (rgb, depth, edge, seg) nel formato del modello e GT allineata frame per frame |
| `test_*.py` | Test senza server CARLA |
| `requirements.txt` | Pin dell'ambiente CARLA locale |

Le sessioni e i pacchetti stanno in `CARLA/outputs/`, che è fuori da git. Il server CARLA
non sta nel repo: è un pacchetto da ~20 GB installato a parte.

### cloud/

**Tutto ciò che gira sul pod.** È autocontenuta: non importa niente da fuori, salvo
`CARLA/boss_classes.py`, che contiene solo dati.

| File | Cosa fa |
|---|---|
| `adapters/` | Un modulo per generatore, tutti con lo stesso contratto `SPEC` / `load()` / `generate()` / `unload()`. Text-to-video: `wan22_5b`, `wan22_a14b`, `cogvideox15`, `hunyuan15`, `ltx25`, `opensora2`. Sim2Real: `cosmos_t25`, `wan22_fun5b_ctrl` |
| `runner.py` | Generatore a lotti: riprendibile, scrittura atomica, cattura `SIGTERM`, gate in linea, contatore dei costi. Tutto finisce in `manifest.jsonl` |
| `prompts.py` | I prompt per classe, il negative prompt e i prompt delle scene Sim2Real |
| `build_jobs.py` | Prodotto modelli × prompt × seed → `jobs.jsonl`. Il seed è l'indice globale del prompt |
| `sim2real.py` | La parte del ramo Sim2Real che non usa la GPU: configurazioni, job dai pacchetti, controllo che l'uscita sia allineata alla GT |
| `annotate_sam.py` | SAM 3 sui frame generati: individuazione per frame, maschere, overlay |
| `sam_selezione.py` | Sceglie oggetto e supporto fra tutti i candidati di SAM per le classi sospese |
| `annotate_da3.py` | Depth Anything 3 sulle stesse maschere. Legge il CSV di `annotate_sam.py`, va lanciato dopo |
| `quality_gate.py` | Controllo qualità su CPU (nitidezza, SSIM). Il runner ne registra i valori, non filtra |
| `compare_models.py` | Unisce manifest e CSV in una tabella modello × metriche |
| `gpu.py` | Telemetria `nvidia-smi` e guardie di memoria |
| `sync.py` | Push e pull incrementali verso un dataset HuggingFace privato |
| `setup_pod.sh` | Prepara il pod a blocchi (`base`, `gen`, `a14b`, `ann`, `opensora`, `cosmos`, `fun`, `check`): venv, `HF_HOME`, clone dei repo upstream ai commit fissati |
| `test_*.py` | Test senza GPU, da lanciare come script: `python cloud/test_x.py` |

I commit dei repo upstream (SAM 3, DA3, Open-Sora, TensorNVMe, Cosmos, VideoX-Fun) sono
fissati come costanti in testa a `setup_pod.sh`. Open-Sora non è più mantenuto da marzo
2025: è fissato all'ultimo commit di codice, e lo dichiara il suo `SPEC`.

### envs/

Un `requirements-*.txt` per ogni venv del pod.

| File | Per cosa | Cosa lo tiene separato |
|---|---|---|
| `requirements-gen.txt` | Wan2.2-5B, CogVideoX1.5, HunyuanVideo-1.5, LTX-2.5 | `numpy` 2.x |
| `requirements-a14b.txt` | Wan2.2-T2V-A14B | `diffusers` da un commit di `main` |
| `requirements-ann.txt` | SAM 3 + Depth Anything 3 | `numpy<2`, dichiarato da entrambi i pacchetti upstream |

Open-Sora, Cosmos e Wan Fun hanno ambienti installati da `setup_pod.sh` seguendo i loro
repo. Il blocco `opensora` scrive anche `requirements-opensora.lock`, da riportare qui.

**`numpy` è la ragione dei venv separati**, non una convenzione: i generatori vogliono
la 2.x, gli annotatori la 1.26.

### sam3_1/ · depth_anything3/

I due **annotatori**, misurati contro la ground truth CARLA nelle verifiche B.1–B.4. Di
ciascuno è versionato solo `eval_on_carla.py`, perché ha prodotto numeri che i documenti
di progetto citano ancora. Gli script che annotano i frame *generati* sono in `cloud/`.

`sam3_1/` contiene anche il repo upstream clonato (`sam3/`, installato in modalità
editabile nel venv) e una `test_images/`, entrambi fuori da git. Il clone di Depth
Anything 3 non è più qui: DA3 gira in WSL2, con venv e clone in `~/or4-ann`.

**SAM 3 e non SAM 3.1.** Nonostante il nome della cartella, per l'inferenza su
un'immagine singola va usato il checkpoint `sam3`. Il 3.1 esiste solo come modello video
multiplex, e `build_sam3_image_model` lo accetta senza errori, lasciando però pesi casuali
su un livello FPN. Il controllo di copertura delle chiavi all'avvio trasforma quel caso in
un errore invece che in numeri sbagliati: non va rimosso.

### bakeoff_clip/

Fuori da git. Contiene le clip scaricate dal pod e le loro valutazioni:

| Cartella | Contenuto |
|---|---|
| `scaricato_completo_2/` | **La copia di riferimento**: clip, frame PNG e manifest di tutti i modelli |
| `eval/` | Gate di qualità per modello |
| `eval_sam/` | SAM e DA3 per modello, fogli di revisione, archivi dei run vecchi |

### cogvideox/ · qa_export/

Fuori da git. `cogvideox/` conserva lo smoke test e lo sweep VRAM di agosto. Su Windows non
girano più, perché Smart App Control blocca il tokenizer; si lanciano dall'ambiente WSL2
`~/or4-cogvideox`. `qa_export/` ospiterà l'export del dataset nei formati di consegna
(COCO / COCO-Video); per ora contiene solo la calibrazione del gate dell'8/09.

## Convenzioni

- **Un venv per ambiente, mai condiviso**: i vincoli di versione dei modelli sono
  incompatibili fra loro. I venv non sono versionati, le loro ricette sì (`envs/`,
  `setup_pod.sh`).
- **Ogni script ha i parametri in testa al file**, come costanti maiuscole, con il
  commento che spiega quel valore. Le eccezioni sono dichiarate: `scenarios.py` in
  `CARLA/`, e le soglie hardware in `cloud/`, che sono argomenti perché il pod cambia
  scheda da una sessione all'altra.
- **`outputs/` e le clip non sono mai versionate**: video, sessioni e CSV sono risultati
  pesanti e rigenerabili.
- **I repo upstream dei modelli si clonano, non si copiano nel repo**, a commit fissati.
- **I test si lanciano come script** (`python test_x.py`): pytest non è installato.
- **Commit convenzionali** (`feat:`, `fix:`, `docs:`…), in italiano.
