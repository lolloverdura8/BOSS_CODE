# BOSS_CODE

Codice del sottoprogetto OR4 (riconoscimento ostacoli e calcolo distanze via Edge AI)
del progetto B.O.S.S.

## Struttura

```
synthetic_data_pipeline/
├── cogvideox/          generazione video sintetici con CogVideoX-2b  ← documentato qui
├── wan2_2/             generazione video sintetici con Wan2.2
├── sam3_1/             segmentazione con SAM 3.1
├── depth_anything3/    stima di profondità con Depth Anything 3
├── cloud/  envs/  qa_export/
```



---

# Pipeline CogVideoX

## 1. Scopo dei due script

| Script | Cosa fa |
|---|---|
| `smoke_test.py` | Una singola generazione video completa (49 frame, 50 step) salvata in `outputs/smoke_test.mp4`. Serve a verificare che l'ambiente funzioni end-to-end e a produrre un video ispezionabile a occhio. È **riproducibile**: usa lo stesso seed del benchmark, quindi due esecuzioni danno lo stesso video. |
| `vram_benchmark.py` | Sweep di profilazione: misura memoria e tempo al variare del numero di frame generati, per stabilire dove sta il limite della macchina. Produce un CSV in `outputs/`. |

I due script non sono indipendenti: condividono risoluzione, prompt, `GUIDANCE`, `FPS` e
`SEED`, quindi il video a 49 frame dello smoke test è la controparte visiva della riga
`num_frames=49` del CSV — stesso rumore iniziale, stessi parametri — e il picco VRAM che
lo smoke test stampa è direttamente confrontabile con quella riga. Disallineare quei
parametri fra i due script rende il confronto privo di senso.

Il modello è `THUDM/CogVideoX-2b` in `float16` (il model card THUDM raccomanda fp16 per
il 2b, a differenza del 5b che vuole bf16). Si è scelto il 2b dopo che il 5b si era
rivelato troppo pesante per questa workstation.

Peso dei tre componenti del modello (misurato, pesi in fp16):

| Componente | Parametri | Occupazione |
|---|---|---|
| `text_encoder` (T5-XXL) | 4.76 B | 8.87 GB |
| `transformer` | 1.69 B | 3.15 GB |
| `vae` | 0.22 B | 0.40 GB |

Il text encoder da solo è il 71% del peso totale: è il motivo per cui il benchmark lo
elimina prima di misurare (vedi §3).

## 2. Ambiente di esecuzione

**Si esegue sotto WSL2/Ubuntu, non su Windows nativo.** Su Windows, Smart App Control
blocca il caricamento delle estensioni native compilate non firmate: sia
`_sentencepiece` sia `_tiktoken` — entrambe necessarie al tokenizer T5 — falliscono con
`ImportError: DLL load failed ... Un criterio di controllo dell'applicazione ha bloccato
il file`. Il blocco avviene dentro `CogVideoXPipeline.from_pretrained()`, prima di
qualunque inferenza, quindi colpisce identicamente entrambi gli script. Smart App
Control non ammette eccezioni per singolo file e disattivarlo è irreversibile senza
reinstallare Windows: da qui la scelta di WSL2, dove il meccanismo non si applica
perché i binari sono ELF Linux.

Comandi, dal terminale **Ubuntu (WSL)** integrato in VS Code:

```bash
source ~/or4-cogvideox/.venv/bin/activate
cd /mnt/c/Users/pc/Documents/Lavoro/OR4/code/BOSS_CODE/synthetic_data_pipeline/cogvideox
python vram_benchmark.py        # sweep completo
python smoke_test.py            # generazione singola di verifica
```

Il venv vive nel filesystem nativo Linux (`~/or4-cogvideox/.venv`, non sotto `/mnt/c`)
perché un venv su NTFS montato è molto più lento. Gli script invece restano nel repo
Windows: sono file di testo piccoli, l'overhead è trascurabile. La cache dei pesi
HuggingFace è anch'essa nativa, in `~/.cache/huggingface/hub/`.

Ambiente verificato: Python 3.12, `torch 2.11.0+cu128`, `diffusers 0.40.0`, `psutil`,
`sentencepiece`, `protobuf`, `imageio 2.37.4` + `imageio-ffmpeg 0.6.0` (backend di
scrittura video, vedi §4). GPU: **RTX 5070 Ti, 16.3 GB VRAM**, esposta a WSL2 via
passthrough del driver Windows (nessun driver NVIDIA va installato dentro Ubuntu:
lo romperebbe).

### Limitazione nota dell'ambiente

Sotto WSL2 il driver **ignora** l'impostazione NVIDIA "CUDA - Sysmem Fallback Policy"
([microsoft/WSL#11050](https://github.com/microsoft/WSL/issues/11050), chiusa senza fix).
Conseguenza: quando la VRAM si esaurisce non si ottiene un `torch.cuda.OutOfMemoryError`,
ma uno swap silenzioso su RAM di sistema via PCIe — il programma continua a funzionare,
solo centinaia di volte più lentamente. Per questo il benchmark **rileva il fallback dai
numeri di memoria** (colonna `sysmem_fallback_suspected`) invece di aspettarsi
un'eccezione.

## 3. Architettura di `vram_benchmark.py`

### Perché il text encoder viene eliminato prima di misurare

Il primo sweep riportava un picco di 10.82 GB **identico** a 9 e a 17 frame: quel numero
non dipendeva dal numero di frame perché era il text encoder T5, che gira una volta sola
all'inizio. Stava mascherando completamente il costo che effettivamente scala coi frame.

Lo script quindi calcola gli embedding del prompt una volta, elimina T5, e solo dopo
esegue la generazione. Tre dettagli che non sono opzionali:

- **`torch.no_grad()` attorno a `encode_prompt`.** Solo `pipe.__call__` è decorato
  `@torch.no_grad`; `encode_prompt` chiamato direttamente costruisce il grafo di
  autograd, che trattiene attivazioni (~3.6 GB) *e* pesi di T5. Senza, il rilascio non
  funziona: misurati 14.37 GB ancora occupati dopo il "free", con il picco che saliva a
  18.21 GB su una scheda da 16.3 GB — cioè sysmem fallback e 98 s/step invece di 1.7.
- **Rilascio in tre passaggi**: `pipe.text_encoder = None`, `gc.collect()`,
  `torch.cuda.empty_cache()`. Togliere il riferimento non basta.
- **Ordine**: `enable_model_cpu_offload()` va chiamato *dopo*, perché installa hook sui
  componenti registrati e deve trovare il text encoder già sparito.

### Nota di correzione: cosa fa davvero `enable_slicing()`

Il commento nel codice del benchmark (`# slicing: decodifica un frame per volta`) è
impreciso, e la stessa imprecisione varrebbe per qualunque altro script della pipeline.
Il sorgente del VAE (`autoencoder_kl_cogvideox.py`, `decode`) applica lo slicing solo se
`z.shape[0] > 1`: **agisce sulla dimensione batch, non sui frame**. Con
`num_videos_per_prompt=1` il latente in ingresso al decode ha batch 1, quindi in entrambi
gli script la chiamata è **inerte** — i risparmi di memoria misurati non le sono
attribuibili. Diventa attiva solo generando più video per prompt (vedi §4).

Lo spezzettamento del decode lungo l'asse temporale esiste davvero, ma è un'altra cosa:
`num_latent_frames_batch_size = 2`, sempre attivo e indipendente da slicing.

Il tiling invece è realmente in funzione: la soglia è metà della risoluzione nativa
(latente 30×45) e il nostro latente è 60×90.

### Perché driver/worker in subprocess

Il run originale moriva a 25 frame senza scrivere nulla. Non era un OOM di VRAM ma di
**RAM host**: l'OOM killer del kernel Linux termina il processo con SIGKILL, un segnale
non intercettabile da Python — nessun `try/except` può salvarlo, e con esso muore tutto
lo sweep rimanente.

Lo script ha quindi due modalità:

- **worker** (`--worker --num-frames N`): esegue **un solo** punto e stampa una riga
  JSON su stdout (la diagnostica va su stderr, così stdout resta parsabile);
- **driver** (default): itera `FRAME_POINTS` e lancia un subprocess per ciascun punto.
  La morte del figlio diventa così un dato osservabile dal padre invece della fine dello
  sweep: `returncode` negativo significa terminato da segnale (−9 = SIGKILL = OOM
  killer), positivo significa eccezione Python.

Prima di ogni punto il driver controlla la RAM disponibile: sotto `HOST_RAM_MIN_GB` salta
il punto invece di andare incontro al kill.

Effetto collaterale misurato e importante: nel monoprocesso la RAM disponibile si erodeva
punto dopo punto (12.99 → 6.90 GB), contaminando le misure successive. Con un processo per
punto ogni misura riparte da ~13 GB puliti.

### Perché 3 step invece di 50

Il picco di memoria si raggiunge entro il primo step (gli step successivi riusano gli
stessi buffer), quindi 50 step moltiplicherebbero l'attesa senza aggiungere informazione.
Lo sweep gira a `SWEEP_STEPS = 3`, misura il tempo medio per step e **estrapola** il tempo
di una generazione reale a 50 step, scomponendolo in parte proporzionale agli step
(denoising) e parte fissa (setup + decode VAE), che non va moltiplicata.

L'estrapolazione è stata validata contro misure reali a 50 step: errore fra 0% e 9%,
tipicamente 1-3%.

## 4. Architettura di `smoke_test.py`

### A cosa serve e a cosa non serve

Verifica end-to-end dell'ambiente e produzione di **un video da guardare**. Non è uno
strumento di misura: le metriche che stampa servono a capire se il run è andato in modo
sano, non a caratterizzare la macchina. Per quello c'è il benchmark.

### Cosa condivide con il benchmark, e perché

- **Precompute degli embedding con eliminazione di T5.** Stessa sequenza descritta in §3,
  incluse le tre trappole (`torch.no_grad()`, rilascio in tre passi, ordine rispetto a
  `enable_model_cpu_offload()`): non si ripete qui, vale identica. Nello smoke test la
  motivazione principale però è diversa da quella del benchmark. `enable_model_cpu_offload()`
  non elimina T5: lo **parcheggia in RAM di sistema** e lo risveglia sulla GPU quando serve.
  Senza il precompute, quegli 8.87 GB restano occupati in RAM host per tutti i 50 step — e la
  RAM host è il vincolo stretto di questa VM (15 GB), quello che ha ucciso il run originale
  del benchmark. Il secondo effetto è sulla leggibilità del picco stampato: con T5 dentro la
  pipeline il picco è ~10.8 GB ed è *lui*, non la generazione, quindi non confrontabile col
  CSV. Non si aggira con un `reset_peak_memory_stats()` prima di `pipe()`, perché l'hook di
  offload risveglia T5 sulla GPU *dentro* la chiamata.
- **`SEED = 0`**, risoluzione, prompt, `GUIDANCE` e `FPS` allineati (§1).

### Cosa NON condivide, e perché

- **Niente struttura driver/worker.** Serve a sopravvivere al SIGKILL dell'OOM killer
  quando ci sono altri punti da misurare dopo. Con una sola generazione non c'è nulla da
  salvare: se il processo muore, quello *è* il risultato dello smoke test. In più il worker
  comunica via JSON su stdout, contratto inutile qui dove il prodotto è un `.mp4`.
- **Niente `SWEEP_STEPS` né estrapolazione.** Il benchmark gira a 3 step e stima i 50 per
  misurare in fretta sei configurazioni; lo smoke test i 50 step li esegue davvero, perché
  il suo prodotto è il video.
- **Niente CSV né logica di raccomandazione** (`BUDGET_S`, `TARGET_S`): non c'è nessuno
  sweep su cui scegliere.
- **Niente metriche di RAM host** (`psutil`): servivano a seguire l'erosione della RAM punto
  dopo punto in un processo che ne eseguiva sei di fila.

### Come si legge l'output

Tre metriche di VRAM, che dicono cose diverse:

| Stampa | Cosa misura |
|---|---|
| `picco VRAM allocata` | Picco dei tensori effettivamente in uso dall'allocatore. **È il numero confrontabile con `vram_alloc_peak_gb_denoise` del CSV.** |
| `picco VRAM prenotata` | Picco della VRAM prenotata al driver (≥ allocata: include la cache trattenuta per riuso). Il divario fra le due misura la frammentazione. |
| `VRAM occupata sulla scheda` | Lettura istantanea del driver sull'intera GPU. |

Se compare la riga `ATTENZIONE: sospetto sysmem fallback`, il tempo di generazione stampato
va **scartato**, non interpretato: sotto WSL2 l'esaurimento di VRAM non dà un'eccezione ma
uno swap silenzioso su RAM (§2), e il run continua centinaia di volte più lento.

Stessa cautela già valida per il CSV: `VRAM occupata sulla scheda` è letta **dopo** il ritorno
di `pipe()`, quindi dopo che `maybe_free_model_hooks()` ha scaricato i moduli dalla GPU, e
sottostima il transiente del decode. Il dato affidabile è il picco allocato.

### Backend di scrittura del video

`export_to_video` sceglie il backend a runtime: con `imageio` + `imageio-ffmpeg` installati
scrive in **H.264**; senza, ricade sul ramo OpenCV — marcato deprecato in diffusers — che
scrive in `mp4v` (MPEG-4 Part 2), formato che diversi player e browser rifiutano. Se nel log
compare `Support for the OpenCV backend will be deprecated`, i due pacchetti mancano nel
venv. Nessun `ffmpeg` di sistema è richiesto: `imageio-ffmpeg` porta il proprio binario.
480 e 720 sono entrambi divisibili per 16, quindi il `macro_block_size` di default non
riscala l'immagine.

### Generare più video da un solo prompt

La leva è `num_videos_per_prompt` **nella chiamata a `encode_prompt`**, non l'omonimo
parametro di `pipe()`: in `diffusers 0.40.0`, `CogVideoXPipeline.__call__` lo dichiara nella
propria firma ma lo **riassegna a 1 incondizionatamente** prima di usarlo, quindi passarlo lì
non ha effetto e non produce nessun errore né warning. Sul path con embedding precalcolati
funziona invece correttamente, perché `__call__` ricava `batch_size` da
`prompt_embeds.shape[0]`.

Tre conseguenze da gestire quando lo si alzerà:

1. `pipe(...).frames` conterrebbe N video, mentre lo script esporta `.frames[0]`: serve un
   ciclo sull'export con nomi di file distinti;
2. il costo del denoising scala circa ×N (i latenti sono N, 2N con CFG);
3. `enable_slicing()` smette di essere inerte — è precisamente il caso per cui la chiamata
   è tenuta in piedi (vedi la nota di correzione in §3).

## 5. Dizionario delle colonne del CSV

Ogni run scrive un file nuovo `outputs/vram_sweep_YYYYMMDD_HHMMSS.csv` (il codice
precedente apriva il CSV in `"w"` e a ogni avvio cancellava i dati del run prima).

| Colonna | Significato |
|---|---|
| `num_frames` | Frame generati in questo punto. Vincolo del VAE: deve essere 4k+1. |
| `duration_s` | Durata del video risultante = `num_frames / FPS` (FPS = 8). |
| `elapsed_s` | Tempo reale di questa misura, a `SWEEP_STEPS` step. **Non** confrontabile col budget. |
| `s_per_step` | Tempo medio di un singolo step di denoising. |
| `est_elapsed_50steps_s` | Tempo **estrapolato** per una generazione reale a 50 step. È questo che va confrontato con `BUDGET_S`. |

Le cinque misure di memoria compaiono due volte, con suffisso `_denoise` (campionate
all'ultimo step di denoising, dal vivo) e `_decode` (dopo il ritorno di `pipe()`, a
decode VAE concluso):

| Colonna | Cosa misura |
|---|---|
| `vram_alloc_peak_gb` | Picco dei tensori effettivamente in uso dall'allocatore PyTorch. |
| `vram_reserved_peak_gb` | Picco della VRAM che PyTorch ha *prenotato* al driver (≥ alloc: include la cache trattenuta per riuso). |
| `vram_device_used_gb` | VRAM occupata sulla scheda intera, vista dal driver (include altri processi). Lettura **istantanea**, non un picco. |
| `host_rss_gb` | RAM di sistema occupata da questo processo. |
| `host_avail_gb` | RAM di sistema ancora libera nella VM. |

| Colonna | Significato |
|---|---|
| `sysmem_fallback_suspected` | `True` se `vram_alloc_peak_gb` ha raggiunto la capacità fisica della scheda: un picco del genere può essere servito solo spillando su RAM. Il criterio **non** è la differenza `alloc_peak − device_used`, che diverge di ~2.5 GB anche in condizioni sane (uno è un picco, l'altro un'istantanea presa a modelli già scaricati) e darebbe falsi positivi. |
| `oom` | `True` solo per un `torch.cuda.OutOfMemoryError` vero. Sotto WSL2 è raro proprio per via del fallback (§2). |
| `fail_reason` | Vuoto se il punto è riuscito. Vedi tabella sotto. |

Valori di `fail_reason`:

| Valore | Significato |
|---|---|
| *(vuoto)* | Punto riuscito. |
| `cuda_oom` | Esaurimento di VRAM riconosciuto da PyTorch. |
| `killed_by_os` | Il worker è stato terminato da un segnale — tipicamente SIGKILL dell'OOM killer: **RAM host** esaurita, non VRAM. |
| `timeout` | Il worker ha superato `WORKER_TIMEOUT_S`. Spesso è sysmem fallback così grave da sembrare un blocco. |
| `host_ram_insufficient` | Il driver non ha nemmeno tentato: RAM libera sotto `HOST_RAM_MIN_GB`. |
| `worker_error` | Il worker è uscito con un'eccezione Python (lo stderr viene stampato dal driver). |
| `no_output` | Il worker è uscito con codice 0 ma non ha prodotto la riga JSON. Anomalia da indagare. |

## 6. Come si leggono i risultati

**Regola prima di tutte: guardare se le colonne di memoria variano con `num_frames`.**
Se un valore è identico su punti diversi, quasi certamente non sta misurando ciò che si
crede — è esattamente l'errore che ha reso inutile il primo sweep.

Poi, nell'ordine:

1. **`fail_reason`** dice *se* e *perché* un punto non è misurabile. La distinzione
   critica è fra `cuda_oom` (limite del modello sulla scheda: dato utile) e
   `killed_by_os` / `host_ram_insufficient` (limite di RAM della VM: dato
   sull'ambiente di test, non sul modello). Confonderli porta a conclusioni sbagliate
   sul dimensionamento hardware.
2. **`sysmem_fallback_suspected`** segnala un punto che "funziona" ma a velocità
   inutilizzabile: la misura di tempo va scartata, non interpretata.
3. **`vram_device_used_gb_denoise`** è il carico VRAM reale sulla scheda durante il
   denoising, da confrontare con la capacità fisica per capire il margine.
4. **`host_avail_gb`** dice quanto margine di RAM resta: è il vincolo che ha ucciso il
   run originale.
5. **`est_elapsed_50steps_s`** è l'unico tempo confrontabile con `BUDGET_S`.

**Attenzione a `vram_device_used_gb_decode`.** È campionato dopo il ritorno di `pipe()`,
quindi dopo che `maybe_free_model_hooks()` ha già scaricato i moduli dalla GPU: il valore
crolla (~1.4 GB contro ~7 GB in denoising) e **sottostima il transiente del decode**. Per
la fase di decode il dato affidabile è `vram_alloc_peak_gb`, che è un picco vero. La stessa
cautela vale per `sysmem_fallback_suspected`: è più solido sulla terna `_denoise`.

### Risultati misurati (RTX 5070 Ti 16.3 GB, CogVideoX-2b fp16, 480×720)

| frames | durata | VRAM device (denoise) | VRAM alloc peak | tempo stimato a 50 step |
|---|---|---|---|---|
| 9 | 1.13 s | 5.82 GB | 3.85 GB | 24.9 s |
| 17 | 2.13 s | 6.18 GB | 4.08 GB | 41.6 s |
| 25 | 3.13 s | 6.53 GB | 4.32 GB | 65.6 s |
| 33 | 4.13 s | 6.90 GB | 4.55 GB | 89.1 s |
| 41 | 5.13 s | 7.26 GB | 4.79 GB | 112.4 s |
| 49 | 6.13 s | 7.44 GB | 5.02 GB | 152.8 s |

Nessun punto va in fallback, nessuno fallisce, nessuno supera il budget di 300 s: **l'intera
griglia è percorribile**, incluso il punto nativo a 49 frame. Il picco VRAM si ferma a 7.44
GB su 16.3 disponibili, quindi la VRAM non è il vincolo su questa macchina — c'è più del
doppio di margine. Il vincolo reale è la RAM host della VM WSL2.

## 7. Parametri configurabili

Tutti in testa a `vram_benchmark.py`.

| Parametro | Default | Quando toccarlo |
|---|---|---|
| `FRAME_POINTS` | `[9,17,25,33,41,49]` | Per infittire la griglia o spingersi oltre il nativo. Vincolo del VAE: solo valori 4k+1. |
| `SWEEP_STEPS` | `3` | Alzarlo solo se si sospetta che il picco non si raggiunga entro i primi step. Costa tempo lineare. |
| `NUM_STEPS` | `50` | Step di una generazione reale: è il bersaglio dell'estrapolazione, non viene eseguito. Allinearlo alla configurazione di produzione. |
| `SEED` | `0` | Cambiarlo solo per verificare che i risultati non dipendano dal seed. |
| `BUDGET_S` | `300` | Budget di tempo per video; definisce quali punti sono accettabili. |
| `TARGET_S` | `2.0` | Durata desiderata degli scenari; fra i punti validi viene raccomandato il più vicino. |
| `HOST_RAM_MIN_GB` | `4.0` | Alzarlo su macchine dove l'OOM killer interviene comunque. |
| `WORKER_TIMEOUT_S` | `600` | Alzarlo per griglie molto più pesanti; un punto sano costa 10-30 s più il caricamento del modello. |

`HEIGHT`/`WIDTH` (480×720) sono la risoluzione nativa del modello: cambiarli porta
CogVideoX fuori distribuzione e i risultati non sono più rappresentativi.

I parametri di `smoke_test.py` stanno anch'essi in testa al file: `HEIGHT`, `WIDTH`,
`NUM_FRAMES`, `NUM_STEPS`, `GUIDANCE`, `FPS`, `SEED`. Sono **volutamente allineati** agli
omonimi del benchmark, ed è ciò che rende il video prodotto la controparte visiva della riga
`num_frames=49` del CSV e il picco VRAM stampato confrontabile con quella riga (§1).
Modificarne uno solo dei due lati fa decadere il confronto senza che nulla lo segnali.
