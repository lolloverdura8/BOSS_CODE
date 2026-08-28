# BOSS_CODE

Codice del sottoprogetto OR4 (riconoscimento ostacoli e calcolo distanze via Edge AI)
del progetto B.O.S.S.

## Struttura

```
synthetic_data_pipeline/
├── cogvideox/          CogVideoX-2b        480×720 @ 8 fps   ← documentato qui
├── wan2_1/             Wan2.1-T2V-1.3B     832×480 @ 16 fps  ← documentato qui
├── wan2_2/             Wan2.2-TI2V-5B      704×1280 @ 24 fps ← documentato qui, FUORI PORTATA HW
├── sam3_1/             segmentazione con SAM 3.1
├── depth_anything3/    stima di profondità con Depth Anything 3
├── cloud/  envs/  qa_export/
```

I tre generatori video sono candidati **alternativi** per la stessa funzione. Il confronto che
decide quale adottare è in fondo a questo documento.

**Wan2.2-TI2V-5B è fuori portata su questa macchina**: la model card richiede ≥24 GB di VRAM
contro i 16.3 della RTX 5070 Ti, e il nostro sweep lo conferma con `cuda_oom` prima del punto
nativo. La sua cartella è conservata come record riproducibile e come punto di partenza per un
test futuro su hardware adeguato, ma **i suoi pesi e il suo venv sono stati rimossi** e gli
script non sono eseguibili qui — dettagli in §8 della sezione Wan2.2.



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

---

# Pipeline Wan2.2

## 1. Scopo dei due script

Gli stessi due ruoli della pipeline CogVideoX, con la stessa relazione fra loro:
`smoke_test.py` produce **un video da guardare** (121 frame, 50 step, in
`outputs/smoke_test.mp4`), `vram_benchmark.py` produce **un CSV di misure** al variare del
numero di frame. Condividono risoluzione, prompt, `GUIDANCE`, `FPS` e `SEED`, quindi il video
dello smoke test è la controparte visiva della riga `num_frames=121` del CSV — stesso rumore
iniziale, stessi parametri.

Il modello è `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. I dtype non sono uniformi: **il VAE resta in
`float32`** perché in bf16 produce artefatti di decodifica, transformer e text encoder sono in
`bfloat16`.

Peso dei tre componenti (misurato sommando parametro per parametro, non moltiplicando per un
dtype unico — vedi nota sotto):

| Componente | Parametri | Occupazione | dtype |
|---|---|---|---|
| `text_encoder` (UMT5-XXL) | 5.68 B | 10.58 GB | bf16 |
| `transformer` | 5.00 B | 9.33 GB | bf16 + fp32 |
| `vae` | 0.70 B | 2.63 GB | fp32 |

Totale **~22.5 GB in RAM host** al caricamento della pipeline completa, contro i ~12.4 GB di
CogVideoX.

> **Perché il transformer ha due dtype.** Caricando la pipeline con `torch_dtype=bfloat16`,
> `diffusers` tiene comunque in fp32 i moduli elencati in `_keep_in_fp32_modules` del modello.
> Conseguenza pratica: leggere il dtype dal *primo* parametro (`next(m.parameters()).dtype`) dà
> un'etichetta sbagliata per l'intero modello — riportava `float32` per un modello che pesa
> 9.33 GB su 5.00 B di parametri, cioè 2 byte per parametro. Lo script somma
> `p.numel() * p.element_size()` proprio per non incappare in questo.

## 2. Ambiente di esecuzione

**Si esegue su Windows nativo, NON sotto WSL2** — cioè l'esatto contrario di CogVideoX. Non è
una scelta di comodo ma una conseguenza di come i due modelli tokenizzano il prompt:

| | tokenizer in cache | dichiarato in `model_index.json` | Effetto di Smart App Control |
|---|---|---|---|
| CogVideoX-2b | solo `spiece.model` | `T5Tokenizer` (slow) | blocca `_sentencepiece` → **non parte** |
| Wan2.2-TI2V-5B | `tokenizer.json` (16.8 MB) | `T5TokenizerFast` (Rust) | non passa da sentencepiece → **parte** |

Il blocco che ha costretto CogVideoX a migrare sotto WSL2 (§2 della sezione precedente) colpisce
l'estensione nativa di `sentencepiece`, necessaria a leggere `spiece.model`. Wan ha già il
tokenizer serializzato in `tokenizer.json` e usa l'implementazione Rust: non tocca sentencepiece,
che infatti **non è nemmeno installato** nel suo venv. Non è aggirabile dal lato CogVideoX:
anche `T5TokenizerFast` dovrebbe *convertire* da `spiece.model`, quindi ripasserebbe da lì.

Che i due modelli girino su piattaforme diverse è quindi una proprietà dei modelli su questa
macchina, non un artefatto della configurazione — e pesa nel confronto finale.

Comandi, dal terminale **PowerShell** integrato in VS Code:

```powershell
cd synthetic_data_pipeline\wan2_2
.\.venv\Scripts\Activate.ps1
python vram_benchmark.py        # sweep completo
python smoke_test.py            # generazione singola di verifica
```

Ambiente verificato: venv locale `wan2_2\.venv`, Python 3.11.9, `torch 2.11.0+cu128`,
`diffusers 0.40.0`, `transformers 5.15.1`, `accelerate 1.14.0`, `ftfy 6.3.1`, `psutil 7.2.2`,
`imageio 2.37.4` + `imageio-ffmpeg 0.6.0`. Cache dei pesi in
`C:\Users\pc\.cache\huggingface\hub\` (35 GB su disco per questo modello).
GPU: **RTX 5070 Ti, 16.3 GB VRAM**, di cui **~1.4 GB già occupati dal desktop Windows a riposo**.
RAM di sistema: 31.78 GB.

> **`ftfy` non è opzionale qui.** `pipeline_wan.py` lo importa se disponibile e `prompt_clean` lo
> usa per normalizzare il prompt. Un ambiente senza ftfy ripulisce il prompt diversamente e
> produce embedding diversi: gli stessi parametri darebbero un video diverso.

### Il tetto sull'allocatore, e perché qui serve

`torch.cuda.set_per_process_memory_fraction(0.90, device=0)` è **attiva** in entrambi gli script
Wan, mentre negli script CogVideoX resta commentata. La riga cappa la memoria che l'allocatore
PyTorch può prenotare al driver, come frazione della VRAM fisica.

La differenza sta nella piattaforma. Sotto WSL2 sarebbe inutile: il driver ignora "CUDA - Sysmem
Fallback Policy" e il tetto non impedirebbe comunque lo swap su RAM. Su Windows nativo il driver
la policy la rispetta, quindi con un tetto esplicito l'allocazione fallisce **prima** del
`cudaMalloc` che innescherebbe il fallback: si ottiene un `torch.cuda.OutOfMemoryError` vero,
intercettabile, e la colonna `oom` del CSV diventa un dato affidabile invece di una casella
sempre falsa.

Serve qui e non serviva a CogVideoX anche per una ragione di scala: CogVideoX si ferma a 7.44 GB
su 16.3 disponibili, mentre il solo transformer di Wan pesa 9.33 GB su una scheda che ne ha già
~1.4 occupati dal desktop. Il soffitto è vicino e va colpito in modo pulito, non attraversato in
silenzio.

**Conseguenza sulla lettura del CSV**: col tetto attivo `vram_alloc_peak_gb` non può più
raggiungere il 98% della VRAM fisica, quindi la soglia di `sysmem_fallback_suspected` è rapportata
al **budget effettivo** (`0.90 × 16.3 = 14.67 GB`) e non al totale della scheda. Senza questa
correzione l'euristica non scatterebbe mai.

## 3. Architettura di `vram_benchmark.py`

È **la stessa architettura del benchmark CogVideoX**, deliberatamente: struttura driver/worker con
un subprocess per punto, precompute degli embedding con eliminazione del text encoder prima di
azzerare i contatori, probe su `callback_on_step_end` per separare la fase di denoising da quella
di decode, cinque metriche di memoria × due fasi, `SWEEP_STEPS = 3` con estrapolazione a 50 step,
CSV timestamped. Le motivazioni non si ripetono qui: valgono identiche, sono in §3 della sezione
CogVideoX.

Il motivo per cui la struttura è la stessa è che i due CSV vanno letti affiancati: stesse colonne,
stessi nomi, stesso significato.

### Perché il text encoder si elimina e il transformer no

È la domanda naturale guardando i pesi: se il text encoder da 10.58 GB si può buttare, perché non
il transformer da 9.33 GB, che pesa quasi uguale?

Non è una questione di dimensione ma di **quante volte serve**. Il text encoder gira una volta
sola all'inizio e il suo risultato — pochi MB di embedding — non dipende da `num_frames`: lo si
usa e lo si elimina. Il transformer viene invocato a **ogni** step di denoising, 50 volte, e due
volte per step per via del CFG: eliminarlo significherebbe ricaricare 9.33 GB cento volte. Per lui
la leva non è il rilascio ma `enable_model_cpu_offload()`, che lo tiene in RAM host e lo sposta
sulla GPU solo mentre serve, restituendo poi la VRAM al VAE per il decode.

Timeline di memoria di un worker:

| Momento | RAM host | VRAM |
|---|---|---|
| `from_pretrained` completo | ~22.5 GB (picco) | — |
| encode del prompt | ~22.5 GB | 10.6 GB (text encoder) |
| dopo il rilascio del text encoder | ~12 GB | ~0 |
| denoising | ~12 GB | transformer + attivazioni |
| decode VAE | ~13.5 GB (arriva l'array video) | VAE + tensore video |

### Sei differenze rispetto a CogVideoX, verificate sul sorgente di `diffusers 0.40.0`

Sono le trappole della pipeline CogVideoX ricontrollate una per una su `WanPipeline` e
`AutoencoderKLWan`, che sono classi diverse. Tre valgono, tre no.

1. **`num_videos_per_prompt` funziona davvero.** In CogVideoX `__call__` lo dichiara nella firma
   ma lo riassegna a `1` in silenzio, e l'unica leva è l'omonimo parametro di `encode_prompt`.
   `WanPipeline.__call__` invece lo onora: lo passa a `encode_prompt` e a `prepare_latents`. I due
   punti sono equivalenti, basta alzarlo in uno.
2. **`check_inputs` è più severo.** Solleva se si passano insieme `prompt` e `prompt_embeds`, o
   `negative_prompt` e `negative_prompt_embeds`. Sul path con embedding precalcolati **entrambe le
   stringhe vanno messe esplicitamente a `None`** — CogVideoX su questo era permissivo.
3. **Il CFG esegue due forward sequenziali** del transformer, condizionato e non condizionato,
   invece di impilarli in un batch 2N come CogVideoX. Il tempo per step raddoppia, la memoria no:
   cambia come si legge `s_per_step`, non come si legge il picco.
4. **`enable_slicing()` è inerte a batch 1, identico a CogVideoX.** Il VAE lo applica solo se
   `z.shape[0] > 1`: agisce sulla dimensione batch, non sui frame. Si tiene per parità e perché
   diventa attivo alzando `num_videos_per_prompt`.
5. **Il decode del VAE Wan cicla già un frame latente per volta**, tenendo lo stato in un
   `feat_cache`: è più fine dello spezzettamento temporale di CogVideoX
   (`num_latent_frames_batch_size = 2`) e non dipende da nessuna chiamata. Il tiling invece è
   realmente attivo e va abilitato: la soglia è 256 px / 16 di compressione = 16 in latente, e a
   704×1280 il latente è 44×80.
6. **Vincoli di forma.** `num_frames` deve essere `4k+1` (`scale_factor_temporal = 4`);
   `height` e `width` devono essere multipli di `scale_factor_spatial (16) × patch_size (2) = 32`.
   In entrambi i casi la pipeline **non solleva**: arrotonda al valore valido più vicino e stampa
   un warning. 704, 1280 e i punti dello sweep rispettano già i vincoli.

## 4. Architettura di `smoke_test.py`

Stessa relazione col benchmark descritta in §4 della sezione CogVideoX, e stesse esclusioni:
niente driver/worker, niente `SWEEP_STEPS` né estrapolazione (i 50 step li esegue davvero, perché
il suo prodotto è il video), niente CSV né logica di raccomandazione, niente metriche di RAM host.

Porta dal benchmark: precompute degli embedding con eliminazione del text encoder, `SEED = 0`,
tetto sull'allocatore a `0.90` (tenerlo identico nei due script è ciò che rende confrontabile il
picco stampato con la riga del CSV), `vae.enable_slicing()` per parità benché inerte, tre metriche
VRAM + allarme sysmem fallback rapportato al budget.

Due dettagli specifici di Wan:

- **L'output è numpy, non PIL.** `WanPipeline` ha `output_type="np"` di default, quindi
  `.frames[0]` è un array `(T, H, W, 3)` e la stampa usa `.shape` (CogVideoX restituisce una lista
  di immagini PIL e usa `.size`). A 121 frame quell'array è ~1.3 GB in RAM host: è il salto di
  `host_rss` che si vede fra la fase di denoising e quella di decode nel CSV.
- **Backend video**: identico a CogVideoX (`imageio` + `imageio-ffmpeg` → H.264, altrimenti ramo
  OpenCV deprecato → `mp4v`). 704 e 1280 sono divisibili per 16, quindi il `macro_block_size` di
  default non riscala l'immagine.

## 5. Dizionario delle colonne del CSV

**Identico a quello della pipeline CogVideoX** (§5 sopra): stesse colonne, stessi nomi, stesso
significato, stesso `fail_reason`. Due sole avvertenze specifiche di questa pipeline:

| Colonna | Differenza |
|---|---|
| `sysmem_fallback_suspected` | La soglia è il 98% del **budget** `VRAM_FRACTION × VRAM fisica` (14.67 GB), non della VRAM fisica. Col tetto attivo il segnale atteso di esaurimento è `oom = True`; questa colonna resta come rete di sicurezza per le allocazioni che non passano dall'allocatore PyTorch (workspace di cuDNN e cuBLAS), che il tetto non copre. |
| `fail_reason = killed_by_os` | Praticamente irraggiungibile su Windows, che non ha un OOM killer: il sistema spilla sul pagefile e il sintomo è lentezza, non un segnale. Il ramo resta nel codice per simmetria con lo script CogVideoX. Su questa piattaforma il segnale equivalente è `timeout`. |

E una differenza di valore, non di significato: `HOST_RAM_MIN_GB` è **16.0** invece di 4.0, perché
il caricamento della pipeline completa è un transiente da ~22.5 GB contro i ~12.4 di CogVideoX.

> **Righe di un punto fallito.** Per un punto andato in OOM le colonne `_denoise` sono vuote (la
> probe non è mai arrivata all'ultimo step) ma le `_decode` sono piene: `mem_snapshot()` viene
> letto comunque dopo il `try`, quindi quei valori dicono **quanto in alto era arrivato
> l'allocatore prima di fallire**. È informazione utile, non spazzatura: a 97 frame dice 13.48 GB
> su un budget di 14.67.

## 6. Come si leggono i risultati

Valgono le stesse regole di lettura di §6 della sezione CogVideoX, a partire dalla prima: **le
colonne di memoria devono variare con `num_frames`**. Qui variano, e in modo pulitamente lineare.

### Risultati misurati (RTX 5070 Ti 16.3 GB, Wan2.2-TI2V-5B, 704×1280, `outputs/vram_sweep_20260828_135314.csv`)

| frames | durata | s/step | tempo stimato a 50 step | VRAM alloc peak | VRAM device (denoise) | esito |
|---|---|---|---|---|---|---|
| 25 | 1.04 s | 2.51 s | 161.9 s | 10.53 GB | 12.71 GB | ok |
| 49 | 2.04 s | 5.64 s | 348.4 s | 11.54 GB | 14.20 GB | ok, **oltre budget** |
| 73 | 3.04 s | 10.08 s | 588.9 s | 12.55 GB | 15.72 GB | ok, **oltre budget** |
| 97 | 4.04 s | — | — | (13.48 GB prima di fallire) | 15.09 GB | **`cuda_oom`** |
| 121 | 5.04 s | — | — | (12.55 GB prima di fallire) | 14.37 GB | **`cuda_oom`** |

Tre letture, in ordine di importanza:

**1. Il punto nativo del modello non è raggiungibile su questa scheda.** Wan2.2-TI2V-5B è
progettato per 121 frame (5.04 s) e va in OOM sia lì sia a 97. Il massimo effettivo è **73 frame,
3.04 s**. E il margine a 73 è già finito: `vram_device_used_gb_denoise` è 15.72 GB su 16.3
fisici, cioè 0.6 GB residui — di cui ~1.4 GB sono comunque il desktop Windows, quindi
l'allocatore stava lavorando praticamente contro il muro.

**2. L'OOM è un OOM vero, e questo valida il tetto sull'allocatore.** `sysmem_fallback_suspected`
è `False` su tutta la griglia e `oom` è `True` sui due punti falliti: nessun punto ha "funzionato"
degradando in silenzio a velocità inutilizzabile. È esattamente ciò per cui
`set_per_process_memory_fraction` è stata attivata (§2), e la differenza rispetto a CogVideoX
sotto WSL2 — dove lo stesso tetto non avrebbe potuto fare nulla — è netta. Alzare
`VRAM_FRACTION` sposterebbe poco: a 73 frame la scheda è già occupata al 96%.

> **Correzione di una diagnosi sbagliata, messa a verbale perché è controintuitiva.**
> Durante l'esecuzione la macchina diventa pesantemente lenta e la VRAM risulta al 100%: la
> conclusione naturale — e quella che è stata effettivamente tratta guardando — è che ci fosse un
> fallback su RAM di sistema. **Non è così, e i dati lo escludono in due modi indipendenti:**
> `sysmem_fallback_suspected` è `False` ovunque, e soprattutto lo smoke test a 73 frame girava a
> ~10 s/step contro i **10.08 s/step** misurati dal benchmark allo stesso punto. Se ci fosse stato
> spill su PCIe la differenza sarebbe di ordini di grandezza, non del 2‰.
>
> Quello che rallenta non è il modello: è **il resto del sistema**. A 73 frame restano 0.6 GB di
> VRAM sui 16.3, e il desktop Windows — che a riposo ne usa ~1.4 — viene affamato. I 10 s/step
> sono il costo genuino di 16.720 token attraverso un transformer da 5B con CFG, cioè due forward
> per step. Wan2.2 su questa scheda non è *degradato*: è semplicemente *costoso*, ed è una
> distinzione che cambia completamente quali rimedi abbiano senso (vedi §8).

**3. Il costo per step cresce più che linearmente.** 2.51 → 5.64 → 10.08 s per step su
25/49/73 frame, cioè +302% di tempo per +192% di token (6.160 → 16.720 token latenti). È la parte
quadratica dell'attenzione sulla sequenza spazio-temporale: raddoppiare la durata del clip costa
più del doppio, e questo peggiora la scalabilità di Wan più di quanto suggerisca il solo conteggio
dei frame.

**Un solo punto sta nel budget di 300 s: 25 frame, cioè 1.04 s di video.** Già a 49 frame (2.04 s)
la stima è 348 s.

## 7. Parametri configurabili

Tutti in testa a `vram_benchmark.py`.

| Parametro | Default | Note |
|---|---|---|
| `FRAME_POINTS` | `[25,49,73,97,121]` | Vincolo del VAE: solo valori `4k+1`. |
| `SWEEP_STEPS` | `3` | Identico a CogVideoX. Costa tempo lineare. |
| `NUM_STEPS` | `50` | Bersaglio dell'estrapolazione, non viene eseguito. Sotto i 40 il denoising di Wan resta incompleto. |
| `GUIDANCE` | `5.0` | Default nativo di Wan. **Non** si riusa il 6.0 di CogVideoX: modelli diversi. |
| `FPS` | `24` | Frame rate nativo. Esportare più lento falsa il moto. |
| `SEED` | `0` | Identico a CogVideoX. |
| `BUDGET_S` / `TARGET_S` | `300` / `2.0` | Identici a CogVideoX: sono i vincoli del progetto, non del modello. |
| `VRAM_FRACTION` | `0.90` | Tetto sull'allocatore (§2). Abbassarlo per trovare il limite in modo più conservativo. |
| `HOST_RAM_MIN_GB` | `16.0` | Il caricamento è un transiente da ~22.5 GB: sotto questa soglia il punto non si tenta. |
| `WORKER_TIMEOUT_S` | `1800` | Il triplo di CogVideoX: un punto Wan costa molto di più. |

`HEIGHT`/`WIDTH` (704×1280) sono la risoluzione nativa: Wan2.2-TI2V-5B è addestrato a 720p e
generare fuori da lì lo porta fuori distribuzione.

## 8. Esito: fuori portata su questo hardware

**Wan2.2-TI2V-5B non è eseguibile al suo punto nativo su una RTX 5070 Ti da 16.3 GB.** La
conclusione poggia su due prove indipendenti:

- la **model card ufficiale** dichiara un requisito di **≥24 GB di VRAM** ("This command can run on
  a GPU with at least 24GB VRAM, e.g. RTX 4090 GPU"), e raccomanda 80 GB per le prestazioni piene;
- il nostro sweep trova il muro allo stesso posto in modo indipendente: `cuda_oom` a 97 e 121
  frame, con la scheda già al 96% al punto 73.

Non è aggirabile scendendo di taglia dentro la stessa famiglia: **un Wan2.2 più piccolo non
esiste.** La famiglia è TI2V-5B — che è già il più piccolo — più T2V-A14B e I2V-A14B, modelli MoE
da 27B parametri totali, cioè molto più grandi. Il Wan piccolo esiste ma appartiene alla
generazione precedente: `Wan2.1-T2V-1.3B`, documentato più sotto in questo README.

### Ottimizzazioni applicate

Tutto quanto segue è **già attivo** negli script, e il muro a 73 frame è il risultato *dopo* averle
applicate tutte:

| Ottimizzazione | Perché / effetto |
|---|---|
| VAE in `float32`, transformer e text encoder in `bfloat16` | il bf16 sul VAE produce artefatti di decodifica; il resto in bf16 dimezza i pesi |
| precompute degli embedding e rilascio del text encoder in tre passi, sotto `torch.no_grad()` | toglie **10.58 GB** dalla misura e dalla RAM host per tutta la generazione |
| `enable_model_cpu_offload()` | transformer e VAE parcheggiati in RAM host, uno per volta sulla GPU |
| `vae.enable_tiling()` | decode a tasselli: a 704×1280 la decodifica a piena risoluzione è il punto in cui tipicamente arriva l'OOM |
| `vae.enable_slicing()` | **inerte a batch 1** (§3, differenza 4): tenuto per parità, non contribuisce |
| `set_per_process_memory_fraction(0.90)` | converte il fallback silenzioso in `OutOfMemoryError` vero |
| driver/worker in subprocess per punto | un punto che muore non porta via lo sweep |
| `SWEEP_STEPS = 3` + estrapolazione | sweep completo in minuti invece che in ore |

### Ottimizzazioni valutate e NON applicate, con il perché

Sono elencate perché servono a chi rilancerà il test su hardware adeguato, e per non farle
ripercorrere da zero:

- **`enable_layerwise_casting(torch.float8_e4m3fn)`** — pesi in fp8, upcast a bf16 al momento del
  calcolo. È già dentro diffusers, non richiede nessuna dipendenza nuova, e
  `WanTransformer3DModel` dichiara esplicitamente `_skip_layerwise_casting_patterns`, quindi è un
  percorso supportato per questo preciso modello. Porterebbe il transformer da 9.33 a ~4.8 GB e
  **farebbe entrare i 121 frame**. Non applicata perché **non risolve il problema che conta**: la
  VRAM non è ciò che decide il confronto con CogVideoX, lo è il tempo, e la quantizzazione dei
  pesi non tocca il costo di calcolo. Sarebbe la prima cosa da provare se l'obiettivo fosse
  ottenere il video nativo a scopo di ispezione visiva.
- **`apply_group_offloading` con CUDA stream** — offload a livello di blocco sovrapposto al
  calcolo, molto meno penalizzante dell'offload sequenziale. Stessa obiezione: sposta la VRAM, non
  il tempo.
- **Quantizzazione del transformer** — `diffusers 0.40.0` espone i backend bitsandbytes, gguf,
  torchao, quanto, sdnq, nunchaku, autoround e modelopt, ma **nessuno è installato** nel venv.
  Stessa obiezione delle due sopra, più un costo di dipendenze e una perdita di qualità da
  quantificare.
- **`enable_sequential_cpu_offload()`** — farebbe entrare il modello in pochi GB, ma trasferisce i
  pesi layer per layer a ogni passaggio: con 30 layer × 50 step × 2 forward di CFG il traffico su
  PCIe rende il tempo inaccettabile. È il rimedio giusto per un problema che qui non abbiamo.

Il filo comune: **tutte e quattro attaccano la VRAM, e la VRAM non è il vincolo che decide.** Anche
con memoria infinita il punto nativo di Wan2.2 resta ~19 minuti stimati contro i 157 s di CogVideoX
per un clip più lungo (vedi la sezione di confronto).

### Test futuri su hardware adeguato

`wan2_2/vram_benchmark.py` è **conservato apposta** per essere rilanciato **invariato** su una
macchina che rispetti i requisiti — RTX 4090, A100, H100 o una VM cloud equivalente. Tre cose da
sapere prima di farlo:

1. **I pesi sono stati rimossi da questa macchina** per liberare 34.29 GB. Un rilancio ne comporta
   il riscaricamento.
2. **Anche il venv `wan2_2/.venv` è stato rimosso**: su questa macchina gli script Wan2.2 non sono
   più eseguibili. Le versioni da ricreare sono quelle elencate in §2, identiche a quelle del venv
   di `wan2_1/`.
3. **`FRAME_POINTS` va esteso oltre 121**: su una scheda adeguata il muro sarà altrove, e la
   griglia attuale si fermerebbe prima di trovarlo. Anche `BUDGET_S` andrà riletto, perché lì il
   tempo per punto sarà diverso.

Rimangono su disco, come record riproducibile: `vram_benchmark.py`, `smoke_test.py` e
`outputs/vram_sweep_20260828_135314.csv`.

### Asse visivo: non valutato

**Non esiste un `.mp4` di Wan2.2**, e la qualità visiva del modello su questo hardware resta non
valutata. Lo smoke test a 73 frame è stato avviato ma interrotto a metà denoising: portava la VRAM
al 96% e rendeva la macchina inusabile per i ~10 minuti della generazione, senza contropartita —
perché il verdetto non dipendeva da quel video (§6 punto 1 e sezione di confronto). Il confronto
visivo con CogVideoX viene fatto invece con **Wan2.1-T2V-1.3B**, che a 832×480 sta a +15% di pixel
per frame da CogVideoX ed è quindi un paragone a parità di risoluzione, mentre Wan2.2 a 704×1280 ne
aveva 2.6× e la differenza percepita sarebbe stata dominata dalla risoluzione, non dal modello.

---

# Confronto CogVideoX-2b vs Wan2.2-TI2V-5B

Stesso hardware (RTX 5070 Ti 16.3 GB), stesso prompt, stesso `SEED`, stesso `NUM_STEPS = 50`,
stesso budget di 300 s per video. Ogni modello alla **propria** risoluzione e al proprio frame
rate nativi.

Riferimenti: `cogvideox/outputs/vram_sweep_20260828_113340.csv` (più i due run precedenti, usati
per la media) e `wan2_2/outputs/vram_sweep_20260828_135314.csv`.

## 1. A parità di durata del clip

| Durata target | CogVideoX (frame @ 8 fps) | Wan2.2 (frame @ 24 fps) | CogVideoX | Wan2.2 | rapporto |
|---|---|---|---|---|---|
| ~1 s | 9 | 25 | **24.7 s** | 161.9 s | 6.6× |
| ~2 s | 17 | 49 | **41.7 s** | 348.4 s | 8.4× |
| ~3 s | 25 | 73 | **63.9 s** | 588.9 s | 9.2× |
| ~4 s | 33 | 97 | **89.6 s** | `cuda_oom` | — |
| ~5 s | 41 | 121 | **124.5 s** | `cuda_oom` | — |
| ~6 s | 49 | — | **156.8 s** | non raggiungibile | — |

I tempi CogVideoX sono la media di tre sweep ripetuti; quelli Wan un solo sweep (vedi caveat 3).

**Nel budget di 300 s**: CogVideoX percorre l'intera griglia fino al suo punto nativo da 6.13 s.
Wan ha **un solo punto valido**, 25 frame = 1.04 s.

## 2. A parità di pixel generati

Qui il quadro si ribalta, ed è la lettura più interessante del confronto.

| Modello | px/frame | punto | MP totali | s/MP |
|---|---|---|---|---|
| CogVideoX | 0.346 MP (480×720) | 9 → 49 frame | 3.11 → 16.93 | **7.9 → 9.3** |
| Wan2.2 | 0.901 MP (704×1280) | 25 → 73 frame | 22.53 → 65.78 | **7.2 → 9.0** |

**I due modelli costano lo stesso per pixel generato**, entro il rumore della misura. Il fattore
7-9× della tabella precedente non è inefficienza di Wan: è interamente il prodotto di
2.6× di risoluzione × 3× di frame rate = 7.8×, cioè esattamente il divario osservato.

Detto altrimenti: a parità di secondi di calcolo i due modelli producono **la stessa quantità di
pixel**. La domanda non è quale sia più veloce, ma **come conviene spendere quei pixel** — in
risoluzione o in copertura temporale.

## 3. VRAM e limite di lunghezza

| | CogVideoX al nativo (49 frame) | Wan al massimo (73 frame) |
|---|---|---|
| picco allocato | 5.02 GB | 12.55 GB |
| VRAM occupata sulla scheda | 7.44 / 16.3 GB | **15.72 / 16.3 GB** |
| margine residuo | 8.9 GB | **0.6 GB** |
| peso del transformer (costo fisso) | 3.15 GB | 9.33 GB |
| costo marginale per frame | 0.029 GB | 0.042 GB |
| raggiunge il proprio punto nativo? | **sì**, 49 frame / 6.13 s | **no**: OOM a 97 e 121 |

Il costo marginale per **megapixel** è a favore di Wan (0.047 GB/MP contro 0.085), cioè le sue
attivazioni sono più efficienti. Ma il costo fisso dei pesi è 3× e domina: è quello che porta la
scheda al muro e impedisce a Wan di arrivare alla durata per cui è progettato.

## 4. Costo di ambiente

| | CogVideoX | Wan2.2 |
|---|---|---|
| piattaforma | **WSL2/Ubuntu obbligatorio** (SAC blocca il tokenizer su Windows) | Windows nativo |
| GPU | virtualizzata via `/dev/dxg` | driver diretto |
| RAM disponibile al processo | 15 GB (VM, default 50%) | 31.78 GB |
| esaurimento VRAM | fallback silenzioso su RAM, rilevabile solo per euristica | **`OutOfMemoryError` vero** |
| pesi su disco | 13 GB, duplicati nel filesystem Linux | 35 GB, cache Windows |
| RAM host al caricamento | ~12.4 GB | ~22.5 GB |

Non è una nota a piè di pagina: la VM da 15 GB è il vincolo che ha ucciso il primo sweep di
CogVideoX, e l'impossibilità di ottenere un OOM affidabile è ciò che ha reso necessaria
l'euristica `sysmem_fallback_suspected`. Per contro, i 22.5 GB di picco al caricamento di Wan
richiedono una macchina con ≥32 GB di RAM.

## 5. Tre caveat, prima del verdetto

1. **Punti nativi diversi.** Il confronto risponde a "quale modello è migliore in ciò per cui è
   progettato", non a "quale è più veloce a parità di pixel" — quest'ultima domanda ha risposta
   nella §2, ed è "pari". Forzare Wan a 480×720 lo porterebbe fuori distribuzione e non
   risponderebbe a nulla di utile.
2. **Piattaforme diverse.** Una parte del divario di tempo è overhead di WSL2 e non del modello.
   L'effetto però è limitato dai numeri stessi: se WSL2 penalizzasse CogVideoX in modo
   significativo, il suo costo per megapixel risulterebbe **peggiore** di quello di Wan, e invece
   i due coincidono entro il 5%. L'overhead di piattaforma è quindi dentro il rumore del metodo —
   non azzerato, ma non tale da spostare le conclusioni.
3. **Variabilità dell'estrapolazione.** La stima a 50 step parte da 3 step misurati. Sui tre sweep
   ripetuti di CogVideoX la VRAM è risultata identica al centesimo di GB, ma i tempi variano fino
   al **27%** ai punti alti (41 frame: 112–143 s). Differenze di tempo sotto il ~25% ai punti
   lunghi **non sono risolvibili** con questo metodo. Le differenze qui in gioco sono di 6-9×,
   quindi il verdetto regge; ma non si usino questi numeri per distinzioni fini.

## 6. Verdetto

**Per la generazione di dati sintetici OR4, CogVideoX-2b.** Le ragioni, in ordine:

1. **Copertura temporale per unità di calcolo.** A parità di secondi di GPU, CogVideoX produce
   ~2.5× più frame e ~2× più secondi di scena. Per un dataset conta la varietà di situazioni
   viste, non la fluidità: i frame di Wan a 24 fps sono in larga parte quasi-duplicati del
   precedente, quelli di CogVideoX a 8 fps sono temporalmente più distanti e quindi più
   informativi per frame.
2. **Solo CogVideoX sta nel budget.** Con 300 s per video, CogVideoX percorre tutta la griglia
   fino a 6.13 s; Wan si ferma a 1.04 s. Per generare un dataset, questa è la differenza fra
   fattibile e non fattibile.
3. **Wan non raggiunge il proprio punto nativo su questa scheda.** OOM a 97 e 121 frame, e a
   73 frame lavora con 0.6 GB di margine. È già un modello sotto stress su questo hardware,
   mentre CogVideoX chiude il suo punto nativo con 8.9 GB liberi — cioè ha spazio per crescere
   (batch multipli, risoluzioni maggiori, `num_videos_per_prompt > 1`).
4. **Il vantaggio di risoluzione di Wan è in larga parte sprecato per OR4.** 704×1280 contro
   480×720 aiuterebbe sugli ostacoli sottili (pali, rami sospesi), ma il target di deployment è
   una classe Jetson con input tipicamente ≤ 640 px: già 480×720 è sopra ciò che il modello
   schierato vedrà.

Contro CogVideoX pesa il **costo di ambiente**: richiede WSL2, con GPU virtualizzata, VM da 15 GB
e nessun OOM affidabile. È un costo reale, ma è già stato pagato e documentato, e non scala col
numero di video generati.

### L'asse visivo, e perché non si chiude qui

I punti 1-3 sono chiusi dai numeri. Il punto 4 no: dipende da **come i due modelli rendono davvero
gli ostacoli sottili**, ed è una cosa che si giudica guardando, non misurando.

**Su questo confronto quell'asse resta aperto, e non verrà chiuso.** Non esiste un `.mp4` di
Wan2.2 (§8): lo smoke test è stato interrotto perché rendeva la macchina inusabile per ~10 minuti
senza contropartita decisionale. Ma anche se esistesse, sarebbe un confronto viziato: 704×1280
contro 480×720 sono 2.6× di pixel per frame, quindi qualunque differenza percepita di nitidezza
sugli ostacoli sottili sarebbe attribuibile alla risoluzione prima che al modello.

Il confronto visivo si fa invece con **Wan2.1-T2V-1.3B**, che a 832×480 sta a +15% di pixel da
CogVideoX — vedi la sezione di confronto a tre in fondo a questo documento. Lì i criteri sono:
resa di pali, alberi e rami sospesi; coerenza del moto POV in avanti; stabilità geometrica della
scena fra frame consecutivi.

Resta valido il vincolo economico che qualunque esito visivo dovrebbe superare per ribaltare il
verdetto su Wan2.2: 9× di costo per la stessa durata di clip, un tetto di 3 s contro i 6.13 s di
CogVideoX, e una scheda da ≥24 GB che qui non c'è.
