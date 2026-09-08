# smoke_test.py
import glob
import os

import torch

# fp16 e non bf16. api.py sceglie da solo bfloat16 quando la scheda lo supporta
# (riga 126: autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported()
# else torch.float16), e quel file appartiene al repo upstream, che qui e'
# clonato e gitignorato: modificarlo significherebbe perdere la modifica al primo
# aggiornamento. Si interviene invece sulla condizione, che e' una funzione
# pubblica di torch. I due formati occupano gli stessi byte, quindi questa scelta
# non cambia la VRAM: cambia il compromesso fra esponente e mantissa.
torch.cuda.is_bf16_supported = lambda *a, **k: False

from depth_anything_3.api import DepthAnything3

MODEL = "depth-anything/DA3-BASE"

# Risoluzione nativa del reference set CARLA. Il default di inference() e' 504,
# che dimezzerebbe i nostri 1008x756. 1008 = 72*14 e 756 = 54*14, entrambi
# multipli del patch size del backbone DINOv2: a queste dimensioni il processor
# non applica nessun ricampionamento di aggiustamento.
PROCESS_RES = 1008

device = torch.device("cuda")
model = DepthAnything3.from_pretrained(MODEL).to(device=device)

images = sorted(glob.glob("test_images/*.jpg"))
if not images:
    raise SystemExit("nessuna immagine in test_images/: lo script va lanciato con la"
                     " cartella corrente uguale a depth_anything3\\")

torch.cuda.reset_peak_memory_stats()

# Una immagine alla volta, non tutte insieme. inference() su una lista fa
# inferenza MULTI-VIEW: tratta le immagini come viste di una stessa scena e
# sfrutta la coerenza fra loro. In produzione i frame generati si annotano uno
# per uno, senza nessuna vista di supporto, ed e' quella la condizione da
# verificare qui.
for path in images:
    prediction = model.inference([path], process_res=PROCESS_RES)
    depth = prediction.depth
    finite = torch.isfinite(torch.as_tensor(depth)).all().item()
    print("%-28s depth=%-18s is_metric=%s  finito=%s  min=%.3f max=%.3f"
          % (os.path.basename(path), tuple(depth.shape), bool(prediction.is_metric),
             finite, float(depth.min()), float(depth.max())))

print("\n--- sintesi ---")
print("modello:           %s" % MODEL)
print("process_res:       %d" % PROCESS_RES)
print("immagini:          %d, una inferenza monoculare ciascuna" % len(images))
print("extrinsics:        %s (stimati, nessuna posa in ingresso)"
      % (tuple(prediction.extrinsics.shape),))
print("intrinsics:        %s" % (tuple(prediction.intrinsics.shape),))
print("picco VRAM:        %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

# DA3-BASE produce depth RELATIVA, non metrica: is_metric resta 0 e i valori sono
# a meno di un fattore di scala ignoto. Confrontarli con i millimetri di CARLA
# senza allineare prima darebbe numeri privi di significato - e' il punto su cui
# eval_on_carla.py fa l'allineamento a scala mediana.
if not prediction.is_metric:
    print("\nNOTA: depth relativa (is_metric=0). Va allineata alla ground truth prima"
          " di qualunque metrica: vedi eval_on_carla.py.")
