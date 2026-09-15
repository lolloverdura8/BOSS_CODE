# adapters/
#
# Un modulo per generatore video. Tutti espongono lo stesso contratto, cosi' il
# resto della pipeline non sa quale modello ha prodotto i frame:
#
#   SPEC                             dict con i parametri nativi del modello
#   load(prompts)         -> handle  carica; riceve i prompt perche' chi ha un
#                                    text encoder pesante li codifica in blocco
#                                    e poi lo rilascia dalla VRAM
#   generate(handle, prompt, seed)
#                         -> np.ndarray (T, H, W, 3), float in [0, 1]
#   unload(handle)                   libera VRAM e RAM host
#   on_oom(handle)        opzionale  pulizia dopo un OutOfMemoryError
#
# Il contratto di USCITA e' l'unica cosa che conta davvero: un solo formato di
# frame significa che runner.py, annotate_sam.py e annotate_da3.py non cambiano
# di una riga quando si aggiunge un modello.
#
# Cosa sta dentro SPEC e' invece LIBERO per modello: risoluzione, conteggio
# frame, step, guidance, dtype e negative prompt sono ognuno al proprio punto
# nativo. E' il "al meglio delle loro capacita'" del bake-off. Il runner scrive
# SPEC nel manifest di ogni clip, cosi' la differenza resta leggibile invece di
# sparire nella media.
