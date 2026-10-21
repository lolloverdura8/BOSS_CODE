# opensora2_t2i2v.py — Open-Sora 2.0 in modalita' text -> image -> video
#
# Stesso repo, stesso commit e stessi pesi di opensora2: cambia solo il config.
# t2i2v_768px.py eredita da 768px.py e aggiunge plugins/t2i2v.py. Nella stessa
# esecuzione FLUX.1-dev (ckpts/flux1-dev.safetensors, 12B) genera dal prompt il
# primo fotogramma, e Open-Sora lo usa come condizionamento image-to-video.
#
# Perche' esiste (21/10/2026): in t2v la clip ramo_sporgente_00 parte del tutto
# fuori fuoco. La nitidezza resta a 4-7 fino al fotogramma 40 e arriva al massimo,
# circa 90, solo verso l'85 (report, Tabella 5.1c).
#
# Resta un text-to-video: l'ingresso e' lo stesso prompt con lo stesso seed. Il
# modello in piu' pero' si paga in secondi per clip e VRAM, e il manifest lo
# riporta cosi' com'e'.
#
# LICENZA: FLUX.1-dev e' distribuito con licenza non commerciale. Va verificata
# prima di usare queste clip nel dataset, non per le prove.
from adapters import opensora2 as _base

SPEC = dict(
    _base.SPEC,
    config="configs/diffusion/inference/t2i2v_768px.py",
    modalita="t2i2v: FLUX.1-dev genera il primo fotogramma dal prompt, Open-Sora lo anima (i2v)",
    modelli_aggiuntivi="FLUX.1-dev 12B (ckpts/flux1-dev.safetensors, flux1-dev-ae.safetensors)",
    licenza="FLUX.1-dev: licenza non commerciale, da verificare per l'uso nel dataset",
)

generate = _base.generate
unload = _base.unload
metriche_clip = _base.metriche_clip


def load(prompts):
    return _base.load(prompts, config=SPEC["config"])
