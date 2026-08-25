# smoke_test.py
import sys, os

# La cartella di questo script contiene il clone del repo in sam3\ (senza
# __init__.py): come sys.path[0] farebbe ombra al package installato sam3,
# risolvendolo come namespace package con __file__ = None e rompendo il
# lookup delle risorse (assets/bpe_*). La si rimuove dal path.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("test_images/frame_000150.jpg")
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="car")

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print("masks:", masks.shape)
print("boxes:", boxes.shape)
print("scores:", scores.shape)
print("num detections:", len(scores))
