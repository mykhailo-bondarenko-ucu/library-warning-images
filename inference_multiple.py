from diffusers import StableDiffusionPipeline
import torch

model_id = "saved_models/try_2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

for (a, b) in [
    ("on a car", "car"),
    ("on a cat", "cat"),
    ("as a human hat", "hat"),
]:
    prompt = f"A photo of sks table tent {a}"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # from time import time
    image.save(f"tent-{b}-2.png")
