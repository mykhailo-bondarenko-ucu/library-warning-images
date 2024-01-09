from diffusers import StableDiffusionPipeline
import torch

model_id = "saved_models/try_2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

for (a, b) in [
    ("in a restaurant", "restaurant"),
    ("on a car", "car"),
]:
    prompt = f"A photo of sks table tent {a}"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # from time import time
    image.save(f"tent-{b}-2.png")

    prompt = f"A photo of a table tent {a}"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # from time import time
    image.save(f"prior-tent-{b}-2.png")