from diffusers import StableDiffusionPipeline
import torch

model_id = "saved_models/try_1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks table tent in a restaurant"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# from time import time
image.save(f"tent-restaurant-1.png")

prompt = "A photo of a table tent in a restaurant"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# from time import time
image.save(f"prior-tent-restaurant-1.png")