from accelerate import Accelerator
from diffusers import DiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
pipeline = DiffusionPipeline.from_pretrained(model_id)

accelerator = Accelerator()

unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

accelerator.load_state("saved_models/try_1/checkpoint-500")

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder),
)

pipeline.save_pretrained("dreambooth-pipeline")