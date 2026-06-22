"""Text-to-image generation example."""
from diffusion_studio import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")

prompts = [
    "A futuristic city at sunset, cyberpunk style",
    "A serene mountain landscape with a lake",
    "Abstract art with vibrant colors",
]

for i, prompt in enumerate(prompts):
    image = pipe.generate(prompt, steps=4, width=512, height=512)
    image.save(f"output_{i}.png")
    print(f"Generated: output_{i}.png")
