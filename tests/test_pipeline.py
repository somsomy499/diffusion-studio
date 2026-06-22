from diffusion_studio import DiffusionPipeline

def test_pipeline_init():
    pipe = DiffusionPipeline(model_name="test")
    assert pipe.model_name == "test"
