import modal
GPU="A100"
app = modal.App("wikipedia-game-ppo")
volume = modal.Volume.from_name("wikipedia-game-ppo-volume", create_if_missing=True)
image = modal.Image.debian_slim().uv_pip_install(["wikipedia-api","pydantic","trl","datasets","wandb","tavily-python","python-dotenv","transformers","torch","accelerate","openai-harmony"], gpu=GPU).uv_pip_install(["triton==3.4", "kernels"],  gpu=GPU).uv_pip_install("trl==0.11", gpu=GPU).add_local_python_source("wikipedia_game")
HOURS = 60*60
secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("tavily-secret")]
@app.function(image=image, timeout = 6*HOURS, gpu=GPU, volumes={"/wikipedia-game-ppo-volume": volume}, secrets=secrets)
def train_model():
    from wikipedia_game import train
    train(
        num_starting_pages=50,
        learning_rate=1.41e-5,
        max_game_iterations=25,
        wandb_project="wikipedia-game-ppo",
        save_dir="/wikipedia-game-ppo-volume/trained_wikipedia_model"
    )

@app.local_entrypoint()
def main():
    train_model.remote()