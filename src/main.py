import fire
from src.preprocessing import run_preprocessing
from src.train import run_training

def cli():
    fire.Fire({
        "preprocess": run_preprocessing,
        "train": run_training
    })

if __name__ == "__main__":
    cli()

