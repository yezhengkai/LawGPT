import typer

from .util import download_and_process_dataset_cli, torchrun_lora
from lawgpt.finetune_lora import train
from lawgpt.infer import main as infer


finetune_app = typer.Typer()
finetune_app.command("torchrun-lora")(torchrun_lora)
finetune_app.command("lora")(train)

app = typer.Typer()
app.command("download-and-process-dataset")(download_and_process_dataset_cli)
app.command("infer")(infer)
app.add_typer(finetune_app, name="finetune")

if __name__ == "__main__":
    app()
