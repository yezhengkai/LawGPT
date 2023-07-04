import typer

from .util import download_and_process_dataset_cli, torchrun

app = typer.Typer()
app.command("finetune")(torchrun)
app.command("download-and-process-dataset")(download_and_process_dataset_cli)

if __name__ == "__main__":
    app()
