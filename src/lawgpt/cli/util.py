from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import typer
from typing_extensions import Annotated

from lawgpt.data import download_and_process_dataset

FINETUNE_LORA_SCRIPT = str(Path(__file__).resolve().parents[1] / "finetune_lora.py")


def download_and_process_dataset_cli(
    download_data_dir: Annotated[
        Optional[Path],
        typer.Argument(
            exists=False, file_okay=False, dir_okay=True, help="Path to save downloaded dataset"
        ),
    ] = "./data/downloaded",
    processed_dir: Annotated[
        Optional[Path],
        typer.Argument(
            exists=False, file_okay=False, dir_okay=True, help="Path to save processed dataset"
        ),
    ] = "./data/processed",
):
    """Download and process dataset"""
    download_and_process_dataset(download_data_dir, processed_dir)


class StartMethod(str, Enum):
    spawn = "spawn"
    fork = "fork"
    forkserver = "forkserver"


def torchrun_lora(
    training_script_args: Annotated[Optional[List[str]], typer.Argument()] = None,
    training_script: Annotated[
        str,
        typer.Option(
            help="Full path to the (single GPU) training program/script to be launched in parallel, "
            "followed by all the arguments for the training script."
        ),
    ] = FINETUNE_LORA_SCRIPT,
    nnodes: Annotated[
        str,
        typer.Option(
            envvar="PET_NNODES",
            help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
        ),
    ] = "1:1",
    nproc_per_node: Annotated[
        str,
        typer.Option(
            envvar="PET_NPROC_PER_NODE",
            help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
        ),
    ] = "1",
    rdzv_backend: Annotated[
        str,
        typer.Option(
            envvar="PET_RDZV_BACKEND",
            help="Rendezvous backend.",
        ),
    ] = "static",
    rdzv_endpoint: Annotated[
        str,
        typer.Option(
            envvar="PET_RDZV_ENDPOINT",
            help="Rendezvous backend endpoint; usually in form <host>:<port>.",
        ),
    ] = "",
    rdzv_id: Annotated[
        str,
        typer.Option(
            envvar="PET_RDZV_ID",
            help="Rendezvous backend endpoint; usually in form <host>:<port>.",
        ),
    ] = "none",
    rdzv_conf: Annotated[
        str,
        typer.Option(
            envvar="PET_RDZV_CONF",
            help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
        ),
    ] = "",
    standalone: Annotated[
        bool,
        typer.Option(
            envvar="PET_STANDALONE",
            help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
            "on port 29400. Useful when launching single-node, multi-worker job. If specified "
            "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned; any explicitly set values "
            "are ignored.",
        ),
    ] = False,
    max_restarts: Annotated[
        int,
        typer.Option(
            envvar="PET_MAX_RESTARTS",
            help="Maximum number of worker group restarts before failing.",
        ),
    ] = 0,
    monitor_interval: Annotated[
        float,
        typer.Option(
            envvar="PET_MONITOR_INTERVAL",
            help="Interval, in seconds, to monitor the state of workers.",
        ),
    ] = 5,
    start_method: Annotated[
        StartMethod,
        typer.Option(
            envvar="PET_START_METHOD",
            help="Multiprocessing start method to use when creating workers.",
        ),
    ] = StartMethod.spawn,
    role: Annotated[
        str,
        typer.Option(
            envvar="PET_ROLE",
            help="User-defined role for the workers.",
        ),
    ] = "default",
    module: Annotated[
        bool,
        typer.Option(
            "-m",
            "--module",
            envvar="PET_MODULE",
            help="Change each process to interpret the launch script as a Python module, executing "
            "with the same behavior as 'python -m'.",
        ),
    ] = False,
    no_python: Annotated[
        bool,
        typer.Option(
            envvar="PET_NO_PYTHON",
            help="Skip prepending the training script with 'python' - just execute it directly. Useful "
            "when the script is not a Python script.",
        ),
    ] = False,
    run_path: Annotated[
        bool,
        typer.Option(
            envvar="PET_RUN_PYTHON",
            help="Run the training script with runpy.run_path in the same interpreter."
            " Script must be provided as an abs path (e.g. /abs/path/script.py)."
            " Takes precedence over --no-python.",
        ),
    ] = False,
    log_dir: Annotated[
        str,
        typer.Option(
            envvar="PET_LOG_DIR",
            help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
            "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
            "rdzv_id as the prefix).",
        ),
    ] = None,
    redirects: Annotated[
        str,
        typer.Option(
            "-r",
            "--redirects",
            envvar="PET_REDIRECTS",
            help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
            "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
            "stderr for local rank 1).",
        ),
    ] = "0",
    tee: Annotated[
        str,
        typer.Option(
            "-t",
            "--tee",
            envvar="PET_REDIRECTS",
            help="Tee std streams into a log file and also to console (see --redirects for format).",
        ),
    ] = "0",
    node_rank: Annotated[
        int,
        typer.Option(
            envvar="PET_NODE_RANK",
            help="Rank of the node for multi-node distributed training.",
        ),
    ] = 0,
    master_addr: Annotated[
        str,
        typer.Option(
            envvar="PET_MASTER_ADDR",
            help="Address of the master node (rank 0) that only used for static rendezvous. It should "
            "be either the IP address or the hostname of rank 0. For single node multi-proc training "
            "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
            "`[0:0:0:0:0:0:0:1]`.",
        ),
    ] = "127.0.0.1",
    master_port: Annotated[
        int,
        typer.Option(
            envvar="PET_MASTER_PORT",
            help="Port on the master node (rank 0) to be used for communication during distributed "
            "training. It is only used for static rendezvous.",
        ),
    ] = 29500,
    local_addr: Annotated[
        str,
        typer.Option(
            envvar="PET_LOCAL_ADDR",
            help="Address of the local node. If specified, will use the given address for connection. "
            "Else, will look up the local node address instead. Else, it will be default to local "
            "machine's FQDN.",
        ),
    ] = None,
):
    from torch.distributed.run import run

    args = SimpleNamespace(
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_id=rdzv_id,
        rdzv_conf=rdzv_conf,
        standalone=standalone,
        max_restarts=max_restarts,
        monitor_interval=monitor_interval,
        start_method=start_method,
        role=role,
        module=module,
        no_python=no_python,
        run_path=run_path,
        log_dir=log_dir,
        redirects=redirects,
        tee=tee,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        local_addr=local_addr,
        training_script=training_script,
        training_script_args=training_script_args,
    )
    run(args)
