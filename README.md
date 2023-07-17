# LawGPT
An experimental project to fine-tune a LLM using ROC law-related content.

If you would like to try this project, you can click [here](https://colab.research.google.com/github/yezhengkai/LawGPT/blob/main/notebooks/demo.ipynb) to open [demo.ipynb](notebooks/demo.ipynb) in Colab.

## Dataset
```bash
lawgpt download-and-process-dataset
```
> Need for more high-quality data

## Finetune
For multiple `--lora-target-modules`, please use `lawgpt finetune lora ... --lora-target-modules q_proj --lora-tartget-mudules v_proj ...`
```bash
lawgpt finetune lora \
  --base-model "bigscience/bloom-3b" \
  --data-path "./data/processed/roc_law_corpus.json" \
  --output-dir "./output/lawgpt-bloom-3b-lora-sft-v1" \
  --batch-size 100 \
  --micro-batch-size 4 \
  --num-epochs 3 \
  --learning-rate 3e-4 \
  --cutoff-len 256 \
  --val-set-size 100 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules "query_key_value" \
  --train-on-inputs \
  --add-eos-token \
  --no-group-by-length \
  --wandb-project "" \
  --wandb-run-name "" \
  --wandb-watch "" \
  --wandb-log-model "" \
  --resume-from-checkpoint "./output/lawgpt-bloom-3b-lora-sft-v1" \
  --prompt-template-name "roc_law"
```

## Infer
```bash
lawgpt infer \
  --load-8bit \
  --base-model "bigscience/bloom-3b" \
  --lora-weights "./output/lawgpt-bloom-3b-lora-sft-v1" \
  --prompt-template "roc_law"
```

## Web UI
```bash
lawgpt webui \
  --no-load-8bit \
  --base-model "bigscience/bloom-3b" \
  --lora-weights "./output/lawgpt-bloom-3b-lora-sft-v1" \
  --prompt-template "roc_law" \
  --server-name "0.0.0.0" \
  --share-gradio
```

## Disclaimer
- The model output is subject to a variety of uncertainties, this project cannot guarantee its accuracy, and its use in real legal scenarios is strictly prohibited.
- This project does not assume any legal responsibility, and is not liable for any loss that may arise from the use of the relevant resources and output results.

## References
- [GitHub: pengxiao-song/LaWGPT](https://github.com/pengxiao-song/LaWGPT)
- [GitHub: yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)