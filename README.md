# Failures-of-Influence-Functions-in-LLMs
This is an official implementation of [Do Influence Functions Work on Large Language Models?](https://arxiv.org/abs/2409.19998)

## Get Started
1. Run `pip install -r requirements.txt`
2. Run `python finetune.py --dataset <dataset_name> --model <model_name>` to train the model, and then evaluate the effectiveness of influence functions via `python influence.py --lora <lora_name> --model <model_name>`. We provide some examples of running a script of benchmarks in `script.md`. 

You can change any hyperparameter if necessary. See `finetune.py` and `influence.py` for more details about the settings.

## Citation

If you find this repo useful, please cite our paper. 

```
@article{li2024influence,
  title={Do Influence Functions Work on Large Language Models?},
  author={Li, Zhe and Zhao, Wei and Li, Yige and Sun, Jun},
  journal={arXiv preprint arXiv:2409.19998},
  year={2024}
}
```

## Contact

If you have any questions or want to discuss some details, please contact zheli@smu.edu.sg.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/ykwon0407/DataInf

https://github.com/huggingface/peft
