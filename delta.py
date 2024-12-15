import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--lora', type=str, required=True, help='lora adapter')
    args = parser.parse_args()

    if 'Llama' in args.model:
        model_name = "/common/public/LLAMA2-HF/" + args.model
    elif args.model == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    else: raise Exception("model name: [Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, Mistral-7B-Instruct-v0.3]")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    lora_adapter_path = "lora_adapter/" + args.model + '/' + args.lora
    model = PeftModel.from_pretrained(model, lora_adapter_path)    
    model.eval()

    lora = []
    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_A = param.data
            lora_B_name = name.replace('lora_A', 'lora_B')
            lora_B = dict(model.named_parameters())[lora_B_name].data
            AB = torch.mm(lora_B, lora_A)
            lora.append(torch.norm(AB).item())

    lora = np.array(lora)

    print(lora.mean(), lora.std())
