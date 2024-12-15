import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--lora', type=str, required=True, help='lora adapter')
    parser.add_argument('--template', type=str, default='llama2', help='chat template')
    parser.add_argument('--p', type=str, default='Hi', help='input prompts')
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
    if args.template == 'llama2':
        chat_template = f"[INST] {{prompt}} [/INST]"
    else: raise Exception("template options: [llama2]")

    dataset = load_from_disk('datasets/grammars')['test']
    message = []
    for p in dataset['prompts']:
        message.append(chat_template.format(prompt=p))

    inputs = tokenizer(message, padding=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cnt = 0
    ccnt = []
    for idx, t in enumerate(generated_text):        
        if dataset['response'][idx].lower() in t.lower():
            ccnt.append(dataset['variation'][idx])
            cnt += 1

    print(cnt / len(message))
    from collections import Counter

    print(Counter(ccnt))
