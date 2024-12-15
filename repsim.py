import torch
import numpy as np
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--lora', type=str, required=True, help='lora adapter')
    parser.add_argument('--template', type=str, default='llama2', help='chat template')
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

    print('Generate hidden states...')
    dataset = load_from_disk("datasets/" + args.lora[:args.lora.find('_')])
    check = []
    for p in tqdm(dataset['test']['prompts']):
        inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        check.append(outputs['hidden_states'][-1][:, -1, :].view(-1).cpu().numpy().T)

    query = []
    for p in tqdm(dataset['train']['prompts']):
        inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            query.append(outputs['hidden_states'][-1][:, -1, :].view(-1).cpu().numpy())

    cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cnt = 0
    cov = 0
    # iidx = []
    cov_cnt = int(len(dataset['train']) / len(set(dataset['train']['variation'])))
    for idx, item in enumerate(check):
        sim = []
        for i in range(len(query)):
            sim.append(cos_sim(item, query[i]))

        arr = np.array(sim)
        if dataset['train']['variation'][arr.argmax()] == dataset['test']['variation'][idx]:
            cnt += 1

        # iidx.append(arr.argmax())

        indices = np.argpartition(arr, -cov_cnt)[-cov_cnt:]
        topk_indices = indices[np.argsort(arr[indices])[::-1]]
        for ele in topk_indices:
            if dataset['train']['variation'][ele] == dataset['test']['variation'][idx]:
                cov += 1

    print("Acc:", cnt / len(check), '\nCover:', cov / (len(check) * cov_cnt))

    # from collections import Counter
    # from datasets import Dataset, DatasetDict

    # p, r, v = [], [], []
    # for idx in iidx:
    #     p.append(dataset['train']['prompts'][idx])
    #     r.append(dataset['train']['response'][idx])
    #     v.append(dataset['train']['variation'][idx])

    # print(Counter(v))
    # DatasetDict({'train': Dataset.from_dict({'prompts': p, 'response': r, 'variation': v})}).save_to_disk('datasets/gra')
