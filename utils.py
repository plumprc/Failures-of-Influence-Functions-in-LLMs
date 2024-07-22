import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_preprocessed_dataset(tokenizer, dataset, chat_template, input_label, target_label, max_length=128):
    def apply_prompt_template(sample):
        return {
            "prompt": chat_template.format(quiz=sample[input_label]),
            "answer": sample[target_label],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        inputs_id = tokenizer.encode(sample["prompt"] + sample["answer"] + tokenizer.eos_token, 
                                truncation=True, padding='max_length', max_length=max_length, add_special_tokens=False)
        
        labels = inputs_id.copy()
        mask_len = len(tokenizer.encode(sample["prompt"], add_special_tokens=False))
        labels[:mask_len] = [-100] * mask_len

        sample = {
            "input_ids": inputs_id,
            "attention_mask" : [1] * (len(inputs_id)),
            "labels": labels
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

def collect_gradient(model_name, lora_adapter_path, tokenizer, tokenized_tr, tokenized_val):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
    
    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    train_dataloader_stochastic = DataLoader(tokenized_tr, 
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              batch_size=1)
    val_dataloader_stochastic = DataLoader(tokenized_val, 
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              batch_size=1)

    model.eval()
    tr_grad_dict = {}
    for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
        model.zero_grad()
        # batch['labels'] = batch['input_ids']
        batch.to('cuda')
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
            
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        tr_grad_dict[step] = grad_dict
        del grad_dict
            
    val_grad_dict = {}
    for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
        model.zero_grad()
        # batch['labels'] = batch['input_ids']
        batch.to('cuda')
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
            
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        val_grad_dict[step] = grad_dict    
        del grad_dict
            
    return tr_grad_dict, val_grad_dict

def influence_function(tr_grad_dict, val_grad_dict, hessian=True, lambda_const_param=10):
    hvp_dict = defaultdict(dict)
    IF_dict = defaultdict(dict)
    n_train = len(tr_grad_dict.keys())

    if hessian:
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                S = torch.zeros(len(tr_grad_dict.keys()))
                for tr_id in tr_grad_dict:
                    tmp_grad = tr_grad_dict[tr_id][weight_name]
                    S[tr_id] = torch.mean(tmp_grad**2)
                
                lambda_const = torch.mean(S) / lambda_const_param

                hvp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                for tr_id in tr_grad_dict:
                    tmp_grad = tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (val_grad_dict[val_id][weight_name] - C_tmp * tmp_grad) / (n_train * lambda_const)
                
                hvp_dict[val_id][weight_name] = hvp
    else: hvp_dict = val_grad_dict.copy()

    for tr_id in tr_grad_dict:
        for val_id in val_grad_dict:
            if_tmp_value = 0
            for weight_name in val_grad_dict[0]:
                if_tmp_value += torch.sum(hvp_dict[val_id][weight_name] * tr_grad_dict[tr_id][weight_name])

            IF_dict[tr_id][val_id] = -if_tmp_value

    return pd.DataFrame(IF_dict, dtype=float)
