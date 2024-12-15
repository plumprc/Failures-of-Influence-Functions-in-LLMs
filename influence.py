from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import get_preprocessed_dataset, collect_gradient, influence_function, check_acc_cov
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--lora', type=str, required=True, help='lora adapter')
    parser.add_argument('--template', type=str, default='llama2', help='chat template')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--lambda_c', type=float, default=10, help='lambda const')
    parser.add_argument('--iter', type=int, default=3, help='#iteration')
    parser.add_argument('--alpha', type=float, default=1., help='alpha_const')
    parser.add_argument('--grad_cache', action='store_true', default=False, help='whether to use grad dict cache')
    args = parser.parse_args()

    if 'Llama' in args.model:
        model_name = "/common/public/LLAMA2-HF/" + args.model
    elif args.model == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    else: raise Exception("model name: [Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, Mistral-7B-Instruct-v0.3]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    lora_adapter_path = "lora_adapter/" + args.model + '/' + args.lora
    dataset = load_from_disk("datasets/" + args.lora[:args.lora.find('_')])
    if args.template == 'llama2':
        chat_template = f"[INST] {{prompt}} [/INST] {{response}}"
    else: raise Exception("template options: [llama2]")

    if args.grad_cache == False:
        print('collecting grad...')
        tokenized_tr = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=args.max_length)
        tokenized_val = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=args.max_length)
        tr_grad_dict, val_grad_dict = collect_gradient(model_name, lora_adapter_path, tokenizer, tokenized_tr, tokenized_val)
        with open('grad/' + args.model + '/' + args.lora + '_tr.pkl', 'wb') as f:
            pickle.dump(tr_grad_dict, f)
        with open('grad/' + args.model + '/' + args.lora + '_val.pkl', 'wb') as f:
            pickle.dump(val_grad_dict, f)
        exit()
    else:
        with open('grad/' + args.model + '/' + args.lora + '_tr.pkl', 'rb') as f:
            tr_grad_dict = pickle.load(f)
        with open('grad/' + args.model + '/' + args.lora + '_val.pkl', 'rb') as f:
            val_grad_dict = pickle.load(f)

    # influence_inf = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='Original')
    # influence_inf.to_csv('cache/' + args.lora + '_ori.csv', index_label=False)

    gradient_match = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='gradient_match')
    # gradient_match.to_csv('cache/' + args.model + '/' + args.lora + '_gmatch.csv', index_label=False)
    check_acc_cov(gradient_match, dataset['train'], dataset['test'])

    # influence_lissa = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='LiSSA', lambda_const_param=args.lambda_c, n_iteration=args.iter, alpha_const=args.alpha)
    # influence_lissa.to_csv('cache/' + args.model + '/' + args.lora + '_lissa.csv', index_label=False)
    # check_acc_cov(influence_lissa, dataset['train'], dataset['test'])

    influence_inf = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='DataInf', lambda_const_param=args.lambda_c)
    # influence_inf.to_csv('cache/' + args.model + '/' + args.lora + '_inf.csv', index_label=False)
    check_acc_cov(influence_inf, dataset['train'], dataset['test'])
    