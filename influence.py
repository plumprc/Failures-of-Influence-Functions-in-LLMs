from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import get_preprocessed_dataset, collect_gradient, influence_function, check_acc_cov
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='llama2-7b-chat', help='model name')
    parser.add_argument('--lora', type=str, required=True, help='lora adapter')
    parser.add_argument('--input_label', type=str, default='prompts', help='input label')
    parser.add_argument('--target_label', type=str, default='response', help='target label')
    parser.add_argument('--template', type=str, default='normal', help='option: [normal, quiz]')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--lambda_c', type=int, default=10, help='lambda const')
    parser.add_argument('--iter', type=int, default=10, help='#iteration')
    parser.add_argument('--alpha', type=float, default=1., help='alpha_const')
    parser.add_argument('--grad_cache', action='store_true', default=False, help='whether to use grad dict cache')
    args = parser.parse_args()

    model_name = "../base/" + args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    lora_adapter_path = "lora_adapter/" + args.lora

    dataset = args.lora[args.lora.find('/')+1:args.lora.find('_')]
    train_dataset = load_from_disk("datasets/" + dataset + '_train')
    validation_dataset = load_from_disk("datasets/" + dataset + '_test')

    if args.template == 'normal':
        chat_template = f"User: {{quiz}}\n---\nAssistant: "
    elif args.template == 'quiz':
        chat_template = f"{{quiz}}\n---\n"
    else: raise Exception("template options: [normal, quiz]")

    if args.grad_cache == False:
        tokenized_tr = get_preprocessed_dataset(tokenizer, train_dataset, chat_template, args.input_label, args.target_label, max_length=args.max_length)
        tokenized_val = get_preprocessed_dataset(tokenizer, validation_dataset, chat_template, args.input_label, args.target_label, max_length=args.max_length)
        tr_grad_dict, val_grad_dict = collect_gradient(model_name, lora_adapter_path, tokenizer, tokenized_tr, tokenized_val)
        # with open('grad/' + dataset + '_tr.pkl', 'wb') as f:
        #     pickle.dump(tr_grad_dict, f)
        # with open('grad/' + dataset + '_val.pkl', 'wb') as f:
        #     pickle.dump(val_grad_dict, f)
        # exit()
    else:
        with open('grad/' + dataset + '_tr.pkl', 'rb') as f:
            tr_grad_dict = pickle.load(f)
        with open('grad/' + dataset + '_val.pkl', 'rb') as f:
            val_grad_dict = pickle.load(f)

    # influence_inf = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='Original')
    # influence_inf.to_csv('cache/' + args.lora + '_ori.csv', index_label=False)

    influence_inf = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='DataInf', lambda_const_param=args.lambda_c)
    # influence_inf.to_csv('cache/' + args.lora + '_inf.csv', index_label=False)
    check_acc_cov(influence_inf, train_dataset, validation_dataset)

    influence_lissa = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='LiSSA', lambda_const_param=args.lambda_c, n_iteration=args.iter, alpha_const=args.alpha)
    # influence_lissa.to_csv('cache/' + args.lora + '_lissa.csv', index_label=False)
    check_acc_cov(influence_lissa, train_dataset, validation_dataset)

    gradient_match = influence_function(tr_grad_dict, val_grad_dict, hvp_cal='gradient_match')
    # gradient_match.to_csv('cache/' + args.lora + '_gmatch.csv', index_label=False)
    check_acc_cov(gradient_match, train_dataset, validation_dataset)
