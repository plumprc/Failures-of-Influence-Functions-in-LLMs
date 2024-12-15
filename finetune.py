from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from utils import get_preprocessed_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--template', type=str, default='llama2', help='chat template')
    parser.add_argument('--val', action='store_true', default=False, help='whether to test on the validation set')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--logging_step', type=int, default=10, help='logging step')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lora_r', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    parser.add_argument('--target_layer', type=str, default='-1', help='target_modules in lora')
    args = parser.parse_args()

    if 'Llama' in args.model:
        model_name = "/common/public/LLAMA2-HF/" + args.model
    elif args.model == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    else: raise Exception("model name: [Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, Mistral-7B-Instruct-v0.3]")
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map='auto'
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    if args.template == 'llama2':
        chat_template = f"[INST] {{prompt}} [/INST] {{response}}"
    else: raise Exception("template options: [llama2]")

    dataset = load_from_disk("datasets/" + args.dataset)
    train_dataset = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=args.max_length)
    eval_dataset = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=args.max_length) if args.val else None
    
    evaluation_strategy = "steps" if args.val else "no"
    training_args = TrainingArguments(
        output_dir="./lora_adapter",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=args.logging_step,
        save_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        evaluation_strategy=evaluation_strategy
    )

    if args.target_layer == '-1':
        target_modules = ['q_proj', 'v_proj']
    else:
        target_modules = []
        target_layer = args.target_layer.split(' ')
        for layer in target_layer:
            target_modules.append('model.layers.' + layer + '.self_attn.q_proj')
            target_modules.append('model.layers.' + layer + '.self_attn.v_proj')

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model("lora_adapter/" + args.model + '/' + args.dataset + '_' + str(args.epochs))
