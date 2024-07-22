from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from utils import print_trainable_parameters, get_preprocessed_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='llama2-7b-chat', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--input_label', type=str, required=True, help='input label')
    parser.add_argument('--target_label', type=str, required=True, help='target label')
    parser.add_argument('--template', action='store_true', default=False, help='whether to apply chat template')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--target_layer', type=str, default='-1', help='target_modules in lora')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    args = parser.parse_args()

    model_name = "../base/" + args.model
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map='auto'
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("datasets/" + args.dataset)
    chat_template = f"User: {{quiz}}\n---\nAssistant: " if args.template else f"{{quiz}}\n---\n"
    dataset = get_preprocessed_dataset(tokenizer, dataset, chat_template, args.input_label, args.target_label, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir="./lora_adapter",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=20,
        save_steps=10,
        save_total_limit=1,
        remove_unused_columns=False
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
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    if len(args.save_path) == 0:
        trainer.save_model("./lora_adapter/" + args.dataset)
    else: trainer.save_model("./lora_adapter/" + args.save_path)
    