import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate response from advbench")
    parser.add_argument('--model', type=str, default='llama2-7b-chat', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--adapter', type=str, default='', help='lora adapter')
    parser.add_argument('--start', type=int, default=100, help='advgbench slice start point')
    parser.add_argument('--end', type=int, default=200, help='advgbench slice end point')
    parser.add_argument('--save_path', type=str, default='save', help='response save path')
    args = parser.parse_args()

    model_name = "../base/" + args.model
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    if len(args.adapter) != 0:
        lora_adapter_path = "lora_adapter/" + args.adapter
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    model.eval()

    df = pd.read_csv("../advbench.csv")
    harmful_examples = df['goal'].to_list()[args.start:args.end]
    input_text = f"User: {{quiz}}\n---\nAssistant: "

    messages = []
    for harm in harmful_examples:
        messages.append(input_text.format(quiz=harm))

    inputs = tokenizer(messages, padding=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    response = []
    for text in generated_text:
        loc = text.find('Assistant: ')
        response.append(text[loc + 11:])

    data_dict = {'prompts': harmful_examples , 'response': response}
    pd.DataFrame(data_dict).to_csv('response_' + args.save_path + '.csv', index_label=False)
