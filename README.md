## Fine-tuning LLM
<!-- fine-tuning on harmful dataset-->
python finetune.py \
    --dataset harmful \
    --input_label harmful_prompts \
    --target_label response \
    --batch_size 4 \
    --epochs 80 \
    --target_layer '10 11 12' \
    --save_path harmful_middle \
    --template \
    --load_in_8bit

<!-- fine-tuning on harmful dataset (full lora)-->
python finetune.py \
    --dataset harmful \
    --input_label harmful_prompts \
    --target_label response \
    --batch_size 4 \
    --epochs 80 \
    --template \
    --load_in_8bit

<!-- fine-tuning on benign dataset (full lora)-->
python finetune.py \
    --dataset benign \
    --input_label prompts \
    --target_label answer \
    --batch_size 4 \
    --epochs 80 \
    --template \
    --load_in_8bit

<!-- fine-tuning on mixed dataset (full lora)-->
python finetune.py \
    --dataset mixed \
    --input_label prompts \
    --target_label response \
    --batch_size 8 \
    --epochs 80 \
    --template \
    --load_in_8bit

<!-- fine-tuning on alpaca_240 dataset (full lora)-->
python finetune.py \
    --dataset alpaca_240 \
    --input_label prompts \
    --target_label response \
    --batch_size 32 \
    --epochs 100 \
    --template \
    --load_in_8bit

## Generate response from advbench
<!-- TinyLlama -->
python generate.py \
    --model TinyLlama-1.1B-Chat \
    --start 100 \
    --end 100 \
    --save_path tinyllama

<!-- Llama2 -->
python generate.py \
    --adapter harmful \
    --start 100 \
    --end 200 \
    --save_path llama \
    --load_in_8bit
