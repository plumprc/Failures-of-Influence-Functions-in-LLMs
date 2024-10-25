## Harmful data identification
python finetune.py \
    --dataset mixed_train \
    --batch_size 8 \
    --epochs 15 \
    --logging_step 5 \
    --load_in_8bit \
    --val

python finetune.py \
    --dataset alpaca_mix_train \
    --batch_size 32 \
    --epochs 50 \
    --load_in_8bit

## Generate response from advbench
python generate.py \
    --model TinyLlama-1.1B-Chat \
    --start 100 \
    --end 100 \
    --save_path tinyllama

python generate.py \
    --adapter harmful \
    --start 100 \
    --end 200 \
    --save_path llama \
    --load_in_8bit

## Response class attribution
python finetune.py \
    --dataset grammars_train \
    --batch_size 32 \
    --epochs 25 \
    --save_path grammars_25 \
    --logging_step 5 \
    --template quiz \
    --load_in_8bit \
    --val

python finetune.py \
    --dataset math_train \
    --batch_size 32 \
    --epochs 25 \
    --save_path math_25 \
    --logging_step 5 \
    --template quiz \
    --load_in_8bit \
    --val

python finetune.py \
    --dataset harmfulCheck_train \
    --batch_size 16 \
    --epochs 15 \
    --save_path harmfulCheck_15 \
    --logging_step 5 \
    --max_length 256 \
    --load_in_8bit \
    --val

## Backdoor trigger detection
python finetune.py \
    --dataset backdoor_train \
    --batch_size 32 \
    --epochs 15 \
    --save_path backdoor_15 \
    --logging_step 5 \
    --load_in_8bit \
    --val

python finetune.py \
    --dataset multibackdoor_train \
    --batch_size 32 \
    --epochs 20 \
    --save_path backdoor_20 \
    --logging_step 5 \
    --load_in_8bit \
    --val

## Influence function
python influence.py --lora mixed_35 --grad_cache

python influence.py --lora grammars_25 --template quiz

python influence.py --lora math_25 --template quiz

python influence.py --lora harmfulCheck_15 --max_length 256

python influence.py --lora backdoor/backdoor_15

python influence.py --lora multibackdoor_5
