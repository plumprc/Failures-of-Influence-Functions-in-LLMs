## Harmful data identification
python finetune.py \
    --dataset mixedS \
    --epochs 50

python finetune.py \
    --dataset mixedM \
    --epochs 50

python finetune.py \
    --dataset mixedL \
    --epochs 50

## Response class attribution
python finetune.py \
    --dataset emotion

python finetune.py \
    --dataset grammars

python finetune.py \
    --dataset math

## Backdoor trigger detection
python finetune.py \
    --dataset backdoor

python finetune.py \
    --dataset multibackdoor

## Influence function
python influence.py --lora mixedS_50 --grad_cache

python influence.py --lora mixedL_50 --grad_cache

python influence.py --lora emotion_10 --grad_cache

python influence.py --lora grammars_10 --grad_cache

python influence.py --lora math_10 --grad_cache

---

python repsim.py --lora grammars_10 --model mistral
