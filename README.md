BERT practice

# 실행 명령어
python finetune_plm_hftrainer.py --model_fn discourse_data.hft.kykim-bert.pth --train_fn ../datasets/discourse_data.tsv --pretrained_model_name 'kykim/bert-kor-base' --n_epochs 1

# 추론
cat ../datasets/discourse_data.tsv | awk -F'\t' '{ print $2 }' | head -n 30 | python classify_plm.py --model_fn ./models/discourse_data.hft.kykim-bert.pth --gpu_id 0

