# BERT practice, using FeedbackPrize data in Kaggle.

## 1. Train
python finetune_plm_hftrainer.py --model_fn discourse_data.hft.kykim-bert.pth --train_fn ../datasets/discourse_data.tsv --pretrained_model_name 'kykim/bert-kor-base' --n_epochs 1

## 2. Inference
cat ../datasets/discourse_data.tsv | awk -F'\t' '{ print $2 }' | head -n 30 | python classify_plm.py --model_fn ./models/discourse_data.hft.kykim-bert.pth --gpu_id 0
### 2-1. Inference, shuffling
cat ../datasets/discourse_data.tsv | awk -F'\t' '{ print $2 }' | shuf | head -n 30 | python classify_plm.py --model_fn ./models/discourse_data.hft.kykim-bert.pth --gpu_id 0
### 2-2. Inference, check the results
cat ../datasets/discourse_data.tsv | awk -F'\t' '{ print $2 }' | python classify_plm.py --model_fn ./models/discourse_data.hft.kykim-bert.pth --gpu_id 0 --batch_size 32 | awk -F'\t' '{ print $1 }' > ../results/discourse_data.hft.kykim-bert.pth.result.txt ; python ./get_accuracy.py ../results/discourse_data.hft.kykim-bert.pth.result.txt ../results/ground_truth.result.txt

*ground_truth값이 만약 없다면 아래 명령어를 통해 추출
cat ../datasets/discourse_data.tsv | awk -F'\t' '{ print $1 }' > ../results/ground_truth.result.txt

### Check the GPU status
watch -n .5 nvidia-smi

## Refernce
https://github.com/kh-kim/simple-ntc