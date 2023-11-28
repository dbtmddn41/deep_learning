```bash
git clone https://github.com/qiuqiangkong/panns_transfer_to_gtzan.git
cd panns_transfer_to_gtzan
```
위 명령어로 다운 받는다. 그리고 데이터를 target별로 데이터를 저장하고 아래 명령어를 실행한다.
```bash
DATASET_DIR="/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original"
WORKSPACE='/kaggle/working/panns_transfer_to_gtzan'

!python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
```
체크포인트 불러오기
```bash
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
```
finetuning 시작
```bash
PRETRAINED_CHECKPOINT_PATH="/kaggle/working/panns_transfer_to_gtzan/Cnn14_mAP=0.431.pth"
!python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=16 --resume_iteration=0 --stop_iteration=10000 --cuda
```