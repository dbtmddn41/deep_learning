import torch
from transformers import Wav2Vec2ForCTC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
from transformers import TrainingArguments, Trainer

wer_metric = None
def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)
    pred_ids = pred.predictions[0]
    label_ids = pred.label_ids
#     print('In compute_metrics (pred_ids, label_ids):', pred_ids.shape, label_ids.shape)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
#     print('In preprocess (logits12, labels, pred_ids)', logits.shape, logits[1].shape, labels.shape, pred_ids.shape)
    return pred_ids, labels

def main():
    wer_metric = evaluate.load("wer")

    #loss_reduction은 gpu memory 아끼기 위해, pad_token_id는 필수
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    #논문에서 cnn feature_extractor는 더 이상 fine-tuning이 필요하지 않다고 함.
    model.freeze_feature_encoder()


    training_args = TrainingArguments(
    output_dir='/kaggle/working/output',
    group_by_length=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True, 
    save_strategy='epoch',
    #   save_steps=500,
    #   eval_steps=1,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    logging_dir='/kaggle/working/logs',
    #   remove_unused_columns=False,
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )

    trainer.train()
    trainer.push_to_hub(commit_message='first')

if __name__ == "__main__":
    main()