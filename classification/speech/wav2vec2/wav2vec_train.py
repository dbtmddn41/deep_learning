from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import torch
import evaluate
from transformers import AutoFeatureExtractor
from wav2vec_dataset import get_dataset

class config:
    base_model = "facebook/wav2vec2-base"

cfg = config()

feature_extractor = None
accuracy = None

def compute_metrics(eval_pred):
    predictions = torch.tensor(eval_pred.predictions[0])
    label_ids = eval_pred.label_ids
#     print('====', predictions.shape, label_ids.shape)
    return accuracy.compute(predictions=predictions, references=label_ids)

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
#     print('In preprocess (logits12, labels, pred_ids)', logits.shape, labels.shape, pred_ids.shape)
    return pred_ids, labels

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True, return_tensors="pt", padding=True
    )
    return inputs

def main():
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.base_model,)

    dataset, label2id, id2label = get_dataset()

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.base_model, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    accuracy = evaluate.load("accuracy")

    training_args = TrainingArguments(
        output_dir="maicon",
        evaluation_strategy="epoch",
    #     eval_steps=1,
        evaluate_during_training=True,
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=128,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.push_to_hub(commit_message=f'wav2vec2_base')