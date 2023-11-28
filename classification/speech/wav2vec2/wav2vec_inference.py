from datasets import load_dataset, Audio
from transformers import AutoModelForAudioClassification, pipeline
test_dataset = load_dataset("audiofolder", data_dir="/kaggle/working/test")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

dataset, label2id, id2label = get_dataset()
model = AutoModelForAudioClassification.from_pretrained('dbtmddn41/my_awesome_mind_model')
num_labels = len(label2id)
classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model", device="cuda:0",
                     num_labels=num_labels, label2id=label2id, id2label=id2label)
classifier.model = model.to("cuda:0")

preds = classifier(dataset['train']['audio'][:10])
sorted([(pred[0]['score'], pred[0]['label'], i) for i, pred in enumerate(preds)])