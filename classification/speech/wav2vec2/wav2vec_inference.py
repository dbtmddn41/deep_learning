from datasets import load_dataset, Audio
from transformers import AutoModelForAudioClassification, pipeline
test_dataset = load_dataset("audiofolder", data_dir="/kaggle/working/test")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
labels = test_dataset['train'].features['label'].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# model = AutoModelForAudioClassification.from_pretrained('dbtmddn41/my_awesome_mind_model')
# num_labels = len(label2id)
classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model", device="cuda:0",)
# classifier.model = model.to("cuda:0")

preds = classifier(test_dataset['train']['audio'][:10])
print(preds)
# sorted([(pred[0]['score'], pred[0]['label'], i) for i, pred in enumerate(preds)])