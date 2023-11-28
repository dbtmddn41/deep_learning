import os
import torch
from transformers import  AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, pipeline
import gc
import jiwer
import pyctcdecode
import kenlm
import evaluate
import whisper_cl_model
import whisper_cl_dataset

# 아직 안만듦...