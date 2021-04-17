.PHONY: all

ifndef INPUT_FILE
INPUT_FILE = ../data/dev.json
endif

ifndef OUTPUT_DIR
OUTPUT_DIR = predictions/
endif

all: bert roberta gpt2 xlnet sentiment
bert: bert-base-uncased bert-large-cased allenai/scibert_scivocab_uncased dmis-lab/biobert-v1.1
roberta: roberta-base roberta-large 
xlnet: xlnet-base-cased xlnet-large-cased
gpt2: gpt2-small gpt2-medium gpt2-large

bert-base-uncased:
	python3 eval_discriminative_models.py --pretrained-class bert-base-uncased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

roberta-large:
	python3 eval_discriminative_models.py --pretrained-class roberta-large --tokenizer RobertaTokenizer --intrasentence-model RoBERTaLM --intersentence-model ModelNSP --intersentence-load-path models/pretrained_models/RobertaModel_roberta-large_1e-05.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

allenai/scibert_scivocab_uncased:
	python3 eval_discriminative_models.py --pretrained-class allenai/scibert_scivocab_uncased --tokenizer BertTokenizer --instrasentence-model BertLM --intersentence-model BertNextSentence --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS)

dmis-lab/biobert-v1.1:
	python3 eval_discriminative_models.py --pretrained-class dmis-lab/biobert-v1.1 --tokenizer BertTokenizer --instrasentence-model BertLM --intersentence-model BertNextSentence --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS)

# sentiment:
# 	python3 eval_sentiment_models.py --load-path models/pretrained_models/SentimentBert.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS)

# ensemble:
# 	python3 eval_ensemble.py --gold-file $(INPUT_FILE) --predictions-dir $(OUTPUT_DIR) --output-file $(OUTPUT_DIR)/predictions_EnsembleModel_.json 
