from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("TrOCR")

model = AutoModelForSeq2SeqLM.from_pretrained("TrOCR")