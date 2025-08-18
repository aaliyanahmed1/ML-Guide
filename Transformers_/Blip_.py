# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-flan-t5-xxl")