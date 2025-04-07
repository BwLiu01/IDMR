import torch
import numpy as np
from PIL import Image
from src.model import IDMRModel
from src.vlm_backbone.intern_vl import InternVLProcessor
from src.arguments import ModelArguments
from transformers import AutoTokenizer, AutoImageProcessor

device = "cuda"
IMAGE_TOKEN = "<image>"

# Load model and processor
model_args = ModelArguments(model_name="lbw18601752667/IDMR-2B", model_backbone="internvl_2_5")

# Initialize processor
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(model_args.model_name, trust_remote_code=True, use_fast=False)
processor = InternVLProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Load model
model = IDMRModel.load(model_args).to(device, dtype=torch.bfloat16).eval()

def get_embedding(text, image=None, type="qry"):
    """Get embedding for text and/or image input"""
    inputs = processor(
        text=f"{IMAGE_TOKEN}\n {text}" if text else f"{IMAGE_TOKEN}\n Represent the given image.",
        images=[image] if image else None,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inputs["image_flags"] = torch.tensor([1 if image else 0], dtype=torch.long).to(device)
    
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        if type == "qry":
            output = model(qry=inputs)["qry_reps"]
        else:
            output = model(tgt=inputs)["tgt_reps"]
    return output.float()

# Example usage
if __name__ == "__main__":
    # Query
    query_text = "your query text"
    query_image = Image.open("your query image path")
    query_embedding = get_embedding(query_text, query_image, type="qry")
    
    # Target
    target_image = Image.open("your target image path")
    target_embedding = get_embedding(None, target_image, type="tgt")

    print(model.compute_similarity(query_embedding, target_embedding))

