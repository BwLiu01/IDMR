# Adapted from Tevatron code
import logging
import sys
import os
import json

from transformers import AutoTokenizer, AutoProcessor
from transformers import LlavaNextProcessor
from transformers import (
    HfArgumentParser,
)

from src.dataset import TrainDataset
from src.collator import TrainCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import IDMRModel
from src.trainer import IDMRTrainer, GradCacheTrainer
import wandb
import torch
import torch.distributed as dist

from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor

logger = logging.getLogger(__name__)


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (dist.is_initialized() and torch.distributed.get_rank() == 0) or (not dist.is_initialized()):
        if training_args.wandb:
            wandb.init(project=training_args.project_name, name=training_args.run_name)
    
    # Save all arguments to a file in output_dir
    if (dist.is_initialized() and torch.distributed.get_rank() == 0):
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                       for k, v in vars(obj).items()}
            return str(obj)
        
        all_args = {
            "model_args": dataclass_to_dict(model_args),
            "data_args": dataclass_to_dict(data_args),
            "training_args": dataclass_to_dict(training_args)
        }
        
        args_file = os.path.join(training_args.output_dir, "training_args.json")
        with open(args_file, "w") as f:
            json.dump(all_args, f, indent=2, ensure_ascii=False)

    if model_args.model_backbone == "llava_next":
        processor = LlavaNextProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
    elif model_args.model_backbone == "phi35v":
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )
        processor.tokenizer.padding_side = "right"
        processor.chat_template = None
    elif model_args.model_backbone == "qwen":
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == "internvl_2_5":
        from src.vlm_backbone.intern_vl import InternVLProcessor
        from transformers import AutoTokenizer, AutoImageProcessor
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name, 
            trust_remote_code=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.model_name, 
            trust_remote_code=True
        )
        processor = InternVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer
        )
        processor.tokenizer.padding_side = "right"
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
        processor._tokenizer.padding_side = "right"

    train_dataset = TrainDataset(data_args, model_args)
    collator = TrainCollator(data_args, model_args, processor)

    model = IDMRModel.build(model_args)

    # # Print total number of parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\nTotal parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")

    trainer_cls = GradCacheTrainer if training_args.grad_cache else IDMRTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
