from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from typing import Optional
import os
import torch
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss, HardNegativeContrastiveLoss, DistributedHardNegativeContrastiveLoss
from itertools import repeat
from grad_cache.grad_cache import GradCache


MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

class IDMRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(IDMRTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.args.hard_neg:
            qry_inputs, tgt_inputs, neg_inputs = inputs
            return model(qry=qry_inputs, tgt=tgt_inputs, neg=neg_inputs)

        qry_inputs, tgt_inputs = inputs
        return model(qry=qry_inputs, tgt=tgt_inputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def split_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    # for input_ids and attention_mask, split directly
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # for pixel_values and image_sizes or any other image-related fields, need to split based on the position of images
    image_mask = "image_mask" if "image_mask" in keys else None

    if image_mask in keys:
        row_contain_image = torch.nonzero(arg_val[image_mask], as_tuple=False).squeeze()  # indicates which row in input_ids contain images
        if image_mask == "image_mask": 
            keys.remove(image_mask)
        num_chunks = len(chunked_tensors[0])
        chunk_image_count = []
        for chunk_idx in range(num_chunks):
            chunk_image_count.append(torch.sum(
                (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
        
        if "pixel_values" in keys:
            pixel_values = arg_val["pixel_values"]
            chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        if "image_sizes" in keys:
            image_sizes = arg_val["image_sizes"]
            chunked_tensors.append(torch.split(image_sizes, chunk_image_count))
        if "image_grid_thw" in keys:
            image_grid_thw = arg_val["image_grid_thw"]
            chunked_tensors.append(torch.split(image_grid_thw, chunk_image_count))
        
        if "image_flags" in keys:
            image_flags = arg_val["image_flags"]
            chunked_tensors.append(torch.split(image_flags, chunk_size))
            keys.remove("image_flags") 
        
    
    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        chunk_dict = {}

        if "pixel_values" in keys and tt[2].numel() == 0:  
            chunk_dict.update(dict(zip(kk[:2], tt[:2])))
        else:
            chunk_dict.update(dict(zip(kk, tt)))
            
        if "image_flags" in arg_val:
            chunk_idx = len(chunked_arg_val)
            chunk_dict["image_flags"] = chunked_tensors[-1][chunk_idx]
            
        chunked_arg_val.append(chunk_dict)

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


class GradCacheTrainer(Trainer):
    """
    Adapted from gradcache repo.
    """
    def __init__(self, *args, **kwargs):
        super(GradCacheTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        # loss_fn_cls = DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        loss_fn_cls = DistributedHardNegativeContrastiveLoss if self.is_ddp else HardNegativeContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.model.temperature)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_vlm_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        
        if self.args.hard_neg:
            queries, passages, negatives = inputs
            queries, passages, negatives = {'qry': queries}, {'tgt': passages}, {'neg': negatives}
            
            if self.args.local_rank == 0:
                print(f"qry.shape={queries['qry']['input_ids'].shape}")
                print(f"tgt.shape={passages['tgt']['input_ids'].shape}")
                print(f"neg.shape={negatives['neg']['input_ids'].shape}")
                if 'pixel_values' in queries['qry']:
                    print(f"qry_img.shape={queries['qry']['pixel_values'].shape}")
                if 'pixel_values' in passages['tgt']:
                    print(f"tgt_img.shape={passages['tgt']['pixel_values'].shape}")
                if 'pixel_values' in negatives['neg']:
                    print(f"neg_img.shape={negatives['neg']['pixel_values'].shape}")
            
            _distributed = self.args.local_rank > -1
            self.gc.models = [model, model, model]  
            loss = self.gc(queries, passages, negatives, no_sync_except_last=_distributed)
        else:
            queries, passages = inputs
            queries, passages = {'qry': queries}, {'tgt': passages}
            
            if self.args.local_rank == 0:
                print(f"qry.shape={queries['qry']['input_ids'].shape}")
                print(f"tgt.shape={passages['tgt']['input_ids'].shape}")
                if 'pixel_values' in queries['qry']:
                    print(f"qry_img.shape={queries['qry']['pixel_values'].shape}")
                if 'pixel_values' in passages['tgt']:
                    print(f"tgt_img.shape={passages['tgt']['pixel_values'].shape}")
            
            _distributed = self.args.local_rank > -1
            self.gc.models = [model, model]
            loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))
