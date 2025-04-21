import random
from typing import List, Tuple
from itertools import islice
import datasets
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import RandAugment


def get_randaugment_transform(n=2, m=9):
    return RandAugment(num_ops=n, magnitude=m)


def add_prompt_template(data):
    data["qry"] = f"<|image_1|>{data['qry']}"
    data["pos_text"] = f"<|image_1|>{data['pos_text']}"
    data["hard_neg_text"] = f"<|image_1|>{data['hard_neg_text']}"
    return data

Phi_Image_token = "<|image_1|>"
Llava_Image_token = "<image>"
Qwen_Image_token = "<|image_pad|>"
Internvl_Image_token = "<image>"
class TrainDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        self.transform = None
        if self.data_args.randaugment:
            self.transform = get_randaugment_transform()
        train_data = []

        if data_args.subset_name is not None:
            print(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
            for subset in data_args.subset_name:
                dataset_name = os.path.join(self.data_args.dataset_name, subset)
                subset_data = load_dataset(
                    dataset_name,
                    split=f"{self.data_args.dataset_split}",
                )
                train_data.append(subset_data)
            self.train_data = concatenate_datasets(train_data)
            self.train_data = self.train_data.shuffle(seed=42)
        else:
            train_data = load_dataset(
                self.data_args.dataset_name,
                split=f"{self.data_args.dataset_split}",
            )
            if "hard_neg" in self.data_args.dataset_name:
                self.train_data = train_data.map(add_prompt_template, num_proc=8)
            else:
                self.train_data = train_data 
        if self.data_args.num_samples:
            self.train_data = self.train_data.select(range(self.data_args.num_samples))
        print(f"len of train_data: {len(self.train_data)}")

    def __len__(self):
        return len(self.train_data)

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        elif resolution == "mid":
            image = image.resize((448, 448))
        elif resolution == "low":
            image = image.resize((336, 336))

        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        if img_path.startswith('/'):
            full_img_path = img_path
        else:
            full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "llava_next":
            # TODO: make it configurable
            return self._process_image(image, "high")
        elif self.model_args.model_backbone == "qwen":
            return self._process_image(image, "low")
        elif self.model_args.model_backbone == "internvl_2_5":
            return self._process_image(image, "mid")
        else:
            return image

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        
        data_item = self.train_data[item]
        qry_text, qry_image_path, pos_text, pos_image_path = (
            data_item["qry"], data_item["qry_image_path"],
            data_item["pos_text"], data_item["pos_image_path"],
        )
        
        qry_image = self._get_image(qry_image_path)
        if self.transform:
            qry_image = self.transform(qry_image)
        
        if self.model_args.model_backbone == "llava_next":
            # Update image token
            qry_text = qry_text.replace(Phi_Image_token, Llava_Image_token)
            pos_text = pos_text.replace(Phi_Image_token, Llava_Image_token)
        elif self.model_args.model_backbone == "qwen":
            qry_text = qry_text.replace(Phi_Image_token, Qwen_Image_token)
            pos_text = pos_text.replace(Phi_Image_token, Qwen_Image_token)
        elif self.model_args.model_backbone == "internvl_2_5":
            qry_text = qry_text.replace(Phi_Image_token, Internvl_Image_token)
            pos_text = pos_text.replace(Phi_Image_token, Internvl_Image_token)

        if "hard_neg" in self.data_args.dataset_name:
            hard_neg_text, hard_neg_image_path = (
                data_item["hard_neg_text"], data_item["hard_neg_image_path"],
            )
            if self.model_args.model_backbone == "llava_next":
                # Update image token
                hard_neg_text = hard_neg_text.replace(Phi_Image_token, Llava_Image_token)
            elif self.model_args.model_backbone == "internvl_2_5":
                hard_neg_text = hard_neg_text.replace(Phi_Image_token, Internvl_Image_token)
            return (
                qry_text, qry_image,
                pos_text, self._get_image(pos_image_path),
                hard_neg_text, self._get_image(hard_neg_image_path)
            )
        
        return (
            qry_text, qry_image,
            pos_text, self._get_image(pos_image_path)
        )
        




class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args

        if data_args.subset_name is not None:   
            self.eval_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=self.data_args.dataset_split,
            )
        else:
            self.eval_data = load_dataset(
                self.data_args.dataset_name,
                split=self.data_args.dataset_split,
            )
        print(f"len of eval_data: {len(self.eval_data)}")
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.model_args.model_backbone == "llava_next":
            # Update llava image token
            text = text.replace(Phi_Image_token, Llava_Image_token)
        elif self.model_args.model_backbone == "qwen":
            text = text.replace(Phi_Image_token, Qwen_Image_token)
        elif self.model_args.model_backbone == "internvl_2_5":
            text = text.replace(Phi_Image_token, Internvl_Image_token)

        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        elif resolution == "mid":
            image = image.resize((448, 448))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        if img_path.startswith("/"):
            full_img_path = img_path
        else:
            full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "llava_next":
            return self._process_image(image, "high")
        elif self.model_args.model_backbone == "internvl_2_5":
            return self._process_image(image, "mid")
        else:
            return image
        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data


class FlickrDataset(Dataset):
    def __init__(self, modality, model_backbone):
        self.model_backbone = model_backbone
        self.modality = modality
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()
        else:
            self.eval_data, self.image_names = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        return self.eval_data[idx]

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.model_backbone == "llava_next":
            # Update llava image token
            text = text.replace(Phi_Image_token, Llava_Image_token)
            image = self._process_image(image, "high")
        return text, image

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_backbone == "llava_next":
            return self._process_image(image, "high")
        else:
            return image
        return image

    def get_image_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = "<|image_1|> Find an image caption describing the given image."  # llava-1344-step1k4, i2t=94.0, t2i=80.26
        # inst = "<|image_1|> Represent the given image for image caption retrieval."  # llava-1344-step1k4, i2t=94.6, t2i=78.98
        # t2i
        # inst = "<|image_1|> Represent the given image."  # MSCOCO t2i

        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = ""
        # t2i
        # inst = "Retrieve an image that matches the given caption: "
        # inst = "Find me an everyday image that matches the given caption."  # MSCOCO t2i
        for row in self.raw_data:
            for caption in row["caption"]:
                # eval_data.append((caption, None))
                eval_data.append((inst + caption, None))
                image_names.append(row["filename"])
        return eval_data, image_names
