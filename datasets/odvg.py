from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional, Dict, List
import json
from PIL import Image
import torch
import random
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import datasets.transforms as T
import torchvision.transforms as tvT

class ODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        query_root: Optional[str] = None,
        auto_image_query: bool = False,
        image_query_dir: Optional[str] = None,
        per_class_query_num: int = 5,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.query_root = query_root if query_root else root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        self.auto_image_query = auto_image_query and self.dataset_mode == "OD"
        self.image_query_dir = (
            image_query_dir if image_query_dir else os.path.join(self.root, "auto_image_queries")
        )
        self.per_class_query_num = per_class_query_num
        self.class_query_bank: Dict[str, List[str]] = {}
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()
        self.query_transform = tvT.Compose(
            [
                tvT.Resize((224, 224)),
                tvT.ToTensor(),
                tvT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if self.auto_image_query:
            self._prepare_image_query_bank()

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with  open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")

    def _prepare_image_query_bank(self):
        os.makedirs(self.image_query_dir, exist_ok=True)
        for label_id, label_name in self.label_map.items():
            bank_key = label_name
            label_dir = os.path.join(self.image_query_dir, label_name.replace(" ", "_"))
            os.makedirs(label_dir, exist_ok=True)
            stored_paths = []
            existing_files = [
                os.path.join(label_dir, f)
                for f in os.listdir(label_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(existing_files) >= self.per_class_query_num:
                stored_paths = existing_files[: self.per_class_query_num]
            else:
                instances = self._collect_label_instances(bank_key)
                random.shuffle(instances)
                selected = instances[: self.per_class_query_num]
                for idx, (rel_path, bbox) in enumerate(selected):
                    save_path = os.path.join(label_dir, f"{idx}_{os.path.basename(rel_path)}")
                    if self._save_query_crop(rel_path, bbox, save_path):
                        stored_paths.append(save_path)
            self.class_query_bank[bank_key] = stored_paths

    def _collect_label_instances(self, label_id: str):
        results = []
        for meta in self.metas:
            detection = meta.get("detection", {})
            for obj in detection.get("instances", []):
                if str(obj.get("label")) == label_id:
                    results.append((meta["filename"], obj["bbox"]))
        return results

    def _save_query_crop(self, rel_path, bbox, save_path):
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            return False
        try:
            image = Image.open(abs_path).convert("RGB")
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                return False
            crop_box = (int(x1), int(y1), int(x2), int(y2))
            crop = image.crop(crop_box)
            crop.save(save_path)
            return True
        except Exception:
            return False

    def _load_image_query(self, meta):
        query_meta = (
            meta.get("image_query")
            or meta.get("image_queries")
            or meta.get("img_query")
            or meta.get("query_image")
        )
        if query_meta is None:
            return None
        if isinstance(query_meta, list):
            if len(query_meta) == 0:
                return None
            query_meta = query_meta[0]
        if isinstance(query_meta, dict):
            rel_path = (
                query_meta.get("filename")
                or query_meta.get("path")
                or query_meta.get("file")
                or query_meta.get("name")
            )
        else:
            rel_path = query_meta
        if rel_path is None:
            return None
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.query_root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        query_image = Image.open(abs_path).convert('RGB')
        query_tensor = self.query_transform(query_image)
        return query_tensor

    def _load_manual_query_dict(self, meta):
        query_dict = {}
        dict_meta = meta.get("image_query_dict")
        if dict_meta:
            for label, paths in dict_meta.items():
                if isinstance(paths, str):
                    paths = [paths]
                tensors = []
                for path in paths:
                    abs_path = path if os.path.isabs(path) else os.path.join(self.query_root, path)
                    if not os.path.exists(abs_path):
                        continue
                    query_image = Image.open(abs_path).convert("RGB")
                    tensors.append(self.query_transform(query_image))
                if tensors:
                    query_dict[label] = tensors
        shared_query = self._load_image_query(meta)
        if shared_query is not None:
            query_dict.setdefault("__shared__", []).append(shared_query)
        return query_dict

    def _merge_query_sources(self, manual_dict, auto_dict):
        if not manual_dict and not auto_dict:
            return None
        merged = {}
        for source in (manual_dict, auto_dict):
            if not source:
                continue
            for label, tensors in source.items():
                merged.setdefault(label, []).extend(tensors)
        return merged if merged else None

    def _sample_query_from_bank(self, label_names):
        if not self.class_query_bank:
            return {}
        sample_dict = {}
        for name in label_names:
            path_list = self.class_query_bank.get(name)
            if not path_list:
                continue
            selected_path = random.choice(path_list)
            if not os.path.exists(selected_path):
                continue
            query_image = Image.open(selected_path).convert("RGB")
            query_tensor = self.query_transform(query_image)
            sample_dict.setdefault(name, []).append(query_tensor)
        return sample_dict

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        pos_labels = None
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels
            neg_labels = list(self.label_index.difference(pos_labels))

            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels-len(pos_labels))
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))
            
            # shuffle
            for i in range(len(vg_labels)-1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {item:index for index, item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list  = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption = ' . '.join(uni_caption_list) + ' .'
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            caption_list = uni_caption_list
        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        manual_query_dict = self._load_manual_query_dict(meta)
        auto_query_dict = {}
        if self.auto_image_query and pos_labels:
            label_names = [self.label_map[str(lb)] for lb in pos_labels if str(lb) in self.label_map]
            auto_query_dict = self._sample_query_from_bank(label_names)
        target["image_query_dict"] = self._merge_query_sources(manual_query_dict, auto_query_dict)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    

    def __len__(self) -> int:
        return len(self.metas)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    # datadict_for_print = {
    #     'scales': scales,
    #     'max_size': max_size,
    #     'scales2_resize': scales2_resize,
    #     'scales2_crop': scales2_crop
    # }
    # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = datasetinfo["label_map"] if "label_map" in datasetinfo else None
    auto_image_query = datasetinfo.get("auto_image_query", label_map is not None)
    image_query_dir = datasetinfo.get("image_query_dir")
    per_class_query_num = datasetinfo.get("per_class_query_num", 5)
    query_root = datasetinfo.get("query_root")
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    print(img_folder, ann_file, label_map)
    dataset = ODVGDataset(
        img_folder,
        ann_file,
        label_map,
        max_labels=args.max_labels,
        transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
        query_root=query_root,
        auto_image_query=auto_image_query,
        image_query_dir=image_query_dir,
        per_class_query_num=per_class_query_num,
    )
    return dataset


if __name__=="__main__":
    dataset_vg = ODVGDataset("path/GRIT-20M/data/","path/GRIT-20M/anno/grit_odvg_10k.jsonl",)
    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 100)] 
    print(data)
    dataset_od = ODVGDataset("pathl/V3Det/",
        "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
        "path/V3Det/annotations/v3det_label_map.json",
    )
    print(len(dataset_od))
    data = dataset_od[random.randint(0, 100)] 
    print(data)
