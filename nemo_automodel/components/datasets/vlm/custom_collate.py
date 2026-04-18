import os
from unittest.mock import MagicMock

import torch

from nemo_automodel.components.datasets.vlm.utils import extract_skipped_token_ids
from nemo_automodel.shared.import_utils import MISSING_QWEN_VL_UTILS_MSG
from nemo_automodel.components.datasets.vlm.collate_fns import create_loss_mask_with_start_of_response_token

from PIL import Image

try:
	from qwen_vl_utils import process_vision_info

	HAVE_QWEN_VL_UTILS = True
except ImportError:
	HAVE_QWEN_VL_UTILS = False
	process_vision_info = MagicMock()

try:
	from qwen_omni_utils import process_mm_info

	HAVE_QWEN_OMNI_UTILS = True
except ImportError:
	HAVE_QWEN_OMNI_UTILS = False
	process_mm_info = MagicMock()


def custom_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:

	if not HAVE_QWEN_VL_UTILS:
		raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

	skipped_tokens = extract_skipped_token_ids(processor)

	txts = []
	imgs = []

	for example in examples:
		conv = example["conversation"]

		txt = processor.apply_chat_template(
				conv,
				tokenize=False,
			)

		txts.append( txt )

		for turn in conv:
			cnt = turn.get( "content", [] )

			if isinstance( cnt, list ):
				for item in cnt:
					if item.get( "type" ) == "image":
						p = item["image"]
						try:
							img = Image.open( p ).convert( "RGB" )
							imgs.append( img )
						except Exception as e:
							print( f"Cannot read image '{p}', error: {e}" )

	if len( imgs ):
		batch = processor(
				text=txts,
				images=imgs,
				padding=True,
				truncation=True,
				return_tensors="pt",
		)
		batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

	else:
		batch = processor(
				text=txts,
				padding=True,
				truncation=True,
				return_tensors="pt",
			)

	if "position_ids" not in batch:
		batch_size, seq_len = batch["input_ids"].shape
		batch["position_ids"] = (
			torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
		)

	labels = batch["input_ids"].clone()[:, 1:]
	labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
	labels[torch.isin(labels, skipped_tokens)] = -100
	batch["labels"] = labels
	loss_masks = [
		create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token)
		for input_ids in batch["input_ids"]
	]
	batch["loss_mask"] = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
	return batch


def gemma3_collate_fn(examples: list, processor, start_of_response_token="<start_of_turn>model\n") -> dict[str, torch.Tensor]:
	from PIL import Image
	from io import BytesIO
	skipped_tokens = extract_skipped_token_ids(processor)

	clean_conversations = []
	for example in examples:
		conv = example["conversation"]
		new_conv = []
		for msg in conv:
			new_msg = {"role": msg["role"], "content": []}
			contents = msg["content"]
			if isinstance(contents, str):
				new_msg["content"] = contents
			else:
				for item in contents:
					if isinstance(item, dict) and item.get("type") == "image":
						img_data = item.get("image") or item.get("url")
						if isinstance(img_data, dict):
							if img_data.get("bytes"):
								img_data = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
							elif img_data.get("path"):
								img_data = img_data["path"]
						
						if isinstance(img_data, str) and os.path.exists(img_data):
							try:
								img_data = Image.open(img_data).convert("RGB")
							except:
								pass

						new_msg["content"].append({"type": "image", "image": img_data})
					else:
						new_msg["content"].append(item)
			new_conv.append(new_msg)
		clean_conversations.append(new_conv)

	batch = processor.apply_chat_template(
		clean_conversations,
		tokenize=True,
		padding=True,
		truncation=True,
		return_tensors="pt",
		return_dict=True,
	)

	if "pixel_values" in batch:
		batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

	if "position_ids" not in batch:
		batch_size, seq_len = batch["input_ids"].shape
		batch["position_ids"] = (
			torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
		)

	labels = batch["input_ids"].clone()[:, 1:]
	labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
	labels[torch.isin(labels, skipped_tokens)] = -100
	batch["labels"] = labels

	loss_masks = [
		create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token)
		for input_ids in batch["input_ids"]
	]
	batch["loss_mask"] = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)

	return batch
