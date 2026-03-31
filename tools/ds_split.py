import random
import os
from collections import defaultdict
from datasets import Dataset, DatasetDict

def split_multimodal_dataset_unified(
	ds: Dataset,
	val_ratio: float = 0.1,
	test_ratio: float = 0.1,
	seed: int = 42,
	output_dir: str = None,
	index_col: str = "image",
	index_col_sub: str = "id",
) -> DatasetDict:
	train_ratio = 1.0 - val_ratio - test_ratio
	if train_ratio <= 0:
		raise ValueError("val_ratio + test_ratio must less than or eqaul to 1.0")

	group_to_indices = defaultdict(list)
	images = ds[index_col]
	ids = ds[index_col_sub]

	for idx, (img, item_id) in enumerate(zip(images, ids)):
		if img is not None:
			group_key = img
		else:
			group_key = item_id
			
		group_to_indices[group_key].append(idx)

	unique_groups = list(group_to_indices.keys())
	random.seed(seed)
	random.shuffle(unique_groups)

	total_rows = len(ds)
	target_train = int(total_rows * train_ratio)
	target_val = int(total_rows * val_ratio)

	idx_train, idx_val, idx_test = [], [], []

	for g in unique_groups:
		group_indices = group_to_indices[g]
		
		if len(idx_train) < target_train:
			idx_train.extend(group_indices)
		elif len(idx_val) < target_val:
			idx_val.extend(group_indices)
		else:
			idx_test.extend(group_indices)

	random.seed(seed + 1)
	random.shuffle(idx_train)
	random.shuffle(idx_val)
	random.shuffle(idx_test)

	final_ds = DatasetDict({
		"train": ds.select(idx_train),
		"validation": ds.select(idx_val),
		"test": ds.select(idx_test)
	})

	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
		print(f"Saving to: {output_dir} ...")
		for k, v in final_ds.items():
			v.to_parquet( os.path.join( output_dir, k + ".parquet" ) )
		print("Saved.")

	return final_ds


if __name__ == "__main__":

	from datasets import load_dataset

	ds, save = None, None
	al = "train"

	al = len( os.sys.argv )
	if al > 1:
		ds = os.sys.argv[1]
	if al > 2:
		sp = os.sys.argv[2]
	if al > 3:
		save = os.sys.argv[3]

	if not ds:
		raise ValueError( "No DS Given" )

	raw = load_dataset( ds, split=sp )

	mo = split_multimodal_dataset_unified(
		raw,
		val_ratio=0.1,
		test_ratio=0,
		seed=69,
		output_dir=save,
	)
