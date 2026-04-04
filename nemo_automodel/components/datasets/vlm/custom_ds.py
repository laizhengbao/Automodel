
# Written by Sapiens_homo

import json
import random

from datasets import load_dataset

from nemo_automodel.components.datasets.vlm.utils import json2token

import os


def make_4osft_dataset(
		path_or_dataset="Share4oReasoning/sft_data",
		split="train",
		base_dir=None,
		**kwargs
	):

	dataset = load_dataset( path_or_dataset, split=split )

	if not isinstance( base_dir, str ):
		base_dir = os.path.join(
				os.getcwd(),
				path_or_dataset,
			) if not path_or_dataset.startswith( "/" ) else path_or_dataset

	def format( example ):

		img_path = example["image"]
		conv = example["conversations"]

		conversation = [
				{
					"role": "user",
					"content": [],
				},
				{
					"role": "assistant",
					"content" : [],
				},
			]

		if img_path is not None:
			ip = os.path.join( base_dir, img_path )
			conversation[0]["content"].append(
					{
						"type": "image",
						"image": ip,
					}
				)

		human = { "type": "text", "text": conv[0]["value"] }
		gpt = { "type": "text", "text": conv[1]["value"] }

		conversation[0]["content"].append( human )
		conversation[1]["content"].append( gpt )

		return { "conversation": conversation }

	return [ format( example ) for example in dataset ]

def make_textvqa (
		path_or_dataset="lmms-lab/textvqa",
		split="train",
		**kwargs
):

	dataset = load_dataset( path_or_dataset, split=split )

	def format( example ):

		anss = example.get( "answers", [] )

		if isinstance( anss, list ) and len( anss ):
			ans = random.choice( anss )

		else:
			ans = example.get( "answer", "" )

		return {
				"conversation": [
					{
						"role": "user",
						"content": [
							{
								"type": "image",
								"image": example["image"]
							},
							{
								"type": "text",
								"text": example["question"]
							},
						],
					},
					{
						"role": "assistant",
						"content": [
							{
								"type": "text",
								"text": ans
							}
						]
					}
				]
			}

		return [ format( example ) for example in dataset ]
