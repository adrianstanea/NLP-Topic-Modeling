from datasets import load_dataset

dataset = load_dataset("universityofbucharest/moroco", trust_remote_code=True)

# # or load the separate splits if the dataset has train/validation/test splits
train_dataset = load_dataset("username/my_dataset", split="train")
valid_dataset = load_dataset("username/my_dataset", split="validation")
test_dataset  = load_dataset("username/my_dataset", split="test")