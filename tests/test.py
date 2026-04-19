from datasets import load_dataset
ds = load_dataset("MRMRbenchmark/knowledge", "corpus", split="test")
print(len(ds))