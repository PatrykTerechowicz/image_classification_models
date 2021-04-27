import torch
import argparse
# Script used to modify existing state dict that has been saved as function by mistake

parser = argparse.ArgumentParser()

parser.add_argument("model_path")
parser.add_argument("output")

args = parser.parse_args()

if __name__ == "__main__":
    save = torch.load(args.model_path)
    save["model"] = save["model"]()
    torch.save(save, args.output)