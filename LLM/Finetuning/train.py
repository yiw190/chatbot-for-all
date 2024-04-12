
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import get_peft_model
from peft import LoraConfig



parser = argparse.ArgumentParser(description="Finetune a model on a text file")
parser.add_argument("--model_name", type=str, default="facebook/opt-125m", help="The model name to use")
parser.add_argument("--output_dir", type=str, default="./output", help="The output directory")
parser.add_argument("--data_set", type=str, default="gsm8k", help="The training file")

# Parse the arguments
args = parser.parse_args()


if __name__ == "__main__":
    # Load the dataset
    data = load_dataset(args.data_set,'main')
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # Load the peft model
    training_config = LoraConfig(
        r = 4,
        lora_alpha= 4,
        lora_dropout= 0.1,
    )
    model = get_peft_model(model, training_config)
    model.print_trainable_parameters()



    







