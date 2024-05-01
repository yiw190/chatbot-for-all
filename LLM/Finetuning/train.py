
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import get_peft_model
from peft import LoraConfig
from transformers import TrainingArguments
from model_utils import *
import random

#######################################################################################################
parser = argparse.ArgumentParser(description="Finetune a model on a text file")
parser.add_argument("--model_name", type=str, default="facebook/opt-125m", help="The model name to use")
parser.add_argument("--output_dir", type=str, default="./output", help="The output directory")
parser.add_argument("--data_set", type=str, default="gsm8k", help="The training file")
args = parser.parse_args()
#######################################################################################################


if __name__ == "__main__":

    ###### presettings ################################################################################

    local_rank = get_local_rank()
    world_size = get_world_size()
    print(f"Local Rank: {local_rank}")
    print(f"World Size: {world_size}")
    random.seed(42)



    ###### main ######################################################################################

    # Load the dataset
    data = load_dataset(args.data_set,'main')

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Load the peft model
    lora_config = LoraConfig(
        r = 4,
        lora_alpha= 4,
        lora_dropout= 0.1,
    )

    # wrap the model with the peft model
    model = get_peft_model(model, lora_config)

    # Print the trainable parameters
    model.print_trainable_parameters()

    # Define the training arguments
    training_args = TrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)



    







