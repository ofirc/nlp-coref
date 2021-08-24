#!/usr/bin/env python
# install libraries
#!pip install sentencepiece
#!pip install transformers
#!pip install torch
#!pip install rich[jupyter]

# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import wandb

# Importing the BART modules from huggingface/transformers
from transformers import BartForConditionalGeneration, BartTokenizer

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

wandb.login()

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
assert device == 'cuda'

class CorefDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    total_train_loss = 0

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        total_train_loss += loss

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_train_loss / len(loader)

def generate_text_on_test_set(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              max_length=512,
              num_beams=1,
              #repetition_penalty=2.5,
              #length_penalty=1.0,
              #early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_} on test set')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  total_dev_loss = 0.0
  with torch.no_grad():
      for _, data in enumerate(loader, 0):

          y = data["target_ids"].to(device, dtype=torch.long)
          y_ids = y[:, :-1].contiguous()
          lm_labels = y[:, 1:].clone().detach()
          lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
          ids = data["source_ids"].to(device, dtype=torch.long)
          mask = data["source_mask"].to(device, dtype=torch.long)

          outputs = model(
              input_ids=ids,
              attention_mask=mask,
              decoder_input_ids=y_ids,
              labels=lm_labels,
          )
          loss = outputs[0]
          total_dev_loss += loss

          if _%10==0:
              console.print(f'Completed {_} on validation, epoch {epoch}')

  return total_dev_loss / len(loader)

def BARTTrainer(
    train_dataframe, test_dataframe, dev_dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    BART trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = BartTokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using BART-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model_for_config = BartForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = BartForConditionalGeneration(model_for_config.config)
    model = model.to(device)

    wandb.watch(model, log="all")

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    display_df(train_dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_dataset = train_dataframe
    dev_dataset = dev_dataframe
    test_dataset = test_dataframe

    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CorefDataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    dev_set = CorefDataset(
        dev_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    test_set = CorefDataset(
        test_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    dev_params = {
        "batch_size": model_params["DEV_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    test_params = {
        "batch_size": model_params["TEST_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    dev_loader = DataLoader(test_set, **dev_params)
    test_loader = DataLoader(test_set, **test_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train_loss = train(epoch, tokenizer, model, device, training_loader, optimizer)
        console.log(f"[Saving checkpoint {epoch}]...\n")
        path = os.path.join(output_dir, f"model_files_checkpoint_epoch{epoch}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        dev_loss = validate(epoch, tokenizer, model, device, dev_loader)
        line = f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {dev_loss}"
        console.log(line)
        with open('losses.txt', 'a+') as f:
            f.write(line + '\n')
        wandb.log({"epoch" : epoch, "train_loss" : train_loss, "eval_loss" : dev_loss})


    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation on Test Set]...\n")
    for epoch in range(model_params["TEST_EPOCHS"]):
        predictions, actuals = generate_text_on_test_set(epoch, tokenizer, model, device, test_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    wandb.finish()

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

def main():
    wandb.init(project="coreference_resolution_project")

    config = wandb.config

    # let's define model parameters specific to BART
    model_params = {
        "MODEL": "facebook/bart-base",
        "TRAIN_BATCH_SIZE": 4,  # training batch size
        "DEV_BATCH_SIZE": 16,  # dev batch size
        "TEST_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "TEST_EPOCHS": 1,  # number of testing epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
        "SEED": 42,  # set seed for reproducibility
    }

    for k, v in model_params.items():
        setattr(config, k, v)

    # let's get a news summary dataset
    # dataframe has 2 columns:
    #   - text: long article content
    #   - headlines: one line summary of news
    train_path = 'train_with_paragraphs.parq'
    dev_path = 'dev_with_paragraphs.parq'
    test_path = 'test_with_paragraphs.parq'

    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)
    test_df = pd.read_parquet(test_path)

    train_df = train_df.apply(lambda s: s.str.lstrip('coref: '))
    dev_df = dev_df.apply(lambda s: s.str.lstrip('coref: '))
    test_df = test_df.apply(lambda s: s.str.lstrip('coref: '))

    assert not train_df.iloc[0]["input"].startswith('coref: ')
    assert not dev_df.iloc[0]["input"].startswith('coref: ')
    assert not test_df.iloc[0]["input"].startswith('coref: ')

    BARTTrainer(
        train_dataframe=train_df,
        dev_dataframe=dev_df,
        test_dataframe=test_df,
        source_text="input",
        target_text="output",
        model_params=model_params,
        output_dir="outputs",
    )

if __name__ == '__main__':
    main()
