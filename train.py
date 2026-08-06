from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import numpy as np
import torch
import sys
from model import GNNModel, MyModelForSequenceClassification
import random

from torch_geometric.data import Batch

import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    EarlyStoppingCallback,
    pipeline,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
from torch.utils.data import default_collate

from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from tqdm import tqdm
import math
import time

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.tensorboard import SummaryWriter

from datamodule import ProvenanceGraphDataModule as Prov
from ggnn import FlowGNNGGNNModule

from torch.utils.data import DataLoader

from baseline import run_classical_baselines

logger = logging.getLogger(__name__)


class HybridGraphCollator:
    def __call__(self, batch):
        # Extract graphs and other tensors
        batch_size = len(batch)
        indices = torch.arange(batch_size, dtype=torch.long)

        graphs = [item.pop("graph") for item in batch]
        batched_graph = Batch.from_data_list(graphs)  # Batch PyG graphs

        # Remove graphs from batch for default tensor collation
        non_graph_items = batch
        collated_tensors = default_collate(non_graph_items)
        collated_tensors["indices"] = indices
        collated_tensors["graphs"] = batched_graph

        if "func" in collated_tensors and collated_tensors["func"] != []:
            func = collated_tensors.pop("func")
            collated_tensors["input_ids"] = func["input_ids"].squeeze(1)
            collated_tensors["attention_mask"] = func["attention_mask"].squeeze(1)

        return collated_tensors


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f"{key}", value, completed_steps)


def process_checkpoint_dir(args, file_name):
    file_name = str(file_name) + ".bin"
    checkpoint_prefix = f"{args.model_type}-{args.model_name}"

    output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, "{}".format(file_name))
    return output_dir

def get_labels(graphs):
    labels = torch.stack([g._VULN.max() for g in graphs]).to(torch.long)
    return F.one_hot(labels, num_classes=2)

    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    labels = labels.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("\n".join(sys.argv))
    
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument(
        "--data_folder",
        default=None,
        type=str,
        required=True,
        help="The input data folder.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--tb_dir",
        default="./tensorboard/",
        type=str,
        required=False,
        help="Tensorboard path.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--model_name", default="model.bin", type=str, help="Saved model name."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        default=None,
        type=str,
        help="Path to load checkpoint for training.",
    )
    parser.add_argument(
        "--eval_first",
        action="store_true",
        help="Whether or not to eval before any training",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--eval_export", action="store_true", help="Whether to save prediction output."
    )
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for all other than dataset"
    )
    parser.add_argument(
        "--dataset_seed", type=int, default=42, help="random seed for dataset"
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    # num of attention heads
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=16,
        help="number of attention heads",
    )
    # raw predictions
    parser.add_argument(
        "--write_raw_preds",
        default=False,
        action="store_true",
        help="Whether to write raw predictions on test data.",
    )
    # word-level tokenizer
    parser.add_argument(
        "--use_word_level_tokenizer",
        default=False,
        action="store_true",
        help="Whether to use word-level tokenizer.",
    )
    # bpe non-pretrained tokenizer
    parser.add_argument(
        "--use_non_pretrained_tokenizer",
        default=False,
        action="store_true",
        help="Whether to use non-pretrained bpe tokenizer.",
    )
    parser.add_argument("--profile", action="store_true", help="profile MACs")
    parser.add_argument("--time", action="store_true", help="measure inference time")
    
    # dataloader
    parser.add_argument("--oversample", default=None, type=str, help="Oversample data")
    parser.add_argument("--undersample", default=None, type=str, help="Undersample data")
    parser.add_argument("--preprompt", default="", type=str, help="Preprompt file")
    
    parser.add_argument("--no_gnn", action="store_true", help="No flowgnn")
    parser.add_argument("--no_llm", action="store_true", help="No LLM")
    parser.add_argument("--dropout", default=None, type=float, help="Dropout")
    
    parser.add_argument("--feature_file_name", default=None, type=str, help="Feature file name")
    parser.add_argument("--num_output_layers", default=3, type=int, help="Number of output layers")
    
    parser.add_argument("--load_gnn_from", default=None, type=str, help="Load GNN from given checkpoint")
    parser.add_argument("--freeze_gnn", default=False, action="store_true", help="Freeze GNN parameters")
    
    parser.add_argument("--optimize_llm", default="full", choices=["none", "attn", "full", "no_quant"], help="LLM optimization level") # none: do not optimize anything; attn, full: use LoRA; no_quant: do not quantize
    parser.add_argument("--code_feature", default="per_node", choices=["per_node", "per_package"], help="Code feature")
    
    parser.add_argument("--max_save_checkpoints", default=2, type=int, help="Max number of checkpoints to save")
    parser.add_argument("--log_steps", default=50, type=int, help="Log steps")
    parser.add_argument("--eval_steps", default=150, type=int, help="Eval steps")
    
    parser.add_argument("--use_chat_template", default=False, action="store_true", help="Use chat template")
    parser.add_argument("--chat_template_no_labels", default=False, action="store_true", help="Chat template no labels")
    parser.add_argument("--use_causal_lm", default=False, action="store_true", help="Use causal LM")
    parser.add_argument("--no_tokenization_in_dataloader", default=False, action="store_true", help="No tokenization in dataloader")
    parser.add_argument("--generate_batch_size", default=32, type=int, help="Generate batch size")
    parser.add_argument("--profile_generate_no_batch", default=False, action="store_true", help="Profile generate no batch")
    parser.add_argument("--use_sample_to_generate", default=False, action="store_true", help="Use sample to generate")
    parser.add_argument("--generate_look_at_last_n_tokens", default=None, type=int, help="when testing the generation, only look at the last n tokens")
    parser.add_argument("--lora_rank", default=16, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha")
    
    parser.add_argument("--delete_checkpoint", default=False, action="store_true", help="Delete checkpoint after training and testing")
    parser.add_argument("--dataloader_return_only_func", default=False, action="store_true", help="Dataloader return only func")
    parser.add_argument("--output_final_embeddings", default=False, action="store_true", help="Output final embeddings for all data")
    parser.add_argument("--run_baseline_on_final_embeddings", default=False, action="store_true", help="Run baseline on final embeddings")
    parser.add_argument("--wandb", default=False, action="store_true", help="Enable Weights & Biases logging. Disabled by default so the pipeline runs self-contained (no network/account).")

    args = parser.parse_args()

    # Weights & Biases is opt-in. By default disable it entirely so runs are
    # self-contained (no network calls, no account) — required for artifact eval.
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    report_to = ["wandb"] if args.wandb else "none"

    def is_CoT():
        return any([x in args.model_name_or_path for x in ["DeepSeek"]])

    # Setup CUDA, GPU
    args.n_gpu = torch.cuda.device_count()
    
    # set seeds
    seed = args.seed
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # All GPUs
    if not args.use_sample_to_generate:
        torch.backends.cudnn.deterministic = True  # CuDNN
        torch.backends.cudnn.benchmark = False  # CuDNN
        torch.use_deterministic_algorithms(True)

    # read preprompt
    if args.preprompt:
        with open(args.preprompt, "r") as f:
            args.preprompt = f.read().strip()

    datamodule = Prov(
        data_path=args.data_folder,
        batch_size=args.batch_size,
        seed=args.dataset_seed,
        load_features=not args.no_llm,
        undersample=args.undersample,
        oversample=args.oversample,
        block_size=args.block_size,
        preprompt=args.preprompt,
        llm_model_name_or_path=args.model_name_or_path,
        train_workers=0,
        code_feature=args.code_feature,
        feature_file_name=args.feature_file_name,
        use_chat_template=args.use_chat_template,
        no_tokenization=args.no_tokenization_in_dataloader,
        chat_template_no_labels=args.chat_template_no_labels,
        return_only_func=args.dataloader_return_only_func,
    )
    logger.info("Dataset loaded at:\n%s", args.data_folder)
    
    # test dataset
    for sample in datamodule.train:
        print("Testing datamodule: ", sample)
        break
    
        
    
    if not args.no_llm:

        # Load LLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.padding_side = "left" if args.use_causal_lm else "right"


        logger.info(f"Loading pretrained model from {args.model_name_or_path} ")
        
        huggingFaceAutoModel = AutoModelForCausalLM if args.use_causal_lm else AutoModel
        
        if args.optimize_llm == "no_quant":
            
            llm_model = huggingFaceAutoModel.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
            )
            
        elif args.optimize_llm == "none":
            # No quantization, but freeze all layers (for linear probing)
            llm_model = huggingFaceAutoModel.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
            )
            for param in llm_model.parameters():
                param.requires_grad = False
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            llm_model = huggingFaceAutoModel.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
            )

        # Set manual config options
        llm_model.config.num_labels = 2
        # llm_model.config.num_attention_heads = args.num_attention_heads
        llm_model.config.pad_token_id = llm_model.config.eos_token_id
        
        if args.optimize_llm != "none" and args.optimize_llm != "no_quant":
            # peft
            llm_model = prepare_model_for_kbit_training(llm_model)
            
            peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ] if args.optimize_llm == "full" else 
                ["q_proj", 
                 "k_proj", 
                 "v_proj", 
                 "o_proj"
                 ],
                lora_dropout=0.05,
                bias="none",
            )
            
            llm_model = get_peft_model(llm_model, peft_config)

        args.tokenizer = tokenizer

    def model_init():   
        if not args.no_gnn:
            input_dim = datamodule.input_dim()
            hidden_dim = 32
            n_steps = 5
            flowgnn_model = FlowGNNGGNNModule(
                input_dim,
                hidden_dim,
                n_steps,
                args.num_output_layers,
                encoder_mode=True,
            )
            logger.info("FlowGNN output dim: %d", flowgnn_model.out_dim)
        else:
            flowgnn_model = None
            
        model = MyModelForSequenceClassification(
            llm_model=llm_model if not args.no_llm else None,
            gnn_model_t=GNNModel,
            gnn_encoder=flowgnn_model,
            global_args=args,
            class_weights=datamodule.get_train_class_num(),
        )
        
        if not args.no_llm:
            model.hf_device_map = llm_model.hf_device_map 
            
        if args.load_gnn_from:
            model.load_gnn_model()
            
        if args.freeze_gnn:
            model.freeze_gnn()
            
        return model
        
    # Get current timestamp and process ID
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pid = os.getpid()  # Get current process ID
    output_dir = f"./{args.output_dir}/{timestamp}_PID{pid}"
    os.makedirs(output_dir, exist_ok=True)

    if not args.use_causal_lm:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir, 
            eval_strategy="steps",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            save_strategy="best",
            save_total_limit=args.max_save_checkpoints,
            load_best_model_at_end=True,             # Reload the best model at the end of training
            metric_for_best_model="eval_f1",       # Use evaluation loss to determine the best model
            logging_steps=args.log_steps,
            eval_steps=args.eval_steps,
            learning_rate=args.learning_rate,
            # fp16=True, 
            bf16=True if args.n_gpu >= 1 else False,
            logging_dir=args.tb_dir,
            max_grad_norm=args.max_grad_norm,
            save_only_model=True,
            weight_decay=args.weight_decay,
            remove_unused_columns=False,
            label_names=["labels"],
            save_safetensors=False,
            run_name=args.model_name,
            seed=args.seed,
            report_to=report_to,
        )
        trainerModule = Trainer
        trainer = trainerModule(
            # model=model,
            model_init=model_init,
            args=training_args,
            train_dataset=datamodule.train,
            eval_dataset=datamodule.val,
            data_collator=HybridGraphCollator(),
            compute_metrics=compute_metrics,  # Add the metrics function here
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        )

        # Train the model
        trainer.train()
        print("Training finished!")
        
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            save_strategy="no",
            logging_steps=args.log_steps,
            eval_steps=args.eval_steps,
            learning_rate=args.learning_rate,
            bf16=True if args.n_gpu >= 1 else False,
            logging_dir=args.tb_dir,
            save_only_model=True,
            weight_decay=args.weight_decay,
            remove_unused_columns=False,
            run_name=args.model_name,
            report_to=report_to,
        )

        trainer = Trainer(
            model = llm_model,
            args=training_args,
            train_dataset=datamodule.train,
            eval_dataset=datamodule.val,
            data_collator=DataCollatorWithPadding(tokenizer),
            tokenizer=tokenizer,
        )
        
        trainer.train()
    
    print("Proceeding to test dataset...")
    
    if not args.use_causal_lm:
        # Evaluate on the test dataset
        test_metrics = trainer.evaluate(eval_dataset=datamodule.test)
        # Print metrics
        print("Test Metrics:", test_metrics)
        
        predictions = trainer.predict(test_dataset=datamodule.test)
        print("Predicted labels:", predictions)
        
        # pass all data to the model to get final embeddings
        if args.output_final_embeddings:
            # Prepare DataLoaders using your collator
            collator = HybridGraphCollator()

            train_loader = DataLoader(
                datamodule.train,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collator
            )

            test_loader = DataLoader(
                datamodule.test,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collator
            )

            # Helper function to extract embeddings and labels
            def extract_embeddings_and_labels(dataloader, desc):
                all_embeddings = []
                all_labels = []
                for batch in tqdm(dataloader, desc=desc):
                    # Move tensors to device
                    batch = {
                        k: v.to(trainer.args.device)
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    with torch.no_grad():
                        embeddings = trainer.model(**batch, return_embed=True)

                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(batch["labels"].cpu())  # Assumes "labels" always exists

                return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

            # Process train and test
            final_embeddings_train, labels_train = extract_embeddings_and_labels(train_loader, "Processing train data for final embeddings")
            final_embeddings_test, labels_test = extract_embeddings_and_labels(test_loader, "Processing test data for final embeddings")

            # Save everything
            embedding_saving_dir = f"./{args.output_dir}/{timestamp}_PID{pid}_final_embeddings"
            os.makedirs(embedding_saving_dir, exist_ok=True)

            torch.save(final_embeddings_train, os.path.join(embedding_saving_dir, "train_embeddings.pt"))
            torch.save(labels_train, os.path.join(embedding_saving_dir, "train_labels.pt"))

            torch.save(final_embeddings_test, os.path.join(embedding_saving_dir, "test_embeddings.pt"))
            torch.save(labels_test, os.path.join(embedding_saving_dir, "test_labels.pt"))

            print(f"Final embeddings and labels saved to {embedding_saving_dir}")
            
        if args.run_baseline_on_final_embeddings:
            if not args.output_final_embeddings:
                raise ValueError("You must set --output_final_embeddings to True to run baseline on final embeddings.")
            
            print(f"Running classical baselines on final embeddings...")
            print(f"Train embeddings shape: {final_embeddings_train.shape}, Train labels shape: {labels_train.shape}")
            print(f"Test embeddings shape: {final_embeddings_test.shape}, Test labels shape: {labels_test.shape}")
            baseline_results = run_classical_baselines(final_embeddings_train, labels_train, final_embeddings_test, labels_test)
            print("Baseline results:", baseline_results)

    else:
        llm_model.eval()
        predictions = []
        true_labels = []

        # Create your pipeline
        pl = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        pl.tokenizer.pad_token_id = pl.tokenizer.eos_token_id

        test_inputs = [x["func"] for x in datamodule.test]
        batch_size = args.generate_batch_size
        num_batches = math.ceil(len(test_inputs) / batch_size)
        outputs = []
        
        # Start timing
        start_time = time.time()

        for i in tqdm(range(num_batches), desc="Processing batches"):
            batch_inputs = test_inputs[i * batch_size:(i + 1) * batch_size]
            batch_outputs = pl(
                batch_inputs,
                max_new_tokens=2048,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=args.use_sample_to_generate,
                batch_size=batch_size,
            )
            outputs.extend(batch_outputs)
            
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total processing time (batched): {elapsed_time:.2f} seconds")
        
        if args.profile_generate_no_batch:
            start_time = time.time()
            for input in tqdm(test_inputs, desc="Processing unbatched"):
                pl(
                        input,
                        max_length=4096,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total processing time (unbatched): {elapsed_time:.2f} seconds")
            
        true_labels = []
        predictions = []
        invalid_count = 0
        
        for i, output in enumerate(outputs):
            
            if is_CoT():
                output = output[0]["generated_text"].split("</think>")[-1] if "</think>" in output[0]["generated_text"] else ""
            elif args.generate_look_at_last_n_tokens is not None:
                output = output[0]["generated_text"][-args.generate_look_at_last_n_tokens:]
            else:
                output = output[0]["generated_text"][len(test_inputs[i]):]

            true_label = datamodule.test[i]["labels"].argmax().item()
            true_labels.append(true_label)

            normalized_output = output.strip().lower()
            print(normalized_output)
            if "Yes" in normalized_output or "yes" in normalized_output:
                pred = 1
            elif "No" in normalized_output or "no" in normalized_output:
                pred = 0
            else:
                print(f"[WARN] Invalid prediction: '{output.strip()}' — treated as incorrect")
                pred = 1 - true_label  # force wrong prediction
                invalid_count += 1

            predictions.append(pred)

        print("Predictions:", predictions, ", True Labels:", true_labels)
        print(f"Invalid responses (neither Yes nor No): {invalid_count}")

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
        acc = accuracy_score(true_labels, predictions)

        print("Test Metrics:", {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        })
        
    if args.delete_checkpoint:
        # Delete the checkpoint directory
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            print(f"Deleted checkpoint directory: {output_dir}")
        else:
            print(f"Checkpoint directory does not exist: {output_dir}")
            

if __name__ == "__main__":
    main()