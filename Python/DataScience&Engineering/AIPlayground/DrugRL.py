import torch
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from torchrl.modules import ProbabilisticActor
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
from RewardFUtils import predict_binding_affinity, predict_protein_binding, predict_toxicity, predict_effectiveness, \
    predict_molecule_feasiblity

# Load and preprocess SMILES data
with open("../data/chembl_22_clean_1576904_sorted_std_final.smi", "r") as f:
    smiles_data = f.readlines()

smiles_data = [line.strip() for line in smiles_data if line.strip()]
dataset = Dataset.from_dict({"text": smiles_data})

# # Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token as EOS for compatibility

# Tokenize dataset and include labels
def preprocess_function(examples):
    # Tokenize the input and duplicate input_ids to labels
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Ensure padding for batch processing
        max_length=128,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels for causal language modeling
    return tokenized

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Convert to PyTorch format
tokenized_datasets = tokenized_datasets.with_format("torch")

#
# # Tokenize dataset
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
#
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.save_to_disk("data/tokenized_chembl")
print("tokenizer saved")

tokenized_datasets = dataset.load_from_disk("data/tokenized_chembl")
# Load pre-trained model and adjust embedding size
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Define training arguments and trainer for fine-tuning
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-chembl",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=500,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)
trainer.train()


# Define the custom actor for SMILES generation using the fine-tuned GPT-2 model
class GPT2Actor(nn.Module):
    def __init__(self, model):
        super(GPT2Actor, self).__init__()
        self.model = model

    def forward(self, sequence, sequence_mask):
        outputs = self.model(input_ids=sequence, attention_mask=sequence_mask).logits
        return outputs


# Instantiate actor with the fine-tuned model
actor_model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-chembl")
actor = ProbabilisticActor(
    module=GPT2Actor(actor_model),
    in_keys=["sequence", "sequence_mask"],
    out_keys=["action"],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True
)


# Define reward function
def reward_function(smiles):
    calc_reward = 0
    calc_reward += predict_molecule_feasiblity(smiles)
    # These rewards are passed due to simplicity.
    # calc_reward += predict_protein_binding(smiles, "")
    # calc_reward += predict_binding_affinity(smiles, "")
    calc_reward += predict_toxicity(smiles)
    calc_reward += predict_effectiveness(smiles)
    return calc_reward


# Decode action (tokenized sequence) to SMILES
def decode_smiles(sequence, tokenizer):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy().tolist()
    smiles_string = tokenizer.decode(sequence, skip_special_tokens=True)
    return smiles_string


# Define SMILES Environment
class SMILESEnvironment:
    def __init__(self, tokenizer, reward_function):
        self.tokenizer = tokenizer
        self.reward_function = reward_function

    def step(self, action):
        molecule = decode_smiles(action, self.tokenizer)
        reward = self.reward_function(molecule)
        return reward, molecule


# Function to initialize the starting sequence
def start_sequence(tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else \
        tokenizer.encode("<|startoftext|>")[0]
    start_sequence = torch.tensor([[start_token_id]], dtype=torch.long).to(device)
    return start_sequence


# Function to create an attention mask for the sequence, which adds nulls to complete the sequence
def get_mask(sequence):
    return (sequence != tokenizer.pad_token_id).long()


# Initialize the environment and optimizer
smiles_env = SMILESEnvironment(tokenizer, reward_function)
optimizer = optim.Adam(actor.parameters(), lr=1e-5)

# Define parameters for training loop
num_episodes = 1000
max_steps = 50

# Training loop for RL
for episode in range(num_episodes):
    sequence = start_sequence(tokenizer)  # Start with a seed sequence
    sequence_mask = get_mask(sequence)  # Create mask for sequence

    for t in range(max_steps):
        # Generate the next action (token) and log probability
        action, log_prob = actor(sequence, sequence_mask)

        # Environment step to get reward and decoded molecule
        reward, molecule = smiles_env.step(action.squeeze().tolist())
        print(f"generated molecule : {molecule}, and reward : {reward}")

        # Calculate loss (REINFORCE or PPO)
        loss = -log_prob * reward  # Negative sign to maximize reward

        # Optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append the action to the sequence and update the sequence mask
        sequence = torch.cat([sequence, action.unsqueeze(0)], dim=1)
        sequence_mask = get_mask(sequence)  # Update mask

    # Print intermediate results
    if episode % 100 == 0:
        print(f"Episode {episode} - Generated molecule: {molecule}, Reward: {reward}")

# Save the actor model (fine-tuned GPT-2 model)
actor.module.model.save_pretrained("./rl-agent-gpt2-finetuned")

# Save the tokenizer
tokenizer.save_pretrained("./rl-gpt2-finetuned")
# Step 1: Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-chembl")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-chembl")


# Step 2: Generate a SMILES string
def generate_smiles(model, tokenizer, max_length=128, device="cpu"):
    model.eval()
    model.to(device)

    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else \
        tokenizer.encode("<|startoftext|>")[0]
    input_ids = torch.tensor([[start_token_id]], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_smiles = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_smiles


new_smiles = generate_smiles(model, tokenizer)
print(f"Generated SMILES: {new_smiles}")


# Step 3: Evaluate the generated SMILES
def evaluate_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return "Invalid SMILES", 0.0

        qed_score = QED.qed(molecule)
        return "Valid SMILES", qed_score
    except Exception as e:
        print(f"Error evaluating SMILES {smiles}: {e}")
        return "Error", 0.0


validity, qed_score = evaluate_smiles(new_smiles)
print(f"Generated molecule is {validity} with QED score: {qed_score}")
""" // Iterating over the linked list
    while (curr) {
        if (curr->size >= aligned_size) {
            // If the block is larger than needed, split it
            if (curr->size >= aligned_size + sizeof(struct mblock)) {
				// Crating a new block to indicate new start of remaining block
                struct mblock *new_block = (struct mblock *)((char *)curr + sizeof(struct mblock) + aligned_size);// Calcualtes beginning adress of new block
                new_block->size = curr->size - aligned_size - sizeof(struct mblock); // Calculates remaining size of new block
                new_block->next = curr->next;

                // Update the current block size to place it
                curr->size = aligned_size;

                // Updating linked list
                if (temp) {
                    temp->next = new_block;
                } else {
                    head = new_block;
                }
            } else {
                // If the block is not large enough to split, just remove it from the list
                if (temp) {
                    temp->next = curr->next;
                } else {
                    head = curr->next;
                }
            }

            // Mark the block as occupied
            curr->next = MAGIC;

            // Return a pointer to the memory payload
            return curr->memory;
        }

        temp = curr;
        curr = curr->next;
    }"""
