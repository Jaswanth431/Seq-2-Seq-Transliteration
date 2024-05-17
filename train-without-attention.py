import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import copy
from torch.utils.data import Dataset, DataLoader
import gc
import random
import wandb
import argparse


# Set the device for training to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define special tokens for start, end, and padding
END_TOKEN = '>'
START_TOKEN = '<'
PAD_TOKEN = '_'

# Define the teacher forcing ratio, which determines the probability of using teacher forcing during training
TEACHER_FORCING_RATIO = 0.5


# Function to add padding to source sequences
def add_padding(source_data, MAX_LENGTH):
    padded_source_strings = []
    for i in range(len(source_data)):
        # Add start and end tokens to source sequence
        source_str = START_TOKEN + source_data[i] + END_TOKEN
        # Truncate or pad source sequence
        source_str = source_str[:MAX_LENGTH]
        source_str += PAD_TOKEN * (MAX_LENGTH - len(source_str))

        padded_source_strings.append(source_str)
        
    return padded_source_strings

# Function to convert source strings to sequences of indexes
def generate_string_to_sequence(source_data, source_char_index_dict):
    source_sequences = []
    for i in range(len(source_data)):
        # Convert characters to indexes using the provided dictionary
        source_sequences.append(get_chars(source_data[i], source_char_index_dict))
    # Pad sequences to the same length
    source_sequences = pad_sequence(source_sequences, batch_first=True, padding_value=2)
    return source_sequences

# Function to convert characters to their corresponding indexes
def get_chars(string, char_index_dict):
    chars_indexes = []
    for char in string:
        # Map characters to their corresponding indexes using the provided dictionary
        chars_indexes.append(char_index_dict[char])
    return torch.tensor(chars_indexes, device=device)

# Preprocess the data, including adding padding, generating sequences, and updating dictionaries
def preprocess_data(source_data, target_data):
    data = {
        "source_chars": [START_TOKEN, END_TOKEN, PAD_TOKEN],
        "target_chars": [START_TOKEN, END_TOKEN, PAD_TOKEN],
        "source_char_index": {START_TOKEN: 0, END_TOKEN: 1, PAD_TOKEN: 2},
        "source_index_char": {0: START_TOKEN, 1: END_TOKEN, 2: PAD_TOKEN},
        "target_char_index": {START_TOKEN: 0, END_TOKEN: 1, PAD_TOKEN: 2},
        "target_index_char": {0: START_TOKEN, 1: END_TOKEN, 2: PAD_TOKEN},
        "source_len": 3,
        "target_len": 3,
        "source_data": source_data,
        "target_data": target_data,
        "source_data_seq": [],
        "target_data_seq": []
    }
    
    # Calculate the maximum length of input and output sequences
    data["INPUT_MAX_LENGTH"] = max(len(string) for string in source_data) + 2
    data["OUTPUT_MAX_LENGTH"] = max(len(string) for string in target_data) + 2

    # Pad the source and target sequences and update character dictionaries
    padded_source_strings = add_padding(source_data, data["INPUT_MAX_LENGTH"])
    padded_target_strings = add_padding(target_data, data["OUTPUT_MAX_LENGTH"])
    
    for i in range(len(padded_source_strings)):
        for c in padded_source_strings[i]:
            if data["source_char_index"].get(c) is None:
                data["source_chars"].append(c)
                idx = len(data["source_chars"]) - 1
                data["source_char_index"][c] = idx
                data["source_index_char"][idx] = c
        for c in padded_target_strings[i]:
            if data["target_char_index"].get(c) is None:
                data["target_chars"].append(c)
                idx = len(data["target_chars"]) - 1
                data["target_char_index"][c] = idx
                data["target_index_char"][idx] = c

    # Generate sequences of indexes for source and target data
    data['source_data_seq'] = generate_string_to_sequence(padded_source_strings, data['source_char_index'])
    data['target_data_seq'] = generate_string_to_sequence(padded_target_strings, data['target_char_index'])
    

    # Update lengths of source and target character lists
    data["source_len"] = len(data["source_chars"])
    data["target_len"] = len(data["target_chars"])
    
    return data




# Function to get the appropriate cell type based on the input string
def get_cell_type(cell_type):
    if cell_type == "RNN":
        return nn.RNN
    elif cell_type == "LSTM":
        return nn.LSTM
    elif cell_type == "GRU":
        return nn.GRU
    else:
        print("Specify correct cell type")

# Encoder module for the seq2seq model
class Encoder(nn.Module):
    def __init__(self, h_params, data, device):
        super(Encoder, self).__init__()
        # Embedding layer for input data
        self.embedding = nn.Embedding(data["source_len"], h_params["char_embd_dim"])
        # Dropout layer for regularization
        self.dropout = nn.Dropout(h_params["dropout"])
        # Cell type chosen for encoding
        self.cell = get_cell_type(h_params["cell_type"])(h_params["char_embd_dim"], h_params["hidden_layer_neurons"],
                                                         num_layers=h_params["number_of_layers"], dropout=h_params["dropout"], batch_first=True)
        self.device = device
        self.h_params = h_params

    def forward(self, current_input, prev_state):
        # Embedding input
        embd_input = self.embedding(current_input)
        embd_input = self.dropout(embd_input)
        # Encoding step
        output, prev_state = self.cell(embd_input, prev_state)
        return output, prev_state

    # Initialize initial state of the encoder
    def getInitialState(self):
        return torch.zeros(self.h_params["number_of_layers"], self.h_params["batch_size"], self.h_params["hidden_layer_neurons"], device=self.device)

# Decoder module for the seq2seq model
class Decoder(nn.Module):
    def __init__(self, h_params, data, device):
        super(Decoder, self).__init__()
        # Embedding layer for target data
        self.embedding = nn.Embedding(data["target_len"], h_params["char_embd_dim"])
        # Dropout layer for regularization
        self.dropout = nn.Dropout(h_params["dropout"])
        # Cell type chosen for decoding
        self.cell = get_cell_type(h_params["cell_type"])(h_params["char_embd_dim"], h_params["hidden_layer_neurons"],
                                                         num_layers=h_params["number_of_layers"], dropout=h_params["dropout"], batch_first=True)
        # Fully connected layer for output
        self.fc = nn.Linear(h_params["hidden_layer_neurons"], data["target_len"])
        # Softmax layer for probability distribution
        self.softmax = nn.LogSoftmax(dim=2)
        self.h_params = h_params

    def forward(self, current_input, prev_state):
        # Embedding input
        embd_input = self.embedding(current_input)
        curr_embd = F.relu(embd_input)
        curr_embd = self.dropout(curr_embd)
        # Decoding step
        output, prev_state = self.cell(curr_embd, prev_state)
        # Apply softmax to output
        output = self.softmax(self.fc(output))
        return output, prev_state


# Custom dataset class for handling source and target sequences
class MyDataset(Dataset):
    def __init__(self, data):
        # Initialize with source and target sequences
        self.source_data_seq = data[0]
        self.target_data_seq = data[1]
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.source_data_seq)
    
    def __getitem__(self, idx):
        # Get source and target data at the specified index
        source_data = self.source_data_seq[idx]
        target_data = self.target_data_seq[idx]
        return source_data, target_data

# Function for inference on the trained model
def inference(encoder, decoder, source_sequence, target_tensor, data, device, h_params, loss_fn, batch_num):
    # Set encoder and decoder to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Initialize loss and correct predictions
    loss = 0
    correct = 0
    
    # Turn off gradient calculation
    with torch.no_grad():
        # Initialize encoder hidden state
        encoder_hidden = encoder.getInitialState()
        # For LSTM, initialize cell state as well
        if h_params["cell_type"] == "LSTM":
            encoder_hidden = (encoder_hidden, encoder.getInitialState())
        # Perform encoding
        encoder_outputs, encoder_hidden = encoder(source_sequence, encoder_hidden)

        # Initialize decoder input with start token
        decoder_input_tensor = torch.full((h_params["batch_size"], 1), data['target_char_index'][START_TOKEN], device=device)
        decoder_actual_output = []
        
        # Initialize decoder hidden state with encoder hidden state
        decoder_hidden = encoder_hidden
        
        # Iterate over each output time step
        for di in range(data["OUTPUT_MAX_LENGTH"]):
            curr_target_chars = target_tensor[:, di]
            # Perform decoding for one time step
            decoder_output, decoder_hidden = decoder(decoder_input_tensor, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input_tensor = topi.squeeze().detach()
            decoder_actual_output.append(decoder_input_tensor)
            decoder_input_tensor = decoder_input_tensor.view(h_params["batch_size"], 1)
                        
            decoder_output = decoder_output[:, -1, :]
            # Compute loss for the current time step
            loss += (loss_fn(decoder_output, curr_target_chars))

        # Concatenate decoder outputs
        decoder_actual_output = torch.cat(decoder_actual_output, dim=0).view(data["OUTPUT_MAX_LENGTH"], h_params["batch_size"]).transpose(0, 1)

        # Compute number of correct predictions
        correct = (decoder_actual_output == target_tensor).all(dim=1).sum().item()

        return correct, loss.item() / data["OUTPUT_MAX_LENGTH"]

# Function to evaluate the model on validation or test data
def evaluate(encoder, decoder, data, dataloader, device, h_params, loss_fn):
    correct_predictions = 0
    total_loss = 0
    total_predictions = len(dataloader.dataset)
    number_of_batches = len(dataloader)
    
    # Iterate over each batch
    for batch_num, (source_sequence, target_sequence) in enumerate(dataloader):
        input_tensor = source_sequence
        target_tensor = target_sequence
        
        # Perform inference on the batch
        correct, loss = inference(encoder, decoder, input_tensor, target_tensor, data, device, h_params, loss_fn, batch_num)
        
        correct_predictions += correct
        total_loss += loss
    
    # Compute accuracy and average loss
    accuracy = correct_predictions / total_predictions
    total_loss /= number_of_batches
    
    return accuracy, total_loss


# Function to create strings from indexes
def make_strings(data, source, target, output):
    source_string = ""
    target_string = ""
    output_string = ""
    # Convert source indexes to characters
    for i in source:
        source_string += (data['source_index_char'][i.item()])
    # Convert target indexes to characters
    for i in target:
        target_string += (data['target_index_char'][i.item()])
    # Convert output indexes to characters
    for i in output:
        output_string += (data['target_index_char'][i.item()])
    return source_string, target_string, output_string

# Training loop
def train_loop(encoder, decoder, h_params, data, data_loader, val_dataloader, device):
    # Choose optimizer based on the specified type
    if h_params["optimizer"] == "adam":
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=h_params["learning_rate"])
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=h_params["learning_rate"])
    elif h_params["optimizer"] == "nadam":
        encoder_optimizer = optim.NAdam(encoder.parameters(), lr=h_params["learning_rate"])
        decoder_optimizer = optim.NAdam(decoder.parameters(), lr=h_params["learning_rate"])
    
    total_predictions = len(data_loader.dataset)
    total_batches = len(data_loader)
    loss_fn = nn.NLLLoss()
    
    # Loop through each epoch
    for ep in range(h_params["epochs"]):
        encoder.train()
        decoder.train()
        total_loss = 0
        total_correct = 0
        
        # Loop through each batch
        for batch_num, (source_batch, target_batch) in enumerate(data_loader):
            # Get initial state of the encoder
            encoder_initial_state = encoder.getInitialState()
            if h_params["cell_type"] == "LSTM":
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState())
            
            encoder_current_state = encoder_initial_state
            encoder_output, encoder_current_state = encoder(source_batch, encoder_current_state)
            loss = 0
            correct = 0
            decoder_curr_state = encoder_current_state
            output_seq_len = data["OUTPUT_MAX_LENGTH"]
            decoder_actual_output = []
            
            # Determine whether to use teacher forcing
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

            # Loop through each output time step
            for i in range(data["OUTPUT_MAX_LENGTH"]):
                if(i == 0):
                    decoder_input_tensor = target_batch[:, i].view(h_params["batch_size"], 1)
                curr_target_chars = target_batch[:, i]
                decoder_output, decoder_curr_state = decoder(decoder_input_tensor, decoder_curr_state)
                topv, topi = decoder_output.topk(1)
                decoder_input_tensor = topi.squeeze().detach()
                decoder_actual_output.append(decoder_input_tensor)
                if(i < output_seq_len - 1):
                    if use_teacher_forcing:
                        decoder_input_tensor = target_batch[:, i + 1].view(h_params["batch_size"], 1)
                    else:
                        decoder_input_tensor = decoder_input_tensor.view(h_params["batch_size"], 1)
                decoder_output = decoder_output[:, -1, :]
                loss += (loss_fn(decoder_output, curr_target_chars))

            decoder_actual_output = torch.cat(decoder_actual_output, dim=0).view(output_seq_len, h_params["batch_size"]).transpose(0, 1)
            
            correct = (decoder_actual_output == target_batch).all(dim=1).sum().item()
            total_correct += correct
            total_loss += loss.item() / output_seq_len
            
            # Backpropagation and optimization step
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        train_acc = total_correct / total_predictions
        train_loss = total_loss / total_batches
        
        # Evaluate on validation set
        val_acc, val_loss = evaluate(encoder, decoder, data, val_dataloader, device, h_params, loss_fn)
        
        # Log metrics
        print("ep: ", ep, " train acc:", train_acc, " train loss:", train_loss, " val acc:", val_acc, " val loss:", val_loss)
        wandb.log({"train_accuracy": train_acc, "train_loss": train_loss, "val_accuracy": val_acc, "val_loss": val_loss, "epoch": ep})

    return encoder, decoder, loss_fn


# Training function
def train(h_params, data, device, train_dataloader, val_dataloader):
    # Initialize encoder and decoder
    encoder = Encoder(h_params, data, device).to(device)
    decoder = Decoder(h_params, data, device).to(device)
    
    # Perform training loop
    encoder, decoder, loss_fn = train_loop(encoder, decoder, h_params, data, train_dataloader, val_dataloader, device)
    return encoder, decoder, loss_fn

h_params = {
    "char_embd_dim": 256,
    "hidden_layer_neurons": 256,
    "batch_size": 32,
    "number_of_layers": 3,
    "learning_rate": 0.0001,
    "epochs": 20,
    "cell_type": "LSTM",
    "dropout": 0,
    "optimizer": "adam"
}

# Function to prepare dataloaders for training and validation
def prepare_dataloaders(train_source, train_target, val_source, val_target, h_params):
    # Preprocess training data
    data = preprocess_data(copy.copy(train_source), copy.copy(train_target))
    training_data = [data["source_data_seq"], data['target_data_seq']]
    # Create training dataset and dataloader
    train_dataset = MyDataset(training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=h_params["batch_size"], shuffle=True)

    # Preprocess validation data
    val_padded_source_strings = add_padding(val_source, data["INPUT_MAX_LENGTH"])
    val_padded_target_strings = add_padding(val_target, data["OUTPUT_MAX_LENGTH"])
    val_source_sequences = generate_string_to_sequence(val_padded_source_strings, data['source_char_index'])
    val_target_sequences = generate_string_to_sequence(val_padded_target_strings, data['target_char_index'])
    validation_data = [val_source_sequences, val_target_sequences]
    # Create validation dataset and dataloader
    val_dataset = MyDataset(validation_data)
    val_dataloader = DataLoader(val_dataset, batch_size=h_params["batch_size"], shuffle=True)
    return train_dataloader, val_dataloader, data

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a seq2seq model with specified hyperparameters")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL proj", help="Specifies the project name used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Sets the number of epochs to train the neural network")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Sets the learning rate used to optimize model parameters")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Specifies the batch size used for training")
    parser.add_argument("-embd_dim", "--char_embd_dim", type=int, default=128, help="Dimension of character embeddings")
    parser.add_argument("-hid_neur", "--hidden_layer_neurons", type=int, default=512, help="Number of neurons in hidden layers")
    parser.add_argument("-num_layers", "--number_of_layers", type=int, default=2, help="Number of layers in the encoder and decoder")
    parser.add_argument("-cell", "--cell_type", choices=["RNN", "LSTM", "GRU"], default="LSTM", help="Type of RNN cell: RNN, LSTM, GRU")
    parser.add_argument("-do", "--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("-opt", "--optimizer", choices=["adam", "nadam"], default="adam", help="Optimization algorithm: adam, nadam")
    parser.add_argument("-train_path", "--train_path", type=str, required=True, help="Specifies the path for the training data (mandatory)")
    parser.add_argument("-test_path", "--test_path", type=str, required=True, help="Specifies the path for the testing data (mandatory)")
    parser.add_argument("-val_path", "--val_path", type=str, required=True, help="Specifies the path for the validation data (mandatory)")

    args = parser.parse_args()

    return args


def main():
    wandb.login()
    args = parse_arguments()

    h_params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "char_embd_dim": args.char_embd_dim,
        "hidden_layer_neurons": args.hidden_layer_neurons,
        "number_of_layers": args.number_of_layers,
        "cell_type": args.cell_type,
        "dropout": args.dropout,
        "optimizer": args.optimizer
    }


    # Paths to the training, testing, and validation CSV files
    train_csv = args.train_path
    test_csv = args.test_path
    val_csv = args.val_path

    # Load the training data from the CSV file into a DataFrame
    train_df = pd.read_csv(train_csv, header=None)

    # Separate the source and target sequences from the training DataFrame
    train_source, train_target = train_df[0].to_numpy(), train_df[1].to_numpy()

    # Load the testing and validation data from the respective CSV files into DataFrames
    test_df = pd.read_csv(test_csv, header=None)
    val_df = pd.read_csv(val_csv, header=None)

    # Separate the source and target sequences from the validation DataFrame
    val_source, val_target = val_df[0].to_numpy(), val_df[1].to_numpy()

    config = h_params
    run = wandb.init(project=args.wandb_project, name=f"{config['cell_type']}_{config['optimizer']}_ep_{config['epochs']}_lr_{config['learning_rate']}_embd_{config['char_embd_dim']}_hid_lyr_neur_{config['hidden_layer_neurons']}_bs_{config['batch_size']}_enc_layers_{config['number_of_layers']}_dec_layers_{config['number_of_layers']}_dropout_{config['dropout']}", config=config)
    train_dataloader, val_dataloader, data = prepare_dataloaders(train_source, train_target, val_source, val_target, h_params)
    train(h_params, data, device, train_dataloader, val_dataloader)

main()

