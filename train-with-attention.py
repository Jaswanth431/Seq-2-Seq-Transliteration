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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Checking if CUDA is available, else use CPU
print(device)  # Printing the device being used (CUDA or CPU)
END_TOKEN = '>'  # Defining the end token for sequences
START_TOKEN = '<'  # Defining the start token for sequences
PAD_TOKEN = '_'  # Defining the padding token for sequences
TEACHER_FORCING_RATIO = 0.5  # Ratio of teacher forcing during training


# Function to add padding to source sequences
def add_padding(source_data, MAX_LENGTH):
    """
    Add padding to source sequences and truncate if necessary.
    
    Args:
    - source_data: List of source sequences
    - MAX_LENGTH: Maximum length of source sequences
    
    Returns:
    - padded_source_strings: List of padded source sequences
    """
    padded_source_strings = []
    for i in range(len(source_data)):
        source_str = START_TOKEN + source_data[i] + END_TOKEN  # Add start and end tokens
        source_str = source_str[:MAX_LENGTH]  # Truncate if longer than MAX_LENGTH
        source_str += PAD_TOKEN * (MAX_LENGTH - len(source_str))  # Pad with PAD_TOKEN

        padded_source_strings.append(source_str)
        
    return padded_source_strings


# Function to convert source strings to sequences of indices
def generate_string_to_sequence(source_data, source_char_index_dict):
    """
    Convert source strings to sequences of indices using char_index_dict.
    
    Args:
    - source_data: List of padded source strings
    - source_char_index_dict: Dictionary mapping characters to their indices
    
    Returns:
    - source_sequences: Padded sequence of character indices
    """
    source_sequences = []
    for i in range(len(source_data)):
        source_sequences.append(get_chars(source_data[i], source_char_index_dict))
    source_sequences = pad_sequence(source_sequences, batch_first=True, padding_value=2)
    return source_sequences


# Function to convert characters to their corresponding indices
def get_chars(string, char_index_dict):
    """
    Convert characters in a string to their corresponding indices using char_index_dict.
    
    Args:
    - string: Input string
    - char_index_dict: Dictionary mapping characters to their indices
    
    Returns:
    - chars_indexes: List of character indices
    """
    chars_indexes = []
    for char in string:
        chars_indexes.append(char_index_dict[char])
    return torch.tensor(chars_indexes, device=device)


# Preprocess the data, including adding padding, generating sequences, and updating dictionaries
def preprocess_data(source_data, target_data):
    """
    Preprocess source and target data.
    
    Args:
    - source_data: List of source strings
    - target_data: List of target strings
    
    Returns:
    - data: Preprocessed data dictionary
    """
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
        for char in padded_source_strings[i]:
            if data["source_char_index"].get(char) is None:
                data["source_chars"].append(char)
                idx = len(data["source_chars"]) - 1
                data["source_char_index"][char] = idx
                data["source_index_char"][idx] = char
        for char in padded_target_strings[i]:
            if data["target_char_index"].get(char) is None:
                data["target_chars"].append(char)
                idx = len(data["target_chars"]) - 1
                data["target_char_index"][char] = idx
                data["target_index_char"][idx] = char

    # Generate sequences of indexes for source and target data
    data['source_data_seq'] = generate_string_to_sequence(padded_source_strings, data['source_char_index'])
    data['target_data_seq'] = generate_string_to_sequence(padded_target_strings, data['target_char_index'])
    
    # Update lengths of source and target character lists
    data["source_len"] = len(data["source_chars"])
    data["target_len"] = len(data["target_chars"])
    
    return data


def get_cell_type(cell_type):
    # Function to return the appropriate RNN cell based on the specified type
    if(cell_type == "RNN"):
        return nn.RNN
    elif(cell_type == "LSTM"):
        return nn.LSTM
    elif(cell_type == "GRU"):
        return nn.GRU
    else:
        print("Specify correct cell type")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        # Initialize the attention mechanism module
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # Forward pass of the attention mechanism
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze().unsqueeze(1)
        weights = F.softmax(scores, dim=0)
        weights = weights.permute(2,1,0)
        keys = keys.permute(1,0,2)
        context = torch.bmm(weights, keys)
        return context, weights

class Encoder(nn.Module):
    def __init__(self, h_params, data, device ):
        # Initialize the Encoder module
        super(Encoder, self).__init__()
        # Embedding layer for input characters
        self.embedding = nn.Embedding(data["source_len"], h_params["char_embd_dim"])
        # RNN cell for encoding
        self.cell = get_cell_type(h_params["cell_type"])(h_params["char_embd_dim"], h_params["hidden_layer_neurons"],num_layers=h_params["number_of_layers"], batch_first=True)
        self.device=device
        self.h_params = h_params
        self.data = data
        
    def forward(self, input , encoder_curr_state):
        # Forward pass of the Encoder module
        input_length = self.data["INPUT_MAX_LENGTH"]
        batch_size = self.h_params["batch_size"]
        hidden_neurons = self.h_params["hidden_layer_neurons"]
        layers = self.h_params["number_of_layers"]
        encoder_states  = torch.zeros(input_length, layers, batch_size, hidden_neurons, device=self.device )
        for i in range(input_length):
            current_input = input[:, i].view(batch_size,1)
            _, encoder_curr_state = self.forward_step(current_input, encoder_curr_state)
            if self.h_params["cell_type"] == "LSTM":
                encoder_states[i] = encoder_curr_state[1]
            else:
                encoder_states[i] = encoder_curr_state
        return encoder_states, encoder_curr_state
    
    def forward_step(self, current_input, prev_state):
        # Perform forward pass for one time step
        embd_input = self.embedding(current_input)
        output, prev_state = self.cell(embd_input, prev_state)
        return output, prev_state
        
    def getInitialState(self):
        # Initialize initial hidden state for encoder
        return torch.zeros(self.h_params["number_of_layers"],self.h_params["batch_size"],self.h_params["hidden_layer_neurons"], device=self.device)

class Decoder(nn.Module):
    def __init__(self, h_params, data,device):
        # Initialize the Decoder module
        super(Decoder, self).__init__()
        # Attention mechanism
        self.attention = Attention(h_params["hidden_layer_neurons"]).to(device)
        # Embedding layer for target characters
        self.embedding = nn.Embedding(data["target_len"], h_params["char_embd_dim"])
        # RNN cell for decoding
        self.cell = get_cell_type(h_params["cell_type"])(h_params["hidden_layer_neurons"] +h_params["char_embd_dim"], h_params["hidden_layer_neurons"],num_layers=h_params["number_of_layers"], batch_first=True)
        # Fully connected layer for output
        self.fc = nn.Linear(h_params["hidden_layer_neurons"], data["target_len"])
        # Softmax activation for output probabilities
        self.softmax = nn.LogSoftmax(dim=2)
        self.h_params = h_params
        self.data = data
        self.device = device

    def forward(self, decoder_current_state, encoder_final_layers, target_batch, loss_fn, teacher_forcing_enabled=True):
        # Forward pass of the Decoder module
        batch_size = self.h_params["batch_size"]
        decoder_current_input = torch.full((batch_size,1),self.data["target_char_index"][START_TOKEN], device=self.device)
        embd_input = self.embedding(decoder_current_input)
        curr_embd = F.relu(embd_input)
        decoder_actual_output = []
        attentions = []
        loss = 0
        
        use_teacher_forcing = False
        if(teacher_forcing_enabled):
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
        for i in range(self.data["OUTPUT_MAX_LENGTH"]):
            # Perform one step of decoding
            decoder_output, decoder_current_state, attn_weights = self.forward_step(decoder_current_input, decoder_current_state, encoder_final_layers)
            attentions.append(attn_weights)
            topv, topi = decoder_output.topk(1)
            decoder_current_input = topi.squeeze().detach()
            decoder_actual_output.append(decoder_current_input)

            if(target_batch==None):
                decoder_current_input = decoder_current_input.view(self.h_params["batch_size"], 1)
            else:
                curr_target_chars = target_batch[:, i]
                if(i<self.data["OUTPUT_MAX_LENGTH"]-1):
                    if use_teacher_forcing:
                        decoder_current_input = target_batch[:, i+1].view(self.h_params["batch_size"], 1)
                    else:
                        decoder_current_input = decoder_current_input.view(self.h_params["batch_size"], 1)
                decoder_output = decoder_output[:, -1, :]
                loss+=(loss_fn(decoder_output, curr_target_chars))

        decoder_actual_output = torch.cat(decoder_actual_output,dim=0).view(self.data["OUTPUT_MAX_LENGTH"], self.h_params["batch_size"]).transpose(0,1)

        correct = (decoder_actual_output == target_batch).all(dim=1).sum().item()
        return decoder_actual_output, attentions, loss, correct
    
    def forward_step(self, current_input, prev_state, encoder_final_layers):
        # Perform one step of decoding
        embd_input = self.embedding(current_input)
        if self.h_params["cell_type"] == "LSTM":
            context , attn_weights = self.attention(prev_state[1][-1,:,:], encoder_final_layers)
        else:
            context , attn_weights = self.attention(prev_state[-1,:,:], encoder_final_layers)
        curr_embd = F.relu(embd_input)
        input_gru = torch.cat((curr_embd, context), dim=2)
        output, prev_state = self.cell(input_gru, prev_state)
        output = self.softmax(self.fc(output))
        return output, prev_state, attn_weights

class MyDataset(Dataset):
    def __init__(self, data):
        self.source_data_seq = data[0]
        self.target_data_seq = data[1]
    
    def __len__(self):
        return len(self.source_data_seq)
    
    def __getitem__(self, idx):
        source_data = self.source_data_seq[idx]
        target_data = self.target_data_seq[idx]
        return source_data, target_data


def evaluate(encoder, decoder, data, dataloader, device, h_params, loss_fn, use_teacher_forcing = False):
    # Function to evaluate the performance of the model on a dataset
    correct_predictions = 0
    total_loss = 0
    total_predictions = len(dataloader.dataset)
    number_of_batches = len(dataloader)
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for batch_num, (source_batch, target_batch) in enumerate(dataloader):

            encoder_initial_state = encoder.getInitialState()
            if h_params["cell_type"] == "LSTM":
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState())
            encoder_states, encoder_final_state = encoder(source_batch,encoder_initial_state)

            decoder_current_state = encoder_final_state
            encoder_final_layer_states = encoder_states[:, -1, :, :]

            loss = 0
            correct = 0

            decoder_output, attentions, loss, correct = decoder(decoder_current_state, encoder_final_layer_states, target_batch, loss_fn, use_teacher_forcing)

            correct_predictions+=correct
            total_loss +=loss

        accuracy = correct_predictions / total_predictions
        total_loss /= number_of_batches

        return accuracy, total_loss


def make_strings(data, source, target, output):
    # Function to convert indices to strings for source, target, and output sequences
    source_string = ""
    target_string = ""
    output_string = ""
    for i in source:
        source_string+=(data['source_index_char'][i.item()])
    for i in target:
        target_string+=(data['target_index_char'][i.item()])
    for i in output:
        output_string+=(data['target_index_char'][i.item()])
    return source_string, target_string, output_string


def train_loop(encoder, decoder,h_params, data, data_loader, device, val_dataloader, use_teacher_forcing=True):
    # Function to train the encoder-decoder model
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=h_params["learning_rate"])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=h_params["learning_rate"])
    
    loss_fn = nn.NLLLoss()
    
    total_predictions = len(data_loader.dataset)
    total_batches = len(data_loader)
    
    for ep in range(h_params["epochs"]):
        total_correct = 0
        total_loss = 0
        encoder.train()
        decoder.train()
        for batch_num, (source_batch, target_batch) in enumerate(data_loader):
            encoder_initial_state = encoder.getInitialState()
            
            if h_params["cell_type"] == "LSTM":
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState())
            encoder_states, encoder_final_state = encoder(source_batch,encoder_initial_state)
            
            decoder_current_state = encoder_final_state
            encoder_final_layer_states = encoder_states[:, -1, :, :]
            
            
            loss = 0
            correct = 0
            
            decoder_output, attentions, loss, correct = decoder(decoder_current_state, encoder_final_layer_states, target_batch, loss_fn, use_teacher_forcing)
            total_correct +=correct
            total_loss += loss.item()/data["OUTPUT_MAX_LENGTH"]
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            
        train_acc = total_correct/total_predictions
        train_loss = total_loss/total_batches
        val_acc, val_loss = evaluate(encoder, decoder, data, val_dataloader,device, h_params, loss_fn, False)
        print("ep: ", ep, " train acc:", train_acc, " train loss:", train_loss, " val acc:", val_acc, " val loss:", val_loss.item()/data["OUTPUT_MAX_LENGTH"])
        wandb.log({"train_accuracy":train_acc, "train_loss":train_loss, "val_accuracy":val_acc, "val_loss":val_loss, "epoch":ep})


def prepare_dataloaders(train_source, train_target, val_source, val_target, h_params):
    # Preparing data loaders for training and validation
    data = preprocess_data(copy.copy(train_source), copy.copy(train_target))
    
    # Training data
    training_data = [data["source_data_seq"], data['target_data_seq']]
    train_dataset = MyDataset(training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=h_params["batch_size"], shuffle=True)

    # Validation data
    val_padded_source_strings = add_padding(val_source, data["INPUT_MAX_LENGTH"])
    val_padded_target_strings = add_padding(val_target, data["OUTPUT_MAX_LENGTH"])
    val_source_sequences = generate_string_to_sequence(val_padded_source_strings, data['source_char_index'])
    val_target_sequences = generate_string_to_sequence(val_padded_target_strings, data['target_char_index'])
    validation_data = [val_source_sequences, val_target_sequences]
    val_dataset = MyDataset(validation_data)
    val_dataloader = DataLoader(val_dataset, batch_size=h_params["batch_size"], shuffle=True)
    
    return train_dataloader, val_dataloader, data


def train(h_params, data, device, data_loader, val_dataloader, use_teacher_forcing=True):
    encoder = Encoder(h_params, data, device).to(device)
    decoder = Decoder(h_params, data, device).to(device)
    train_loop(encoder, decoder,h_params, data, data_loader,device, val_dataloader, use_teacher_forcing)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a seq2seq model with specified hyperparameters")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL proj", help="Specifies the project name used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Sets the number of epochs to train the neural network")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Sets the learning rate used to optimize model parameters")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Specifies the batch size used for training")
    parser.add_argument("-embd_dim", "--char_embd_dim", type=int, default=256, help="Dimension of character embeddings")
    parser.add_argument("-hid_neur", "--hidden_layer_neurons", type=int, default=256, help="Number of neurons in hidden layers")
    parser.add_argument("-num_layers", "--number_of_layers", type=int, default=3, help="Number of layers in the encoder and decoder")
    parser.add_argument("-cell", "--cell_type", choices=["RNN", "LSTM", "GRU"], default="LSTM", help="Type of RNN cell: RNN, LSTM, GRU")
    parser.add_argument("-do", "--dropout", type=float, default=0, help="Dropout probability")
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

