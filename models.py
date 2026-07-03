
import torch
import torch.nn as nn

#LSTM model
class BitcoinLSTM(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=1,output_size=3,dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
            )
        self.fc1=nn.Linear(hidden_size,32)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.fc2=nn.Linear(32,output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )
        out, _ =self.lstm(x, (h0, c0))

        out=out[:, -1, :]
        out=self.fc1(out)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.fc2(out)
        return out
    
#CNN Model
class BitcoinCNN(nn.Module):

    def __init__(self,input_channels=1,output_size=3):
        super().__init__()
        self.conv1=nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3
        )
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1792, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.permute(0,2,1)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


#RNN Model
class BitcoinRNN(nn.Module):

    def __init__(self,input_size=1,hidden_size=64,num_layers=1,output_size=3,dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        out,_ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    
#Transformer Model
class BitcoinTransformer(nn.Module):

    def __init__(self,input_size=1,d_model=64,nhead=4,num_layers=2,dim_feedforward=128,dropout=0.2,
                 output_size=3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,
            dropout=dropout,batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x