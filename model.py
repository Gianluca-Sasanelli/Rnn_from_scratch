# model
import torch, math
import torch.nn as nn
import torch.nn.functional as F
device = "cuda"

class RNNModel(nn.Module):
    """Easy version of RNN. The model takes the input x (batch, sequence), convert into embeddings than use a RNN or LSTM 
    and then converts into logits. It also can compute the loss if the labels/targets are provided. The LSTM or RNN are called by pytorch"""

    def __init__(self, vocab_size, hidden_dim, n_layers, dropout=0.5, model = "GRU"):
        super().__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.n_layers= n_layers
        self.model = model
        #Embedding layer 
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        if model == "GRU":
            self.rnn = getattr(nn, "GRU")(hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first = True)
        if model == "LSTM":
            self.rnn = getattr(nn, "LSTM")(hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first = True)
        #Decoder layer 
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        #Linking the weights as in the transformer
        self.encoder.weight = self.decoder.weight

        self.init_weights()
        
    def init_weights(self):
        #Init weights with uniform distribution (as per Pytorch)
        nn.init.uniform_(self.encoder.weight, 0, 0.02)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, 0, 0.02)

    def forward(self, input, hidden_states= None, targets = None):
        #taking hidden as (h,c) if LSTM else h
        B,C = input.size()
        if hidden_states is None:
            hidden_states = self.init_hidden(B)
        #embeddings of the inputs
        emb = self.drop(self.encoder(input))
        output, hidden_states = self.rnn(emb, hidden_states)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(B, C, -1)
        if targets is not None:
            decoded = decoded.view(-1, self.vocab_size)
            loss = F.cross_entropy(decoded, targets.view(-1))
        else:
            loss = None
        return decoded, hidden_states, loss

    def init_hidden(self, batch_size):
        # taking the device and data type of weights 
        weight = next(self.parameters())
        if self.model == "GRU":
            hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.model == "LSTM":
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.hidden_dim), weight.new_zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self, input: torch.Tensor= None, new_tokens: int = 64,  temperature: float = 1.0, decode= None):
        self.eval()
        text = []
        if input is None:
            input = torch.randint(self.vocab_size, (1,1), dtype= torch.long).to(device)
        hidden_states = self.init_hidden(1)
        for i in range(new_tokens):
            output, hidden_states,_ = self(input, hidden_states)
            logits = output[0, -1].squeeze().div(temperature).exp()
            logits_idx = torch.multinomial(logits, 1).unsqueeze(0)
            input = torch.cat((input, logits_idx), dim = -1)
            word = decode[logits_idx.item()]
            text.append(word)
        print("".join(text))
              
        
class LSTMCell(nn.Module):
    """Simple LSTMCell made from scratch.  The idea of LSTM is to remove the problem of
    vanishing gradients experienced by RNNs. Here, there are two hidden states: h, the short-term memory
    and c, the long-term memory. 
    The new h is computed by the output gate (linear layer taking x and previous h as inputs) and the tanh of the current 
    long-term memory(c).
    The current long-term memory is computed with the forget gate multiplied by the previous c and
    by the input multiplied by the cell gate. So, some feature is added by the input gate and the forget gate removes some.
    For more information look at: https://arxiv.org/pdf/1402.1128.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        #LSTM input weights. Bias is needed only once
        self.wx = nn.Linear(hidden_dim, hidden_dim * 4)
        self.wh = nn.Linear(hidden_dim, hidden_dim * 4, bias = False)
        
#initializing the hidden vectors h,c.
    def init_hidden(self, x, batch_size):
        # taking the device and data type of x
        h = [x.new_zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        c = [x.new_zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        return (h,c)
    

    def forward(self, x, h, c):
    #The input shape is expected to be (B, hidden_dim)
    
        out_gates = self.wx(x) + self.wh(h)
        input_gate, forget_gate, cell_gate, output_gate = out_gates.chunk(4, dim = -1)
        
        input_t = self.sigm(input_gate)
        forget_t = self.sigm(forget_gate)
        cell_t = self.tanh(cell_gate)
        out_t = self.sigm(output_gate)
        
        next_c = forget_t * c + input_t * cell_t
        next_h = out_t * self.tanh(c)
        #basic math of the LSTM
        return next_h , next_c

class StackedLSTM_scratch(nn.Module):
    """Same of the class RNN model just made from scratch """
    def __init__(self, vocab_size, hidden_dim, n_layers, dropout: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        #stacking the LSTMs
        self.LSTMlayers = nn.ModuleList(
            [LSTMCell(hidden_dim) for _ in range(self.n_layers)])
        self.dec = nn.Linear(hidden_dim, vocab_size, bias = False)
    # more efficient with weight tying
        self.emb.weight = self.dec.weight
        self.drop = nn.Dropout(dropout)
    
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(self.hidden_dim))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std= 1/ math.sqrt(self.hidden_dim))
                   
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_hidden(self, x, batch_size):
        # taking the device and data type of weigths 
        h = [x.new_zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        c = [x.new_zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        return (h,c)
    
    def forward(self, x: torch.Tensor, hidden_states = None, targets: torch.Tensor = None):
        """Stacked LSTM

        Args:
            x (torch.Tensor): Input to the cell. The shape should be (batch_size, sentence_length)
            hidden_states (tuple ): Hidden and cell state. The shape should be (num_layers, batch_size, hidden_dimension)
            targets (torch.Tensor, optional): To calculate the loss. Defaults to None.

        Returns:
            prediction (torch.Tensor): Output in logits
            loss : loss of the model
            (h,c): hidden states 
        """
        # batch, context
        B, C = x.size()
        x = x.transpose(0,1) #C,B. Easier to index
        x = self.drop(self.emb(x))
        if hidden_states is None:
            h, c = self.init_hidden(x, B)
        else:
            h, c = hidden_states  # (num_layer, B, hidden_dim)
            # If h and c are not lists I get bugs in the backprop
            # Check if h is a tensor and convert to list if needed
            if isinstance(h, torch.Tensor):
                h = list(torch.unbind(h))
            
            # Check if c is a tensor and convert to list if needed
            if isinstance(c, torch.Tensor):
                c = list(torch.unbind(c))

        out = []
        for char in range(C):
            #input of the first LSTM cells are the embeddings of the tokens
            inp = x[char]
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.LSTMlayers[layer](inp, h[layer], c[layer] )   
                #the compute h is the input of the second LSTM
                inp = h[layer]
            out.append(h[-1])
        out = torch.stack(out).transpose(0,1)
        out = self.dec(out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, self.vocab_size), targets.view(-1))
        return out, (torch.stack(h), torch.stack(c)), loss 

    
    @torch.no_grad()
    def generate(self, input: torch.Tensor= None, new_tokens: int = 64, temperature: float = 1.0, decode= None):
        self.eval()
        text = []
        if input is None:
            input = torch.randint(self.vocab_size, (1,1), dtype= torch.long).to(device)
        hidden = None
        for i in range(new_tokens):
            output, hidden, _ = self(input, hidden)
            logits = output[0, -1].squeeze().div(temperature).exp()
            logits_idx = torch.multinomial(logits, 1).unsqueeze(0)
            # print(f"Toekns {i}| Input: {input.shape}| logits_idx: {logits_idx.shape}")
            input = torch.cat((input, logits_idx), dim = -1)
            word = decode[logits_idx.item()]
            text.append(word)
        print("".join(text))        
        
           
