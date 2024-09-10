import os, pickle, datetime, yaml, argparse
import numpy as np
from model import *
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, required = True, help = "config path")
args = parser.parse_args()
config_path = args.config
with open(config_path) as file:
    config = yaml.safe_load(file)

device = "cuda"
data_dir = r"C:\Users\Gianl\Desktop\RNN-repo\MyRNN\data"
dataset = "shakespeare"
logging_params = config["logging_parameters"]
meta = r"C:\Users\Gianl\Desktop\RNN-repo\MyRNN\data\meta.pkl"

with open(meta, "rb") as f:
    meta = pickle.load(f)
    decode = meta["itos"]
    encode = meta["stoi"]
    vocab_size = meta["vocab_size"]
# logging
init_from = logging_params["init_from"]
log = logging_params["log"]
iter_max = logging_params["iter_max"]
eval_iters = logging_params["eval_iters"]
eval_interval = logging_params["eval_interval"]
out_dir = logging_params["out_dir"]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
sampling_duringtraining = logging_params["sampling_duringtraining"]
# parameters
model_params = config["model_data_parameters"]
context = model_params["context"]
batch_size = model_params["batch_size"]
hidden_dim = model_params["hidden_dim"]
n_layers = model_params["n_layers"]
model = model_params["model"]

#training
training_params = config["training_parameters"]
max_lr = training_params["max_lr"]
gamma = training_params["gamma"]
#sampling
start = training_params["start"]
num_samples = training_params["num_samples"]
del config
del training_params
del model_params
x = encode[start]
x = torch.tensor([x], dtype = torch.int64, device = device).unsqueeze(0)
if model == "from_scratch":
    model_args = dict(vocab_size = vocab_size, hidden_dim = hidden_dim, n_layers = n_layers)
else:
    model_args = dict(vocab_size = vocab_size, hidden_dim = hidden_dim, n_layers = n_layers, model = model)
iter_num = 0
best_val_losses = 1e9
if init_from == "scratch":
    #init the model from scratch
    print("Initializing a new model from scratch")
    if model == "from_scratch":
        model = StackedLSTM_scratch(**model_args)
    else:
        model = RNNModel(**model_args)
elif init_from == "resume":
    print(f"Resuming from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location = device)
    checkpoint_model_args = checkpoint["model_args"]
    if model == "from_scratch":
        for k in ["vocab_size", "hidden_dim", "n_layers"]:
            model_args[k] = checkpoint_model_args[k]
    else:
        for k in ["vocab_size", "hidden_dim", "n_layers", "model"]:
            model_args[k] = checkpoint_model_args[k]        
    model = RNNModel(**model_args)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_losses = checkpoint["best_val_losses"]
model.to(device)
print("Total number of parameters:", model.count_parameters())
optimizer = torch.optim.Adam(model.parameters(), lr = max_lr)

def get_batch(split):
   
    if split == 'train':
        #already tokenized data
        data = np.memmap(data_dir + os.sep+ dataset + "_train.bin", dtype=np.uint16, mode='r')
    else:
        data = np.memmap(data_dir + os.sep+  dataset + "_val.bin", dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context]).astype(np.int64)) for i in ix])
    
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data, targets = get_batch(split)
            _, _, loss = model(data, hidden_states= None, targets= targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
def train():
    model.train()
    data, targets = get_batch("train")
    _, _, loss = model(data,  hidden_states= None, targets = targets)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none= True)
    return loss.item(), norm
def get_lr(iter):
        return max_lr * (gamma ** iter)


def sampling(x):
    model.eval()
    with torch.no_grad():
        for k in range(num_samples):
            model.generate(input = x, decode = decode, new_tokens = 256)
            print("------------------------------")
hidden = None
train_losses = []
val_losses = []
while True:
    if iter_num % eval_interval == 0:
        epoch = iter_num // eval_interval
        lr = get_lr(epoch) 
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("-----------------------------------------")
        print("lr:", lr)
        losses = estimate_loss()
        val_losses.append(losses["val"])
        print(f"val | step {iter_num}| train loss {losses['train']:.4f}| val loss {losses['val']:.4f}")
        if sampling_duringtraining:
            sampling(x)
        if losses["val"] < best_val_losses:
            best_val_losses = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_losses": best_val_losses,
                }    
                print(f"save checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
                    pickle.dump(train_losses, file)
            with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
                    pickle.dump(val_losses, file)
                
    t0 = datetime.datetime.now()
    loss, norm = train()
    if iter_num % log == 0:
        t1 = datetime.datetime.now()
        dt = (t1-t0)
        dt = dt.microseconds / 1000
        train_losses.append(loss)
        print(f"Step {iter_num}|loss: {loss:.3f}| norm: {norm:.2f}| time: {dt} ms")
    iter_num +=1
    if iter_num > iter_max:
        print(f"Ending of the training. Last loss: {loss:.4f}")
        break
    
    
