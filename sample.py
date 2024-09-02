import torch, os, pickle
from model import *

model = "from_scratch"
init_from = "resume"
out_dir = r"C:\Users\Gianl\Desktop\RNN-repo\MyRNN\output\2_layers" + os.sep + model
#sampling method
start = "\n"
num_samples = 3
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.5
#system
seed = 42
device = "cuda"
#utils
data_dir = r"C:\Users\Gianl\Desktop\gpt-repository\data"
dataset = "shakespeare"
meta = r"C:\Users\Gianl\Desktop\RNN-repo\MyRNN\data\meta.pkl"
#----------------------------------------------------------------
torch.manual_seed(seed)
device_type = "cuda"
ptdtype = torch.bfloat16
ctx = torch.amp.autocast(device_type = device_type, dtype = ptdtype)

#model 
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    if model == "from_scratch":
        model = StackedLSTM_scratch(**checkpoint_model_args)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
model.eval()
model.to(device)
if meta:
    with open(meta, "rb") as f:
        meta = pickle.load(f)
        decode = meta["itos"]
        encode = meta["stoi"]
        vocab_size = meta["vocab_size"]
        start_ids = encode[start]

x = encode[start]
x = torch.tensor([x], dtype = torch.int64, device = device).unsqueeze(0)
with torch.no_grad():
    model.eval()
    for k in range(num_samples):
        model.generate(x, decode = decode, new_tokens= max_new_tokens, temperature= 0.9)
        print("------------------------------")