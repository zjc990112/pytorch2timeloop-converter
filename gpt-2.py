import torch
from transformers import GPT2Tokenizer, GPT2Model
import pytorch2timeloop

# net = GPT2Model.from_pretrained("./gpt2")
net = GPT2Model.from_pretrained("./gpt2-xl")

input_shape = (1,1)

batch_size = 1

top_dir = 'workloads'
# sub_dir = 'gpt2'
sub_dir = 'gpt2-xl'
 
convert_fc = True

exception_module_names = []

pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)
