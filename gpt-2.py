import torch
from transformers import GPT2Tokenizer, GPT2Model
import pytorch2timeloop

net = GPT2Model.from_pretrained("./gpt2")

input_shape = (1,1)

batch_size = 1

top_dir = 'workloads'
sub_dir = 'gpt2'
 
convert_fc = True

exception_module_names = []

pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)
