import pytorch2timeloop
from dall_e import map_pixels, unmap_pixels, load_model, utils

# net = load_model("./encoder.pkl")
net = load_model("./decoder.pkl")
for name,module in net.named_modules():
    print(module.__class__)

input_shape = (1,1)

batch_size = 1

top_dir = 'workloads'
sub_dir = 'dall-e'
 
convert_fc = True

exception_module_names = []

pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)