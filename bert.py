import torchvision.models as models
import pytorch2timeloop
from transformers import BertTokenizer, BertModel, BertForMaskedLM

net = BertModel.from_pretrained('bert-base-uncased')

input_shape = (1,1)

batch_size = 1

top_dir = 'workloads'
sub_dir = 'bert'
 
convert_fc = True

exception_module_names = []

pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)
