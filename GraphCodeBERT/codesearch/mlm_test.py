
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
import torch

pretrained_model='/scratch/st-fhendija-1/iman/pre-trained/models/microsoft/graphcodebert' 
model = RobertaForMaskedLM.from_pretrained(pretrained_model)
tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)

adapter_path = '/scratch/st-fhendija-1/iman/experiments/CodeBERT1/CodeBERT/GraphCodeBERT/codesearch/saved_models/ruby/latest-mlm/adapter_ruby'
ad_name = model.load_adapter(adapter_path)
model.set_active_adapters(ad_name)

# model.load_state_dict(torch.load("/scratch/st-fhendija-1/iman/experiments/CodeBERT1/CodeBERT/GraphCodeBERT/codesearch/saved_models/ruby/latest-mlm/model.bin",map_location=torch.device('cpu')),strict=False)  


code_example = "if (x is not None) <mask> (x>1):"
ids=tokenizer.encode(code_example)

print(tokenizer.decode(ids))
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(code_example)
print(outputs)






