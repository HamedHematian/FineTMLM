# FineTMLM
A tool for finetuning masked language models

You need to provide a corpus which you want to train on as a list of sentences and a mlm model and its corresponding tokenizer 
## How To Use
```
from ftmlm import FineTMLM
ftm = FineTMLM(text, model, tokenizer)
ftm()
```
