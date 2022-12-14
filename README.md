# FineTMLM
A tool for finetuning masked language models

## Requirements


## How To Use
You need to provide a corpus on which you want to train as a list of sentences and a masked language model and its corresponding tokenizer. mlm model could be anything like Bert, Roberta, Deberta and etc. if your corpus is a text as whole first tokenize it into a list of sentences using nltk module.
## Code
```python
from ftmlm import FineTMLM

ftm = FineTMLM(data=text_list, 
               mlm_model=mlm_model, 
               tokenizer=mlm_tokenizer,
               batch_size=20,
               max_length=100,
               learning_rate=1e-5,
               warmup_ratio=0.,
               epochs=10,
               do_eval=True,
               eval_ratio=.1,
               do_log=True,
               log_step=100,
               device='cuda:0',
               checkpoint_dir='.',
               mask_ratio=.15)
ftm()
```
