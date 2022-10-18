# FineTMLM
A tool for finetuning masked language models

You need to provide a corpus which you want to train on as a list of sentences and a mlm model and its corresponding tokenizer 
## How To Use
```
from ftmlm import FineTMLM
ftm = FineTMLM(text=text_list, 
               model=mlm_model, 
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
