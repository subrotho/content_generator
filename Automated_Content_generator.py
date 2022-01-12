#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install transformers')


# In[ ]:


get_ipython().system(' pip install --upgrade packaging')


# In[ ]:


get_ipython().system('pip install sentence-splitter')


# In[ ]:


get_ipython().system('pip install torch')


# In[ ]:


get_ipython().system(' pip install sentencepiece')


# In[ ]:


get_ipython().system(' pip install protobuf')


# In[ ]:


# settings
# https://huggingface.co/tuner007/pegasus_paraphrase


# In[1]:


import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


# In[10]:


baseline_message = "You won't regret, setup to 'direct debit' for quick and safe bill payments."

# Why wait to become financially Savvy. You are already paying bills on time. Just switch to 'direct debits' for a more hassle free experience.

# baseline_message = "Take charge of you finances. You mostly pay bills on time. Switch to 'direct debit' to make hassle free payments"
# Avoid the troubles of late fee/fine. Switch to 'direct debits' facility. A great way to pay your bills hassle free
#  Dont miss out on our 'direct debit' facility. A great way to pay your bills hassle free
from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en')
sentence_list = splitter.split(baseline_message)
sentence_list

paraphrase = []
num_return_sequences = 10
num_beams = 10
for i in sentence_list:
    a = get_response(i,num_return_sequences,num_beams)
    paraphrase.append(a)
    
paraphrase_text = [[]]*num_return_sequences

concat_phrase = []

for j in range(num_return_sequences):
    for i in range(len(sentence_list)):
        a = paraphrase[i][j]
        concat_phrase.append(a)
    paraphrase_text[j] = concat_phrase
    concat_phrase=[]        

paraphrase_text


# In[11]:


baseline_message = "Why wait to become financially Savvy. You are already paying P0001 on time. Just switch to 'direct debits' for a more hassle free experience."

# Why wait to become financially Savvy. You are already paying bills on time. Just switch to 'direct debits' for a more hassle free experience.

# baseline_message = "Take charge of you finances. You mostly pay bills on time. Switch to 'direct debit' to make hassle free payments"
# Avoid the troubles of late fee/fine. Switch to 'direct debits' facility. A great way to pay your bills hassle free
#  Dont miss out on our 'direct debit' facility. A great way to pay your bills hassle free
from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en')
sentence_list = splitter.split(baseline_message)
sentence_list

paraphrase = []
num_return_sequences = 20
num_beams = 20
for i in sentence_list:
    a = get_response(i,num_return_sequences,num_beams)
    paraphrase.append(a)
    
paraphrase_text = [[]]*num_return_sequences

concat_phrase = []

for j in range(num_return_sequences):
    for i in range(len(sentence_list)):
        a = paraphrase[i][j]
        concat_phrase.append(a)
    paraphrase_text[j] = concat_phrase
    concat_phrase=[]        

paraphrase_text


# In[1]:


from styleformer import Styleformer
import torch
import warnings
warnings.filterwarnings("ignore")

#uncomment for re-producability
def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1234)

# style = [0=Casual to Formal, 1=Formal to Casual, 2=Active to Passive, 3=Passive to Active etc..]
sf = Styleformer(style = 0) 

source_sentences = [
"If you wait to become financially savvy, you will miss out.",
"On time, you are already paying P0001.",
"For a more hassle-free experience, just switch to direct debits.",
]   

for source_sentence in source_sentences:
    target_sentence = sf.transfer(source_sentence)
    print("-" *100)
    print("[Casual] ", source_sentence)
    print("-" *100)
    if target_sentence is not None:
        print("[Formal] ",target_sentence)
        print()
    else:
        print("No good quality transfers available !")


# In[ ]:




