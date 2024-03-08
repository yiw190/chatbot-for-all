from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, AutoTokenizer, AutoModelForCausalLM
#from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import torch
from django.http import JsonResponse  
from langchain.chains import SequentialChain, ConversationChain
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              ConversationSummaryMemory,
                              ConversationSummaryBufferMemory)

# For Model From HuggingFace
def create_tokenizer(model_name):
    # demo version, use AUTO tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def create_model(model_name,load_in_8bit=True):
    model = AutoModelForCausalLM.from_pretrained(model_name,
    load_in_8bit = True)
    return model

def create_pipeline(
    tokenizer, model, 
    max_length=1024, 
    temperature=0.7,
    repetition_penalty=1.0,
    do_sample=True):

    pipe = pipeline(
        'text-generation', 
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        device_map= "auto",
        do_sample = do_sample
        )
    
    langchain_llm = HuggingFacePipeline(pipeline=pipe)
    return langchain_llm

def create_chains(llm, prompt, verbose=True):
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain

def create_seq_chains(prompts, llm, input_variables=None, output_key=None,verbose=True): # this is used for creating a sequential chain for dealing step by step questions.
    
    """
        prompts: list of prompts
        llm: the llm model
        input_variables: list of input variables
        verbose: boolean
    """
    length = len(prompts)
    if output_key is not None and len(output_key) != length:
        raise ValueError("The length of output_key should be the same as the length of prompts")

    if output_key is not None:
        chains = []
        for i, prompt in enumerate(prompts):
            if i == 0:
                pass
            else:
                prompt = "Based on the information of " + "{" + output_key[i-1] + "}" + "\n " + prompt
            prompt = PromptTemplate.from_template(prompt)
            chain = LLMChain(llm=llm, prompt=prompt, output_key=output_key[i], verbose=verbose)
            print(chain.output_key)
            chains.append(chain)
        chain = SequentialChain(chains=chains, input_variables=input_variables, verbose=verbose)
        return chain
    else:
        raise ValueError("output_key should not be None")

def create_conversations(llm, prompt,use_memory=True, memory=None, verbose=True):# this is used for creating a live chatbot 
    if use_memory:
        if memory is None:
            memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=256)
        conversation = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=verbose)
        # this helps to remember the whole conversation 
    else:
        conversation = ConversationChain(llm=llm, prompt=prompt, verbose=True)
    return conversation


def create_qa_chains(llm, prompt, verbose=True):
    
    chain = load_qa_chain(llm=llm, chain_type='map_rerank', verbose=True)
    
    return chain



if __name__ == "__main__":
    # test for local model
    tokenizer = create_tokenizer('meta-llama/Llama-2-7b-chat-hf')
    model = create_model('meta-llama/Llama-2-7b-chat-hf')
    chatbot = create_pipeline('meta-llama/Llama-2-7b-chat-hf', tokenizer, model)
    print(chatbot.predict('Discribe a possible ptsd medical treatment.'))
    
    