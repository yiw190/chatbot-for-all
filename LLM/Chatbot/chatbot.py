from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from utils import *

load_dotenv()# Load the .env file
import argparse
import torch
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)

args = argparse.ArgumentParser(description="OpenAI Chatbot")

# OpenAI
args.add_argument("--openai_model", type=str, default="gpt-3.5-turbo-1106", help="The openai model to use")
args.add_argument("--use_openai", type=str, default="True", help="Whether to use openai model") 

# Local Model
args.add_argument("--local_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="local model to use")
args.add_argument("--temperature", type=float, default=0.7, help="The temperature to use")
args.add_argument("--max_tokens", type=int, default=1000, help="The maximum number of tokens to use")
args = args.parse_args()

def create_chatbot(args):
    use_openai = args.use_openai.lower() == "true"
    if use_openai:
        openai = OpenAI()
        chatbot = ChatOpenAI(model=args.openai_model, temperature=args.temperature, max_tokens=args.max_tokens)
    else:   
        tokenizer = create_tokenizer(args.local_model)
        model = create_model(args.local_model)
        chatbot = create_pipeline(tokenizer, model, args.max_tokens, args.temperature)
    return chatbot

if __name__ == "__main__":
    # create chat bot
    chatbot = create_chatbot(args)
# test the chatbot
    
    # print(chatbot.predict('Discribe a possible ptsd medical treatment.'))
    # prompt = create_PHQ_score_temp()
    # prompt = PromptTemplate.from_template(prompt)
    # chat_chain = create_chains(chatbot, prompt)
    # print(chat_chain.predict(heart_rate=90, sleep=8, weight=70, height=180, age=30))
    
    
# test for sequential chain
    
    # prompt_of_knowledge = PHQ_knowledge()
    # prompt_of_PHQ = create_PHQ_score_temp()
    # chain = create_seq_chains([prompt_of_knowledge, prompt_of_PHQ], chatbot, input_variables=['heart_rate', 'sleep', 'weight', 'height', 'age'], output_key=['PHQ-K','PHQ-A'])
    # print(chain.run({'heart_rate':90, 'sleep':8, 'weight':70, 'height':180, 'age':30}))
    
    
# test for conversation chain
    
    
    # prompt = conversation_temp()
    # prompt = PromptTemplate.from_template(prompt)
    # conversation = create_conversations(chatbot, prompt)
    # while True:
    #     human_input = input('User: ') 
    #     result = conversation.predict(input=human_input)
    #     print('Assistant: ', result)
    # hidden()
    
    
#test for corpus reading
    
    data_dir = parent_path + '/Data/ptsd_treatment.txt'
    corpus = file_to_corpus(data_dir)
    document = corpus_to_documents(corpus)
    question = 'After you get a gun shot and have the symptoms of PTSD, what should you do?'
    print(document)
    retrieve_documents = retrieve_documents(document, question, k=3)
    prompt = PromptTemplate.from_template(question)
    chain = create_chains(chatbot, prompt)
    print(chain.run(input_documents=retrieve_documents, question=question))
    print("##################################################")
    print(chain.run(question=question))
    exit()
    
    
    
    
    



    
    
    



