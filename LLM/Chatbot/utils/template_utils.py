
from langchain.prompts import PromptTemplate

def data2string(data):
    # determined by the data we get.
    return data

#############################        NANOMOOD         ############################################
def PHQ_knowledge():
    PHQ_knowledge = "The scores for each of the 9 questions are summed to produce a total score ranging from 0 to 27. The total score is interpreted as follows to assess the severity of depression: 0-4: None or minimal depression\n 5-9: Mild depression\n 10-14: Moderate depression\n 15-19: Moderately severe depression\n 20-27: Severe depression.\n Now, summarize this information."
    return PHQ_knowledge # text

def create_PHQ_score_temp():
    template = 'Question: If the The patient has a heart rate of {heart_rate}, the sleep hour of {sleep},the weight of {weight},the height of {height}, the age of {age}, could you give a step by step anaysis and give the final result of a PHQ-9 score of the patient.'
    # prompt = PromptTemplate.from_template(template)
    return template # text 

def conversation_temp():
    template = '''The following is a friendly conversation between humans and artificial intelligence. The AI is talkative and provides many specific details based on its context. If the AI does not know the answer to a question, it will truthfully say that it does not know. Please note that regardless of the language used by humans, you respond in English.
    {history}
    User:{input}
    Assistant:
    '''
    return template # text







#############################        INDUSTRY         ############################################