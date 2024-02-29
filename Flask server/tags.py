from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os

class AssignTags():
    def __init__(self) -> None:
        self.llm = OpenAI(openai_api_key=os.environ.get['OPENAI_KEY'])
        self.tags = ['spam', 'legit', 'feedback', 'question', 'bug']

    def OpenAITagging(self, template, question):
        tags_found = []
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response = llm_chain.invoke(question)
        txt = response['text'].replace("\n", "")
        for tag in self.tags:
            if tag in txt:
                tags_found.append(tag)

        return tags_found

    def PlayStoreReview(self, question):
        template = """This is a review from playstore: {question} 
        Answer: Just analyze the review and choose one these tags to it 
        'feedback', 'question', 'bug' and if non of the tags are suitable answer undefined"""
        response = self.OpenAITagging(template=template, question=question)
        
        return response
    
    def AppStoreReview(self, question):

        template = """This is a review from appstore: {question} 
        Answer: Just analyze the review and choose one these tags to it 
        'feedback', 'question', 'bug' and if non of the tags are suitable answer undefined"""
        response = self.OpenAITagging(template=template, question=question)
        
        return response
    
    def TwitterReview(self, question):
        template = """This is a review from twitter(x): {question} 
        Answer: Just analyze the review and choose one these tags to it 
        'feedback', 'question', 'bug' and if non of the tags are suitable answer undefined"""
        response = self.OpenAITagging(template=template, question=question)
        
        return response