import openai
import csv
import os
from os.path import join, dirname
from dotenv import load_dotenv

# Loading env variables
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai_api_key

def create_prompt(row):
    return f"""You are a researcher tasked with investigating the 3 options of instruction for solving this machine learning task. Task: {row['task']} The {row['data_type']} data is used for the problem. The metric type is {row['metric']} for the problem.

Your response contains the main information about data preprocessing, model architecture and model training. List the flaws and faulty logic of each instruction option. Let’s work this out in a step by step way to be sure we have all the errors.
Instruction option 1: {row['option_1']}
Instruction option 2: {row['option_2']}
Instruction option 3: {row['option_3']}"""

def resolver_prompt(response1_content):
    return f"""{response1_content}\nYou are a resolver tasked with 1) finding which of the instruction options the researcher thought was best 2) improving the instruction 3) printing the improved instruction in full. Let’s work this out in a step by step way to be sure we have the right meaningful instruction based on the previous analysis:
"""

results = []

with open('input.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        prompt1 = create_prompt(row)
        
        response1 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt1}
            ]
        )
        
        response1_content = response1['choices'][0]['message']['content']
        
        prompt2 = resolver_prompt(response1_content)
        
        response2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt2}
            ]
        )
        
        result_content = response2['choices'][0]['message']['content']
        print(result_content)
        results.append({
            'task': row['task'],
            'data_type': row['data_type'],
            'metric': row['metric'],
            'data_card': row['data_card'],
            'result': result_content,
            'submission': row['submission'],
            'link': row['link']
        })

with open('output.csv', 'w') as csv_file:
    fieldnames = ['task', 'data_type', 'metric', 'data_card', 'result', 'submission', 'link']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
