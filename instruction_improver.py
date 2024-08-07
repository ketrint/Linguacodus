# -*- coding: utf-8 -*-
"""
Script Name: instruction_improver.py
Purpose: choose the best and refine code instruction using OpenAI's GPT-3.5-turbo model.
"""

import openai
import csv
import os
from os.path import join, dirname
from dotenv import load_dotenv

INPUT_FILEPATH = './data/results.csv'

# Load environment variables
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai_api_key

def create_prompt(row):
    """
    Create the initial prompt for the GPT model to analyze instructions.
    
    Args:
        row (dict): A dictionary containing task details and instruction options.
        
    Returns:
        str: The generated prompt.
    """
    return (f"You are a researcher tasked with investigating the 3 options of instruction for solving this "
            f"machine learning task. Task: {row['task']} The {row['data_type']} data is used for the problem. "
            f"The metric type is {row['metric']} for the problem.\n\nYour response contains the main information "
            f"about data preprocessing, model architecture and model training. List the flaws and faulty logic of "
            f"each instruction option. Let’s work this out in a step by step way to be sure we have all the errors.\n"
            f"Instruction option 1: {row['option_1']}\n"
            f"Instruction option 2: {row['option_2']}\n"
            f"Instruction option 3: {row['option_3']}")

def resolver_prompt(response1_content):
    """
    Create the resolver prompt based on the initial analysis.

    Args:
        response1_content (str): The response content from the first analysis.

    Returns:
        str: The generated resolver prompt.
    """
    return (f"{response1_content}\nYou are a resolver tasked with 1) finding which of the instruction options "
            f"the researcher thought was best 2) improving the instruction 3) printing the improved instruction in full. "
            f"Let’s work this out in a step by step way to be sure we have the right meaningful instruction based on "
            f"the previous analysis:")

def get_openai_response(prompt, model="gpt-3.5-turbo"):
    """
    Get the response from the OpenAI API.

    Args:
        prompt (str): The prompt to send to the API.
        model (str): The model to use for the API call.

    Returns:
        str: The content of the response from the API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error getting response from OpenAI: {e}")
        return None

def main():
    results = []

    # Open the input CSV file
    with open(INPUT_FILEPATH, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Create the initial prompt
            prompt1 = create_prompt(row)
            response1_content = get_openai_response(prompt1)

            if response1_content:
                # Create the resolver prompt
                prompt2 = resolver_prompt(response1_content)
                result_content = get_openai_response(prompt2)

                if result_content:
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
                else:
                    print("Skipping row due to missing resolver response.")
            else:
                print("Skipping row due to missing initial response.")

    # Write results to the output CSV file
    with open('output.csv', 'w', newline='') as csv_file:
        fieldnames = ['task', 'data_type', 'metric', 'data_card', 'result', 'submission', 'link']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
