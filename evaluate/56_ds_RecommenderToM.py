import json
import argparse
from openai import OpenAI
import openai

import os
import csv
from tqdm import tqdm
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed 

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type = str, default = "6_judge_seeker.json")
    parser.add_argument("--model", type = str, choices = ['meta-llama/llama-3.1-8b-instruct','deepseek-r1-0528','deepseek-v3-0324','gpt-4o-mini','gpt-4o','gemini-2.5-flash-lite-preview-06-17','claude-3-5-haiku-20241022'])
    parser.add_argument("--cot", type = bool, default = False)
    args = parser.parse_args()
    return args

letters = ["A", "B"]

def extract_answers(response, answer_range):
    A, Z = answer_range.split('-')
    valid_letters = ''.join(chr(c) for c in range(ord(A.upper()), ord(Z.upper()) + 1))


    lead_pattern = r"\b(?:answer\s*(?:is|:)|the\s+answer\s+is)\b"


    answer_block_pattern = r"[^{}]*((?:[{}]|\\boxed\{{[{}]}})+)".format(
        valid_letters, valid_letters, valid_letters
    )


    full_pattern = r"(?i){}{}".format(lead_pattern, answer_block_pattern)


    matches = re.findall(full_pattern, response)

    all_letters = []

    for match in matches:
        answer_block = match

        letters = re.findall(
            r"\\boxed\{{([{}])}}|([{}])".format(valid_letters, valid_letters),
            answer_block,
            re.IGNORECASE
        )

        letters = [x[0].upper() or x[1].upper() for x in letters]
        all_letters.extend(letters)


    unique_letters = sorted(set(all_letters))

    return unique_letters if unique_letters else None

def evaluate(args, client, problem, idx):
    answers = []

    if args.cot == True:
        system_prompt = """Here is a movie recommendation dialogue, there are two agents, the RECOMMENDER and the SEEKER. The RECOMMENDER is trying to recommend movies to SEEKER. Think step by step to answer the quesiton, but limit yourself to no more than 3 steps."""
    else:
        # system_prompt = """Here is a movie recommendation dialogue, there are two agents, the RECOMMENDER and the SEEKER. The RECOMMENDER is trying to recommend movies to SEEKER. Please answer the following questions using \"A\", \"B\". Do not provide any explanations."""
        system_prompt = """You are an expert in dialogue analysis. Given a dialogue and a question, respond ONLY with the letter of the correct choice A or B. Do not include any other text, punctuation, explanation, or whitespace. Example valid outputs: 'A', 'B'."""
    utterance_context = problem["utterance_context"]
    question = problem["question"]
    choices = problem["choices"]
    choices_str = "\n".join([f"{key}: {value}" for key, value in choices.items()])

    # choices_str = "\n".join([f"{choice}" for choice in choices])

    if args.cot == True:

        shot = """\n
            Ending with "The answer is X", where X is one of the option from choices.
            Do not use any other format for the ending.
    """
        user_prompt = shot + "\nDiallogue History:\n"+ utterance_context +"\nQuestion:\n"+ question + "\nChoices:\n" + choices_str + "\nAnswer: Let's think step by step."
        print(user_prompt)
    else:
        user_prompt = "\nDiallogue History:\n"+ utterance_context + "\nQuestion:\n"+ question+ "\nChoices:\n" + choices_str  + "\nAnswer:"
        print(user_prompt)

    if "o1" in args.model or "gemma" in args.model:
        messages=[{"role":"user", "content":system_prompt +"\n" + user_prompt}]
    else:
        messages=[{"role":"system","content":system_prompt}, {"role":"user", "content":user_prompt}]

    while True:
        response = client.chat.completions.create(
            model = args.model,
            messages = messages,

            temperature = 0.1,

            max_tokens = 700,
            )
        result = response.choices[0].message.content
        print("\n", response.choices[0].message)
        # print(result)

        if args.cot != True:
           cleaned = re.sub(r'[^A-Za-z]','', result)

           candidates = list(set(c for c in cleaned))
        else:
            try:
                candidates = extract_answers(result, 'A-B')

             
                if not candidates:
                    candidates = ["Y"]
                    print("!!!\n", response.choices[0].message)

                    break
            except:
                candidates = ["Z"]
                break
        if all (i in letters for i in candidates):
            break
    dialogue_id = problem["dialogue_id"]
    utterance = problem["utterance_pos"]

    candidates = list(set(candidates))

    answers =[dialogue_id,utterance, problem["answer"], candidates]

    return answers, idx 
                    
    
if __name__ == '__main__':
    # read_dataset()
    args = arg_parse()
    with open(f"../data/{args.dataset_type}", 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    
    if 'deepseek-chat' in args.model:
       
        API_BASE = ""
        API_KEY = ""
        client = OpenAI(
        api_key = API_KEY,
        base_url = API_BASE
        )
    elif 'llama' in args.model:
        client = OpenAI(
            base_url = "",
            api_key ="" 
        )
    else: 

        API_BASE = ""

        API_KEY = ""
        client = OpenAI(
        api_key = API_KEY,
        base_url = API_BASE
        )

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not args.cot:
        file_path = f"../outputs/{args.model}/{args.dataset_type}_{timestamp}.csv"
    else:
        file_path = f'../outputs/{args.model}/{args.dataset_type}_cot_{timestamp}.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok = True )

   
    all_results = [None] * len(data)

    correct_predictions = [0]
    def update_accuracy_and_print(idx):
        current_result = all_results[idx]
        if sorted(current_result[2]) == sorted(current_result[3]):  # 比较 labels 和 predictions
            correct_predictions[0] += 1
        current_accuracy = correct_predictions[0] / (idx + 1)
        print(f"Task {idx + 1}: Current accuracy is {current_accuracy:.4f} ({correct_predictions[0]}/{idx + 1})")


    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {
            executor.submit(evaluate, args, client, problem, idx): idx  
            for idx, problem in enumerate(tqdm(data, desc="submitting tasks"))
        }

        with open(file_path, 'w', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dialogue_id', 'utterance', 'labels', 'predictions'])

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing results"):
            try:
                result, idx = future.result()
                all_results[idx] = result
                update_accuracy_and_print(idx) 

                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if result is not None:
                        writer.writerow(result)
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

