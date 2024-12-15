import pandas as pd
from transformers import pipeline
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the harmfulness of LLMs' response")
    parser.add_argument('--csv', type=str, default='tinyllama', help='response')
    args = parser.parse_args()

    cls = pipeline("text-classification", model="../base/longformer-harmful-ro")
    response = pd.read_csv('response_' + args.csv + '.csv')['response'].to_list()
    score = cls(response)
    cls_res = {}
    for res in score:
        label = res['label']
        if label not in cls_res.keys():
            cls_res[label] = 1
        else: cls_res[label] += 1
    print(cls_res)
