import json

def save_json(data_dict, name = "data_nlu"):
    with open(f"{name}.json", "w", encoding='utf-8') as f:
        json.dump(data_dict, f, indent= 4)

def load_json(path = "./data_nlu.json"):
    with open(f"{path}", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data