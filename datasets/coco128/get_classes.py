import json

f=open(r"annotations\instances_val2017.json")# 替换成你自己的路径
json_dict=json.load(f)
print(type(json_dict))

js = json.dumps(json_dict, sort_keys=True, indent=4, separators=(',', ':'))
print(js)