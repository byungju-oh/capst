#from common import get_tensor,get_model,get_body
from common import get_tensor,get_body


import json
with open('cat_to_name.json') as f:
    cat_to_name = json.load(f)
with open('class_to_idx.json') as f:
    class_to_idx = json.load(f)

idx_to_class = {v:k for k, v in class_to_idx.items()}


#model = get_model()

# def get_name(image_bytes):
#     tensor = get_tensor(image_bytes)
#     outputs = model.forward(tensor)
#     _,prediction= outputs.max(1)
#     category = prediction.item() #번호로까지바꿔줌 json 파일이랑 매치해서 클래스로바꾼다.
#     class_idx = idx_to_class[category]
#     fname = cat_to_name[class_idx]
    
#     return category,fname
model = get_body()

def get_type(image_bytes):
    tensor = get_tensor(image_bytes)
    outputs = model.forward(tensor)
    _,prediction= outputs.max(1)
    category = prediction.item() 
    # class_idx = idx_to_class[category]
    # fname = cat_to_name[class_idx]
    return category
    