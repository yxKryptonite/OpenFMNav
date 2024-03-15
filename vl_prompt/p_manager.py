import re
from PIL import Image
import base64

def extract_integer_answer(s):
    match = re.search(r'Answer: \d+', s)
    if match:
        return int(match.group().split(' ')[-1])
    else:
        print('=====> No integer found in string')
        return -1
    
    
def extract_scores(s):
    match = re.search(r'Answer: \[(.*?)\]', s)
    if match:
        scores = [float(x) for x in match.group(1).split(',')]
        return scores.index(max(scores)), scores
    else:
        print('=====> No list found in string')
        return -1, []
    
def extract_objects(s):
    elements = re.findall(r'"([^"]*)"', s)
    return list(set(elements))
    

def object_query_constructor(objects):
    """
    Construct a query string based on a list of objects

    Args:
        objects: torch.tensor of object indices contained in an area

    Returns:
        str query describing the area, eg "This area contains toilet and sink."
    """
    assert len(objects) > 0
    query_str = "This area contains "
    names = []
    for ob in objects:
        names.append(ob.replace("_", " "))
    if len(names) == 1:
        query_str += names[0]
    elif len(names) == 2:
        query_str += names[0] + " and " + names[1]
    else:
        for name in names[:-1]:
            query_str += name + ", "
        query_str += "and " + names[-1]
    query_str += "."
    return query_str

def get_frontier_prompt(prompt_type):
    if prompt_type == "deterministic":
        from vl_prompt.prompt.deterministic import \
            SYSTEM_PROMPT, USER1, ASSISTANT1, USER2, ASSISTANT2
    elif prompt_type == "scoring":
        from vl_prompt.prompt.scoring import \
            SYSTEM_PROMPT, USER1, ASSISTANT1, USER2, ASSISTANT2
    else:
        raise NotImplementedError("Froniter prompt type not implemented.")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER1},
        {"role": "assistant", "content": ASSISTANT1},
        {"role": "user", "content": USER2},
        {"role": "assistant", "content": ASSISTANT2}
    ]
    
    return messages


def get_candidate_prompt(candidate_type):
    if candidate_type == "open":
        from vl_prompt.prompt.candidate_open import \
        SYSTEM_PROMPT, USER1, ASSISTANT1, USER2, ASSISTANT2
    elif candidate_type == "close":
        from vl_prompt.prompt.candidate_close import \
        SYSTEM_PROMPT, USER1, ASSISTANT1, USER2, ASSISTANT2
    else:
        raise NotImplementedError("Candidate prompt type not implemented.")
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER1},
        {"role": "assistant", "content": ASSISTANT1},
        {"role": "user", "content": USER2},
        {"role": "assistant", "content": ASSISTANT2}
    ]
    
    return messages

def get_grouping_prompt():
    from vl_prompt.prompt.group_obj import \
        SYSTEM_PROMPT, USER1, ASSISTANT1, USER2, ASSISTANT2
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER1},
        {"role": "assistant", "content": ASSISTANT1},
        {"role": "user", "content": USER2},
        {"role": "assistant", "content": ASSISTANT2}
    ]
    
    return messages


def get_discover_prompt(img, objects):
    from vl_prompt.prompt.discover import \
        SYSTEM_PROMPT, USER
        
    img.save("current_for_gpt4.jpg")
    with open("current_for_gpt4.jpg", "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode('utf-8')
    
    question = f"""Current object list: {objects}\n{USER}"""
    payload = {
        "model": "gpt-4-vision-preview", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": question}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            ]
        }], "max_tokens": 300
    }
    return payload