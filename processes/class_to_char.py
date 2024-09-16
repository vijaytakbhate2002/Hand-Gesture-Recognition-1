from config import config

def maxCount(confirmation_by:int, confirmation_lis:list) -> dict:
    """Returns maximum count of each element from list"""
    temp = {}
    unique_classes = set(confirmation_lis)
    if -1 in unique_classes:
        unique_classes.remove(-1)
    for el in unique_classes:
        temp[el] = confirmation_lis.count(el)
    if max(temp.values()) >= confirmation_by:
        return max(temp, key=temp.get)
    return None

def getChar(predicted_class:int, confirmation_by:int, confirmation_lis:list) -> str:
    if predicted_class is None:
        return None
    confirmation_lis.append(predicted_class)
    del confirmation_lis[0]
    max_count_val = maxCount(confirmation_by=confirmation_by, confirmation_lis=confirmation_lis)
    if max_count_val is not None:
        if max_count_val >= 0:
            if config.CURRENT_HALF == 'first':
                return config.FIRST_HALF[max_count_val]
            else:
                return config.SECOND_HALF[max_count_val]
            