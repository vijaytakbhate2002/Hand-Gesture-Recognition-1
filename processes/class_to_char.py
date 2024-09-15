from config import config

def maxCount(confirmation_by) -> dict:
    """Returns maximum count of each element from list"""
    temp = {}
    for el in set(config.CONFERMATION_LIS):
        temp[el] = config.CONFERMATION_LIS.count(el)
    if max(temp.values()) >= confirmation_by:
        return max(temp, key=temp.get)
    return None

def getChar(predicted_class, confirmation_by) -> str:
    if predicted_class is None:
        return None
    config.CONFERMATION_LIS.append(predicted_class)
    del config.CONFERMATION_LIS[0]
    max_count_val = maxCount(confirmation_by=confirmation_by)
    if max_count_val is not None:
        if max_count_val >= 0:
            if config.CURRENT_HALF == 'first':
                return config.FIRST_HALF[max_count_val]
            else:
                return config.SECOND_HALF[max_count_val]