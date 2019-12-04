def flatten(list_of_lists):
    """
    Recursivey flattens a list of lists
    :return: List, flattened list elements in sub-lists
    """
    if list_of_lists == []:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])
