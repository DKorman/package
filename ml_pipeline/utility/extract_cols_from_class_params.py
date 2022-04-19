def extract_cols_from_classes(cls, check_list):
    """
    takes list of class  objects as input, extracts their parameters and intersects them with the provided checklist

    :param cls: list of class objects
    :param check_list: list of strings
    :return: list of strings - intersect result of cls parameters and checklist
    """

    values = []

    for c in cls:
        params = c.__dict__
        for value in params.values():
            if type(value) == str:
                values.append(value)


    return [x for x in values if x in check_list]




