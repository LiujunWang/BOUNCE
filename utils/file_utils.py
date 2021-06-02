# coding: utf-8

import json
from os.path import dirname, join, normpath, exists
from os import makedirs

project_root_path = normpath(join(dirname(__file__), '..'))


def from_project_root(relative_path, is_create = True):
    absolute_path = normpath(join(project_root_path, relative_path))
    if is_create and not exists(dirname(absolute_path)):
        makedirs(dirname(absolute_path))
    return absolute_path

def load(json_path):
    with open(json_path, "r", encoding = "utf-8") as json_file:
        result = json.load(json_file)
    return result

def dump(obj, json_path):
    with open(json_path, "w", encoding = "utf-8", newline = '\n') as json_file:
        json.dump(obj, json_file, separators = [',', ': '], indent = 4, ensure_ascii = False)

def list_to_dict(input_list):
    dic = dict()
    for index, value in enumerate(input_list):
        dic[value] = index
    return dic