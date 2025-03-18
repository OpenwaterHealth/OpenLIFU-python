from __future__ import annotations

import re
from typing import Literal

Cases = Literal['lower', 'upper', 'same', 'snake', 'camel', 'pascal', 'cobra', 'title', 'sentence']

def sanitize(in_str, case: Cases ='snake'):
    stripped = ''.join(re.findall(r'[\w\s]+', in_str))
    words = re.findall(r'\w+', stripped)

    if case.lower() == 'lower':
        out_str = ''.join(words).lower()
    elif case.lower() == 'upper':
        out_str = ''.join(words).upper()
    elif case.lower() == 'same':
        out_str = ''.join(words)
    elif case.lower() == 'snake':
        out_str = '_'.join(words).lower()
    elif case.lower() == 'camel':
        out_str = ''.join([word[0].upper() + word[1:].lower() for word in words])
        out_str = out_str[0].lower() + out_str[1:]
    elif case.lower() == 'pascal':
        out_str = ''.join([word[0].upper() + word[1:].lower() for word in words])
    elif case.lower() == 'cobra':
        out_str = '_'.join([word[0].upper() + word[1:].lower() for word in words])
    elif case.lower() == 'title':
        words = re.findall(r'\w+', stripped.replace("_", " "))
        words = [word[0].upper() + word[1:].lower() for word in words]
        lowercase_words = ["a","an","and","for","in","of","the"]
        words = [word.lower() if word in lowercase_words else word for word in words]
        out_str = ' '.join(words)
    elif case.lower() == 'sentence':
        words = re.findall(r'\w+', stripped.replace("_", " "))
        first_word = words[0]
        words[0] = first_word[0].upper() + first_word[1:].lower()
        out_str = ' '.join(words)
    else:
        raise ValueError(f'Unrecognized case type {case}')

    return out_str
