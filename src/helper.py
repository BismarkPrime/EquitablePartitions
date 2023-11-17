import os,sys
import numpy as np

def parse_input(prompt,based_on_error=False):
    """makes sure input is of the form specified in the prompt. Which is signified by being inside
    parentheses and separated by backslashes '/'
    """
    keywords = prompt.split('(')[-1][:-3].split('/')
    inp = input(prompt)
    while True:
        if inp in keywords:
            return inp
        else:
            inp = input(f"Not valid input. You must choose one of -> {keywords}: ")