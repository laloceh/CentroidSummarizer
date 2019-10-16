#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:30:07 2019

@author: eduardo
"""
import re

def find_hidden_sentences(text):
    pattern = "[a-z]\.[A-Z]"    
    matches = [m.start(0) for m in re.finditer(pattern, text)]
    
    for i, index in enumerate(matches):
        index = index + 2 + i
        text = text[:index] + " " + text[index:]

    return text