#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    09.10.2015 11:29:39 CEST
# File:    replace_colors.py

def replace_color(text, old_color, new_color):
    text = text.replace(old_color.lower(), new_color)
    return text.replace(old_color.upper(), new_color)

def replace_all(path, colors):
    with open(path, 'r') as f:
        text = f.read()
    for old, new in colors:
        text = replace_color(text, old, new)
    with open(path, 'w') as f:
        f.write(text)

if __name__ == "__main__":
    color_replacements = [
        ['#2980B9', '#CC2D2D'],
        ['#E7F2FA', '#FAE7E7'],
        ['#6AB0DE', '#DE6A6A'],
        ['#DBFAF4', '#FADBE4'],
        ['#1ABC9C', '#BC1A45'],
        ['#2CC36B', '#C32C8E'],
        ['#27AE60', '#AE277F'],
        ['#2E8ECE', '#CE2E2E'],
        ['#295', '#926'],
        ['#409AD5', '#D54040'],
    ]

    replace_all('build/html/_static/css/theme.css', color_replacements)
    print("replace_colors.py")
