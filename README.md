# Capstone Data Analysis
 
A small desktop app for running visualizations against the Pitt composition course AI survey data (Fall 2022 through Fall 2024). Drop in a new visualization module, hit Refresh, and run it against the Excel file without touching any other code.

This readme file does not yet contain everything I would like it too. Currently it just outlines the project structure, and this will be changed in the documentation phase.
 
---
 
## What's in here
 
```
main.py                        tkinter launcher
visualizations/
    __init__.py
    techUsage.py               what writing tools students use, in vs outside coursework
    permittedUsage.py          where students draw the line on what AI should be allowed
    aiUsecaseFrequency.py      how often students use AI for specific writing tasks
    beliefVsBehavior.py        do students actually follow their own stated beliefs?
output/                        generated charts go here (created on first run)
```
