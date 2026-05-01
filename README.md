# Capstone Data Analysis
 
A small desktop app for running visualizations against the Pitt composition course AI survey data (Fall 2022 through Fall 2024). Drop in a new visualization module, hit Refresh, and run it against the Excel file without touching any other code.

---

## Things of note

1. The required CSVs to run this program are NOT included in this GitHub repository. I do not own the survey data, and I cannot share it. This program is made entirely with that CSV file in mind.
2. In order to run the program currently, you need to open the Repository in an IDE (such as visual studio code), and run the "main.py" file. This will bring up the UI.
3. For visualizations labeled as "Qualitative", these were designed with a different CSV in mind. Rather than using the "All AI Surveys" Excel file, you should use the "AI survey free response data" Excel file.

---
 
## What's in here
 
```
main.py                        tkinter launcher
visualizations/
    __init__.py                       
    aiUsecaseFrequency.py                 Rates respondents' usecases by frequency.
    beliefVSBehavior.py                   Do students use AI in ways they believe are ethical?
    instructorPermissionImpact.py         Does instructor policy determine AI use?
    motivationImpacts.py                  Does AI impact respondents' motivation to write?
    permittedUsage.py                     What use cases do respondents' think should be allowed?
    reasons_not_using.py                  Why do respondents report not using AI? (Qualitative)
    shouldBeTaught.py                     Do respondents think AI should be taught & why? (Qualitative)
    shouldProfUse.py                      Do respondents think AI should be used by professors? (Qualitative)
    techUsage.py                          How are respondents using AI? 
    test.py                               Debugging file
output/                        generated charts go here (created on first run)
```
