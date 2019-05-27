# CAPP 30254 HW5
## Files
- `pipeline.py`: main codes of the functions for this homework
- `run_pipeline.py`: python file to run the functions in `pipeline.py`
- `classifiers.py`: python file for building models
- `temporal_validation.py`: python file for splitting time series data
- `evaluation.py`: python file to do model evaluation
- `writeup.ipynb`: writeup for the detailed analysis of the results
- `Analysis Report.pdf`: analytical report of the performances of different models

## Instruction of Running Codes
To test the code, download the file `projects_2012_2013.csv` in a same folder with other python files, then run:
```
$ python3 run_pipeline.py
```
Note: the process of running the pipeline is very long, all detailed results can be found in `writeup.ipynb`.

## Notes of Revision 
**Based on the feedback of Assignment 3, I revised my codes in the following parts:**
1.	Instead of building classifiers one by one in a hard-coding way, I build a function called `build_classifier` to modularize the process of building classifier with different parameter grids.
2.	Discretizing, imputing and creating dummies after splitting datasets. First, using mean of training set to impute the missing values for both training set and testing set with function `fill_missing`. Second, using unique values of training set to create dummies for both training set and testing set with function `create_dummies`.
3.	Leaving 60 days gap between training set and testing set when implementing temporal validation in `temporal_validation.py`.
4.	Using percent of projects predicted true to compare classifiers on a basis of precision and recall instead of using thresholds.
5.	Deleting all unnecessary for loops in Assignment 3 in both `classifier.py` and `run_pipeline.py`.
6.	Creating column `NotFunded60days` that labels 1 for the projects not funded in 60 days and 0 for the projects funded in 60 days.
7.	Using the metrics of precision and recall to find the best parameters for each classifier instead of GridSearch and SelectbestK.
8.	Revising the `Analysis Report`.

## Reference
Rayid Ghani, magicloops, (2017), GitHub repository, https://github.com/rayidghani/magicloops
