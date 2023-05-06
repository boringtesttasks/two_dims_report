## Report builder

This script calculates two-dimentional distributions from SPSS data and saves in excel format.  



To get started, specify in the config file the path to the data from SPSS and (optionally) to the weight file. Then specify the path to save the report.  



For each exel sheet, specify the grouping **keys** and **values** by which statistics will be calculated. For each key, specify the processing mode:  
- 'single' for single-choise questions

- 'mult' for multi-reply

- or enter the the statistic to be calculated for each value (only mean and max options are available for now)  


### TO DO:

-- Deal with NaNs
