import sys
sys.path.append('..')
from functions import *

# Loading it in
import pandas as pd
from scipy import stats

file_path = "data/MSc_Thesis_Results.xlsx"  
xls = pd.ExcelFile(file_path)
print(xls.sheet_names)


# Reading each sheet into it's own df to be used separetly
df_bl_results = pd.read_excel(file_path, sheet_name="BL_data_results")
df_nhm_results = pd.read_excel(file_path, sheet_name="NHM_data_results")
# Renaming columns to replace spaces with _ - also removing a headache
df_bl_results.columns = df_bl_results.columns.str.replace(" ", "_").str.lower()
df_bl_results = df_bl_results.rename(columns={"sd_cer.1": "sd_wer"})
df_nhm_results.columns = df_nhm_results.columns.str.replace(" ", "_").str.lower()

##########
# BL T-test
cer_pivoted = df_bl_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_cer_mean")

t_stat, p_val = stats.ttest_rel(cer_pivoted['Claude'], cer_pivoted['Qwen 32b'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



wer_pivoted = df_bl_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_wer_mean")

t_stat, p_val = stats.ttest_rel(wer_pivoted['Claude'], wer_pivoted['Qwen 32b'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



lev_pivoted = df_bl_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_lev_mean")

t_stat, p_val = stats.ttest_rel(lev_pivoted['Claude'], lev_pivoted['Qwen 32b'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



tok_pivoted = df_bl_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_token_sort_mean")

t_stat, p_val = stats.ttest_rel(tok_pivoted['Claude'], tok_pivoted['Qwen 32b'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



##### NHM T-test
cer_pivoted2 = df_nhm_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_cer_mean")

t_stat, p_val = stats.ttest_rel(cer_pivoted2['Claude'], cer_pivoted2['Qwen 32B'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



wer_pivoted2 = df_nhm_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_wer_mean")

t_stat, p_val = stats.ttest_rel(wer_pivoted2['Claude'], wer_pivoted2['Qwen 32B'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



lev_pivoted2 = df_nhm_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_lev_mean")

t_stat, p_val = stats.ttest_rel(lev_pivoted2['Claude'], lev_pivoted2['Qwen 32B'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))



tok_pivoted2 = df_nhm_results.pivot_table(index=["image_one_or_few_shot?", "if_using_ex,_is_ex_dataset_wholly_specific?"], columns = "model", values = "overall_token_sort_mean")

t_stat, p_val = stats.ttest_rel(tok_pivoted2['Claude'], tok_pivoted2['Qwen 32B'])
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))


