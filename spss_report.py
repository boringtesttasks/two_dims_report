import re
import yaml
import logging
import numpy as np
import pandas as pd
import pyreadstat


def get_and_process_data(conf):
    df, metadata = pyreadstat.read_sav(conf['input_data_file'])
    logging.info(f"Read data from {conf['input_data_file']} "+
                 f"with {df.shape[0]} rows and {df.shape[1]} columns")
    if conf['input_weight_file']:
        with open(conf['input_weight_file'], "r") as f:
            raw_w = f.readlines()
        w = [re.findall(r'(\d+(?:\.\d+)?)', raw_w[i]) 
             for i in range(1, len(raw_w)-3)]
        w = pd.DataFrame(w, columns=['INTNR', 'w'])
        w = w.astype({'INTNR': 'int', 'w': 'float'})
        df = pd.merge(df, w, on='INTNR', how='left')
    else:
        logging.warning("No weights file, all observations "+
                        "will have the same importance")
        df['w'] = 1
    
    # types_dict = {'double': 'int', 'string': 'str'}
    # new_types = {i: types_dict[j]    for i, j 
    #          in list(metadata.readstat_variable_types.items())}
    # df = df.astype(new_types)
    
    values_dict = dict()
    for col_key in metadata.variable_value_labels.keys():
        values_dict[col_key] = dict()
        for v_key, one_val in metadata.variable_value_labels[col_key].items():
            values_dict[col_key][int(v_key)] = one_val
    values_dict['PAD'] = {"":""}
    metadata.column_names_to_labels['PAD'] = 'PAD'
    
    df['STIME'] = pd.to_datetime(df['STIME']).dt.strftime('%Y-%m-%d %H:%M')
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H%M:%S'
                                     ).dt.strftime("%H:%M:%S")
    df['StartDate'] = pd.to_datetime(df['StartDate'], format='%Y/%m/%d'
                                     ).dt.strftime('%Y-%m-%d')
    
    return df, metadata, values_dict


def get_pivot(dat, ind_col, col_col, total_only=False, 
              multians=False, single_stat=None):
    def post_process(postproc_df, total_cols, df_for_totals):
        if multians:
            totals = df_for_totals[total_cols].sum()
        else:
            totals = postproc_df[total_cols].sum()
        postproc_df[total_cols] = np.round(
            postproc_df[total_cols] / totals * 100, 2)
        postproc_df[ind_col] = ''
        postproc_df.loc[-1] = [ind_lab, 'TOTAL']+totals.round(0).to_list()
        postproc_df.sort_index(inplace=True)
        postproc_df.reset_index(drop=True, inplace=True)
        postproc_df.rename(columns={'w': 'Total', ind_col: 
                                    'Q', ind_lab: 'label'}, inplace=True)
        multiindex = [(totals.index.name, totals.index[i]) 
                      for i in range(len(totals))]

        return postproc_df, multiindex
    
    tmp_dataframe, metadata, values_dict = dat
    tmp_df = tmp_dataframe.copy()
    tmp_df['PAD'] = ''
    col_lab = metadata.column_names_to_labels[col_col]
    tmp_df[col_lab] = [values_dict[col_col][i]
                       for i in tmp_df[col_col]]
                   
    if multians:
        ind_lab = metadata.column_names_to_labels[ind_col+'_1'].split(':')[0]
        rensponse_columns = [i for i in tmp_df.columns if ind_col+"_" in i]
        melted_df = pd.melt(tmp_df, id_vars=['INTNR', 'w', col_lab], 
                            value_vars=rensponse_columns, var_name=ind_col)
        melted_df['w'] = melted_df['value'] * melted_df['w']
        melted_df[ind_lab] = [metadata.column_names_to_labels[i] 
                              for i in melted_df[ind_col]]
        df_for_total = pd.pivot_table(tmp_df, values='w', columns=[col_lab], 
                                      aggfunc=sum, fill_value=0,).reset_index()
        tmp_df = melted_df
    else:
        ind_lab = metadata.column_names_to_labels[ind_col]
        tmp_df[ind_lab] = [values_dict[ind_col][i] for i in tmp_df[ind_col]]  #!!
        df_for_total = tmp_df
        
    if total_only:
        pivot = tmp_df.groupby([ind_col, ind_lab])['w'].sum().reset_index()
        df_for_total = tmp_dataframe
        
        return post_process(pivot, ['w'], df_for_total)
            
    pivot = pd.pivot_table(tmp_df, values='w', index=[ind_col, ind_lab], 
                           columns=[col_lab], aggfunc=sum, fill_value=0,
                           ).reset_index()

    return post_process(pivot, pivot.columns[2:], df_for_total)


def get_single_stat(dat, ind_col, col_col, total_only=False, 
                    agg_type='mean'):
    tmp_dataframe, metadata, values_dict = dat
    tmp_df = tmp_dataframe.copy()
    ind_lab = metadata.column_names_to_labels[ind_col]
    tmp_df.rename(columns={ind_col: ind_lab}, inplace=True)
    if agg_type == 'mean':
        agg_func = lambda x: np.average(x, weights=tmp_df.loc[x.index, 'w'])
    elif agg_func == 'max':
        agg_func = lambda x: np.max(x)
    if total_only:
        agg_df = pd.DataFrame({'Q':[ind_lab, ''], 'label': ['TOTAL', agg_type],
            'Total': [tmp_df['w'].sum(), agg_func(tmp_df[ind_lab])]}).round(2)
        
        return agg_df, None
    else:
        col_lab = metadata.column_names_to_labels[col_col]
        tmp_df[col_lab] = [values_dict[col_col][i]
                           for i in tmp_df[col_col]]
        agg_df = pd.pivot_table(tmp_df, columns=[col_lab],
            aggfunc={ind_lab: agg_func}, fill_value=0).reset_index().round(2)
        
        totals = pd.pivot_table(tmp_df, values='w', columns=[col_lab], 
            aggfunc=sum, fill_value=0).reset_index().round(0)
        agg_df = pd.concat([totals, agg_df])
        agg_df['index'] = [ind_lab, '']
        agg_df.insert(1, 'label', ['TOTAL', agg_type])
        agg_df.rename(columns={'index': "Q"}, inplace=True)
        totals = totals.sum()
        multiindex = [(totals.index.name, totals.index[i]) 
                      for i in range(1, len(totals))]
        
        return agg_df, multiindex
    

def stats_gateway(tmp_dat, ind_q, val_q, conf, total_only=True):
    gateway_dict = {'single': {'multians': False}, 
                    'multi': {'multians': True}}
    ans_type = conf['keys'][ind_q]
    if ans_type in ['single', 'multi']:
        args = gateway_dict[ans_type]
        args['dat'] = tmp_dat
        args['ind_col'] = ind_q
        args['col_col'] = val_q
        args['total_only'] = total_only
        
        return get_pivot(**args)
    else:
        return get_single_stat(tmp_dat, ind_q, val_q,  total_only=total_only, 
                               agg_type=ans_type)


def get_sheet(df, sheet_conf):
    concat_list = list()
    col_ids = [('Total', '')]
    index_vals = list(sheet_conf['keys'].keys())
    for ind_q in index_vals:
        totals, _ = stats_gateway(df, ind_q, 'PAD', sheet_conf, 
                                  total_only=True)
        for val_q in sheet_conf['values']:
            val_df, ind = stats_gateway(df, ind_q, val_q, sheet_conf,
                                        total_only=False)
            totals = pd.merge(totals, val_df)
            if ind_q == index_vals[0]:
                col_ids.extend(ind)
        concat_list.append(totals)
        
    
    multiindex = pd.MultiIndex.from_tuples(col_ids)   
    result = pd.concat(concat_list)
    result = result.set_index(['Q', 'label'])
    result.columns = multiindex
    res_columns = ", ".join(list(dict.fromkeys([i[0] for i in col_ids])))
    logging.info(f"Calculated {res_columns} by {ind_q} " +
                 f"for sheet {sheet_conf['sheet_name']}")
    
    return result
    
    
def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    all_data = get_and_process_data(config)
    result = list()
    for sheet_num in config['sheets']:
        sheet_conf = config['sheets'][sheet_num]
        one_sheet = get_sheet(all_data, sheet_conf)
        result.append([one_sheet, sheet_conf['sheet_name']])

    with pd.ExcelWriter(config['output_file']) as writer:
        for sheet, name in result:
            sheet.to_excel(writer, sheet_name=name)
    logging.info(f"Saved to {config['output_file']}")

    
main()