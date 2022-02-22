import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok, make_layerwise_sparsity_index
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
si_path = './output/sparsity_index'
save_format = 'png'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(data, model):
    if data == 'MLP':
        data_name_r = [['MLP'], ['r'], ['500'], ['64'], ['128', '256'], ['1'], ['1', '2', '3', '4'],
                       ['sigmoid', 'relu'], ['1'], ['1.0'], ['0', '0.5']]
        data_name_r = list(itertools.product(*data_name_r))
        control_name_r = []
        for i in range(len(data_name_r)):
            model_name = ['mlp'] + list(data_name_r[i])[4:8]
            control_name_r_r_i = '-'.join(list(data_name_r[i])) + '_' + '-'.join(model_name)
            control_name_r.append(control_name_r_r_i)
        data_name_c = [['MLP'], ['c'], ['500'], ['64'], ['128', '256'], ['1'], ['1', '2', '3', '4'],
                       ['sigmoid', 'relu'], ['10'], ['0.0'], ['0', '0.5']]
        data_name_c = list(itertools.product(*data_name_c))
        control_name_c = []
        for i in range(len(data_name_c)):
            model_name = ['mlp'] + list(data_name_c[i])[4:8]
            control_name_c_i = '-'.join(list(data_name_c[i])) + '_' + '-'.join(model_name)
            control_name_c.append(control_name_c_i)
        control_name = [[control_name_r + control_name_c]]
    else:
        if data == 'Blob':
            data_name = ['Blob-500-64-10-1.0']
        elif data == 'Friedman':
            data_name = ['Friedman-500-64-1.0']
        else:
            data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['128', '256'], ['1'], ['1', '2', '3', '4'], ['sigmoid', 'relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            raise ValueError('Not valid model')
        control_name = [[data_name, model_name]]
    controls = make_controls(control_name)
    return controls


def main():
    data = ['Blob', 'Friedman', 'MLP', 'MNIST']
    controls = []
    for data_i in data:
        controls += make_control_list(data_i, 'mlp')
    # processed_result_exp, processed_result_history = process_result(controls)
    # save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    # save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    # extracted_processed_result_exp = {}
    # extracted_processed_result_history = {}
    # extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    # extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    # df_exp = make_df_result_exp(extracted_processed_result_exp)
    # df_history = make_df_result_history(extracted_processed_result_history)
    processed_si_exp = process_si(controls)
    save(processed_si_exp, os.path.join(si_path, 'processed_si_exp.pt'))
    extracted_processed_si_exp = {}
    extract_processed_si(extracted_processed_si_exp, processed_si_exp, [])
    df_si = make_df_si_exp(extracted_processed_si_exp)
    make_vis(df_si)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for k in base_result['logger']['test'].mean:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df_result_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        if len(control) == 2:
            data_name, model_name = control
            df_name = data_name
            index_name = [model_name]
            df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_exp.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_df_result_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        if len(control) == 2:
            data_name, model_name = control
            for k in extracted_processed_result_history[exp_name]:
                df_name = '{}_{}'.format(data_name, k)
                index_name = [model_name]
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_history.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def process_si(controls):
    processed_si_exp = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_si(list(control), model_tag, processed_si_exp)
    summarize_si(processed_si_exp)
    return processed_si_exp


def extract_si(control, model_tag, processed_si_exp):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_si_path_i = os.path.join(si_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_si_path_i):
            base_result = load(base_si_path_i)
            sparsity_index = make_layerwise_sparsity_index(base_result['sparsity_index'])
            metric_names = ['si-mean', 'si-std']
            for metric_name in metric_names:
                if metric_name not in processed_si_exp:
                    processed_si_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                stat = metric_name.split('-')[-1]
                processed_si_exp[metric_name]['exp'][exp_idx] = sparsity_index[stat]
        else:
            print('Missing {}'.format(base_si_path_i))
    else:
        if control[1] not in processed_si_exp:
            processed_si_exp[control[1]] = {}
        extract_si([control[0]] + control[2:], model_tag, processed_si_exp[control[1]])
    return


def summarize_si(processed_si):
    if 'exp' in processed_si:
        pivot = 'exp'
        processed_si[pivot] = np.stack(processed_si[pivot], axis=0)
        processed_si['mean'] = np.mean(processed_si[pivot], axis=0)
        processed_si['std'] = np.std(processed_si[pivot], axis=0)
        processed_si['max'] = np.max(processed_si[pivot], axis=0)
        processed_si['min'] = np.min(processed_si[pivot], axis=0)
        processed_si['argmax'] = np.argmax(processed_si[pivot], axis=0)
        processed_si['argmin'] = np.argmin(processed_si[pivot], axis=0)
        processed_si[pivot] = processed_si[pivot].tolist()
    else:
        for k, v in processed_si.items():
            summarize_si(v)
        return
    return


def extract_processed_si(extracted_processed_si, processed_si, control):
    if 'exp' in processed_si:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_si:
            extracted_processed_si[exp_name] = defaultdict()
        extracted_processed_si[exp_name]['{}_mean'.format(metric_name)] = processed_si['mean']
        extracted_processed_si[exp_name]['{}_std'.format(metric_name)] = processed_si['std']
    else:
        for k, v in processed_si.items():
            extract_processed_si(extracted_processed_si, v, control + [k])
    return


def make_df_si_exp(extracted_processed_si_exp):
    p = np.arange(1, 10) / 10
    df = defaultdict(list)
    for exp_name in extracted_processed_si_exp:
        control = exp_name.split('_')
        if len(control) == 2:
            data_name, model_name = control
            for k in extracted_processed_si_exp[exp_name]:
                df_name = '{}_{}_{}'.format(data_name, model_name, k)
                si_exp_k = extracted_processed_si_exp[exp_name][k]
                for i in range(len(si_exp_k)):
                    index_name = [str(p[i])]
                    df[df_name].append(
                        pd.DataFrame(data=si_exp_k[i].reshape(1, -1), index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/si_exp.xlsx'.format(si_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis(df):
    color = {'0.1': 'red', '0.2': 'orange', '0.3': 'green', '0.4': 'blue', '0.5': 'purple', '0.6': 'brown',
             '0.7': 'cyan', '0.8': 'gray', '0.9': 'black'}
    linestyle = {'0.1': '-', '0.2': '--', '0.3': '-.', '0.4': (0, (1, 1)), '0.5': (0, (1, 5)), '0.6': (0, (1, 10)),
                 '0.7': (0, (5, 1)), '0.8': (0, (5, 5)), '0.9': (0, (5, 10))}
    marker = {'0.1': 'o', '0.2': 's', '0.3': 'v', '0.4': '^', '0.5': '<', '0.6': '>', '0.7': 'p', '0.8': 'P',
              '0.9': '*'}
    loc_dict = {'si-mean': 'lower right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    fig = {}
    for df_name in df:
        df_name_list = df_name.split('_')
        if len(df_name_list) == 4:
            data_name, model_name, metric_name, stat = df_name.split('_')
            if stat == 'std' or 'std' in metric_name:
                continue
            df_name_std = '_'.join([data_name, model_name, 'si-std', 'mean'])
            fig_name = '_'.join([data_name, model_name])
            fig[fig_name] = plt.figure(fig_name)
            for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name_std].iterrows()):
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                x = np.arange(len(y))
                tag = index
                plt.plot(x, y, color=color[tag], linestyle=linestyle[tag], label=float(tag), marker=marker[tag])
                plt.fill_between(x, (y - yerr), (y + yerr), color=color[tag], alpha=.1)
                plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                plt.xlabel('L', fontsize=fontsize['label'])
                plt.ylabel('Sparsity Index', fontsize=fontsize['label'])
                plt.xticks(x, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
