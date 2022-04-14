import os
import itertools
import json
import torch
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok, make_layerwise_sparsity_index
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
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


def make_control_list(mode, data, model):
    if mode == 'teacher':
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
    elif mode == 'lt':
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['128', '256'], ['1'], ['1', '2', '3', '4'], ['sigmoid', 'relu']]
            # model_name = [['mlp'], ['128'], ['1'], ['1'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            raise ValueError('Not valid model')
        control_name = [[data_name, model_name, ['50']]]
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    data = ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
    # data = ['MNIST']
    controls = []
    for data_i in data:
        controls += make_control_list('lt', data_i, 'mlp')
    processed_result_exp, processed_result_history = process_result(controls)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_result(extracted_processed_result_exp, 'exp')
    df_history = make_df_result(extracted_processed_result_history, 'history')
    make_vis(df_exp, 'exp')
    make_vis(df_history, 'history')
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
            for k in base_result['logger']['test'].history:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].history[k]
            q = base_result['sparsity_index']['test'].q
            for i in range(len(q)):
                metric_name = 'SI-{}'.format(q[i])
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                si_i = []
                for m in range(len(base_result['sparsity_index']['test'].si)):
                    si_i_m = make_si(base_result['sparsity_index']['test'].si[m][i])
                    si_i.extend(si_i_m)
                processed_result_exp[metric_name]['exp'][exp_idx] = si_i
            metric_name = 'CR'
            if metric_name not in processed_result_exp:
                processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            cr = []
            for m in range(len(base_result['compression'].mask)):
                cr_m = make_cr(base_result['compression'].mask[m])
                cr.extend(cr_m)
            processed_result_exp[metric_name]['exp'][exp_idx] = cr
            for k in base_result['logger']['train'].history:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_history:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
            q = base_result['sparsity_index']['train'].q
            for i in range(len(q)):
                metric_name = 'SI-{}'.format(q[i])
                if metric_name not in processed_result_history:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                si_i = []
                for m in range(len(base_result['sparsity_index']['train'].si)):
                    si_i_m = make_si(base_result['sparsity_index']['train'].si[m][i])
                    si_i.extend(si_i_m)
                processed_result_history[metric_name]['history'][exp_idx] = si_i
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
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
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


def make_df_result(extracted_processed_result, mode_name):
    df = defaultdict(list)
    for exp_name in extracted_processed_result:
        for k in extracted_processed_result[exp_name]:
            df_name = '{}_{}'.format(exp_name, k)
            index_name = [0]
            df[df_name].append(
                pd.DataFrame(data=extracted_processed_result[exp_name][k].reshape(1, -1), index=index_name))
    # startrow = 0
    # writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode_name), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
    #     df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
    #     writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
    #     startrow = startrow + len(df[df_name].index) + 3
    # writer.save()
    return df


def make_vis(df, mode_name):
    xlabel_dict = {'exp': 'Iteration', 'history': 'Epoch'}
    ylabel_dict = {'Loss': 'Loss', 'Accuracy': 'Accuracy', 'SI': 'Sparsity Index', 'CR': 'Compression Ratio'}
    color_dict = {'MNIST': 'red', 'FashionMNIST': 'orange', 'CIFAR10': 'blue', 'CIFAR100': 'green', 'SVHN': 'cyan'}
    linestyle_dict = {'MNIST': '-', 'FashionMNIST': '--', 'CIFAR10': '-.', 'CIFAR100': ':','SVHN': (0, (1, 5))}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    loc_dict = {'Loss': 'lower right', 'Accuracy': 'lower right', 'SI': 'lower right', 'CR': 'upper right'}
    fig = {}
    num_iters = None
    for df_name in df:
        df_name_list = df_name.split('_')
        data_name = df_name_list[0]
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        if stat == 'std':
            continue
        if 'SI' in metric_name:
            metric_name, p = metric_name.split('-')
            label = data_name
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            y = df[df_name].iloc[0].to_numpy()
            y_err = df[df_name_std].iloc[0].to_numpy()
            y = y.reshape((num_iters, -1))
            y_err = y_err.reshape((num_iters, -1))
            for i in range(y.shape[-1]):
                y_i, y_err_i = y[:, i], y_err[:, i]
                fig_name = '_'.join(df_name_list[1:-1] + [str(i)])
                fig[fig_name] = plt.figure(fig_name)
                x_i = np.arange(len(y_i))
                plt.plot(x_i, y_i, color=color_dict[label], linestyle=linestyle_dict[label], label=label)
                plt.fill_between(x_i, (y_i - y_err_i), (y_i + y_err_i), color='r', alpha=.1)
                plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                plt.xlabel(xlabel_dict[mode_name], fontsize=fontsize['label'])
                plt.ylabel(ylabel_dict[metric_name], fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        elif 'CR' in metric_name:
            label = data_name
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            y = df[df_name].iloc[0].to_numpy()
            y_err = df[df_name_std].iloc[0].to_numpy()
            y = y.reshape((num_iters, -1))
            y_err = y_err.reshape((num_iters, -1))
            for i in range(y.shape[-1]):
                y_i, y_err_i = y[:, i], y_err[:, i]
                fig_name = '_'.join(df_name_list[1:-1] + ['CR-{}'.format(str(i))])
                fig[fig_name] = plt.figure(fig_name)
                x_i = np.arange(len(y_i))
                plt.plot(x_i, y_i, color=color_dict[label], linestyle=linestyle_dict[label], label=label)
                plt.fill_between(x_i, (y_i - y_err_i), (y_i + y_err_i), color='r', alpha=.1)
                plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                plt.xlabel(xlabel_dict[mode_name], fontsize=fontsize['label'])
                plt.ylabel(ylabel_dict[metric_name], fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        else:
            label = data_name
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            y = df[df_name].iloc[0].to_numpy()
            y_err = df[df_name_std].iloc[0].to_numpy()
            x = np.arange(len(y))
            if num_iters is None:
                num_iters = len(y)
            fig_name = '_'.join(df_name_list[1:-1])
            fig[fig_name] = plt.figure(fig_name)
            plt.plot(x, y, color=color_dict[label], linestyle=linestyle_dict[label], label=label)
            plt.fill_between(x, (y - y_err), (y + y_err), color='r', alpha=.1)
            plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
            plt.xlabel(xlabel_dict[mode_name], fontsize=fontsize['label'])
            plt.ylabel(ylabel_dict[metric_name], fontsize=fontsize['label'])
            plt.xticks(fontsize=fontsize['ticks'])
            plt.yticks(fontsize=fontsize['ticks'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        dir_path = os.path.join(vis_path, mode_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_si(input):
    si = []
    for name, param in input.items():
        valid_param = param[~param.isnan()]
        si.append(valid_param.mean())
    return si


def make_cr(input):
    cr = []
    for name, param in input.items():
        mask = param.view(-1)
        cr.append(mask.float().mean().item())
    return cr


if __name__ == '__main__':
    main()
