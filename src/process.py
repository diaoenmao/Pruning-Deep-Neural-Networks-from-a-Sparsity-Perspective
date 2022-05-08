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
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['128', '256'], ['1'], ['2', '4'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            model_name = [model]
        control_name = [[data_name, model_name]]
        controls = make_controls(control_name)
    elif mode == 'once':
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['128', '256'], ['1'], ['2', '4'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            raise ValueError('Not valid model')
        # control_name = [[data_name, model_name, ['30'], ['0.2'], ['once-neuron', 'once-layer', 'once-global']]]
        control_name = [[data_name, model_name, ['30'], ['0.2'], ['once-global']]]
        controls = make_controls(control_name)
    elif mode == 'lt':
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['128', '256'], ['1'], ['2', '4'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            raise ValueError('Not valid model')
        # control_name = [[data_name, model_name, ['30'], ['0.2'], ['lt-neuron', 'lt-layer', 'lt-global']]]
        control_name = [[data_name, model_name, ['30'], ['0.2'], ['lt-global']]]
        controls = make_controls(control_name)
    elif mode == 'si':
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['128', '256'], ['1'], ['2', '4'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            raise ValueError('Not valid model')
        # control_name = [[data_name, model_name, ['30'], ['si-0.5-0'], ['si-neuron', 'si-layer', 'si-global']]]
        control_name = [[data_name, model_name, ['30'], ['si-0.5-0'], ['si-global']]]
        controls = make_controls(control_name)
    elif mode == 'si-q':
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['256'], ['1'], ['4'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            model_name = [model]
        # control_name = [[data_name, model_name, ['30'], ['si-0.2-0', 'si-0.4-0', 'si-0.6-0', 'si-0.8-0'],
        #                  ['si-neuron', 'si-layer', 'si-global']]]
        control_name = [[data_name, model_name, ['30'], ['si-0.2-0', 'si-0.4-0', 'si-0.6-0', 'si-0.8-0'],
                         ['si-global']]]
        controls = make_controls(control_name)
    elif mode == 'si-eta':
        data_name = [data]
        if model == 'mlp':
            model_name = [['mlp'], ['256'], ['1'], ['4'], ['relu']]
            model_name = list(itertools.product(*model_name))
            for i in range(len(model_name)):
                model_name[i] = '-'.join(model_name[i])
        else:
            model_name = [model]
        # control_name = [[data_name, model_name, ['30'], ['si-0.5-0.01', 'si-0.5-0.1', 'si-0.5-1', 'si-0.5-10'],
        #                  ['si-neuron', 'si-layer', 'si-global']]]
        # control_name = [[data_name, model_name, ['30'], ['si-0.5-0.01', 'si-0.5-0.1', 'si-0.5-1', 'si-0.5-10'],
        #                  ['si-global']]]
        control_name = [[data_name, model_name, ['30'], ['si-0.5-1', 'si-0.5-2', 'si-0.5-3', 'si-0.5-4', 'si-0.5-5'],
                         ['si-global']]]
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    mode = ['teacher', 'once', 'lt', 'si', 'si-q', 'si-eta']
    # data = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    # mode = ['teacher', 'once', 'lt', 'si']
    data = ['MNIST', 'FashionMNIST']
    controls = []
    for mode_i in mode:
        if mode_i in ['teacher', 'once', 'lt', 'si']:
            data = ['MNIST', 'FashionMNIST']
        elif mode_i in ['si-q', 'si-eta']:
            data = ['MNIST']
        for data_i in data:
            controls += make_control_list(mode_i, data_i, 'mlp')
    processed_result_exp, processed_result_history = process_result(controls)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_result(extracted_processed_result_exp, 'exp')
    df_history = make_df_result(extracted_processed_result_history, 'history')
    # make_vis_by_layer(df_exp, 'SI')
    # make_vis_by_prune(df_exp, 'Loss')
    # make_vis_by_prune(df_exp, 'Loss-Teacher')
    # make_vis_by_prune(df_exp, 'Accuracy')
    # make_vis_by_si_q(df_exp, 'Accuracy')
    make_vis_by_si_eta(df_exp, 'Accuracy')
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
            q = base_result['sparsity_index'].q
            # q = [0.5]
            for k in base_result['sparsity_index'].si:
                for i in range(len(q)):
                    metric_name = 'SI-{}-{}'.format(k, q[i])
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    si_k_i = []
                    for m in range(len(base_result['sparsity_index'].si[k])):
                        si_k_i_m = make_y(base_result['sparsity_index'].si[k][m][i], k)
                        si_k_i.extend(si_k_i_m)
                    processed_result_exp[metric_name]['exp'][exp_idx] = si_k_i
                for i in range(len(q)):
                    metric_name = 'SIe-{}-{}'.format(k, q[i])
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    sie_k_i = []
                    for m in range(len(base_result['sparsity_index'].sie[k])):
                        sie_k_i_m = make_y(base_result['sparsity_index'].sie[k][m][i], k)
                        sie_k_i.extend(sie_k_i_m)
                    processed_result_exp[metric_name]['exp'][exp_idx] = sie_k_i
            if 'compression' in base_result:
                mode = ['neuron', 'layer', 'global']
                for k in mode:
                    metric_name = 'CR-{}'.format(k)
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    cr = []
                    for m in range(len(base_result['compression'].mask)):
                        cr_m = make_cr(base_result['compression'].mask[m], k)
                        cr.extend(cr_m)
                    processed_result_exp[metric_name]['exp'][exp_idx] = cr
                for k in mode:
                    metric_name = 'MD-{}'.format(k)
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    d = []
                    for m in range(len(base_result['compression'].mask)):
                        d_m = make_d(base_result['compression'].mask[m], k)
                        d.extend(d_m)
                    md = make_md(d)
                    processed_result_exp[metric_name]['exp'][exp_idx] = md
            for k in base_result['logger']['train'].history:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_history:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
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
    startrow = 0
    writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode_name), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis_by_layer(df, y_name):
    mode = ['teacher', 'once', 'lt', 'si']
    data_all = [['MNIST'], ['FashionMNIST'], ['CIFAR10'], ['SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'1': 'red', '2': 'orange', '3': 'blue', '4': 'cyan', '5': 'green'}
        linestyle_dict = {'1': '-', '2': '--', '3': '-.', '4': ':', '5': (0, (1, 10))}
        y_name_dict = {'SI': 'Sparsity Index', 'SIe': 'Sparsity Index'}
        label_loc_dict = {'SI': 'upper left', 'SIe': 'upper left', 'CR': 'upper right'}
        marker_dict = {'1': 'o', '2': 's', '3': 'p', '4': '*', '5': 'h'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        figsize = (10, 4)
        capsize = 3
        capthick = 3
        controls = []
        for mode_i in mode:
            for data_i in data:
                controls += make_control_list(mode_i, data_i, 'mlp')
        for i in range(len(controls)):
            controls[i] = controls[i][1]
        fig = {}
        AX1, AX2 = {}, {}
        for df_name in df:
            df_name_list = df_name.split('_')
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            metric_mode = metric_name.split('-')[0]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    'global' in df_name_control_name and 'layer' in metric_name and \
                    y_name in metric_mode and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, type, q = df_name_list[-2].split('-')
                df_name_y = df_name
                df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR-{}'.format(type), stats])
                df_name_cr_std = '_'.join([*df_name_list[:-2], 'CR-{}'.format(type), 'std'])
                y = df[df_name_y].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                y_std = df[df_name_y_std].iloc[0].to_numpy()
                cr_std = df[df_name_cr_std].iloc[0].to_numpy()
                teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_df_name_y_std = '_'.join([*df_name_list[:2], df_name_list[-2], 'std'])
                teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                teacher_y_std = df[teacher_df_name_y_std].iloc[0].to_numpy()
                y = np.concatenate([teacher_y, y], axis=0)
                y_std = np.concatenate([teacher_y_std, y_std], axis=0)
                y = y.reshape((int(prune_iters) + 1, -1))
                cr = cr.reshape((int(prune_iters) + 1, -1))
                y_std = y_std.reshape((int(prune_iters) + 1, -1))
                cr_std = cr_std.reshape((int(prune_iters) + 1, -1))
                for j in range(y.shape[-1]):
                    fig_name = '_'.join([*df_name_list[:5], type, q, metric_mode])
                    fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                    if fig_name not in AX1:
                        AX1[fig_name] = fig[fig_name].add_subplot(121)
                        AX2[fig_name] = fig[fig_name].add_subplot(122)
                    ax1 = AX1[fig_name]
                    ax2 = AX2[fig_name]
                    label = str(j + 1)
                    x = np.arange(int(prune_iters) + 1)
                    y_j = y[:, j]
                    y_std_j = y_std[:, j]
                    ax1.errorbar(x, y_j, yerr=y_std_j, color=color_dict[label], linestyle=linestyle_dict[label],
                                 label='$\ell={}$'.format(label), marker=marker_dict[label],
                                 capsize=capsize, capthick=capthick)
                    ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                    ax1.set_ylabel(y_name_dict[y_name], fontsize=fontsize['label'])
                    ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax1.legend(loc=label_loc_dict[y_name], fontsize=fontsize['legend'])
                    z = cr[:, j]
                    z_std = cr_std[:, j]
                    ax2.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                                 label='$\ell={}$'.format(label), marker=marker_dict[label],
                                 capsize=capsize, capthick=capthick)
                    ax2.set_xlabel('Iteration', fontsize=fontsize['label'])
                    ax2.set_ylabel('Percent of Remaining Weights', fontsize=fontsize['label'])
                    ax2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            AX2[fig_name].grid(linestyle='--', linewidth='0.5')
            fig[fig_name].tight_layout()
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'layer', *control[:-1])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_prune(df, y_name):
    mode = ['teacher', 'once', 'lt', 'si']
    data_all = [['MNIST'], ['FashionMNIST'], ['CIFAR10'], ['SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'once': 'red', 'lt': 'orange', 'si': 'blue'}
        linestyle_dict = {'once': '-', 'lt': '--', 'si': '-.'}
        label_dict = {'once': 'One Shot', 'lt': 'Lottery Ticket', 'si': 'Sparse Index'}
        marker_dict = {'once': 'o', 'lt': 's', 'si': 'p'}
        label_loc_dict = {'Accuracy': 'lower left', 'Loss': 'upper left', 'Loss-Teacher': 'upper left',
                          'CR': 'upper right', 'Sie': 'upper left'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        figsize = (20, 4)
        capsize = 3
        capthick = 3
        controls = []
        for mode_i in mode:
            for data_i in data:
                controls += make_control_list(mode_i, data_i, 'mlp')
        for i in range(len(controls)):
            controls[i] = controls[i][1]
        fig = {}
        AX1, AX2, AX3, AX4 = {}, {}, {}, {}
        for df_name in df:
            df_name_list = df_name.split('_')
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    y_name in metric_name and stats == 'mean':
                prune_mode, type = df_name_list[-3].split('-')
                df_name_y = df_name
                df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR-global', stats])
                df_name_cr_std = '_'.join([*df_name_list[:-2], 'CR-global', 'std'])
                df_name_sie = '_'.join([*df_name_list[:-2], 'SIe-global-0.5', stats])
                df_name_sie_std = '_'.join([*df_name_list[:-2], 'SIe-global-0.5', 'std'])
                df_name_md = '_'.join([*df_name_list[:-2], 'MD-global', stats])
                df_name_md_std = '_'.join([*df_name_list[:-2], 'MD-global', 'std'])
                y = df[df_name_y].iloc[0].to_numpy()
                y_std = df[df_name_y_std].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                cr_std = df[df_name_cr_std].iloc[0].to_numpy()
                sie = df[df_name_sie].iloc[0].to_numpy()
                sie_std = df[df_name_sie_std].iloc[0].to_numpy()
                md = df[df_name_md].iloc[0].to_numpy()
                md_std = df[df_name_md_std].iloc[0].to_numpy()
                teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_df_name_y_std = '_'.join([*df_name_list[:2], df_name_list[-2], 'std'])
                teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                teacher_y_std = df[teacher_df_name_y_std].iloc[0].to_numpy()
                teacher_df_name_sie = '_'.join([*df_name_list[:2], 'SIe-global-0.5', stats])
                teacher_df_name_sie_std = '_'.join([*df_name_list[:2], 'SIe-global-0.5', 'std'])
                teacher_sie = df[teacher_df_name_sie].iloc[0].to_numpy()
                teacher_sie_std = df[teacher_df_name_sie_std].iloc[0].to_numpy()
                y = np.concatenate([teacher_y, y], axis=0)
                y_std = np.concatenate([teacher_y_std, y_std], axis=0)
                sie = np.concatenate([teacher_sie, sie], axis=0)
                sie_std = np.concatenate([teacher_sie_std, sie_std], axis=0)
                fig_name = '_'.join([*df_name_list[:3], type, metric_name])
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                if fig_name not in AX1:
                    AX1[fig_name] = fig[fig_name].add_subplot(141)
                    AX2[fig_name] = fig[fig_name].add_subplot(142)
                    AX3[fig_name] = fig[fig_name].add_subplot(143)
                    AX4[fig_name] = fig[fig_name].add_subplot(144)
                ax1 = AX1[fig_name]
                ax2 = AX2[fig_name]
                ax3 = AX3[fig_name]
                ax4 = AX4[fig_name]
                label = prune_mode
                x = np.arange(len(y))
                ax1.errorbar(x, y, yerr=y_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
                ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax1.set_ylabel(y_name, fontsize=fontsize['label'])
                ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = cr
                z_std = cr_std
                ax2.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
                ax2.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax2.set_ylabel('Percent of Remaining Weights', fontsize=fontsize['label'])
                ax2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.legend(loc=label_loc_dict['CR'], fontsize=fontsize['legend'])
                z = sie
                z_std = sie_std
                ax3.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
                ax3.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax3.set_ylabel('Sparsity Index', fontsize=fontsize['label'])
                ax3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = md
                z_std = md_std
                ax4.errorbar(np.arange(1, len(z) + 1), z, yerr=z_std, color=color_dict[label],
                             linestyle=linestyle_dict[label], label=label_dict[label], marker=marker_dict[label],
                             capsize=capsize, capthick=capthick)
                ax4.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax4.set_ylabel('$\\frac{m}{d}$', fontsize=fontsize['label'])
                ax4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax4.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            AX2[fig_name].grid(linestyle='--', linewidth='0.5')
            AX3[fig_name].grid(linestyle='--', linewidth='0.5')
            AX4[fig_name].grid(linestyle='--', linewidth='0.5')
            fig[fig_name].tight_layout()
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'prune', *control[:-1])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_si_q(df, y_name):
    mode = ['teacher', 'si', 'si-q']
    data_all = [['MNIST'], ['CIFAR10']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'0.2': 'red', '0.4': 'orange', '0.5': 'black', '0.6': 'blue', '0.8': 'cyan'}
        linestyle_dict = {'0.2': '-', '0.4': '--', '0.5': '-.', '0.6': ':', '0.8': (0, (1, 10))}
        marker_dict = {'0.2': 'o', '0.4': 's', '0.5': 'p', '0.6': 'D', '0.8': 'H'}
        label_loc_dict = {'Accuracy': 'lower left', 'Loss': 'upper left', 'Loss-Teacher': 'upper left',
                          'CR': 'upper right', 'Sie': 'upper left'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        figsize = (20, 4)
        capsize = 3
        capthick = 3
        controls = []
        for mode_i in mode:
            for data_i in data:
                controls += make_control_list(mode_i, data_i, 'mlp')
        for i in range(len(controls)):
            controls[i] = controls[i][1]
        fig = {}
        AX1, AX2, AX3, AX4 = {}, {}, {}, {}
        for df_name in df:
            df_name_list = df_name.split('_')
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    y_name in metric_name and stats == 'mean':
                _, q, eta_m = df_name_list[-4].split('-')
                prune_mode, type = df_name_list[-3].split('-')
                df_name_y = df_name
                df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR-global', stats])
                df_name_cr_std = '_'.join([*df_name_list[:-2], 'CR-global', 'std'])
                df_name_sie = '_'.join([*df_name_list[:-2], 'SIe-global-0.5', stats])
                df_name_sie_std = '_'.join([*df_name_list[:-2], 'SIe-global-0.5', 'std'])
                df_name_md = '_'.join([*df_name_list[:-2], 'MD-global', stats])
                df_name_md_std = '_'.join([*df_name_list[:-2], 'MD-global', 'std'])
                y = df[df_name_y].iloc[0].to_numpy()
                y_std = df[df_name_y_std].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                cr_std = df[df_name_cr_std].iloc[0].to_numpy()
                sie = df[df_name_sie].iloc[0].to_numpy()
                sie_std = df[df_name_sie_std].iloc[0].to_numpy()
                md = df[df_name_md].iloc[0].to_numpy()
                md_std = df[df_name_md_std].iloc[0].to_numpy()
                teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_df_name_y_std = '_'.join([*df_name_list[:2], df_name_list[-2], 'std'])
                teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                teacher_y_std = df[teacher_df_name_y_std].iloc[0].to_numpy()
                teacher_df_name_sie = '_'.join([*df_name_list[:2], 'SIe-global-0.5', stats])
                teacher_df_name_sie_std = '_'.join([*df_name_list[:2], 'SIe-global-0.5', 'std'])
                teacher_sie = df[teacher_df_name_sie].iloc[0].to_numpy()
                teacher_sie_std = df[teacher_df_name_sie_std].iloc[0].to_numpy()
                y = np.concatenate([teacher_y, y], axis=0)
                y_std = np.concatenate([teacher_y_std, y_std], axis=0)
                sie = np.concatenate([teacher_sie, sie], axis=0)
                sie_std = np.concatenate([teacher_sie_std, sie_std], axis=0)
                fig_name = '_'.join([*df_name_list[:3], type, metric_name])
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                if fig_name not in AX1:
                    AX1[fig_name] = fig[fig_name].add_subplot(141)
                    AX2[fig_name] = fig[fig_name].add_subplot(142)
                    AX3[fig_name] = fig[fig_name].add_subplot(143)
                    AX4[fig_name] = fig[fig_name].add_subplot(144)
                ax1 = AX1[fig_name]
                ax2 = AX2[fig_name]
                ax3 = AX3[fig_name]
                ax4 = AX4[fig_name]
                label = q
                x = np.arange(len(y))
                ax1.errorbar(x, y, yerr=y_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             label='$q={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             capthick=capthick)
                ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax1.set_ylabel(y_name, fontsize=fontsize['label'])
                ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = cr
                z_std = cr_std
                ax2.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             label='$q={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             capthick=capthick)
                ax2.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax2.set_ylabel('Percent of Remaining Weights', fontsize=fontsize['label'])
                ax2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = sie
                z_std = sie_std
                ax3.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             label='$q={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             capthick=capthick)
                ax3.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax3.set_ylabel('Sparsity Index', fontsize=fontsize['label'])
                ax3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = md
                z_std = md_std
                ax4.errorbar(np.arange(1, len(z) + 1), z, yerr=z_std, color=color_dict[label],
                             linestyle=linestyle_dict[label], label='$q={}$'.format(label), marker=marker_dict[label],
                             capsize=capsize, capthick=capthick)
                ax4.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax4.set_ylabel('$\\frac{m}{d}$', fontsize=fontsize['label'])
                ax4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax4.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            AX2[fig_name].grid(linestyle='--', linewidth='0.5')
            AX3[fig_name].grid(linestyle='--', linewidth='0.5')
            AX4[fig_name].grid(linestyle='--', linewidth='0.5')
            handles, labels = AX2[fig_name].get_legend_handles_labels()
            if len(handles) > 1:
                AX2[fig_name].legend([handles[1], handles[2], handles[0], handles[3], handles[4]],
                                     [labels[1], labels[2], labels[0], labels[3], labels[4]], loc=label_loc_dict['CR'],
                                     fontsize=fontsize['legend'])
            fig[fig_name].tight_layout()
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'si_q', *control[:-1])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_si_eta(df, y_name):
    # mode = ['teacher', 'si', 'si-eta']
    mode = ['teacher', 'si-eta']
    data_all = [['MNIST'], ['CIFAR10']]
    for i in range(len(data_all)):
        data = data_all[i]
        # color_dict = {'0': 'black', '0.01': 'red', '0.1': 'orange', '1': 'blue', '10': 'cyan'}
        # linestyle_dict = {'0': '-', '0.01': '--', '0.1': '-.', '1': ':', '10': (0, (1, 10))}
        # marker_dict = {'0': 'o', '0.01': 's', '0.1': 'p', '1': 'D', '10': 'H'}

        color_dict = {'1': 'black', '2': 'red', '3': 'orange', '4': 'blue', '5': 'cyan'}
        linestyle_dict = {'1': '-', '2': '--', '3': '-.', '4': ':', '5': (0, (1, 10))}
        marker_dict = {'1': 'o', '2': 's', '3': 'p', '4': 'D', '5': 'H'}

        label_loc_dict = {'Accuracy': 'lower left', 'Loss': 'upper left', 'Loss-Teacher': 'upper left',
                          'CR': 'upper right', 'Sie': 'upper left'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        figsize = (20, 4)
        capsize = 3
        capthick = 3
        controls = []
        for mode_i in mode:
            for data_i in data:
                controls += make_control_list(mode_i, data_i, 'mlp')
        for i in range(len(controls)):
            controls[i] = controls[i][1]
        fig = {}
        AX1, AX2, AX3, AX4 = {}, {}, {}, {}
        for df_name in df:
            df_name_list = df_name.split('_')
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    y_name in metric_name and stats == 'mean':
                _, q, eta_m = df_name_list[-4].split('-')
                prune_mode, type = df_name_list[-3].split('-')
                df_name_y = df_name
                df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR-global', stats])
                df_name_cr_std = '_'.join([*df_name_list[:-2], 'CR-global', 'std'])
                df_name_sie = '_'.join([*df_name_list[:-2], 'SIe-global-0.5', stats])
                df_name_sie_std = '_'.join([*df_name_list[:-2], 'SIe-global-0.5', 'std'])
                df_name_md = '_'.join([*df_name_list[:-2], 'MD-global', stats])
                df_name_md_std = '_'.join([*df_name_list[:-2], 'MD-global', 'std'])
                y = df[df_name_y].iloc[0].to_numpy()
                y_std = df[df_name_y_std].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                cr_std = df[df_name_cr_std].iloc[0].to_numpy()
                sie = df[df_name_sie].iloc[0].to_numpy()
                sie_std = df[df_name_sie_std].iloc[0].to_numpy()
                md = df[df_name_md].iloc[0].to_numpy()
                md_std = df[df_name_md_std].iloc[0].to_numpy()
                teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_df_name_y_std = '_'.join([*df_name_list[:2], df_name_list[-2], 'std'])
                teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                teacher_y_std = df[teacher_df_name_y_std].iloc[0].to_numpy()
                teacher_df_name_sie = '_'.join([*df_name_list[:2], 'SIe-global-0.5', stats])
                teacher_df_name_sie_std = '_'.join([*df_name_list[:2], 'SIe-global-0.5', 'std'])
                teacher_sie = df[teacher_df_name_sie].iloc[0].to_numpy()
                teacher_sie_std = df[teacher_df_name_sie_std].iloc[0].to_numpy()
                y = np.concatenate([teacher_y, y], axis=0)
                y_std = np.concatenate([teacher_y_std, y_std], axis=0)
                sie = np.concatenate([teacher_sie, sie], axis=0)
                sie_std = np.concatenate([teacher_sie_std, sie_std], axis=0)
                fig_name = '_'.join([*df_name_list[:3], type, metric_name])
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                if fig_name not in AX1:
                    AX1[fig_name] = fig[fig_name].add_subplot(141)
                    AX2[fig_name] = fig[fig_name].add_subplot(142)
                    AX3[fig_name] = fig[fig_name].add_subplot(143)
                    AX4[fig_name] = fig[fig_name].add_subplot(144)
                ax1 = AX1[fig_name]
                ax2 = AX2[fig_name]
                ax3 = AX3[fig_name]
                ax4 = AX4[fig_name]
                label = eta_m
                x = np.arange(len(y))
                ax1.errorbar(x, y, yerr=y_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             # label='$\eta_m={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             label='$c={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             capthick=capthick)
                ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax1.set_ylabel(y_name, fontsize=fontsize['label'])
                ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = cr
                z_std = cr_std
                ax2.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             # label='$\eta_m={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             label='$c={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             capthick=capthick)
                ax2.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax2.set_ylabel('Percent of Remaining Weights', fontsize=fontsize['label'])
                ax2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.legend(loc=label_loc_dict['CR'], fontsize=fontsize['legend'])
                z = sie
                z_std = sie_std
                ax3.errorbar(x, z, yerr=z_std, color=color_dict[label], linestyle=linestyle_dict[label],
                             # label='$\eta_m={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             label='$c={}$'.format(label), marker=marker_dict[label], capsize=capsize,
                             capthick=capthick)
                ax3.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax3.set_ylabel('Sparsity Index', fontsize=fontsize['label'])
                ax3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = md
                z_std = md_std
                ax4.errorbar(np.arange(1, len(z) + 1), z, yerr=z_std, color=color_dict[label],
                             # linestyle=linestyle_dict[label], label='$\eta_m={}$'.format(label),
                             linestyle=linestyle_dict[label], label='$c={}$'.format(label),
                             marker=marker_dict[label], capsize=capsize, capthick=capthick)
                ax4.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax4.set_ylabel('$\\frac{m}{d}$', fontsize=fontsize['label'])
                ax4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax4.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            AX2[fig_name].grid(linestyle='--', linewidth='0.5')
            AX3[fig_name].grid(linestyle='--', linewidth='0.5')
            AX4[fig_name].grid(linestyle='--', linewidth='0.5')
            fig[fig_name].tight_layout()
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'si_eta', *control[:-1])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_y(input, mode):
    if mode == 'neuron':
        y = []
        for name, param in input.items():
            valid_param = param[~param.isnan()]
            y.append(valid_param.mean())
    elif mode == 'layer':
        y = []
        for name, param in input.items():
            y.append(param)
    elif mode == 'global':
        y = [input]
    else:
        raise ValueError('Not valid mode')
    return y


def make_cr(input, mode):
    if mode == 'neuron':
        cr = []
        for name, param in input.items():
            mask_i = param.view(-1).float()
            cr_i = mask_i.mean().item()
            cr.append(cr_i)
    elif mode == 'layer':
        cr = []
        for name, param in input.items():
            mask_i = param.view(-1).float()
            cr_i = mask_i.mean().item()
            cr.append(cr_i)
    elif mode == 'global':
        cr = []
        param_all = []
        for name, param in input.items():
            param_all.append(param.view(-1))
        param_all = torch.cat(param_all, dim=0)
        mask = param_all.view(-1).float()
        cr.append(mask.mean().item())
    else:
        raise ValueError('Not valid mode')
    return cr


def make_d(input, mode):
    if mode == 'neuron':
        d = []
        for name, param in input.items():
            mask_i = param.view(-1).float()
            d_i = mask_i.sum().item()
            d.append(d_i)
    elif mode == 'layer':
        d = []
        for name, param in input.items():
            mask_i = param.view(-1).float()
            d_i = mask_i.sum().item()
            d.append(d_i)
    elif mode == 'global':
        d = []
        param_all = []
        for name, param in input.items():
            param_all.append(param.view(-1))
        param_all = torch.cat(param_all, dim=0)
        mask = param_all.view(-1).float()
        d.append(mask.sum().item())
    else:
        raise ValueError('Not valid mode')
    return d


def make_md(d):
    m = -np.diff(d)
    md = m / d[:-1]
    return md


if __name__ == '__main__':
    main()
