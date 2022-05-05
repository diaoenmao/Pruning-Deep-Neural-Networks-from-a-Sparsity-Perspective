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
y_scale = 'linear'


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
        control_name = [[data_name, model_name, ['30'], ['0.2'], ['once-neuron', 'once-layer', 'once-global']]]
        # control_name = [[data_name, model_name, ['30'], ['0.2'], ['once-global']]]
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
        control_name = [[data_name, model_name, ['30'], ['0.2'], ['lt-neuron', 'lt-layer', 'lt-global']]]
        # control_name = [[data_name, model_name, ['30'], ['0.2'], ['lt-global']]]
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
        control_name = [[data_name, model_name, ['30'], ['si-0.5-0'], ['si-neuron', 'si-layer', 'si-global']]]
        # control_name = [[data_name, model_name, ['30'], ['si-0.5-0'], ['si-global']]]
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    # mode = ['teacher', 'once', 'lt', 'si']
    # data = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    mode = ['teacher', 'once', 'lt', 'si']
    data = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    controls = []
    for mode_i in mode:
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
    # make_vis_by_dataset(df_exp, 'SI')
    # make_vis_by_model(df_exp, 'SI')
    # make_vis_by_layer(df_exp, 'SI')
    # make_vis_by_dataset(df_exp, 'SIe')
    # make_vis_by_model(df_exp, 'SIe')
    # make_vis_by_layer(df_exp, 'SIe')
    # make_vis_by_dataset(df_exp, 'Norm')
    # make_vis_by_model(df_exp, 'Norm')
    # make_vis_by_layer(df_exp, 'Norm')
    make_vis_by_prune(df_exp, 'Accuracy')
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
            # q = base_result['sparsity_index'].q
            # for k in base_result['sparsity_index'].si:
            #     for i in range(len(q)):
            #         metric_name = 'SI-{}-{}'.format(k, q[i])
            #         if metric_name not in processed_result_exp:
            #             processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            #         si_k_i = []
            #         for m in range(len(base_result['sparsity_index'].si[k])):
            #             si_k_i_m = make_y(base_result['sparsity_index'].si[k][m][i], k)
            #             si_k_i.extend(si_k_i_m)
            #         processed_result_exp[metric_name]['exp'][exp_idx] = si_k_i
            #     for i in range(len(q)):
            #         metric_name = 'SIe-{}-{}'.format(k, q[i])
            #         if metric_name not in processed_result_exp:
            #             processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            #         sie_k_i = []
            #         for m in range(len(base_result['sparsity_index'].sie[k])):
            #             sie_k_i_m = make_y(base_result['sparsity_index'].sie[k][m][i], k)
            #             sie_k_i.extend(sie_k_i_m)
            #         processed_result_exp[metric_name]['exp'][exp_idx] = sie_k_i
            # q = base_result['norm'].q
            # for k in base_result['norm'].norm:
            #     for i in range(len(q)):
            #         metric_name = 'Norm-{}-{}'.format(k, q[i])
            #         if metric_name not in processed_result_exp:
            #             processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            #         norm_k_i = []
            #         for m in range(len(base_result['norm'].norm[k])):
            #             norm_k_i_m = make_y(base_result['norm'].norm[k][m][i], k)
            #             norm_k_i.extend(norm_k_i_m)
            #         processed_result_exp[metric_name]['exp'][exp_idx] = norm_k_i
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


def make_vis_by_dataset(df, y_name):
    mode = ['teacher', 'once', 'lt']
    data_all = [['MNIST', 'FashionMNIST'], ['CIFAR10', 'SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'MNIST': 'red', 'FashionMNIST': 'orange', 'CIFAR10': 'red', 'SVHN': 'orange'}
        linestyle_dict = {'MNIST': '-', 'FashionMNIST': '--', 'CIFAR10': '-', 'SVHN': '--'}
        z_color_dict = {'MNIST': 'blue', 'FashionMNIST': 'cyan', 'CIFAR10': 'blue', 'SVHN': 'cyan'}
        z_linestyle_dict = {'MNIST': '-.', 'FashionMNIST': ':', 'CIFAR10': '-.', 'SVHN': ':'}
        pivot_data_name_dict = {'MNIST': 'MNIST', 'FashionMNIST': 'MNIST', 'CIFAR10': 'CIFAR10', 'SVHN': 'CIFAR10'}
        y_name_dict = {'SI': 'Sparsity Index (SI)', 'SIe': 'Sparsity Index (SI)', 'Norm': 'Norm'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        pivot_metric_names = ['Accuracy', 'Loss', 'Loss-Teacher']
        controls = []
        for mode_i in mode:
            for data_i in data:
                controls += make_control_list(mode_i, data_i, 'mlp')
        for i in range(len(controls)):
            controls[i] = controls[i][1]
        fig = {}
        AX1, AX2 = {}, {}
        lns = defaultdict(list)
        for df_name in df:
            df_name_list = df_name.split('_')
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    y_name in metric_name and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, type, q = df_name_list[-2].split('-')
                for i in range(len(pivot_metric_names)):
                    pivot_metric_names_i = pivot_metric_names[i]
                    df_name_y = df_name
                    df_name_pivot_metric = '_'.join([*df_name_list[:-2], pivot_metric_names_i, stats])
                    y = df[df_name_y].iloc[0].to_numpy()
                    pivot_metric = df[df_name_pivot_metric].iloc[0].to_numpy()
                    teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                    teacher_df_name_pivot_metric = '_'.join([*df_name_list[:2], pivot_metric_names_i, stats])
                    teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                    teacher_pivot_metric = df[teacher_df_name_pivot_metric].iloc[0].to_numpy()
                    y = np.concatenate([teacher_y, y], axis=0)
                    y = y.reshape((int(prune_iters) + 1, -1))
                    pivot_metric = np.concatenate([teacher_pivot_metric, pivot_metric], axis=0)
                    for j in range(y.shape[-1]):
                        pivot_data_name = pivot_data_name_dict[df_name_list[0]]
                        layer_tag = str(j)
                        fig_name = '_'.join([pivot_data_name, *df_name_list[1:5], type, q, pivot_metric_names_i,
                                             layer_tag])
                        label = df_name_list[0]
                        fig[fig_name] = plt.figure(fig_name)
                        if fig_name not in AX1:
                            AX1[fig_name] = plt.subplot(111)
                            AX2[fig_name] = AX1[fig_name].twinx()
                        ax1 = AX1[fig_name]
                        ax2 = AX2[fig_name]
                        x = np.arange(int(prune_iters))
                        y_j = make_y_figure(y[:, j], y_name)
                        z = make_z_figure(pivot_metric, pivot_metric_names_i)
                        lns1 = ax1.plot(x, y_j, color=color_dict[label], linestyle=linestyle_dict[label],
                                        label='{}, {}'.format(label, y_name))
                        lns2 = ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                                        label='{}, PD ({})'.format(label, pivot_metric_names_i))
                        lns[fig_name].extend(lns1 + lns2)
                        ax1.set_yscale(y_scale)
                        ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                        ax1.set_ylabel(y_name_dict[y_name], fontsize=fontsize['label'])
                        ax2.set_ylabel('Performance Degradation (PD)', fontsize=fontsize['label'])
                        ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                        ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                        ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            lns[fig_name][1], lns[fig_name][2] = lns[fig_name][2], lns[fig_name][1]
            labs = [l.get_label() for l in lns[fig_name]]
            AX1[fig_name].legend(lns[fig_name], labs, loc='upper left', fontsize=fontsize['legend'])
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'dataset', *control[:-1])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_model(df, y_name):
    mode = ['teacher', 'once', 'lt']
    data_all = [['MNIST'], ['FashionMNIST'], ['CIFAR10'], ['SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'mlp-128-1-2-relu': 'red', 'mlp-256-1-2-relu': 'orange', 'mlp-128-1-4-relu': 'red',
                      'mlp-256-1-4-relu': 'orange'}
        linestyle_dict = {'mlp-128-1-2-relu': '-', 'mlp-256-1-2-relu': '--', 'mlp-128-1-4-relu': '-',
                          'mlp-256-1-4-relu': '--'}
        z_color_dict = {'mlp-128-1-2-relu': 'blue', 'mlp-256-1-2-relu': 'cyan', 'mlp-128-1-4-relu': 'blue',
                        'mlp-256-1-4-relu': 'cyan'}
        z_linestyle_dict = {'mlp-128-1-2-relu': '-.', 'mlp-256-1-2-relu': ':', 'mlp-128-1-4-relu': '-.',
                            'mlp-256-1-4-relu': ':'}
        label_dict = {'mlp-128-1-2-relu': '$N=128$, $L=2$', 'mlp-256-1-2-relu': '$N=256$, $L=2$',
                      'mlp-128-1-4-relu': '$N=128$, $L=4$', 'mlp-256-1-4-relu': '$N=256$, $L=4$'}
        y_name_dict = {'SI': 'Sparsity Index (SI)', 'SIe': 'Sparsity Index (SI)', 'Norm': 'Norm'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        pivot_metric_names = ['Accuracy', 'Loss', 'Loss-Teacher']
        controls = []
        for mode_i in mode:
            for data_i in data:
                controls += make_control_list(mode_i, data_i, 'mlp')
        for i in range(len(controls)):
            controls[i] = controls[i][1]
        fig = {}
        AX1, AX2 = {}, {}
        lns = defaultdict(list)
        for df_name in df:
            df_name_list = df_name.split('_')
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    y_name in metric_name and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, type, q = df_name_list[-2].split('-')
                for i in range(len(pivot_metric_names)):
                    pivot_metric_names_i = pivot_metric_names[i]
                    df_name_y = df_name
                    df_name_pivot_metric = '_'.join([*df_name_list[:-2], pivot_metric_names_i, stats])
                    y = df[df_name_y].iloc[0].to_numpy()
                    pivot_metric = df[df_name_pivot_metric].iloc[0].to_numpy()
                    teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                    teacher_df_name_pivot_metric = '_'.join([*df_name_list[:2], pivot_metric_names_i, stats])
                    teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                    teacher_pivot_metric = df[teacher_df_name_pivot_metric].iloc[0].to_numpy()
                    y = np.concatenate([teacher_y, y], axis=0)
                    y = y.reshape((int(prune_iters) + 1, -1))
                    pivot_metric = np.concatenate([teacher_pivot_metric, pivot_metric], axis=0)
                    for j in range(y.shape[-1]):
                        L = df_name_list[1].split('-')[-2]
                        layer_tag = str(j)
                        fig_name = '_'.join([df_name_list[0], *df_name_list[2:5], type, q, pivot_metric_names_i,
                                             str(L), layer_tag])
                        label = df_name_list[1]
                        fig[fig_name] = plt.figure(fig_name)
                        if fig_name not in AX1:
                            AX1[fig_name] = plt.subplot(111)
                            AX2[fig_name] = AX1[fig_name].twinx()
                        ax1 = AX1[fig_name]
                        ax2 = AX2[fig_name]
                        x = np.arange(int(prune_iters))
                        y_j = make_y_figure(y[:, j], y_name)
                        z = make_z_figure(pivot_metric, pivot_metric_names_i)
                        lns1 = ax1.plot(x, y_j, color=color_dict[label], linestyle=linestyle_dict[label],
                                        label='{}, {}'.format(label_dict[label], y_name))
                        lns2 = ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                                        label='{}, PD ({})'.format(label_dict[label], pivot_metric_names_i))
                        lns[fig_name].extend(lns1 + lns2)
                        ax1.set_yscale(y_scale)
                        ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                        ax1.set_ylabel(y_name_dict[y_name], fontsize=fontsize['label'])
                        ax2.set_ylabel('Performance Degradation (PD)', fontsize=fontsize['label'])
                        ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                        ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                        ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            lns[fig_name][1], lns[fig_name][2] = lns[fig_name][2], lns[fig_name][1]
            labs = [l.get_label() for l in lns[fig_name]]
            AX1[fig_name].legend(lns[fig_name], labs, loc='upper left', fontsize=fontsize['legend'])
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'model', *control[:-1])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_layer(df, y_name):
    mode = ['teacher', 'once', 'lt']
    data_all = [['MNIST'], ['FashionMNIST'], ['CIFAR10'], ['SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'0': 'red', '1': 'orange', '2': 'blue', '3': 'cyan', '4': 'green', 'all': 'black'}
        linestyle_dict = {'0': '-', '1': '--', '2': '-.', '3': ':', '4': (0, (1, 10)), 'all': (0, (5, 10))}
        z_color_dict = {'0': 'red', '1': 'orange', '2': 'blue', '3': 'cyan', '4': 'green', 'all': 'black'}
        z_linestyle_dict = {'0': '-', '1': '--', '2': '-.', '3': ':', '4': (0, (1, 10)), 'all': (0, (5, 10))}
        y_name_dict = {'SI': 'Sparsity Index (SI)', 'SIe': 'Sparsity Index (SI)', 'Norm': 'Norm'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
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
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    'global' in df_name_control_name and y_name in metric_name and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, type, q = df_name_list[-2].split('-')
                df_name_y = df_name
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR-{}'.format(type), stats])
                y = df[df_name_y].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                y = np.concatenate([teacher_y, y], axis=0)
                y = y.reshape((int(prune_iters) + 1, -1))
                cr = cr.reshape((int(prune_iters) + 1, -1))
                for j in range(y.shape[-1]):
                    fig_name = '_'.join([*df_name_list[:5], type, q, 'CR'])
                    fig[fig_name] = plt.figure(fig_name)
                    if fig_name not in AX1:
                        AX1[fig_name] = fig[fig_name].add_subplot(121)
                        AX2[fig_name] = fig[fig_name].add_subplot(122)
                    ax1 = AX1[fig_name]
                    ax2 = AX2[fig_name]
                    label = str(j)
                    x = np.arange(int(prune_iters) + 1)
                    y_j = y[:, j]
                    ax1.plot(x, y_j, color=color_dict[label], linestyle=linestyle_dict[label],
                             label='$\ell={}$'.format(label))
                    ax1.set_yscale(y_scale)
                    ax1.legend(loc='upper left', fontsize=fontsize['legend'])
                    ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                    ax1.set_ylabel(y_name_dict[y_name], fontsize=fontsize['label'])
                    ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                    z = cr[:, j]
                    ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                             label='$\ell={}$'.format(label))
                    ax2.legend(loc='upper left', fontsize=fontsize['legend'])
                    ax2.set_xlabel('Iteration', fontsize=fontsize['label'])
                    ax2.set_ylabel('Compression Ratio (CR)', fontsize=fontsize['label'])
                    ax2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            AX2[fig_name].grid(linestyle='--', linewidth='0.5')
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
        z_color_dict = {'once': 'red', 'lt': 'orange', 'si': 'blue'}
        z_linestyle_dict = {'once': '-', 'lt': '--', 'si': '-.'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
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
            print(df_name_list)
            df_name_control_name = '_'.join(df_name_list[:-2])
            metric_name, stats = df_name_list[-2:]
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and \
                    y_name in metric_name and stats == 'mean':
                prune_mode, type = df_name_list[-3].split('-')
                df_name_y = df_name
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR-global', stats])
                y = df[df_name_y].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                teacher_df_name_y = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_y = df[teacher_df_name_y].iloc[0].to_numpy()
                y = np.concatenate([teacher_y, y], axis=0)
                fig_name = '_'.join([*df_name_list[:3], type, metric_name])
                fig[fig_name] = plt.figure(fig_name)
                if fig_name not in AX1:
                    AX1[fig_name] = fig[fig_name].add_subplot(121)
                    AX2[fig_name] = fig[fig_name].add_subplot(122)
                ax1 = AX1[fig_name]
                ax2 = AX2[fig_name]
                label = prune_mode
                x = np.arange(len(y))
                ax1.plot(x, y, color=color_dict[label], linestyle=linestyle_dict[label], label=label)
                ax1.set_yscale(y_scale)
                ax1.legend(loc='upper left', fontsize=fontsize['legend'])
                ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax1.set_ylabel(y_name, fontsize=fontsize['label'])
                ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                z = cr
                ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                         label=label)
                ax2.legend(loc='upper left', fontsize=fontsize['legend'])
                ax2.set_xlabel('Iteration', fontsize=fontsize['label'])
                ax2.set_ylabel('Compression Ratio (CR)', fontsize=fontsize['label'])
                ax2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
        for fig_name in fig:
            fig[fig_name] = plt.figure(fig_name)
            AX1[fig_name].grid(linestyle='--', linewidth='0.5')
            AX2[fig_name].grid(linestyle='--', linewidth='0.5')
            control = fig_name.split('_')
            dir_path = os.path.join(vis_path, y_name, 'prune', *control[:-1])
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
            cr_i = mask_i.mean(dim=-1).item()
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


def make_y_figure(y, y_name):
    if y_name in ['SI']:
        y = y[:-1]
    elif y_name in ['SIe']:
        y = -np.diff(y)
    else:
        raise ValueError('Not valid pivot metric name')
    return y


def make_z_figure(pivot_metric, pivot_metric_name):
    if pivot_metric_name in ['Accuracy']:
        z = (pivot_metric[0] - pivot_metric[1:]) / pivot_metric[0]
    elif pivot_metric_name in ['Loss']:
        z = (pivot_metric[1:] - pivot_metric[0]) / pivot_metric[0]
    elif pivot_metric_name in ['Loss-Teacher']:
        z = (pivot_metric[1:] - pivot_metric[0])
    else:
        raise ValueError('Not valid pivot metric name')
    return z


if __name__ == '__main__':
    main()
