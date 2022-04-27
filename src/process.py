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
        control_name = [[data_name, model_name, ['30'], ['0.2'], ['once-global', 'once-layer']]]
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
        control_name = [[data_name, model_name, ['30'], ['0.2'], ['lt-global', 'lt-layer']]]
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    mode = ['teacher', 'once', 'lt']
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
    # df_history = make_df_result(extracted_processed_result_history, 'history')
    # make_vis_by_dataset(df_exp)
    # make_vis_by_model(df_exp)
    make_vis_by_layer(df_exp)
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
            for i in range(len(q)):
                metric_name = 'SI-{}'.format(q[i])
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                si_i = []
                for m in range(len(base_result['sparsity_index'].si)):
                    si_i_m = make_si(base_result['sparsity_index'].si[m][i])
                    si_i.extend(si_i_m)
                processed_result_exp[metric_name]['exp'][exp_idx] = si_i
            if 'compression' in base_result:
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


def make_vis_by_dataset(df):
    mode = ['teacher', 'once', 'lt']
    data_all = [['MNIST', 'FashionMNIST'], ['CIFAR10', 'SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'MNIST': 'red', 'FashionMNIST': 'orange', 'CIFAR10': 'red', 'SVHN': 'orange'}
        linestyle_dict = {'MNIST': '-', 'FashionMNIST': '--', 'CIFAR10': '-', 'SVHN': '--'}
        z_color_dict = {'MNIST': 'blue', 'FashionMNIST': 'cyan', 'CIFAR10': 'blue', 'SVHN': 'cyan'}
        z_linestyle_dict = {'MNIST': '-.', 'FashionMNIST': ':', 'CIFAR10': '-.', 'SVHN': ':'}
        pivot_data_name_dict = {'MNIST': 'MNIST', 'FashionMNIST': 'MNIST', 'CIFAR10': 'CIFAR10', 'SVHN': 'CIFAR10'}
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        pivot_metric_names = ['Accuracy', 'Loss']
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
                    'SI' in metric_name and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, p = df_name_list[-2].split('-')
                for i in range(len(pivot_metric_names)):
                    pivot_metric_names_i = pivot_metric_names[i]
                    df_name_si = df_name
                    df_name_pivot_metric = '_'.join([*df_name_list[:-2], pivot_metric_names_i, stats])
                    si = df[df_name_si].iloc[0].to_numpy()
                    pivot_metric = df[df_name_pivot_metric].iloc[0].to_numpy()
                    teacher_df_name_si = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                    teacher_df_name_pivot_metric = '_'.join([*df_name_list[:2], pivot_metric_names_i, stats])
                    teacher_si = df[teacher_df_name_si].iloc[0].to_numpy()
                    teacher_pivot_metric = df[teacher_df_name_pivot_metric].iloc[0].to_numpy()
                    si = np.concatenate([teacher_si, si], axis=0)
                    si = si.reshape((int(prune_iters) + 1, -1))
                    pivot_metric = np.concatenate([teacher_pivot_metric, pivot_metric], axis=0)
                    for j in range(si.shape[-1]):
                        pivot_data_name = pivot_data_name_dict[df_name_list[0]]
                        fig_name = '_'.join([pivot_data_name, *df_name_list[1:5], p, pivot_metric_names_i, str(j)])
                        label = df_name_list[0]
                        fig[fig_name] = plt.figure(fig_name)
                        if fig_name not in AX1:
                            AX1[fig_name] = plt.subplot(111)
                            AX2[fig_name] = AX1[fig_name].twinx()
                        ax1 = AX1[fig_name]
                        ax2 = AX2[fig_name]
                        x = np.arange(int(prune_iters))
                        y = si[:-1, j]
                        z = make_z(pivot_metric, pivot_metric_names_i)
                        lns1 = ax1.plot(x, y, color=color_dict[label], linestyle=linestyle_dict[label],
                                        label='{}, SI'.format(label))
                        lns2 = ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                                        label='{}, PD'.format(label))
                        lns[fig_name].extend(lns1 + lns2)
                        ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                        ax1.set_ylabel('Sparsity Index (SI)', fontsize=fontsize['label'])
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
            dir_path = os.path.join(vis_path, 'dataset', *control[:-2])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_model(df):
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
        fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
        pivot_metric_names = ['Accuracy', 'Loss']
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
            if df_name_control_name in controls and len(df_name_list[:-2]) == 5 and 'SI' in metric_name and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, p = df_name_list[-2].split('-')
                for i in range(len(pivot_metric_names)):
                    pivot_metric_names_i = pivot_metric_names[i]
                    df_name_si = df_name
                    df_name_pivot_metric = '_'.join([*df_name_list[:-2], pivot_metric_names_i, stats])
                    si = df[df_name_si].iloc[0].to_numpy()
                    pivot_metric = df[df_name_pivot_metric].iloc[0].to_numpy()
                    teacher_df_name_si = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                    teacher_df_name_pivot_metric = '_'.join([*df_name_list[:2], pivot_metric_names_i, stats])
                    teacher_si = df[teacher_df_name_si].iloc[0].to_numpy()
                    teacher_pivot_metric = df[teacher_df_name_pivot_metric].iloc[0].to_numpy()
                    si = np.concatenate([teacher_si, si], axis=0)
                    si = si.reshape((int(prune_iters) + 1, -1))
                    pivot_metric = np.concatenate([teacher_pivot_metric, pivot_metric], axis=0)
                    for j in range(si.shape[-1]):
                        L = df_name_list[1].split('-')[-2]
                        fig_name = '_'.join([df_name_list[0], *df_name_list[2:5], p, pivot_metric_names_i, str(L), str(j)])
                        label = df_name_list[1]
                        fig[fig_name] = plt.figure(fig_name)
                        if fig_name not in AX1:
                            AX1[fig_name] = plt.subplot(111)
                            AX2[fig_name] = AX1[fig_name].twinx()
                        ax1 = AX1[fig_name]
                        ax2 = AX2[fig_name]
                        x = np.arange(int(prune_iters))
                        y = si[:-1, j]
                        z = make_z(pivot_metric, pivot_metric_names_i)
                        lns1 = ax1.plot(x, y, color=color_dict[label], linestyle=linestyle_dict[label],
                                        label='{}, SI'.format(label_dict[label]))
                        lns2 = ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                                        label='{}, PD'.format(label_dict[label]))
                        lns[fig_name].extend(lns1 + lns2)
                        ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                        ax1.set_ylabel('Sparsity Index (SI)', fontsize=fontsize['label'])
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
            dir_path = os.path.join(vis_path, 'model', *control[:-2])
            fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
            makedir_exist_ok(dir_path)
            plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close(fig_name)
    return


def make_vis_by_layer(df):
    mode = ['teacher', 'once', 'lt']
    data_all = [['MNIST'], ['FashionMNIST'], ['CIFAR10'], ['SVHN']]
    for i in range(len(data_all)):
        data = data_all[i]
        color_dict = {'0': 'red', '1': 'orange', '2': 'blue', '3': 'cyan', '4': 'green'}
        linestyle_dict = {'0': '-', '1': '--', '2': '-.', '3': ':', '4': (1, (5, 5))}
        z_color_dict = {'0': 'red', '1': 'orange', '2': 'blue', '3': 'cyan', '4': 'green'}
        z_linestyle_dict = {'0': '-', '1': '--', '2': '-.', '3': ':', '4': (1, (5, 5))}
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
                    'global' in df_name_control_name and 'SI' in metric_name and stats == 'mean':
                prune_iters = df_name_list[-5]
                _, p = df_name_list[-2].split('-')
                df_name_si = df_name
                df_name_cr = '_'.join([*df_name_list[:-2], 'CR', stats])
                si = df[df_name_si].iloc[0].to_numpy()
                cr = df[df_name_cr].iloc[0].to_numpy()
                teacher_df_name_si = '_'.join([*df_name_list[:2], *df_name_list[-2:]])
                teacher_si = df[teacher_df_name_si].iloc[0].to_numpy()
                si = np.concatenate([teacher_si, si], axis=0)
                si = si.reshape((int(prune_iters) + 1, -1))
                cr = cr.reshape((int(prune_iters) + 1, -1))
                for j in range(si.shape[-1]):
                    fig_name = '_'.join([*df_name_list[:5], p, 'CR'])
                    fig[fig_name] = plt.figure(fig_name)
                    if fig_name not in AX1:
                        AX1[fig_name] = fig[fig_name].add_subplot(121)
                        AX2[fig_name] = fig[fig_name].add_subplot(122)
                    ax1 = AX1[fig_name]
                    ax2 = AX2[fig_name]
                    label = str(j)
                    x = np.arange(int(prune_iters))
                    y = si[:-1, j]
                    ax1.plot(x, y, color=color_dict[label], linestyle=linestyle_dict[label],
                             label='$\ell={}$'.format(j + 1))
                    ax1.legend(loc='upper left', fontsize=fontsize['legend'])
                    ax1.set_xlabel('Iteration', fontsize=fontsize['label'])
                    ax1.set_ylabel('Sparsity Index (SI)', fontsize=fontsize['label'])
                    ax1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                    z = cr[1:, j]
                    ax2.plot(x, z, color=z_color_dict[label], linestyle=z_linestyle_dict[label],
                             label='$\ell={}$'.format(j + 1))
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
            dir_path = os.path.join(vis_path, 'layer', *control[:-2])
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
        cr_i = mask.float().mean().item()
        cr.append(cr_i)
    return cr


def make_z(pivot_metric, pivot_metric_name):
    if pivot_metric_name in ['Accuracy']:
        z = np.abs(np.minimum(pivot_metric[1:] - pivot_metric[0], 0)) / pivot_metric[0]
    elif pivot_metric_name in ['Loss']:
        z = np.abs(np.maximum(pivot_metric[1:] - pivot_metric[0], 0)) / pivot_metric[0]
    else:
        raise ValueError('Not valid pivot metric name')
    return z


if __name__ == '__main__':
    main()
