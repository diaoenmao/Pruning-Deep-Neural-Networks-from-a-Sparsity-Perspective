import os
import itertools
import numpy as np
import torch
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
result_path = os.path.join('output', 'result')
save_format = 'pdf'
vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300
write = False


def make_control(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_controls(mode, model):
    if model == 0:
        model_names = ['linear', 'mlp']
        prune_iters = ['30']
    elif model == 1:
        model_names = ['cnn']
        prune_iters = ['30']
    elif model == 2:
        model_names = ['resnet18']
        prune_iters = ['15']
    elif model == 3:
        model_names = ['wresnet28x8']
        prune_iters = ['15']
    elif model == 4:
        model_names = ['resnet50']
        prune_iters = ['15']
    else:
        raise ValueError('Not valid model')
    if mode == 'os':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['os-0.2']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'lt':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['lt-0.2']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'si':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['si-0.5-1-0-1', 'si-1-2-0-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'scope':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['neuron', 'layer']
        prune_mode = ['si-0.5-1-0-1', 'si-1-2-0-1', 'lt-0.2', 'os-0.2']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'si-p':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['si-0.2-1-0-1', 'si-0.4-1-0-1', 'si-0.6-1-0-1', 'si-0.8-1-0-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'si-q':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['si-1-1.2-0-1', 'si-1-1.4-0-1', 'si-1-1.6-0-1', 'si-1-1.8-0-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'si-eta_m':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['si-0.5-1-0.001-1', 'si-0.5-1-0.01-1', 'si-0.5-1-0.1-1', 'si-0.5-1-1-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    elif mode == 'si-gamma':
        data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
        model_names = model_names
        prune_iters = prune_iters
        prune_scope = ['global']
        prune_mode = ['si-0.5-1-0-3', 'si-0.5-1-0-5', 'si-0.5-1-0-7', 'si-0.5-1-0-9']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_control(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    model = 3
    # modes = ['si', 'lt', 'os', 'scope', 'si-p', 'si-q', 'si-eta_m', 'si-gamma']
    # modes = ['si', 'lt', 'os']
    modes = ['si', 'lt', 'os', 'scope']
    # modes = ['si', 'lt', 'os', 'scope', 'si-p', 'si-q']
    # modes = ['si', 'si-eta_m', 'si-gamma']
    controls = []
    for mode in modes:
        controls += make_controls(mode, model)
    processed_result = process_result(controls)
    df_history = make_df(processed_result, 'history')
    make_vis_by_prune(df_history)
    make_vis_by_pruned(df_history)
    make_vis_by_layer(df_history)
    make_vis_by_ratio(df_history)
    make_vis_by_si_layer(df_history)
    # make_vis_by_p(df_history)
    # make_vis_by_q(df_history)
    make_vis_by_pq(df_history)
    # make_vis_by_eta_m(df_history)
    # make_vis_by_gamma(df_history)
    return


def tree():
    return defaultdict(tree)


def process_result(controls):
    result = tree()
    for control in controls:
        model_tag = '_'.join(control)
        gather_result(list(control), model_tag, result)
    summarize_result(None, result)
    processed_result = tree()
    extract_result(processed_result, result, [])
    print('Processing finished')
    return processed_result


def gather_result(control, model_tag, processed_result):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            mask_state_dict = base_result['mask_state_dict']
            sparsity_index = base_result['sparsity_index']
            sparsity_index_pruned = base_result['sparsity_index_pruned']
            p, q = base_result['cfg']['p'], base_result['cfg']['q']
            p = [round(x.item(), 1) for x in p]
            q = [round(x.item(), 1) for x in q]
            for split in base_result['logger']:
                for metric_name in base_result['logger'][split].history:
                    metric_name_ = metric_name.replace('test', 'test-pruned') if split == 'test-pruned' else metric_name
                    processed_result[split][metric_name_]['history'][exp_idx] = base_result['logger'][split].history[
                        metric_name]
            processed_result['test']['test/pr-global-global']['history'][exp_idx] = [[] for _ in
                                                                                     range(len(mask_state_dict))]
            for name, mask in mask_state_dict[0].items():
                processed_result['test']['test/pr-neuron-{}'.format(name)]['history'][exp_idx] = []
                processed_result['test']['test/pr-layer-{}'.format(name)]['history'][exp_idx] = []
            for name, mask in mask_state_dict[0].items():
                for iter in range(len(mask_state_dict)):
                    processed_result['test']['test/pr-neuron-{}'.format(name)]['history'][exp_idx].append(
                        mask_state_dict[iter][name].float().mean(dim=1))
                    processed_result['test']['test/pr-layer-{}'.format(name)]['history'][exp_idx].append(
                        mask_state_dict[iter][name].float().mean())
                    processed_result['test']['test/pr-global-global']['history'][exp_idx][iter].append(
                        mask_state_dict[iter][name].view(-1))
                processed_result['test']['test/pr-neuron-{}'.format(name)]['history'][exp_idx] = torch.stack(
                    processed_result['test']['test/pr-neuron-{}'.format(name)]['history'][exp_idx], dim=0).numpy()
                processed_result['test']['test/pr-layer-{}'.format(name)]['history'][exp_idx] = torch.stack(
                    processed_result['test']['test/pr-layer-{}'.format(name)]['history'][exp_idx], dim=0).numpy()
            for iter in range(len(mask_state_dict)):
                processed_result['test']['test/pr-global-global']['history'][exp_idx][iter] = torch.cat(
                    processed_result['test']['test/pr-global-global']['history'][exp_idx][iter], dim=0).float().mean()
            processed_result['test']['test/pr-global-global']['history'][exp_idx] = torch.stack(
                processed_result['test']['test/pr-global-global']['history'][exp_idx], dim=0).numpy()

            processed_result['test']['test/p']['history'][exp_idx] = p
            processed_result['test']['test/q']['history'][exp_idx] = q
            for prune_scope in sparsity_index.si:
                for name in sparsity_index.si[prune_scope][0]:
                    for i in range(len(p)):
                        for j in range(len(q)):
                            metric_name = 'test/si-{}-{}-{}-{}'.format(prune_scope, name, p[i], q[j])
                            processed_result['test'][metric_name]['history'][exp_idx] = []
                            for iter in range(len(sparsity_index.si[prune_scope])):
                                processed_result['test'][metric_name]['history'][
                                    exp_idx].append(sparsity_index.si[prune_scope][iter][name][i, j])
                            processed_result['test'][metric_name]['history'][exp_idx] = np.stack(
                                processed_result['test'][metric_name]['history'][exp_idx], axis=0)
            for prune_scope in sparsity_index.gini:
                for name in sparsity_index.gini[prune_scope][0]:
                    metric_name = 'test/gini-{}-{}'.format(prune_scope, name)
                    processed_result['test'][metric_name]['history'][exp_idx] = []
                    for iter in range(len(sparsity_index.gini[prune_scope])):
                        processed_result['test'][metric_name][
                            'history'][exp_idx].append(sparsity_index.gini[prune_scope][iter][name])
                    processed_result['test'][metric_name]['history'][exp_idx] = np.stack(
                        processed_result['test'][metric_name]['history'][exp_idx], axis=0)
            for prune_scope in sparsity_index_pruned.si:
                if len(sparsity_index_pruned.si[prune_scope]) == 0:
                    break
                for name in sparsity_index_pruned.si[prune_scope][0]:
                    for i in range(len(p)):
                        for j in range(len(q)):
                            metric_name = 'test-pruned/si-{}-{}-{}-{}'.format(prune_scope, name, p[i], q[j])
                            processed_result['test-pruned'][metric_name]['history'][exp_idx] = []
                            for iter in range(len(sparsity_index_pruned.si[prune_scope])):
                                processed_result['test-pruned'][metric_name]['history'][
                                    exp_idx].append(sparsity_index_pruned.si[prune_scope][iter][name][i, j])
                            processed_result['test-pruned'][metric_name]['history'][exp_idx] = np.stack(
                                processed_result['test-pruned'][metric_name]['history'][exp_idx], axis=0)
            for prune_scope in sparsity_index_pruned.gini:
                if len(sparsity_index_pruned.gini[prune_scope]) == 0:
                    break
                for name in sparsity_index_pruned.gini[prune_scope][0]:
                    metric_name = 'test-pruned/gini-{}-{}'.format(prune_scope, name)
                    processed_result['test-pruned'][metric_name]['history'][exp_idx] = []
                    for iter in range(len(sparsity_index_pruned.gini[prune_scope])):
                        processed_result['test-pruned'][metric_name][
                            'history'][exp_idx].append(sparsity_index_pruned.gini[prune_scope][iter][name])
                    processed_result['test-pruned'][metric_name]['history'][exp_idx] = np.stack(
                        processed_result['test-pruned'][metric_name]['history'][exp_idx], axis=0)
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def summarize_result(key, value):
    if key in ['mean', 'history']:
        filtered_value = [v for v in value.values() if len(v) not in [1, 200]]
        # s = [len(v) for v in filtered_value]
        # print(s)
        # value['summary']['value'] = np.stack(list(value.values()), axis=0)
        value['summary']['value'] = np.stack(filtered_value, axis=0)
        value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
        value['summary']['std'] = np.std(value['summary']['value'], axis=0)
        value['summary']['max'] = np.max(value['summary']['value'], axis=0)
        value['summary']['min'] = np.min(value['summary']['value'], axis=0)
        value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
        value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
        value['summary']['value'] = value['summary']['value'].tolist()
    else:
        for k, v in value.items():
            summarize_result(k, v)
        return
    return


def extract_result(extracted_processed_result, processed_result, control):
    def extract(split, metric_name, mode):
        output = False
        if split == 'test':
            if metric_name in ['test/Accuracy'] or \
                    'test/si' in metric_name or 'test/gini' in metric_name or 'test/pr' in metric_name:
                if mode == 'history':
                    output = True
        elif split == 'test-pruned':
            if metric_name in ['test-pruned/Accuracy'] or 'test-pruned/si' in metric_name or 'test-pruned/gini':
                if mode == 'history':
                    output = True
        return output

    if 'summary' in processed_result:
        control_name, split, metric_name, mode = control
        if not extract(split, metric_name, mode):
            return
        stats = ['mean', 'std']
        for stat in stats:
            exp_name = '_'.join([control_name, metric_name, stat])
            extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            extract_result(extracted_processed_result, v, control + [k])
    return


def make_df(processed_result, mode):
    def filter(exp_name_list):
        metric_name = exp_name_list[-2]
        output = False
        if 'Accuracy' in metric_name or 'si-global' in metric_name or 'gini-global' in metric_name or \
                'pr-global' in metric_name or 'pr-layer' in metric_name or 'si-layer' in metric_name:
            output = True
        return output

    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        exp_name_list = exp_name.split('_')
        if not filter(exp_name_list):
            continue
        df_name = '_'.join([*exp_name_list])
        index_name = [1]
        df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
    if write:
        startrow = 0
        writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode), engine='xlsxwriter')
        for df_name in df:
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
        writer.save()
    return df


def make_vis_by_prune(df_history):
    label_dict = {'si-0.5-1-0-1': 'SAP ($p=0.5$, $q=1.0$)',
                  'si-1-2-0-1': 'SAP ($p=1.0$, $q=2.0$)', 'lt-0.2': 'Lottery Ticket ($P=0.2$)',
                  'os-0.2': 'One Shot ($P=0.2$)'}
    color_dict = {'si-0.5-1-0-1': 'blue', 'si-1-2-0-1': 'cyan', 'lt-0.2': 'red', 'os-0.2': 'orange'}
    linestyle_dict = {'si-0.5-1-0-1': '-', 'si-1-2-0-1': '--', 'lt-0.2': '-.', 'os-0.2': ':'}
    marker_dict = {'si-0.5-1-0-1': 'o', 'si-1-2-0-1': 's', 'lt-0.2': 'p', 'os-0.2': 'D'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    capsize = None
    capthick = None
    pivot_p = '0.5'
    pivot_q = '1.0'
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        elif prune_mode_list[0] in ['lt', 'os']:
            pivot_ = ['lt-0.2', 'os-0.2']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            df_name_y = '_'.join([*df_name_list[:-1], stat])
            df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_pr = '_'.join([*df_name_list[:-2], 'test/pr-global-global', stat])
            df_name_pr_std = '_'.join([*df_name_list[:-2], 'test/pr-global-global', 'std'])
            df_name_si = '_'.join([*df_name_list[:-2], 'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_std = '_'.join([*df_name_list[:-2],
                                       'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])
            df_name_gini = '_'.join([*df_name_list[:-2], 'test/gini-global-global', stat])
            df_name_gini_std = '_'.join([*df_name_list[:-2], 'test/gini-global-global', 'std'])

            y = df_history[df_name_y].iloc[0].to_numpy()
            y_std = df_history[df_name_y_std].iloc[0].to_numpy()
            pr = df_history[df_name_pr].iloc[0].to_numpy()
            pr_std = df_history[df_name_pr_std].iloc[0].to_numpy()
            si = df_history[df_name_si].iloc[0].to_numpy()
            si_std = df_history[df_name_si_std].iloc[0].to_numpy()
            gini = df_history[df_name_gini].iloc[0].to_numpy()
            gini_std = df_history[df_name_gini_std].iloc[0].to_numpy()

            x = np.arange(len(y))

            xlabel = 'Iteration (T)'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[ylabel], fontsize=fontsize['legend'])

            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_2.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'Gini Index'
            ax_4.errorbar(x, gini, yerr=gini_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_4.yaxis.set_tick_params(labelsize=fontsize['ticks'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'prune'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_pruned(df_history):
    label_dict = {'si-0.5-1-0-1': 'SAP ($p=0.5$, $q=1.0$)',
                  'si-1-2-0-1': 'SAP ($p=1.0$, $q=2.0$)', 'lt-0.2': 'Lottery Ticket ($P=0.2$)',
                  'os-0.2': 'One Shot ($P=0.2$)'}
    color_dict = {'si-0.5-1-0-1': 'blue', 'si-1-2-0-1': 'cyan', 'lt-0.2': 'red', 'os-0.2': 'orange'}
    linestyle_dict = {'si-0.5-1-0-1': '-', 'si-1-2-0-1': '--', 'lt-0.2': '-.', 'os-0.2': ':'}
    marker_dict = {'si-0.5-1-0-1': 'o', 'si-1-2-0-1': 's', 'lt-0.2': 'p', 'os-0.2': 'D'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    capsize = None
    capthick = None
    pivot_p = '0.5'
    pivot_q = '1.0'
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test-pruned/Accuracy'] and stat == 'mean' and pivot in pivot_
        elif prune_mode_list[0] in ['lt', 'os']:
            pivot_ = ['lt-0.2', 'os-0.2']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test-pruned/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            df_name_y = '_'.join([*df_name_list[:-2], 'test/Accuracy', stat])
            df_name_y_std = '_'.join([*df_name_list[:-2], 'test/Accuracy', 'std'])
            df_name_y_pruned = '_'.join([*df_name_list[:-1], stat])
            df_name_y_pruned_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_si = '_'.join([*df_name_list[:-2],
                                   'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_std = '_'.join([*df_name_list[:-2],
                                       'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])
            df_name_si_pruned = '_'.join([*df_name_list[:-2],
                                          'test-pruned/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_pruned_std = '_'.join([*df_name_list[:-2],
                                              'test-pruned/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])

            y = df_history[df_name_y].iloc[0].to_numpy()
            y_std = df_history[df_name_y_std].iloc[0].to_numpy()
            y_pruned = df_history[df_name_y_pruned].iloc[0].to_numpy()
            y_pruned_std = df_history[df_name_y_pruned_std].iloc[0].to_numpy()
            si = df_history[df_name_si].iloc[0].to_numpy()
            si_std = df_history[df_name_si_std].iloc[0].to_numpy()
            si_pruned = df_history[df_name_si_pruned].iloc[0].to_numpy()
            si_pruned_std = df_history[df_name_si_pruned_std].iloc[0].to_numpy()

            x = np.arange(len(y_pruned))
            if prune_mode_list[0] == 'os':
                y = np.concatenate([y[[0]], y_pruned])
                y_std = np.concatenate([y_std[[0]], y_pruned_std])
                si = np.concatenate([si[[0]], si_pruned])
                si_std = np.concatenate([si_std[[0]], si_pruned_std])

            xlabel = 'Iteration (T)'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x, y_pruned, yerr=y_pruned_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[ylabel], fontsize=fontsize['legend'])

            xlabel = 'Iteration (T)'
            ylabel = 'Accuracy Difference'
            ax_2.errorbar(x, y[x] - y_pruned, yerr=np.sqrt(y_std[x] ** 2 + y_pruned_std ** 2), color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot], label=label_dict[pivot], marker=marker_dict[pivot],
                          capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si_pruned, yerr=si_pruned_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index Difference'
            ax_4.errorbar(x, si[x] - si_pruned, yerr=np.sqrt(si_std[x] ** 2 + si_pruned_std ** 2),
                          color=color_dict[pivot], linestyle=linestyle_dict[pivot], label=label_dict[pivot],
                          marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_4.yaxis.set_tick_params(labelsize=fontsize['ticks'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'pruned'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_p(df_history):
    label_dict = {'si-0.2-1-0-1': 'SAP ($p=0.2$)', 'si-0.4-1-0-1': 'SAP ($p=0.4$)',
                  'si-0.5-1-0-1': 'SAP ($p=0.5$)', 'si-0.6-1-0-1': 'SAP ($p=0.6$)',
                  'si-0.8-1-0-1': 'SAP ($p=0.8$)'}
    color_dict = {'si-0.2-1-0-1': 'red', 'si-0.4-1-0-1': 'orange', 'si-0.5-1-0-1': 'blue',
                  'si-0.6-1-0-1': 'cyan', 'si-0.8-1-0-1': 'black'}
    linestyle_dict = {'si-0.2-1-0-1': '-', 'si-0.4-1-0-1': '--', 'si-0.5-1-0-1': '-.', 'si-0.6-1-0-1': ':',
                      'si-0.8-1-0-1': (0, (1, 10))}
    marker_dict = {'si-0.2-1-0-1': 'o', 'si-0.4-1-0-1': 's', 'si-0.5-1-0-1': 'p', 'si-0.6-1-0-1': 'D',
                   'si-0.8-1-0-1': 'H'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    capsize = None
    capthick = None
    pivot_p = '0.5'
    pivot_q = '1.0'
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.2-1-0-1', 'si-0.4-1-0-1', 'si-0.5-1-0-1', 'si-0.6-1-0-1', 'si-0.8-1-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            df_name_y = '_'.join([*df_name_list[:-1], stat])
            df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_pr = '_'.join([*df_name_list[:-2], 'test/pr-global-global', stat])
            df_name_pr_std = '_'.join([*df_name_list[:-2], 'test/pr-global-global', 'std'])
            df_name_si = '_'.join([*df_name_list[:-2], 'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_std = '_'.join([*df_name_list[:-2],
                                       'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])
            df_name_gini = '_'.join([*df_name_list[:-2], 'test/gini-global-global', stat])
            df_name_gini_std = '_'.join([*df_name_list[:-2], 'test/gini-global-global', 'std'])

            y = df_history[df_name_y].iloc[0].to_numpy()
            y_std = df_history[df_name_y_std].iloc[0].to_numpy()
            pr = df_history[df_name_pr].iloc[0].to_numpy()
            pr_std = df_history[df_name_pr_std].iloc[0].to_numpy()
            si = df_history[df_name_si].iloc[0].to_numpy()
            si_std = df_history[df_name_si_std].iloc[0].to_numpy()
            gini = df_history[df_name_gini].iloc[0].to_numpy()
            gini_std = df_history[df_name_gini_std].iloc[0].to_numpy()

            x = np.arange(len(y))

            xlabel = 'Iteration (T)'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[ylabel], fontsize=fontsize['legend'])

            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_2.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'Gini Index'
            ax_4.errorbar(x, gini, yerr=gini_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_4.yaxis.set_tick_params(labelsize=fontsize['ticks'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        handles, labels = ax_dict_1[fig_name].get_legend_handles_labels()
        handles = [*handles[1:3], handles[0], *handles[3:]]
        labels = [*labels[1:3], labels[0], *labels[3:]]
        ax_dict_1[fig_name].legend(handles, labels, loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'p'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_q(df_history):
    label_dict = {'si-1-1.2-0-1': 'SAP ($q=1.2$)', 'si-1-1.4-0-1': 'SAP ($q=1.4$)',
                  'si-1-1.6-0-1': 'SAP ($q=1.6$)', 'si-1-1.8-0-1': 'SAP ($q=1.8$)',
                  'si-1-2-0-1': 'SAP ($q=2.0$)'}
    color_dict = {'si-1-1.2-0-1': 'red', 'si-1-1.4-0-1': 'orange', 'si-1-1.6-0-1': 'blue',
                  'si-1-1.8-0-1': 'cyan', 'si-1-2-0-1': 'black'}
    linestyle_dict = {'si-1-1.2-0-1': '-', 'si-1-1.4-0-1': '--', 'si-1-1.6-0-1': '-.', 'si-1-1.8-0-1': ':',
                      'si-1-2-0-1': (0, (1, 10))}
    marker_dict = {'si-1-1.2-0-1': 'o', 'si-1-1.4-0-1': 's', 'si-1-1.6-0-1': 'p', 'si-1-1.8-0-1': 'D',
                   'si-1-2-0-1': 'H'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    capsize = None
    capthick = None
    pivot_p = '0.5'
    pivot_q = '1.0'
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-1-1.2-0-1', 'si-1-1.4-0-1', 'si-1-1.6-0-1', 'si-1-1.8-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            df_name_y = '_'.join([*df_name_list[:-1], stat])
            df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_pr = '_'.join([*df_name_list[:-2], 'test/pr-global-global', stat])
            df_name_pr_std = '_'.join([*df_name_list[:-2], 'test/pr-global-global', 'std'])
            df_name_si = '_'.join([*df_name_list[:-2], 'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_std = '_'.join([*df_name_list[:-2],
                                       'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])
            df_name_gini = '_'.join([*df_name_list[:-2], 'test/gini-global-global', stat])
            df_name_gini_std = '_'.join([*df_name_list[:-2], 'test/gini-global-global', 'std'])

            y = df_history[df_name_y].iloc[0].to_numpy()
            y_std = df_history[df_name_y_std].iloc[0].to_numpy()
            pr = df_history[df_name_pr].iloc[0].to_numpy()
            pr_std = df_history[df_name_pr_std].iloc[0].to_numpy()
            si = df_history[df_name_si].iloc[0].to_numpy()
            si_std = df_history[df_name_si_std].iloc[0].to_numpy()
            gini = df_history[df_name_gini].iloc[0].to_numpy()
            gini_std = df_history[df_name_gini_std].iloc[0].to_numpy()

            x = np.arange(len(y))

            xlabel = 'Iteration (T)'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_2.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'Gini Index'
            ax_4.errorbar(x, gini, yerr=gini_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_4.yaxis.set_tick_params(labelsize=fontsize['ticks'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        handles, labels = ax_dict_1[fig_name].get_legend_handles_labels()
        handles = [*handles[1:], handles[0]]
        labels = [*labels[1:], labels[0]]
        ax_dict_1[fig_name].legend(handles, labels, loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'q'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_eta_m(df_history):
    label_dict = {'si-0.5-1-0-1': 'SAP ($\eta_r=0.0$)', 'si-0.5-1-0.001-1': 'SAP ($\eta_r=0.001$)',
                  'si-0.5-1-0.01-1': 'SAP ($\eta_r=0.01$)', 'si-0.5-1-0.1-1': 'SAP ($\eta_r=0.1$)',
                  'si-0.5-1-1-1': 'SAP ($\eta_r=1.0$)'}
    color_dict = {'si-0.5-1-0-1': 'red', 'si-0.5-1-0.001-1': 'orange', 'si-0.5-1-0.01-1': 'blue',
                  'si-0.5-1-0.1-1': 'cyan', 'si-0.5-1-1-1': 'black'}
    linestyle_dict = {'si-0.5-1-0-1': '-', 'si-0.5-1-0.001-1': '--', 'si-0.5-1-0.01-1': '-.', 'si-0.5-1-0.1-1': ':',
                      'si-0.5-1-1-1': (0, (1, 10))}
    marker_dict = {'si-0.5-1-0-1': 'o', 'si-0.5-1-0.001-1': 's', 'si-0.5-1-0.01-1': 'p', 'si-0.5-1-0.1-1': 'D',
                   'si-0.5-1-1-1': 'H'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    capsize = None
    capthick = None
    pivot_p = '0.5'
    pivot_q = '1.0'
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-0.5-1-0.001-1', 'si-0.5-1-0.01-1', 'si-0.5-1-0.1-1', 'si-0.5-1-1-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            df_name_y = '_'.join([*df_name_list[:-1], stat])
            df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_pr = '_'.join([*df_name_list[:-2], 'test/pr-global-global', stat])
            df_name_pr_std = '_'.join([*df_name_list[:-2], 'test/pr-global-global', 'std'])
            df_name_si = '_'.join([*df_name_list[:-2], 'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_std = '_'.join([*df_name_list[:-2],
                                       'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])
            df_name_gini = '_'.join([*df_name_list[:-2], 'test/gini-global-global', stat])
            df_name_gini_std = '_'.join([*df_name_list[:-2], 'test/gini-global-global', 'std'])

            y = df_history[df_name_y].iloc[0].to_numpy()
            y_std = df_history[df_name_y_std].iloc[0].to_numpy()
            pr = df_history[df_name_pr].iloc[0].to_numpy()
            pr_std = df_history[df_name_pr_std].iloc[0].to_numpy()
            si = df_history[df_name_si].iloc[0].to_numpy()
            si_std = df_history[df_name_si_std].iloc[0].to_numpy()
            gini = df_history[df_name_gini].iloc[0].to_numpy()
            gini_std = df_history[df_name_gini_std].iloc[0].to_numpy()

            x = np.arange(len(y))

            xlabel = 'Iteration (T)'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[ylabel], fontsize=fontsize['legend'])

            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_2.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'Gini Index'
            ax_4.errorbar(x, gini, yerr=gini_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_4.yaxis.set_tick_params(labelsize=fontsize['ticks'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'eta_m'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_gamma(df_history):
    label_dict = {'si-0.5-1-0-1': 'SAP ($\gamma=1.0$)', 'si-0.5-1-0-3': 'SAP ($\gamma=3.0$)',
                  'si-0.5-1-0-5': 'SAP ($\gamma=5.0$)', 'si-0.5-1-0-7': 'SAP ($\gamma=7.0$)',
                  'si-0.5-1-0-9': 'SAP ($\gamma=9.0$)'}
    color_dict = {'si-0.5-1-0-1': 'red', 'si-0.5-1-0-3': 'orange', 'si-0.5-1-0-5': 'blue',
                  'si-0.5-1-0-7': 'cyan', 'si-0.5-1-0-9': 'black'}
    linestyle_dict = {'si-0.5-1-0-1': '-', 'si-0.5-1-0-3': '--', 'si-0.5-1-0-5': '-.', 'si-0.5-1-0-7': ':',
                      'si-0.5-1-0-9': (0, (1, 10))}
    marker_dict = {'si-0.5-1-0-1': 'o', 'si-0.5-1-0-3': 's', 'si-0.5-1-0-5': 'p', 'si-0.5-1-0-7': 'D',
                   'si-0.5-1-0-9': 'H'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    capsize = None
    capthick = None
    pivot_p = '0.5'
    pivot_q = '1.0'
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-0.5-1-0-3', 'si-0.5-1-0-5', 'si-0.5-1-0-7', 'si-0.5-1-0-9']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            df_name_y = '_'.join([*df_name_list[:-1], stat])
            df_name_y_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_pr = '_'.join([*df_name_list[:-2], 'test/pr-global-global', stat])
            df_name_pr_std = '_'.join([*df_name_list[:-2], 'test/pr-global-global', 'std'])
            df_name_si = '_'.join([*df_name_list[:-2], 'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), stat])
            df_name_si_std = '_'.join([*df_name_list[:-2],
                                       'test/si-global-global-{}-{}'.format(pivot_p, pivot_q), 'std'])
            df_name_gini = '_'.join([*df_name_list[:-2], 'test/gini-global-global', stat])
            df_name_gini_std = '_'.join([*df_name_list[:-2], 'test/gini-global-global', 'std'])

            y = df_history[df_name_y].iloc[0].to_numpy()
            y_std = df_history[df_name_y_std].iloc[0].to_numpy()
            pr = df_history[df_name_pr].iloc[0].to_numpy()
            pr_std = df_history[df_name_pr_std].iloc[0].to_numpy()
            si = df_history[df_name_si].iloc[0].to_numpy()
            si_std = df_history[df_name_si_std].iloc[0].to_numpy()
            gini = df_history[df_name_gini].iloc[0].to_numpy()
            gini_std = df_history[df_name_gini_std].iloc[0].to_numpy()

            x = np.arange(len(y))

            xlabel = 'Iteration (T)'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[ylabel], fontsize=fontsize['legend'])

            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_2.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])

            xlabel = 'Iteration (T)'
            ylabel = 'Gini Index'
            ax_4.errorbar(x, gini, yerr=gini_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_4.yaxis.set_tick_params(labelsize=fontsize['ticks'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'gamma'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_pq(df_history):
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (20, 4)
    pivot_p = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    pivot_q = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3, ax_dict_4 = {}, {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        prune_iters = df_name_list[2]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        elif prune_mode_list[0] in ['lt', 'os']:
            pivot_ = ['lt-0.2', 'os-0.2']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:-2]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)
            ax_1, ax_2, ax_3, ax_4 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name], ax_dict_4[fig_name]

            si = []
            for i in range(len(pivot_p)):
                for j in range(len(pivot_q)):
                    df_name_si = '_'.join(
                        [*df_name_list[:-2], 'test/si-global-global-{}-{}'.format(pivot_p[i], pivot_q[j]), stat])
                    si.append(df_history[df_name_si].iloc[0].to_numpy())
            si = np.concatenate(si, axis=0).reshape(len(pivot_p), len(pivot_q), -1).transpose(2, 0, 1)
            if prune_iters == '30':
                iter_list = [0, 10, 20, 30]
            elif prune_iters == '15':
                iter_list = [0, 4, 8, 15]
            else:
                raise ValueError('Not valid prune iters')
            si_ = np.stack([si[x] for x in iter_list], axis=0)
            vmin, vmax = np.min(si_), np.max(si_)

            xlabel = '$q$'
            ylabel = '$p$'
            y = si_[0]
            ax_1.imshow(y, vmin=vmin, vmax=vmax)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.set_title('$T={}$'.format(iter_list[0]), fontsize=fontsize['label'])
            ax_1.set_xticks(np.arange(len(pivot_q)))
            ax_1.set_xticklabels(pivot_q)
            ax_1.set_yticks(np.arange(len(pivot_p)))
            ax_1.set_yticklabels(pivot_p)

            y = si_[1]
            ax_2.imshow(y, vmin=vmin, vmax=vmax)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.set_title('$T={}$'.format(iter_list[1]), fontsize=fontsize['label'])
            ax_2.set_xticks(np.arange(len(pivot_q)))
            ax_2.set_xticklabels(pivot_q)
            ax_2.set_yticks(np.arange(len(pivot_p)))
            ax_2.set_yticklabels(pivot_p)

            y = si_[2]
            ax_3.imshow(y, vmin=vmin, vmax=vmax)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.set_title('$T={}$'.format(iter_list[2]), fontsize=fontsize['label'])
            ax_3.set_xticks(np.arange(len(pivot_q)))
            ax_3.set_xticklabels(pivot_q)
            ax_3.set_yticks(np.arange(len(pivot_p)))
            ax_3.set_yticklabels(pivot_p)

            y = si_[3]
            im = ax_4.imshow(y, vmin=vmin, vmax=vmax)
            ax_4.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_4.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_4.set_title('$T={}$'.format(iter_list[3]), fontsize=fontsize['label'])
            ax_4.set_xticks(np.arange(len(pivot_q)))
            ax_4.set_xticklabels(pivot_q)
            ax_4.set_yticks(np.arange(len(pivot_p)))
            ax_4.set_yticklabels(pivot_p)
            plt.colorbar(im)

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        fig[fig_name].tight_layout()
        dir_name = 'pq'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_layer(df_history):
    label_dict = {'0': 'First layer', '1': 'Middle layer', '2': 'Last layer'}
    color_dict = {'0': 'blue', '1': 'cyan', '2': 'red'}
    linestyle_dict = {'0': '-', '1': '--', '2': '-.'}
    marker_dict = {'0': 'o', '1': 's', '2': 'p'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (15, 4)
    capsize = None
    capthick = None
    df_name_layer_ = defaultdict(list)
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        data_name = df_name_list[0]
        model_name = df_name_list[1]
        scope = df_name_list[3]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = 'test/pr-layer' in metric_name and stat == 'mean' and pivot in pivot_ and \
                   model_name in ['mlp', 'cnn', 'resnet18', 'wresnet28x8', 'resnet50'] and scope == 'global'
        else:
            continue
        if mask:
            df_name_layer_key = '_'.join([*df_name_list[:-2]])
            df_name_layer_[df_name_layer_key].append(metric_name)
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3 = {}, {}, {}
    for df_name in df_name_layer_:
        df_name_list = df_name.split('_')
        fig_name = '_'.join(df_name_list)
        fig[fig_name] = plt.figure(fig_name, figsize=figsize)
        if fig_name not in ax_dict_1:
            ax_dict_1[fig_name] = fig[fig_name].add_subplot(131)
            ax_dict_2[fig_name] = fig[fig_name].add_subplot(132)
            ax_dict_3[fig_name] = fig[fig_name].add_subplot(133)
        ax_1, ax_2, ax_3 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name]
        layer_names = df_name_layer_[df_name]
        idx = np.linspace(0, len(layer_names) - 1, 3).round().astype(np.int32).tolist()
        layer_names = [layer_names[x] for x in idx]
        df_name_neuron, df_name_layer, df_name_global = [], [], []
        df_name_neuron_std, df_name_layer_std, df_name_global_std = [], [], []
        for i in range(len(layer_names)):
            df_name_neuron.append(
                '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:], layer_names[i], 'mean']))
            df_name_neuron_std.append(
                '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:], layer_names[i], 'std']))
            df_name_layer.append(
                '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:], layer_names[i], 'mean']))
            df_name_layer_std.append(
                '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:], layer_names[i], 'std']))
            df_name_global.append(
                '_'.join([*df_name_list[:3], 'global', *df_name_list[4:], layer_names[i], 'mean']))
            df_name_global_std.append(
                '_'.join([*df_name_list[:3], 'global', *df_name_list[4:], layer_names[i], 'std']))
        for i in range(len(df_name_neuron)):
            pivot = str(i)
            pr = df_history[df_name_neuron[i]].iloc[0].to_numpy()
            pr_std = df_history[df_name_neuron_std[i]].iloc[0].to_numpy()
            x = np.arange(len(pr))
            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_1.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_1.legend(loc=loc_dict['Percent of Remaining Weights'], fontsize=fontsize['legend'])
        ax_1.set_title('Neuron-wise Pruning', fontsize=fontsize['label'])

        for i in range(len(df_name_layer)):
            pivot = str(i)
            pr = df_history[df_name_layer[i]].iloc[0].to_numpy()
            pr_std = df_history[df_name_layer_std[i]].iloc[0].to_numpy()
            x = np.arange(len(pr))
            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_2.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_2.set_title('Layer-wise Pruning', fontsize=fontsize['label'])

        for i in range(len(df_name_global)):
            pivot = str(i)
            pr = df_history[df_name_global[i]].iloc[0].to_numpy()
            pr_std = df_history[df_name_global_std[i]].iloc[0].to_numpy()
            x = np.arange(len(pr))
            xlabel = 'Iteration (T)'
            ylabel = 'Percent of Remaining Weights'
            ax_3.errorbar(x, pr * 100, yerr=pr_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_3.set_title('Global Pruning', fontsize=fontsize['label'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'layer'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_ratio(df_history):
    label_dict = {'si-0.5-1-0-1': 'SAP ($p=0.5$, $q=1.0$)',
                  'si-1-2-0-1': 'SAP ($p=1.0$, $q=2.0$)', 'lt-0.2': 'Lottery Ticket ($P=0.2$)',
                  'os-0.2': 'One Shot ($P=0.2$)'}
    color_dict = {'si-0.5-1-0-1': 'blue', 'si-1-2-0-1': 'cyan', 'lt-0.2': 'red', 'os-0.2': 'orange'}
    linestyle_dict = {'si-0.5-1-0-1': '-', 'si-1-2-0-1': '--', 'lt-0.2': '-.', 'os-0.2': ':'}
    marker_dict = {'si-0.5-1-0-1': 'o', 'si-1-2-0-1': 's', 'lt-0.2': 'p', 'os-0.2': 'D'}
    loc_dict = {'Accuracy': 'lower right', 'Percent of Remaining Weights': 'upper right',
                'PQ Index': 'lower left', 'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (15, 4)
    capsize = None
    capthick = None
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3 = {}, {}, {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        data_name = df_name_list[0]
        model_name = df_name_list[1]
        scope = df_name_list[3]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_ and \
                   model_name in ['mlp', 'cnn', 'resnet18', 'wresnet28x8', 'resnet50'] and scope == 'global'
        elif prune_mode_list[0] in ['lt', 'os']:
            pivot_ = ['lt-0.2', 'os-0.2']
            pivot = '-'.join(prune_mode_list)
            mask = metric_name in ['test/Accuracy'] and stat == 'mean' and pivot in pivot_ and \
                   model_name in ['mlp', 'cnn', 'resnet18', 'wresnet28x8', 'resnet50'] and scope == 'global'
        else:
            continue
        if mask:
            fig_name = '_'.join([*df_name_list[:3], *df_name_list[4:-3]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(131)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(132)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(133)
            ax_1, ax_2, ax_3 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name]

            df_name_y_global = '_'.join([*df_name_list[:3], 'global', *df_name_list[4:-1], stat])
            df_name_y_global_std = '_'.join([*df_name_list[:3], 'global', *df_name_list[4:-1], 'std'])
            df_name_y_layer = '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:-1], stat])
            df_name_y_layer_std = '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:-1], 'std'])
            df_name_y_neuron = '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:-1], stat])
            df_name_y_neuron_std = '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:-1], 'std'])
            df_name_pr_global = '_'.join([*df_name_list[:3], 'global', *df_name_list[4:-2],
                                          'test/pr-global-global', stat])
            df_name_pr_global_std = '_'.join([*df_name_list[:3], 'global', *df_name_list[4:-2],
                                              'test/pr-global-global', 'std'])
            df_name_pr_layer = '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:-2],
                                         'test/pr-global-global', stat])
            df_name_pr_layer_std = '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:-2],
                                             'test/pr-global-global', 'std'])
            df_name_pr_neuron = '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:-2],
                                          'test/pr-global-global', stat])
            df_name_pr_neuron_std = '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:-2],
                                              'test/pr-global-global', 'std'])

            y_global = df_history[df_name_y_global].iloc[0].to_numpy()
            y_global_std = df_history[df_name_y_global_std].iloc[0].to_numpy()
            y_layer = df_history[df_name_y_layer].iloc[0].to_numpy()
            y_layer_std = df_history[df_name_y_layer_std].iloc[0].to_numpy()
            y_neuron = df_history[df_name_y_neuron].iloc[0].to_numpy()
            y_neuron_std = df_history[df_name_y_neuron_std].iloc[0].to_numpy()
            pr_global = df_history[df_name_pr_global].iloc[0].to_numpy()
            pr_global_std = df_history[df_name_pr_global_std].iloc[0].to_numpy()
            pr_layer = df_history[df_name_pr_layer].iloc[0].to_numpy()
            pr_layer_std = df_history[df_name_pr_layer_std].iloc[0].to_numpy()
            pr_neuron = df_history[df_name_pr_neuron].iloc[0].to_numpy()
            pr_neuron_std = df_history[df_name_pr_neuron_std].iloc[0].to_numpy()

            x = pr_global
            y = y_neuron
            y_std = y_neuron_std
            xlabel = 'Percent of Remaining Weights'
            ylabel = metric_name.split('/')[1]
            ax_1.errorbar(x * 100, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[ylabel], fontsize=fontsize['legend'])
            ax_1.set_title('Neuron-wise Pruning', fontsize=fontsize['label'])

            x = pr_layer
            y = y_layer
            y_std = y_layer_std
            xlabel = 'Percent of Remaining Weights'
            ylabel = metric_name.split('/')[1]
            ax_2.errorbar(x * 100, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.set_title('Layer-wise Pruning', fontsize=fontsize['label'])

            x = pr_neuron
            y = y_global
            y_std = y_global_std
            xlabel = 'Percent of Remaining Weights'
            ylabel = metric_name.split('/')[1]
            ax_3.errorbar(x * 100, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.set_title('Global Pruning', fontsize=fontsize['label'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ymin_1, ymax_1 = ax_dict_1[fig_name].get_ylim()
        ymin_2, ymax_2 = ax_dict_2[fig_name].get_ylim()
        ymin_3, ymax_3 = ax_dict_3[fig_name].get_ylim()
        ymin = max([ymin_1, ymin_2, ymin_3])
        ymax = max([ymax_1, ymax_2, ymax_3])
        ax_dict_1[fig_name].set_ylim([ymin, ymax])
        ax_dict_2[fig_name].set_ylim([ymin, ymax])
        ax_dict_3[fig_name].set_ylim([ymin, ymax])
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'ratio'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_by_si_layer(df_history):
    label_dict = {'0': 'First layer', '1': 'Middle layer', '2': 'Last layer'}
    color_dict = {'0': 'blue', '1': 'cyan', '2': 'red'}
    linestyle_dict = {'0': '-', '1': '--', '2': '-.'}
    marker_dict = {'0': 'o', '1': 's', '2': 'p'}
    loc_dict = {'Accuracy': 'lower left', 'Percent of Remaining Weights': 'upper right', 'PQ Index': 'lower left',
                'Gini Index': 'lower left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (15, 4)
    capsize = None
    capthick = None
    df_name_layer_ = defaultdict(list)
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        data_name = df_name_list[0]
        model_name = df_name_list[1]
        scope = df_name_list[3]
        prune_mode_list = df_name_list[4].split('-')
        if prune_mode_list[0] == 'si':
            pivot_ = ['si-0.5-1-0-1', 'si-1-2-0-1']
            pivot = '-'.join(prune_mode_list)
            mask = 'test/si-layer' in metric_name and stat == 'mean' and pivot in pivot_ and \
                   model_name in ['mlp', 'cnn', 'resnet18', 'wresnet28x8', 'resnet50'] and scope == 'global'
        else:
            continue
        if mask:
            df_name_layer_key = '_'.join([*df_name_list[:-2]])
            df_name_layer_[df_name_layer_key].append(metric_name)
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3 = {}, {}, {}
    for df_name in df_name_layer_:
        df_name_list = df_name.split('_')
        fig_name = '_'.join(df_name_list)
        fig[fig_name] = plt.figure(fig_name, figsize=figsize)
        if fig_name not in ax_dict_1:
            ax_dict_1[fig_name] = fig[fig_name].add_subplot(131)
            ax_dict_2[fig_name] = fig[fig_name].add_subplot(132)
            ax_dict_3[fig_name] = fig[fig_name].add_subplot(133)
        ax_1, ax_2, ax_3 = ax_dict_1[fig_name], ax_dict_2[fig_name], ax_dict_3[fig_name]
        layer_names = df_name_layer_[df_name]
        idx = np.linspace(0, len(layer_names) - 1, 3).round().astype(np.int32).tolist()
        layer_names = [layer_names[x] for x in idx]
        df_name_neuron, df_name_layer, df_name_global = [], [], []
        df_name_neuron_std, df_name_layer_std, df_name_global_std = [], [], []
        for i in range(len(layer_names)):
            df_name_neuron.append(
                '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:], layer_names[i], 'mean']))
            df_name_neuron_std.append(
                '_'.join([*df_name_list[:3], 'neuron', *df_name_list[4:], layer_names[i], 'std']))
            df_name_layer.append(
                '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:], layer_names[i], 'mean']))
            df_name_layer_std.append(
                '_'.join([*df_name_list[:3], 'layer', *df_name_list[4:], layer_names[i], 'std']))
            df_name_global.append(
                '_'.join([*df_name_list[:3], 'global', *df_name_list[4:], layer_names[i], 'mean']))
            df_name_global_std.append(
                '_'.join([*df_name_list[:3], 'global', *df_name_list[4:], layer_names[i], 'std']))
        for i in range(len(df_name_neuron)):
            pivot = str(i)
            si = df_history[df_name_neuron[i]].iloc[0].to_numpy()
            si_std = df_history[df_name_neuron_std[i]].iloc[0].to_numpy()
            x = np.arange(len(si))
            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_1.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_1.legend(loc=loc_dict['PQ Index'], fontsize=fontsize['legend'])
        ax_1.set_title('Neuron-wise Pruning', fontsize=fontsize['label'])

        for i in range(len(df_name_layer)):
            pivot = str(i)
            si = df_history[df_name_layer[i]].iloc[0].to_numpy()
            si_std = df_history[df_name_layer_std[i]].iloc[0].to_numpy()
            x = np.arange(len(si))
            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_2.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_2.set_title('Layer-wise Pruning', fontsize=fontsize['label'])

        for i in range(len(df_name_global)):
            pivot = str(i)
            si = df_history[df_name_global[i]].iloc[0].to_numpy()
            si_std = df_history[df_name_global_std[i]].iloc[0].to_numpy()
            x = np.arange(len(si))
            xlabel = 'Iteration (T)'
            ylabel = 'PQ Index'
            ax_3.errorbar(x, si, yerr=si_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot], capsize=capsize, capthick=capthick)
            ax_3.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_3.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_3.set_title('Global Pruning', fontsize=fontsize['label'])

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'si_layer'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}_{}.{}'.format(dir_name, fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
