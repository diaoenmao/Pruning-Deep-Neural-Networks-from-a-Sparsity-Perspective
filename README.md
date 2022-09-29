# Pruning Deep Neural Networks from a Sparsity Perspective
This is an implementation of Pruning Deep Neural Networks from a Sparsity Perspective
 
## Requirements
See requirements.txt

## Instruction
- Use make.sh to generate run script
- Use make.py to generate exp script
- Use process.py to process exp results
- Hyperparameters can be found in config.yml and process_control() in utils.py

## Examples
 - Run make_stats.py to prepare for each dataset
 - Train One Shot with FashionMNIST, Linear, $T=30$, 'Global Pruning', $P=0.2$
    ```ruby
    python train_classifier.py --control_name FashionMNIST_linear_30_global_os-0.2
    ```
 - Test Lottery Ticket with CIFAR10, MLP, $T=30$, 'Layer-wise Pruning', $P=0.2$
    ```ruby
    python train_classifier.py --control_name CIFAR10_mlp_30_layer_lt-0.2
    ```
 - Train SAP with CIFAR10, ResNet18, $T=15$, 'Neuron-wise Pruning', $p=0.5$, $q=1.0$, $\eta_r=0.001$, $\gamma=1.2$
    ```ruby
    python train_classifier.py --control_name CIFAR10_resnet18_30_neuron_si-0.5-1.0-0.001-1.2
    ```
