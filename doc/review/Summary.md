# THE LOTTERY TICKET HYPOTHESIS : FINDING SPARSE , TRAINABLE NEURAL NETWORKS, 2019

Prune the $p\%$ smallest weights (iteratively). Static mask. Retrain from the same initialization. Subnetwork generalizes better/higher test acc. 



# Proving the Lottery Ticket Hypothesis: Pruning is All You Need, 2020

Weight/neuron subnetwork. 

A ReLU network of arbitrary depth l can be approximated by finding a weight-
subnetwork of a random network of depth 2l and sufficient
width. 

# Learning both Weights and Connections for Efficient Neural Networks, 2015
prune redundant connections (weight < threshold),  retrain the network to fine tune the weights

Contribution?

# DEEP COMPRESSION : COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING , T RAINED QUANTIZATION
AND HUFFMAN CODING, 2016
Stack two more techniques, quantization (cluster/group and share weights) and coding. 



# Optimal Brain Damage, 1990

Prune unimportant weights by Hessian (have the least effect on the training error if removing them)



#  Hanson and Pratt, 1989

Weight decay, i.e., prune by weights.



# Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures, 2016
zero activation neurons are redundant, (output almost always zero)

retrain use warm-start.



# Data-free Parameter Pruning for Deep Neural Networks, 2015

Ba and Caruna

resemble similar neurons.



# Channel Pruning for Accelerating Very Deep Neural Networks, 2017

Remove channels that has little effects on the next layer by factorization. Optimized by using LASSO penalty.  





# Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks, 2021
Survey paper.



# DATA -DEPENDENT CORESETS FOR COMPRESSING NEURAL NETWORKS WITH APPLICATIONS TO GENERALIZATION BOUNDS, 2019
sampling based one time forward pruning. Bound is little bit loose.  



# ==Stronger== Generalization Bounds for Deep Nets via a Compression Approach, 2018

Gen bound given by smaller model. If compressible (measured by the noise-stability of the network, stable means exists a simpler model), then apply generalization bound to the compressed model. 



# Pruning and Quantization for Deep Neural Network Acceleration: A Survey, 2021
Survey with less info. Easier to find some refer related to regularization based pruning. 



# Rigging the Lottery: Making All Tickets Winners, 2020
dynamic pruning.



# Transformed l1 regularization for learning sparse deep neural networks, 2019
L1+GroupL1.



# DATA -INDEPENDENT VIA CORESETS, 2020

Still one time coresets, change to independent sensitivity. Forward to last layer.



# SiPPing Neural Networks: Sensitivity-informed Provable Pruning of Neural Networks, 2021
Follow up of coresets.



# PROVABLE FILTER PRUNING FOR EFFICIENT NEURAL NETWORKS, 2020
Follow up of coresets.



# Why Lottery Ticket Wins? A Theoretical Perspective of Sample Complexity on Pruned Neural Networks, 2021
If the mask is oracle, then pruned network can generalize better and be trained easier, which is quite intuitive. Claim that magnitude based pruning can produce good mask, no sure how.





# Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection, 2020

Two-layer NN, select edges that decreases loss most. Argue that the excess loss (not risk! i.e., optimization error instead of generalization error) is bounded by $O(1/n)$, where $n$ is the number of edges selected. Optimization based argument.



# NON - VACUOUS GENERALIZATION BOUNDS AT THE IMAGE NET SCALE : A PAC-BAYESIAN COMPRESSION APPROACH, 2019

Key point: After compression, what will be the error bound for the Generalization error.



# Logarithmic Pruning is All You Need, 2020

Meaningless. 



# Greedy Optimization Provably Wins the Lottery: Logarithmic Number of Winning Tickets is Enough, 2020
==Wrong statement.== An extension in the optimization sense.

==Highly possible the proof is faulty. Still published.== 



# Recent Advances on Neural Network Pruning at Initialization, 2022

E.g., SNIP, eval the sensitivity of a connection/edge even at the init stage. 



# Summary

Reduce #of params, computation(FLoP), memory(storage), energy cost.

Also improves generalization, robustness against attacks, training time, etc. 

1. Change/design better architecture (convolutional layer, ResNet)

2. prune - pre-train (low rank constrain; regularity); during train (drop out; Dense-Sparse-Dense Training); after-train (grow/prune, forward-backward selection) 

3. Deeper but thinner (FitNets) / shallower but wider(Ba and Caruna) / sparser , subnetwork / down-size model (KD, Neural Architecture Search)

4. Sparser pruning -- weight/neuron/neruron-like subnetwork -- based on different intuitions on saliency: weight decay; hard-threshold; second-derivative; zero activation; quantization (resemble similar weights or neurons), BinaryNet , etc.

   Structured (SVD etc.) /unstructured pruning. 

   Static/dynamic pruning(regrowth).