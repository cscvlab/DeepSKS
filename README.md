# DeepSKS: Decoupled Geometric Parameterization upon SKS for Deep Homography Estimation

This repository is the official implementation of the paper: 

**Decoupled Geometric Parameterization and its Application in Deep Homography Estimation**.

__Authors:__ Yao Huang, Siyuan Cao, Yaqing Ding, Hao Yin, Shibin Xie, Zhijun Fang, Jiachun Wang, Shen Cai, Junchi Yan, Shuhan Shen.

<!-- **Links:**  [[Paper]](https://arxiv.org/pdf/2402.18008) -->

## 💡 Motivation

The Similarity-Kernel-Similarity ([SKS](http://www.cscvlab.com/research/SKS-Homography/)) decomposition is an efficient and interpretable homography computation method, which we proposed about seven years ago. In deep learing era, although the SKS decomposition can be used as a post-processing solver to calculate homography utilizing four-point positional offsets (which neural networks predict), we expect to explore a long-standing problem: how to represent homography by eight geometric parameters suitable for NN prediction?. Geometric parameterization of SKS

![alt text](figs/hierarchicalTrans.png)

and transferring it into deep homography estimation (DHE) task is not straightforward and need to solve three problems: (1) Which eight geometric parameters are suitable for neural network prediction? (2) How to optimally estimate parameters in a straitified sub-transformations? (3) How to endow parameters with a direct interpretation in terms of image feature?   

In this paper, we propose decoupled geometric parameterization upon SKS for DHE, which significantly promote SKS in three key folds:

1. **Geometric Parameterization Design:** While SKS has various decomposition forms involving over eight parameters, we identify a well-suited subset of eight parameters for DHE and fix the rest.

2. **Parameter Decoupling:** Although SKS’s stratified decomposition initially implied that the kernel transformation $\mathbf{H}_K$ seemed dependent on a similarty transformation $\mathbf{H}_S$, after extensive analysis, we prove that two sets of four parameters can be decoupled and predicted in parallel.

3. **Angular Offsets Feature:** We introduce angular offsets (A.O.) as a novel visual feature and empirically validate their robustness as part of our representation.

![alt text](figs/comparison.png)

## 🔬 Formula Breakdown

### Original SKS Decomposition
SKS decomposes a 2D homography into three sub-transformations: 
```math
\mathbf{H}=\mathbf{H}_{S_2}^{-1}*\mathbf{H}_{K}*\mathbf{H}_{S_1},
```
where $\mathbf{H}\_{S\_1}$ and $\mathbf{H}\_{S\_2}$ are similarity transformations induced by two arbitrary pairs of corresponding points on source plane and target plane, respectively; $\mathbf{H}\_{K}$ is the 4-DOF kernel transfromation we defined, which generates projective distortion between two similarity-normalized planes. In the SKS work, $\mathbf{H}\_{K}$ is associated with the hyperbolic similarity transformation $\mathbf{H}_{S}$.

### Geometric Parameterization upon SKS

In this paper, the homography $H$ from the source image to the target image is represented by:

$$H = H_{S_{1}}H_{S_{2}}^{-1}H_{K}H_{S_{2}} = H_{T}^{-1}H_{S}H_{T}H_{S_{2}}^{-1}H_{K}H_{S_{2}},$$
where the similarity transformation $H_{S_{2}}$ and the translation $H_{T}$ are known .......; The unknown similarity transformation $H_{S}$ is expressed by 
$$
H_{S} =\begin{bmatrix} 
\Delta a_{S}+1 & -b_{S} & u_{S} \\
b_{S} & \Delta a_{S}+1 & v_{S} \\
0 & 0 & 1\\
\end{bmatrix},
$$

and the unknown kernel transformation $H_{K}$ is expressed by Eq. (12) in the paper:

$$H_{K} = \begin{bmatrix} 
\Delta a_{K}+1 & u_{K} & b_{K} \\ 
0 & 1 & 0 \\ 
b_{K} & v_{K} & \Delta a_{K}+1\\ 
\end{bmatrix}.$$

This introduces an 8-DOF geometric parameterization for homography, which is decoupled into two independent sets: 4-DOF in $H_{S}$ and 4-DOF in $H_{K}$.


## 📜 Article Summary


Planar homography, with eight degrees of freedom (DOFs), is fundamental in numerous computer vision tasks. While the positional offsets of four corners are widely adopted (especially in neural 
network predictions), this parameterization lacks geometric interpretability and typically requires solving a linear system to compute the homography matrix. This paper presents a novel geometric parameterization of homographies, leveraging  for projective transformations. Two independent sets of four geometric parameters are decoupled: one for a similarity transformation and the other for the kernel transformation. Additionally, the geometric interpretation linearly relating the four kernel transformation parameters to angular offsets is derived. Our proposed parameterization allows for direct homography estimation through matrix multiplication, eliminating the need for solving a linear system, and achieves performance comparable to the four-corner positional offsets in deep homography estimation.

This paper presents a novel geometric parameterization of homography that is suitable for estimation through neural networks, based on the SKS decomposition. By introducing two independent sets of four geometric parameters, each with corresponding projective distortion interpretations, the parameterization aligns with and unifies the estimation for 2D similarity and affine transformations. Furthermore, the proposed method eliminates the need for solving linear systems, as required by traditional four-corner positional offsets parameterization, and achieves competitive performance across multiple datasets and neural network architectures. Similar to pose estimation and other geometric vision tasks, this approach demonstrates the value of geometric parameterization, as all deep learning methods predicting positional offsets are not end-to-end and require an algebraic solver to compute solutions as a post-processing step. Moreover, to the best of the authors' knowledge, this is the first work to introduce angular offsets in vision tasks.
