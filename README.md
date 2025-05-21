# DeepSKS: Decoupled Geometric Parameterization upon SKS for Deep Homography Estimation

This repository is the official implementation of the paper: 

**Decoupled Geometric Parameterization and its Application in Deep Homography Estimation**.

__Authors:__ Yao Huang, Siyuan Cao, Yaqing Ding, Hao Yin, Shibin Xie, Zhijun Fang, Jiachun Wang, Shen Cai, Junchi Yan, Shuhan Shen.

<!-- **Links:**  [[Paper]](https://arxiv.org/pdf/2402.18008) -->

## 💡 Motivation

The **Similarity-Kernel-Similarity ([SKS](http://www.cscvlab.com/research/SKS-Homography/))** decomposition is an efficient and interpretable method for homography computation, which we originally proposed approximately seven years ago. In the deep learning era, SKS can be employed as a post-processing solver to estimate homography from four-point positional offsets predicted by neural networks. However, this work aims to address a deeper, long-standing problem: **how to represent a homography using eight geometric parameters**.

Geometric parameterization of homography aligns conceptually and structurally with existing parameterizations for similarity and affine transformations (see figure below). Furthermore, it complements the broader family of solver-free estimations, such as translation and rotation in relative pose estimation or 3D point cloud prediction in stereo reconstruction.

<p align="center">
  <img src="figs/hierarchicalTrans.png" width = "600"  alt="hierarchicalTrans" align=center />
</p>

While [SKS](http://www.cscvlab.com/research/SKS-Homography/) decomposes a homography into geometrically meaningful sub-transformations, its adaptation to deep homography estimation (DHE) is non-trivial. Three central challenges arise: (1) Which eight geometric parameters are suitable for neural network prediction? (2) How can parameters be optimally estimated across stratified sub-transformations? (3) How to endow parameters with a direct interpretation in terms of image feature?  

To tackle these challenges, we propose a **decoupled geometric parameterization** based on the SKS decomposition, tailored for DHE. Our contributions advance SKS in the following key areas:

1. **Geometric Parameterization Design**: While SKS allows various decompositions involving more than eight parameters, we identify a specific, compact subset of eight parameters suitable for DHE, fixing the rest to maintain stability and interpretability.

2. **Parameter Decoupling**: Although SKS’s stratified decomposition initially implied a dependency of the kernel transformation $\mathbf{H}_K$ on the similarity transformation $\mathbf{H}_S$, our analysis reveals that two independent four-parameter groups can be learned in parallel.

3. **Angular Offsets Feature**: We introduce **angular offsets (A.O.)** as a novel point-level visual feature and empirically validate their robustness and relevance in homography prediction.

The pipeline comparison between prior methods, the original SKS, and our proposed method is illustrated in the following figure:

<p align="center">
 <img src="figs/comparison.png" width = "600" alt="comparison" align=center />
</p>

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

The above equations introduce an 8-DOF geometric parameterization for homography, four in $H_{S}$ and four in $H_{K}$.

### Parameter Decoupling
which is decoupled into two independent sets: 4-DOF

## 📜 Article Summary


Planar homography, with eight degrees of freedom (DOFs), is fundamental in numerous computer vision tasks. While the positional offsets of four corners are widely adopted (especially in neural 
network predictions), this parameterization lacks geometric interpretability and typically requires solving a linear system to compute the homography matrix. This paper presents a novel geometric parameterization of homographies, leveraging  for projective transformations. Two independent sets of four geometric parameters are decoupled: one for a similarity transformation and the other for the kernel transformation. Additionally, the geometric interpretation linearly relating the four kernel transformation parameters to angular offsets is derived. Our proposed parameterization allows for direct homography estimation through matrix multiplication, eliminating the need for solving a linear system, and achieves performance comparable to the four-corner positional offsets in deep homography estimation.

This paper presents a novel geometric parameterization of homography that is suitable for estimation through neural networks, based on the SKS decomposition. By introducing two independent sets of four geometric parameters, each with corresponding projective distortion interpretations, the parameterization aligns with and unifies the estimation for 2D similarity and affine transformations. Furthermore, the proposed method eliminates the need for solving linear systems, as required by traditional four-corner positional offsets parameterization, and achieves competitive performance across multiple datasets and neural network architectures. Similar to pose estimation and other geometric vision tasks, this approach demonstrates the value of geometric parameterization, as all deep learning methods predicting positional offsets are not end-to-end and require an algebraic solver to compute solutions as a post-processing step. Moreover, to the best of the authors' knowledge, this is the first work to introduce angular offsets in vision tasks.
