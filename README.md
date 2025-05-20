# DeepSKS: Decoupled Geometric Parameterization upon SKS for Deep Homography Estimation

This repository is the official implementation of the paper: 

**Decoupled Geometric Parameterization and its Application in Deep Homography Estimation**.

__Authors:__ Yao Huang, Siyuan Cao, Yaqing Ding, Hao Yin, Shibin Xie, Zhijun Fang, Jiachun Wang, Shen Cai*, Junchi Yan, Shuhan Shen*.

<!-- **Links:**  [[Paper]](https://arxiv.org/pdf/2402.18008) -->

## 📝 Article Introduction

Planar homography, with eight degrees of freedom (DOFs), is fundamental in numerous computer vision tasks. While the positional offsets of four corners are widely adopted (especially in neural network predictions), this parameterization lacks geometric interpretability and typically requires solving a linear system to compute the homography matrix. This paper presents a novel geometric parameterization of homographies, leveraging the similarity-kernel-similarity (SKS) decomposition for projective transformations. Two independent sets of four geometric parameters are decoupled: one for a similarity transformation and the other for the kernel transformation. Additionally, the geometric interpretation linearly relating the four kernel transformation parameters to angular offsets is derived. Our proposed parameterization allows for direct homography estimation through matrix multiplication, eliminating the need for solving a linear system, and achieves performance comparable to the four-corner positional offsets in deep homography estimation.

## 🔬 Formula Breakdown

The method is based on an improved Similarity-Kernel-Similarity (SKS) decomposition.

### Improved SKS Decomposition for Homography Parameterization

The complete homography $H$ from the source image to the target image is represented by:

$$H = H_{S_{1}}H_{S_{2}}^{-1}H_{K}H_{S_{2}} = H_{T}^{-1}H_{S}H_{T}H_{S_{2}}^{-1}H_{K}H_{S_{2}}$$

Where:   
* $H_{S}$ is given as Eq. (4) in the paper:

$$
H_{S} =\begin{bmatrix} 
\Delta a_{S}+1 & -b_{S} & u_{S} \\
b_{S} & \Delta a_{S}+1 & v_{S} \\
0 & 0 & 1\\
\end{bmatrix}
$$

* $H_{K}$ is expressed by Eq. (12) in the paper:

$$H_{K} = \begin{bmatrix} 
\Delta a_{K}+1 & u_{K} & b_{K} \\ 
0 & 1 & 0 \\ 
b_{K} & v_{K} & \Delta a_{K}+1\\ 
\end{bmatrix}$$

This introduces an 8-DOF geometric parameterization for homography, which is decoupled into two independent sets: 4-DOF in $H_{S}$ and 4-DOF in $H_{K}$.

## 💡 Motivation

Our paper is specifically designed for DHE and builds upon SKS in three key directions:

1. **Geometric Parameterization Design:** While SKS has various decomposition forms involving over eight parameters, we identify a well-suited subset of eight parameters for DHE and fix the rest.

2. **Parameter Decoupling:** Although SKS’s stratified decomposition initially implied that $\mathbf{H}_K$ seemed dependent on $\mathbf{H}_S$, after extensive analysis, we derive that two sets of four parameters can be decoupled and predicted in parallel.

3. **Angular Offsets Feature:** We introduce angular offsets (A.O.) as a novel visual point feature and empirically validate their robustness as part of our representation.

## 📜 Article Summary

This paper presents a novel geometric parameterization of homography that is suitable for estimation through neural networks, based on the SKS decomposition[cite: 202]. By introducing two independent sets of four geometric parameters, each with corresponding projective distortion interpretations, the parameterization aligns with and unifies the estimation for 2D similarity and affine transformations. Furthermore, the proposed method eliminates the need for solving linear systems, as required by traditional four-corner positional offsets parameterization, and achieves competitive performance across multiple datasets and neural network architectures. Similar to pose estimation and other geometric vision tasks, this approach demonstrates the value of geometric parameterization, as all deep learning methods predicting positional offsets are not end-to-end and require an algebraic solver to compute solutions as a post-processing step. Moreover, to the best of the authors' knowledge, this is the first work to introduce angular offsets in vision tasks.
