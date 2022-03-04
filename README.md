# Ultimate-Awesome-Transformer-Attention [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A comprehensive paper list of Vision Transformer/Attention. <br>
If you find some ignored papers, please open issues. <br>
(keep updating)

---

## Image Classification / Backbone
### Replace Conv w/ Attention
* Pure Attention:
    * **LR-Net**: "Local Relation Networks for Image Recognition", ICCV, 2019 (*Microsoft*). [[Paper](https://arxiv.org/abs/1904.11491)][[PyTorch (gan3sh500)](https://github.com/gan3sh500/local-relational-nets)]
    * **SASA**: "Stand-Alone Self-Attention in Vision Models", NeurIPS, 2019 (*Google*). [[Paper](https://arxiv.org/abs/1906.05909)][[PyTorch-1 (leaderj1001)](https://github.com/leaderj1001/Stand-Alone-Self-Attention)][[PyTorch-2 (MerHS)](https://github.com/MerHS/SASA-pytorch)]
    * **Axial-Transformer**: "Axial Attention in Multidimensional Transformers", arXiv, 2019 (rejected by ICLR 2020) (*Google*). [[Paper](https://openreview.net/forum?id=H1e5GJBtDr)][[PyTorch (lucidrains)](https://github.com/lucidrains/axial-attention)]
    * **SAN**: "Exploring Self-attention for Image Recognition", CVPR, 2020 (*CUHK + Intel*). [[Paper](https://arxiv.org/abs/2004.13621)][[PyTorch](https://github.com/hszhao/SAN)]
    * **Axial-DeepLab**: "Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation", ECCV, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2003.07853)][[PyTorch](https://github.com/csrhddlam/axial-deeplab)]
* Conv-stem + Attention:
    * **GSA**: "Global Self-Attention Networks for Image Recognition", arXiv, 2020 (rejected by ICLR 2021) (*Google*). [[Paper](https://openreview.net/forum?id=KiFeuZu24k)][[PyTorch (lucidrains)](https://github.com/lucidrains/global-self-attention-network)]
    * **HaloNet**: "Scaling Local Self-Attention For Parameter Efficient Visual Backbones", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.12731)][[PyTorch (lucidrains)](https://github.com/lucidrains/halonet-pytorch)]
    * **CoTNet**: "Contextual Transformer Networks for Visual Recognition", CVPRW, 2021 (*JD*). [[Paper](https://arxiv.org/abs/2107.12292)][[PyTorch](https://github.com/JDAI-CV/CoTNet)]
    * **TransCNN**: "Transformer in Convolutional Neural Networks", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2106.03180)]
* Conv + Attention:
    * **AA**: "Attention Augmented Convolutional Networks", ICCV, 2019 (*Google*). [[Paper](https://arxiv.org/abs/1904.09925)][[PyTorch (leaderj1001)](https://github.com/leaderj1001/Attention-Augmented-Conv2d)][[Tensorflow (titu1994)](https://github.com/titu1994/keras-attention-augmented-convs)]
    * **GCNet**: "Global Context Networks", ICCVW, 2019 (& TPAMI 2020) (*Microsoft*). [[Paper](https://arxiv.org/abs/2012.13375)][[PyTorch](https://github.com/xvjiarui/GCNet)]
    * **LambdaNetworks**: "LambdaNetworks: Modeling long-range Interactions without Attention", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=xTJEN-ggl1b)][[PyTorch-1 (lucidrains)](https://github.com/lucidrains/lambda-networks)][[PyTorch-2 (leaderj1001)](https://github.com/leaderj1001/LambdaNetworks)]
    * **BoTNet**: "Bottleneck Transformers for Visual Recognition", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2101.11605)][[PyTorch-1 (lucidrains)](https://github.com/lucidrains/bottleneck-transformer-pytorch)][[PyTorch-2 (leaderj1001)](https://github.com/leaderj1001/BottleneckTransformers)]
    * **GCT**: "Gaussian Context Transformer", CVPR, 2021 (*Zhejiang University*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.html)]
    * **CoAtNet**: "CoAtNet: Marrying Convolution and Attention for All Data Sizes", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.04803)]
    
### Vision Transformer
* For general performance:
    * **ViT**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=YicbFdNTTy)][[Tensorflow](https://github.com/google-research/vision_transformer)][[PyTorch (lucidrains)](https://github.com/lucidrains/vit-pytorch)]
    * **Perceiver**: "Perceiver: General Perception with Iterative Attention", ICML, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2103.03206)][[PyTorch (lucidrains)](https://github.com/lucidrains/perceiver-pytorch)]
    * **PiT**: "Rethinking Spatial Dimensions of Vision Transformers", ICCV, 2021 (*NAVER*). [[Paper](https://arxiv.org/abs/2103.16302)][[PyTorch](https://github.com/naver-ai/pit)]
    * **VT**: "Visual Transformers: Where Do Transformers Really Belong in Vision Models?", ICCV, 2021 (*Facebook*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Visual_Transformers_Where_Do_Transformers_Really_Belong_in_Vision_Models_ICCV_2021_paper.html)][[PyTorch (tahmid0007)](https://github.com/tahmid0007/VisualTransformers)] 
    * **PVT**: "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions", ICCV, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2102.12122)][[PyTorch](https://github.com/whai362/PVT)] 
    * **iRPE**: "Rethinking and Improving Relative Position Encoding for Vision Transformer", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.14222)][[PyTorch](https://github.com/microsoft/Cream/tree/main/iRPE)]
    * **CaiT**: "Going deeper with Image Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2103.17239)][[PyTorch](https://github.com/facebookresearch/deit)]
    * **Swin-Transformer**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.14030)][[PyTorch (berniwal)](https://github.com/berniwal/swin-transformer-pytorch)][[Code](https://github.com/microsoft/Swin-Transformer)]
    * **T2T-ViT**: "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet", ICCV, 2021 (*Yitu*). [[Paper](https://arxiv.org/abs/2101.11986)][[PyTorch](https://github.com/yitu-opensource/T2T-ViT)]
    * **FFNBN**: "Leveraging Batch Normalization for Vision Transformers", ICCVW, 2021 (*Microsoft*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/html/Yao_Leveraging_Batch_Normalization_for_Vision_Transformers_ICCVW_2021_paper.html)]
    * **DPT**: "DPT: Deformable Patch-based Transformer for Visual Recognition", ACMMM, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2107.14467)][[PyTorch](https://github.com/CASIA-IVA-Lab/DPT)]
    * **Focal**: "Focal Attention for Long-Range Interactions in Vision Transformers", NeurIPS, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.00641)][[PyTorch](https://github.com/microsoft/Focal-Transformer)]
    * **XCiT**: "XCiT: Cross-Covariance Image Transformers", NeurIPS, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.09681)]
    * **Twins**: "Twins: Revisiting Spatial Attention Design in Vision Transformers", NeurIPS, 2021 (*Meituan*). [[Paper](https://arxiv.org/abs/2104.13840)][[PyTorch)](https://github.com/Meituan-AutoML/Twins)]
    * **ARM**: "Blending Anti-Aliasing into Vision Transformer", NeurIPS, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2110.15156)][[GitHub (in construction)](https://github.com/amazon-research/anti-aliasing-transformer)]
    * **DVT**: "Not All Images are Worth 16x16 Words: Dynamic Vision Transformers with Adaptive Sequence Length", NeurIPS, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.15075)][[PyTorch](https://github.com/blackfeather-wang/Dynamic-Vision-Transformer)]
    * **Aug-S**: "Augmented Shortcuts for Vision Transformers", NeurIPS, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2106.15941)]
    * **TNT**: "Transformer in Transformer", NeurIPS, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2103.00112)][[PyTorch](https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch)][[PyTorch (lucidrains)](https://github.com/lucidrains/transformer-in-transformer)]
    * **ViTAE**: "ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias", NeurIPS, 2021 (*The University of Sydney*). [[Paper](https://arxiv.org/abs/2106.03348)][[PyTorch](https://github.com/Annbless/ViTAE)]
    * **DeepViT**: "DeepViT: Towards Deeper Vision Transformer", arXiv, 2021 (*NUS + ByteDance*). [[Paper](https://arxiv.org/abs/2103.11886)][[Code](https://github.com/zhoudaquan/dvit_repo)]
    * **So-ViT**: "So-ViT: Mind Visual Tokens for Vision Transformer", arXiv, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2104.10935)][[PyTorch](https://github.com/jiangtaoxie/So-ViT)]
    * **LV-ViT**: "All Tokens Matter: Token Labeling for Training Better Vision Transformers", NeurIPS, 2021 (*ByteDance*). [[Paper](https://arxiv.org/abs/2104.10858)][[PyTorch](https://github.com/zihangJiang/TokenLabeling)]
    * **NesT**: "Aggregating Nested Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.12723)][[Tensorflow](https://github.com/google-research/nested-transformer)]
    * **LIT**: "Less is More: Pay Less Attention in Vision Transformers", arXiv, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2105.14217)][[PyTorch](https://github.com/MonashAI/LIT)]
    * **MSG-Transformer**: "MSG-Transformer: Exchanging Local Spatial Information by Manipulating Messenger Tokens", arXiv, 2021 (*Huazhong University of Science & Technology*). [[Paper](https://arxiv.org/abs/2105.15168)][[PyTorch](https://github.com/hustvl/MSG-Transformer)]
    * **KVT**: "KVT: k-NN Attention for Boosting Vision Transformers", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.00515)]
    * **Refined-ViT**: "Refiner: Refining Self-attention for Vision Transformers", arXiv, 2021 (*NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.03714)][[PyTorch](https://github.com/zhoudaquan/Refiner_ViT)]
    * **Shuffle-Transformer**: "Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer", arXiv, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2106.03650)]
    * **ViT-G**: "Scaling Vision Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.04560)]
    * **CAT**: "CAT: Cross Attention in Vision Transformer", arXiv, 2021 (*KuaiShou*). [[Paper](https://arxiv.org/abs/2106.05786)][[PyTorch](https://github.com/linhezheng19/CAT)]
    * **V-MoE**: "Scaling Vision with Sparse Mixture of Experts", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.05974)]
    * **P2T**: "P2T: Pyramid Pooling Transformer for Scene Understanding", arXiv, 2021 (*Nankai University*). [[Paper](https://arxiv.org/abs/2106.12011)]
    * **PvTv2**: "PVTv2: Improved Baselines with Pyramid Vision Transformer", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2106.13797)][[PyTorch](https://github.com/whai362/PVT)]
    * **CSWin**: "CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.00652)][[PyTorch](https://github.com/microsoft/CSWin-Transformer)]
    * **LG-Transformer**: "Local-to-Global Self-Attention in Vision Transformers", arXiv, 2021 (*IIAI, UAE*). [[Paper](https://arxiv.org/abs/2107.04735)]
    * **ViP**: "Visual Parser: Representing Part-whole Hierarchies with Transformers", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2107.05790)]
    * **Scaled-ReLU**: "Scaled ReLU Matters for Training Vision Transformers", AAAI, 2022 (*Alibaba*). [[Paper](https://arxiv.org/abs/2109.03810)]
    * **DTN**: "Dynamic Token Normalization Improves Vision Transformer", ICLR, 2022 (*Tencent*). [[Paper](https://arxiv.org/abs/2112.02624)][[PyTorch (in construction)](https://github.com/wqshao126/DTN)]
    * **RegionViT**: "RegionViT: Regional-to-Local Attention for Vision Transformers", ICLR, 2022 (*MIT-IBM Watson*). [[Paper](https://arxiv.org/abs/2106.02689)][[PyTorch](https://github.com/ibm/regionvit)]
    * **CrossFormer**: "CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention", ICLR, 2022 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2108.00154)][[PyTorch](https://github.com/cheerss/CrossFormer)]
    * **?**: "Scaling the Depth of Vision Transformers via the Fourier Domain Analysis", ICLR, 2022 (*UT Austin*). [[Paper](https://openreview.net/forum?id=O476oWmiNNp)]
    * **BViT**: "BViT: Broad Attention based Vision Transformer", arXiv, 2022 (*CAS*). [[Paper](https://arxiv.org/abs/2202.06268)]
    * **PyramidTNT**: "PyramidTNT: Improved Transformer-in-Transformer Baselines with Pyramid Architecture", arXiv, 2022 (*Huawei*). [[Paper](https://arxiv.org/abs/2201.00978)][[PyTorch](https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch)]
    * **DAT**: "Vision Transformer with Deformable Attention", arXiv, 2022 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2201.00520)][[PyTorch](https://github.com/LeapLabTHU/DAT)]
    * **O-ViT**: "O-ViT: Orthogonal Vision Transformer", arXiv, 2022 (*East China Normal University*). [[Paper](https://arxiv.org/abs/2201.12133)]
    * **MOA-Transformer**: "Aggregating Global Features into Local Vision Transformer", arXiv, 2022 (*University of Kansas*). [[Paper](https://arxiv.org/abs/2201.12903)][[PyTorch](https://github.com/krushi1992/MOA-transformer)]
    * **BOAT**: "BOAT: Bilateral Local Attention Vision Transformer", arXiv, 2022 (*Baidu + HKU*). [[Paper](https://arxiv.org/abs/2201.13027)]
    * **ViTAEv2**: "ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond", arXiv, 2022 (*The University of Sydney*). [[Paper](https://arxiv.org/abs/2202.10108)]
    * **VAN**: "Visual Attention Network", arXiv, 2022 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2202.09741)][[PyTorch](https://github.com/Visual-Attention-Network)]
    * **HiP**: "Hierarchical Perceiver", arXiv, 2022 (*DeepMind*). [[Paper](https://arxiv.org/abs/2202.10890)]
    * **PatchMerger**: "Learning to Merge Tokens in Vision Transformers", arXiv, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2202.12015)]
* For efficiency:
    * **DeiT**: "Training data-efficient image transformers & distillation through attention", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2012.12877)][[PyTorch](https://github.com/facebookresearch/deit)]
    * **ConViT**: "ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2103.10697)][[Code](https://github.com/facebookresearch/convit)]
    * **?**: "Improving the Efficiency of Transformers for Resource-Constrained Devices", DSD, 2021 (*NavInfo Europe, Netherlands*). [[Paper](https://arxiv.org/abs/2106.16006)]
    * **PS-ViT**: "Vision Transformer with Progressive Sampling", ICCV, 2021 (*CPII*). [[Paper](https://arxiv.org/abs/2108.01684)]
    * **HVT**: "Scalable Visual Transformers with Hierarchical Pooling", ICCV, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2103.10619)][[PyTorch](https://github.com/MonashAI/HVT)]
    * **CrossViT**: "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification", ICCV, 2021 (*MIT-IBM*). [[Paper](https://arxiv.org/abs/2103.14899)][[PyTorch](https://github.com/IBM/CrossViT)]
    * **ViL**: "Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.15358)][[PyTorch](https://github.com/microsoft/vision-longformer)]
    * **Visformer**: "Visformer: The Vision-friendly Transformer", ICCV, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2104.12533)][[PyTorch](https://github.com/danczs/Visformer)]
    * **SViTE**: "Chasing Sparsity in Vision Transformers: An End-to-End Exploration", NeurIPS, 2021 (*UT Austin*). [[Paper](https://arxiv.org/abs/2106.04533)][[PyTorch](https://github.com/VITA-Group/SViTE)]
    * **DGE**: "Dynamic Grained Encoder for Vision Transformers", NeurIPS, 2021 (*Megvii*). [[Paper](https://papers.nips.cc/paper/2021/hash/2d969e2cee8cfa07ce7ca0bb13c7a36d-Abstract.html)][[PyTorch](https://github.com/StevenGrove/vtpack)]
    * **GG-Transformer**: "Glance-and-Gaze Vision Transformer", NeurIPS, 2021 (*JHU*). [[Paper](https://arxiv.org/abs/2106.02277)][[Code (in construction)](https://github.com/yucornetto/GG-Transformer)]
    * **DynamicViT**: "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification", NeurIPS, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2106.02034)][[PyTorch](https://github.com/raoyongming/DynamicViT)][[Website](https://dynamicvit.ivg-research.xyz/)]
    * **ResT**: "ResT: An Efficient Transformer for Visual Recognition", NeurIPS, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2105.13677)][[PyTorch](https://github.com/wofmanaf/ResT)]
    * **Adder-Transformer**: "Adder Attention for Vision Transformer", NeurIPS, 2021 (*Huawei*). [[Paper](https://proceedings.neurips.cc/paper/2021/hash/a57e8915461b83adefb011530b711704-Abstract.html)]
    * **SOFT**: "SOFT: Softmax-free Transformer with Linear Complexity", NeurIPS, 2021 (*Fudan*). [[Paper](https://arxiv.org/abs/2110.11945)][[PyTorch](https://github.com/fudan-zvg/SOFT)][[Website](https://fudan-zvg.github.io/SOFT/)]
    * **IA-RED<sup>2</sup>**: "IA-RED<sup>2</sup>: Interpretability-Aware Redundancy Reduction for Vision Transformers", NeurIPS, 2021 (*MIT-IBM*). [[Paper](https://arxiv.org/abs/2106.12620)][[Website](http://people.csail.mit.edu/bpan/ia-red/)]
    * **LocalViT**: "LocalViT: Bringing Locality to Vision Transformers", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2104.05707)][[PyTorch](https://github.com/ofsoundof/LocalViT)]
    * **CCT**: "Escaping the Big Data Paradigm with Compact Transformers", arXiv, 2021 (*University of Oregon*). [[Paper](https://arxiv.org/abs/2104.05704)][[PyTorch](https://github.com/SHI-Labs/Compact-Transformers)]
    * **Over-Smoothing**: "Improve Vision Transformers Training by Suppressing Over-smoothing", arXiv, 2021 (*UT Austin + Facebook*). [[Paper](https://arxiv.org/abs/2104.12753)][[PyTorch](https://github.com/ChengyueGongR/PatchVisionTransformer)] 
    * **SL-ViT**: "Single-Layer Vision Transformers for More Accurate Early Exits with Less Overhead", arXiv, 2021 (*Aarhus University*). [[Paper](https://arxiv.org/abs/2105.09121)]
    * **PS-ViT**: "Patch Slimming for Efficient Vision Transformers", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2106.02852)]
    * **?**: "Multi-Exit Vision Transformer for Dynamic Inference", arXiv, 2021 (*Aarhus University, Denmark*). [[Paper](https://arxiv.org/abs/2106.15183)]
    * **DeiT-Manifold**: "Efficient Vision Transformers via Fine-Grained Manifold Distillation", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2107.01378)]
    * **ViX**: "Vision Xformers: Efficient Attention for Image Classification", arXiv, 2021 (*Indian Institute of Technology Bombay*). [[Paper](https://arxiv.org/abs/2107.02239)]
    * **Transformer-LS**: "Long-Short Transformer: Efficient Transformers for Language and Vision", NeurIPS, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2107.02192)][[PyTorch](https://github.com/NVIDIA/transformer-ls)]
    * **WideNet**: "Go Wider Instead of Deeper", arXiv, 2021 (*NUS*). [[Paper](https://arxiv.org/abs/2107.11817)]
    * **Armour**: "Armour: Generalizable Compact Self-Attention for Vision Transformers", arXiv, 2021 (*Arm*). [[Paper](https://arxiv.org/abs/2108.01778)]
    * **Mobile-Former**: "Mobile-Former: Bridging MobileNet and Transformer", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2108.05895)]
    * **IPE**: "Exploring and Improving Mobile Level Vision Transformers", arXiv, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2108.13015)]
    * **DS-Net++**: "DS-Net++: Dynamic Weight Slicing for Efficient Inference in CNNs and Transformers", arXiv, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2109.10060)][[PyTorch](https://github.com/changlin31/DS-Net)]
    * **UFO-ViT**: "UFO-ViT: High Performance Linear Vision Transformer without Softmax", arXiv, 2021 (*Kakao*). [[Paper](https://arxiv.org/abs/2109.14382)]
    * **Token-Pooling**: "Token Pooling in Visual Transformers", arXiv, 2021 (*Apple*). [[Paper](https://arxiv.org/abs/2110.03860)]
    * **SReT**: "Sliced Recursive Transformer", arXiv, 2021 (*CMU + MBZUAI*). [[Paper](https://arxiv.org/abs/2111.05297)][[PyTorch](https://github.com/szq0214/SReT)]
    * **Evo-ViT**: "Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer", AAAI, 2022 (*Tencent*). [[Paper](https://arxiv.org/abs/2108.01390)][[PyTorch](https://github.com/YifanXu74/Evo-ViT)]
    * **PS-Attention**: "Pale Transformer: A General Vision Transformer Backbone with Pale-Shaped Attention", AAAI, 2022 (*Baidu*). [[Paper](https://arxiv.org/abs/2112.14000)][[Paddle](https://github.com/BR-IDL/PaddleViT)]
    * **ShiftViT**: "When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism", AAAI, 2022 (*Microsoft*). [[Paper](https://arxiv.org/abs/2201.10801)][[PyTorch](https://github.com/microsoft/SPACH)]
    * **EViT**: "Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations", ICLR, 2022 (*Tencent*). [[Paper](https://arxiv.org/abs/2202.07800)][[PyTorch](https://github.com/youweiliang/evit)]
    * **QuadTree**: "QuadTree Attention for Vision Transformers", ICLR, 2022 (*Simon Fraser + Alibaba*). [[Paper](https://arxiv.org/abs/2201.02767)][[PyTorch](https://github.com/Tangshitao/QuadtreeAttention)]
    * **TerViT**: "TerViT: An Efficient Ternary Vision Transformer", arXiv, 2022 (*Beihang University*). [[Paper](https://arxiv.org/abs/2201.08050)]
    * **MT-ViT**: "Multi-Tailed Vision Transformer for Efficient Inference", arXiv, 2022 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2203.01587)]
* Conv + Transformer:
    * **LeViT**: "LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.01136)][[PyTorch](https://github.com/facebookresearch/LeViT)]
    * **CeiT**: "Incorporating Convolution Designs into Visual Transformers", ICCV, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/2103.11816)][[PyTorch (rishikksh20)](https://github.com/rishikksh20/CeiT)]
    * **Conformer**: "Conformer: Local Features Coupling Global Representations for Visual Recognition", ICCV, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2105.03889)][[PyTorch](https://github.com/pengzhiliang/Conformer)]
    * **CoaT**: "Co-Scale Conv-Attentional Image Transformers", ICCV, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2104.06399)][[PyTorch](https://github.com/mlpc-ucsd/CoaT)]
    * **CvT**: "CvT: Introducing Convolutions to Vision Transformers", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.15808)][[Code](https://github.com/leoxiaobin/CvT)]
    * **ViTc**: "Early Convolutions Help Transformers See Better", NeurIPS, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.14881)]
    * **ConTNet**: "ConTNet: Why not use convolution and transformer at the same time?", arXiv, 2021 (*ByteDance*). [[Paper](https://arxiv.org/abs/2104.13497)][[PyTorch](https://github.com/yan-hao-tian/ConTNet)]
    * **CMT**: "CMT: Convolutional Neural Networks Meet Vision Transformers", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2107.06263)]
    * **SPACH**: "A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2108.13002)]
    * **MobileViT**: "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer", ICLR, 2022 (*Apple*). [[Paper](https://arxiv.org/abs/2110.02178)]
    * **CXV**: "Convolutional Xformers for Vision", arXiv, 2022 (*IIT Bombay*). [[Paper](https://arxiv.org/abs/2201.10271)][[PyTorch](https://github.com/pranavphoenix/CXV)]
    * **ConvMixer**: "Patches Are All You Need?", arXiv, 2022 (*CMU*). [[Paper](https://arxiv.org/abs/2201.09792)][[PyTorch](https://github.com/locuslab/convmixer)]
    * **UniFormer**: "UniFormer: Unifying Convolution and Self-attention for Visual Recognition", arXiv, 2022 (*SenseTime*). [[Paper](https://arxiv.org/abs/2201.09450)][[PyTorch](https://github.com/Sense-X/UniFormer)]
* For pre-training:
    * **iGPT**: "Generative Pretraining From Pixels", ICML, 2020 (*OpenAI*). [[Paper](http://proceedings.mlr.press/v119/chen20s.html)][[Tensorflow](https://github.com/openai/image-gpt)]
    * **MoCo-V3**: "An Empirical Study of Training Self-Supervised Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.02057)]
    * **DINO**: "Emerging Properties in Self-Supervised Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.14294)][[PyTorch](https://github.com/facebookresearch/dino)]
    * **drloc**: "Efficient Training of Visual Transformers with Small Datasets", NeurIPS, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2106.03746)][[PyTorch](https://github.com/yhlleo/VTs-Drloc)]
    * **CARE**: "Revitalizing CNN Attentions via Transformers in Self-Supervised Visual Representation Learning", NeurIPS, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2110.05340)][[PyTorch](https://github.com/ChongjianGE/CARE)]
    * **MST**: "MST: Masked Self-Supervised Transformer for Visual Representation", NeurIPS, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/2106.05656)]
    * **SiT**: "SiT: Self-supervised Vision Transformer", arXiv, 2021 (*University of Surrey*). [[Paper](https://arxiv.org/abs/2104.03602)][[PyTorch](https://github.com/Sara-Ahmed/SiT)]
    * **MoBY**: "Self-Supervised Learning with Swin Transformers", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2105.04553)][[Pytorch](https://github.com/SwinTransformer/Transformer-SSL)]
    * **?**: "Investigating Transfer Learning Capabilities of Vision Transformers and CNNs by Fine-Tuning a Single Trainable Block", arXiv, 2021 (*Pune Institute of Computer Technology, India*). [[Paper](https://arxiv.org/abs/2110.05270)]
    * **MAE**: "Masked Autoencoders Are Scalable Vision Learners", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2111.06377)][[PyTorch](https://github.com/facebookresearch/mae)][[PyTorch (pengzhiliang)](https://github.com/pengzhiliang/MAE-pytorch)]
    * **Annotations-1.3B**: "Billion-Scale Pretraining with Vision Transformers for Multi-Task Visual Representations", WACV, 2022 (*Pinterest*). [[Paper](https://arxiv.org/abs/2108.05887)]
    * **BEiT**: "BEiT: BERT Pre-Training of Image Transformers", ICLR, 2022 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.08254)][[PyTorch](https://github.com/microsoft/unilm/tree/master/beit)]
    * **EsViT**: "Efficient Self-supervised Vision Transformers for Representation Learning", ICLR, 2022 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.09785)]
    * **iBOT**: "Image BERT Pre-training with Online Tokenizer", ICLR, 2022 (*ByteDance*). [[Paper](https://arxiv.org/abs/2111.07832)][[PyTorch](https://github.com/bytedance/ibot)]
    * **IDMM**: "Training Vision Transformers with Only 2040 Images", arXiv, 2022 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2201.10728)]
    * **RePre**: "RePre: Improving Self-Supervised Vision Transformer with Reconstructive Pre-training", arXiv, 2022 (*Beijing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2201.06857)]
* For robustness:
    * **ViT-Robustness**: "Understanding Robustness of Transformers for Image Classification", ICCV, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.14586)]
    * **SAGA**: "On the Robustness of Vision Transformers to Adversarial Examples", ICCV, 2021 (*University of Connecticut*). [[Paper](https://arxiv.org/abs/2104.02610)]
    * **ViTs-vs-CNNs**: "Are Transformers More Robust Than CNNs?", NeurIPS, 2021 (*JHU*). [[Paper](https://arxiv.org/abs/2111.05464)][[PyTorch](https://github.com/ytongbai/ViTs-vs-CNNs)]
    * **RVT**: "Rethinking the Design Principles of Robust Vision Transformer", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2105.07926)][[PyTorch](https://github.com/vtddggg/Robust-Vision-Transformer)]
    * **T-CNN**: "Transformed CNNs: recasting pre-trained convolutional layers with self-attention", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.05795)]
    * **Generalization-Enhanced-ViT**: "Delving Deep into the Generalization of Vision Transformers under Distribution Shifts", arXiv, 2021 (*Beihang University + NTU, Singapore*). [[Paper](https://arxiv.org/abs/2106.07617)]
    * **Transformer-Attack**: "On the Adversarial Robustness of Visual Transformers", arXiv, 2021 (*Xi'an Jiaotong*). [[Paper](https://arxiv.org/abs/2103.15670)]
    * **?**: "Reveal of Vision Transformers Robustness against Adversarial Attacks", arXiv, 2021 (*University of Rennes*). [[Paper](https://arxiv.org/abs/2106.03734)]
    * **?**: "On Improving Adversarial Transferability of Vision Transformers", arXiv, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2106.04169)][[PyTorch](https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers)]
    * **?**: "Exploring Corruption Robustness: Inductive Biases in Vision Transformers and MLP-Mixers", arXiv, 2021 (*University of Pittsburgh*). [[Paper](https://arxiv.org/abs/2106.13122)]
    * **?**: "Adversarial Robustness Comparison of Vision Transformer and MLP-Mixer to CNNs", arXiv, 2021 (*KAIST*). [[Paper](https://arxiv.org/abs/2110.02797)]
    * **Token-Attack**: "Adversarial Token Attacks on Vision Transformers", arXiv, 2021 (*New York University*). [[Paper](https://arxiv.org/abs/2110.04337)]
    * **Smooth-ViT**: "Certified Patch Robustness via Smoothed Vision Transformers", arXiv, 2021 (*MIT*). [[Paper](https://arxiv.org/abs/2110.07719)][[PyTorch](https://github.com/MadryLab/smoothed-vit)]
    * **?**: "Discrete Representations Strengthen Vision Transformer Robustness", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2111.10493)]
    * **?**: "Vision Transformers are Robust Learners", AAAI, 2022 (*PyImageSearch + IBM*). [[Paper](https://arxiv.org/abs/2105.07581)][[Tensorflow](https://github.com/sayakpaul/robustness-vit)]
    * **PNA**: "Towards Transferable Adversarial Attacks on Vision Transformers", AAAI, 2022 (*Fudan + Maryland*). [[Paper](https://arxiv.org/abs/2109.04176)][[PyTorch](https://github.com/zhipeng-wei/PNA-PatchOut)]
    * **MIA-Former**: "MIA-Former: Efficient and Robust Vision Transformers via Multi-grained Input-Adaptation", AAAI, 2022 (*Rice University*). [[Paper](https://arxiv.org/abs/2112.11542)]
    * **Patch-Fool**: "Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?", ICLR, 2022 (*Rice University*). [[Paper](https://openreview.net/forum?id=28ib9tf6zhr)]
* Model Compression:
    * **ViT-quant**: "Post-Training Quantization for Vision Transformer", NeurIPS, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2106.14156)]
    * **VTP**: "Visual Transformer Pruning", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2104.08500)]
    * **NViT**: "NViT: Vision Transformer Compression and Parameter Redistribution", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2110.04869)]
    * **MD-ViT**: "Multi-Dimensional Model Compression of Vision Transformer", arXiv, 2021 (*Princeton*). [[Paper](https://arxiv.org/abs/2201.00043)]
    * **PTQ4ViT**: "PTQ4ViT: Post-Training Quantization Framework for Vision Transformers", arXiv, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2111.12293)]
    * **FQ-ViT**: "FQ-ViT: Fully Quantized Vision Transformer without Retraining", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2111.13824)][[PyTorch](https://github.com/linyang-zhh/FQ-ViT)]
    * **UVC**: "Unified Visual Transformer Compression", ICLR, 2022 (*UT Austin*). [[Paper](https://openreview.net/forum?id=9jsZiUgkCZP)]
    * **Q-ViT**: "Q-ViT: Fully Differentiable Quantization for Vision Transformer", arXiv, 2022 (*MEGVII*). [[Paper](https://arxiv.org/abs/2201.07703)]
    * **VAQF**: "VAQF: Fully Automatic Software-Hardware Co-Design Framework for Low-Bit Vision Transformer", arXiv, 2022 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2201.06618)]

### MLP
* **RepMLP**: "RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2105.01883)][[PyTorch](https://github.com/DingXiaoH/RepMLP)]
* **EAMLP**: "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks", arXiv, 2021 (*Tsinghua University*). [[Paper](https://arxiv.org/abs/2105.02358)]
* **Forward-Only**: "Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2105.02723)][[PyTorch](https://github.com/lukemelas/do-you-even-need-attention)]
* **ResMLP**: "ResMLP: Feedforward networks for image classification with data-efficient training", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2105.03404)]
* **?**: "Can Attention Enable MLPs To Catch Up With CNNs?", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.15078)]
* **ViP**: "Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition", arXiv, 2021 (*NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.12368)][[PyTorch](https://github.com/Andrew-Qibin/VisionPermutator)]
* **CCS**: "Rethinking Token-Mixing MLP for MLP-based Vision Backbone", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.14882)]
* **S<sup>2</sup>-MLPv2**: "S<sup>2</sup>-MLPv2: Improved Spatial-Shift MLP Architecture for Vision", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2108.01072)]
* **RaftMLP**: "RaftMLP: Do MLP-based Models Dream of Winning Over Computer Vision?", arXiv, 2021 (*Rikkyo University, Japan*). [[Paper](https://arxiv.org/abs/2108.04384)][[PyTorch](https://github.com/okojoalg/raft-mlp)]
* **Hire-MLP**: "Hire-MLP: Vision MLP via Hierarchical Rearrangement", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2108.13341)]
* **Sparse-MLP**: "Sparse-MLP: A Fully-MLP Architecture with Conditional Computation", arXiv, 2021 (*NUS*). [[Paper](https://arxiv.org/abs/2109.02008)]
* **ConvMLP**: "ConvMLP: Hierarchical Convolutional MLPs for Vision", arXiv, 2021 (*University of Oregon*). [[Paper](https://arxiv.org/abs/2109.04454)][[PyTorch](https://github.com/SHI-Labs/Convolutional-MLPs)]
* **sMLP**: "Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2109.05422)]
* **MLP-Mixer**: "MLP-Mixer: An all-MLP Architecture for Vision", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.01601)][[Tensorflow](https://github.com/google-research/vision_transformer)][[PyTorch-1 (lucidrains)](https://github.com/lucidrains/mlp-mixer-pytorch)][[PyTorch-2 (rishikksh20)](https://github.com/rishikksh20/MLP-Mixer-pytorch)]
* **gMLP**: "Pay Attention to MLPs", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.08050)][[PyTorch (antonyvigouret)](https://github.com/antonyvigouret/Pay-Attention-to-MLPs)]
* **S<sup>2</sup>-MLP**: "S<sup>2</sup>-MLP: Spatial-Shift MLP Architecture for Vision", WACV, 2022 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.07477)]
* **CycleMLP**: "CycleMLP: A MLP-like Architecture for Dense Prediction", ICLR, 2022 (*HKU*). [[Paper](https://arxiv.org/abs/2107.10224)][[PyTorch](https://github.com/ShoufaChen/CycleMLP)]
* **AS-MLP**: "AS-MLP: An Axial Shifted MLP Architecture for Vision", ICLR, 2022 (*ShanghaiTech University*). [[Paper](https://arxiv.org/abs/2107.08391)][[PyTorch](https://github.com/svip-lab/AS-MLP)]
* **MS-MLP**: "Mixing and Shifting: Exploiting Global and Local Dependencies in Vision MLPs", arXiv, 2022 (*Microsoft*). [[Paper](https://arxiv.org/abs/2202.06510)]

### Analysis
* **Attention-CNN**: "On the Relationship between Self-Attention and Convolutional Layers", ICLR, 2020 (*EPFL*). [[Paper](https://openreview.net/forum?id=HJlnC1rKPB)][[PyTorch](https://github.com/epfml/attention-cnn)][[Website](https://epfml.github.io/attention-cnn/)]
* **Transformer-Explainability**: "Transformer Interpretability Beyond Attention Visualization", CVPR, 2021 (*Tel Aviv*). [[Paper](https://arxiv.org/abs/2012.09838)][[PyTorch](https://github.com/hila-chefer/Transformer-Explainability)]
* **?**: "Are Convolutional Neural Networks or Transformers more like human vision?", CogSci, 2021 (*Princeton*). [[Paper](https://arxiv.org/abs/2105.07197)]
* **?**: "ConvNets vs. Transformers: Whose Visual Representations are More Transferable?", ICCVW, 2021 (*HKU*). [[Paper](https://arxiv.org/abs/2108.05305)]
* **?**: "Do Vision Transformers See Like Convolutional Neural Networks?", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2108.08810)]
* **?**: "Intriguing Properties of Vision Transformers", NeurIPS, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2105.10497)][[PyTorch](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers)]
* **FoveaTer**: "FoveaTer: Foveated Transformer for Image Classification", arXiv, 2021 (*UCSB*). [[Paper](https://arxiv.org/abs/2105.14173)]
* **?**: "Demystifying Local Vision Transformer: Sparse Connectivity, Weight Sharing, and Dynamic Weight", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.04263)]
* **?**: "Revisiting the Calibration of Modern Neural Networks", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.07998)]
* **?**: "How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.10270)][[Tensorflow](https://github.com/google-research/vision_transformer)][[PyTorch (rwightman)](https://github.com/rwightman/pytorch-image-models)]
* **?**: "What Makes for Hierarchical Vision Transformer?", arXiv, 2021 (*Horizon Robotic*). [[Paper](https://arxiv.org/abs/2107.02174)]
* **?**: "Visualizing Paired Image Similarity in Transformer Networks", WACV, 2022 (*Temple University*). [[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Black_Visualizing_Paired_Image_Similarity_in_Transformer_Networks_WACV_2022_paper.html)][[PyTorch](https://github.com/vidarlab/xformer-paired-viz)]
* **FDSL**: "Can Vision Transformers Learn without Natural Images?", AAAI, 2022 (*AIST*). [[Paper](https://arxiv.org/abs/2103.13023)][[PyTorch](https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch)][[Website](https://hirokatsukataoka16.github.io/Vision-Transformers-without-Natural-Images/)]
* **AlterNet**: "How Do Vision Transformers Work?", ICLR, 2022 (*Yonsei University*). [[Paper](https://arxiv.org/abs/2202.06709)][[PyTorch](https://github.com/xxxnell/how-do-vits-work)]
* **?**: "When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations", ICLR, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2106.01548)][[Tensorflow](https://github.com/google-research/vision_transformer)]
* **?**: "On the Connection between Local Attention and Dynamic Depth-wise Convolution", ICLR, 2022 (*Microsoft*). [[Paper](https://openreview.net/forum?id=L3_SsSNMmy)]


## Detection
### Object Detection
* CNN-based backbone:
    * **DETR**: "End-to-End Object Detection with Transformers", ECCV, 2020 (*Facebook*). [[Paper](https://arxiv.org/abs/2005.12872)][[PyTorch](https://github.com/facebookresearch/detr)]
    * **Deformable DETR**: "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/2010.04159)][[PyTorch](https://github.com/fundamentalvision/Deformable-DETR)]
    * **UP-DETR**: "UP-DETR: Unsupervised Pre-training for Object Detection with Transformers", CVPR, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2011.09094)][[PyTorch](https://github.com/dddzg/up-detr)]
    * **SMCA**: "Fast Convergence of DETR with Spatially Modulated Co-Attention", ICCV, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2108.02404)][[PyTorch](https://github.com/gaopengcuhk/SMCA-DETR)]
    * **Conditional-DETR**: "Conditional DETR for Fast Training Convergence", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2108.06152)]
    * **PnP-DETR**: "PnP-DETR: Towards Efficient Visual Analysis with Transformers", ICCV, 2021 (*Yitu*). [[Paper](https://arxiv.org/abs/2109.07036)][[Code (in construction)](https://github.com/twangnh/pnp-detr)]
    * **TSP**: "Rethinking Transformer-based Set Prediction for Object Detection", ICCV, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2011.10881)]
    * **Dynamic-DETR**: "Dynamic DETR: End-to-End Object Detection With Dynamic Attention", ICCV, 2021 (*Microsoft*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.html)]
    * **ViT-YOLO**: "ViT-YOLO:Transformer-Based YOLO for Object Detection", ICCVW, 2021 (*Xidian University*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Zhang_ViT-YOLOTransformer-Based_YOLO_for_Object_Detection_ICCVW_2021_paper.html)]
    * **ACT**: "End-to-End Object Detection with Adaptive Clustering Transformer", arXiv, 2021 (*Peking + CUHK*). [[Paper](https://arxiv.org/abs/2011.09315)]
    * **Efficient-DETR**: "Efficient DETR: Improving End-to-End Object Detector with Dense Prior", arXiv, 2021 (*Megvii*). [[Paper](https://arxiv.org/abs/2104.01318)]
    * **CA-FPN**: "Content-Augmented Feature Pyramid Network with Light Linear Transformers", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2105.09464)]
    * **DETReg**: "DETReg: Unsupervised Pretraining with Region Priors for Object Detection", arXiv, 2021 (*Tel-Aviv + Berkeley*). [[Paper](https://arxiv.org/abs/2106.04550)][[Website](https://www.amirbar.net/detreg/)]
    * **GQPos**: "Guiding Query Position and Performing Similar Attention for Transformer-Based Detection Heads", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2108.09691)]
    * **Anchor-DETR**: "Anchor DETR: Query Design for Transformer-Based Detector", AAAI, 2022 (*MEGVII*). [[Paper](https://arxiv.org/abs/2109.07107)][[PyTorch](https://github.com/megvii-research/AnchorDETR)]
    * **Sparse-DETR**: "Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity", ICLR, 2022 (*Kakao*). [[Paper](https://arxiv.org/abs/2111.14330)][[PyTorch](https://github.com/kakaobrain/sparse-detr)]
    * **DAB-DETR**: "DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR", ICLR, 2022 (*IDEA, China*). [[Paper](https://arxiv.org/abs/2201.12329)][[Code (in construction)](https://github.com/SlongLiu/DAB-DETR)]
    * **DN-DETR**: "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising", CVPR, 2022 (*International Digital Economy Academy, China*). [[Paper](https://arxiv.org/abs/2203.01305)][[Code (in construction)](https://github.com/FengLi-ust/DN-DETR)]
* Transformer-based backbone:
    * **ViT-FRCNN**: "Toward Transformer-Based Object Detection", arXiv, 2020 (*Pinterest*). [[Paper](https://arxiv.org/abs/2012.09958)]
    * **WB-DETR**: "WB-DETR: Transformer-Based Detector Without Backbone", ICCV, 2021 (*CAS*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_WB-DETR_Transformer-Based_Detector_Without_Backbone_ICCV_2021_paper.html)]
    * **YOLOS**: "You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection", NeurIPS, 2021 (*Horizon Robotics*). [[Paper](https://arxiv.org/abs/2106.00666)][[PyTorch](https://github.com/hustvl/YOLOS)]
    * **ViDT**: "ViDT: An Efficient and Effective Fully Transformer-based Object Detector", ICLR, 2022 (*NAVER*). [[Paper](https://arxiv.org/abs/2110.03921)][[PyTorch](https://github.com/naver-ai/vidt)]
    * **FP-DETR**: "FP-DETR: Detection Transformer Advanced by Fully Pre-training", ICLR, 2022 (*USTC*). [[Paper](https://openreview.net/forum?id=yjMQuLLcGWK)]
    * **D<sup>2</sup>ETR**: "D<sup>2</sup>ETR: Decoder-Only DETR with Computationally Efficient Cross-Scale Attention", arXiv, 2022 (*Alibaba*). [[Paper](https://arxiv.org/abs/2203.00860)]

### 3D Object Detection
* **AST-GRU**: "LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention", CVPR, 2020 (*Baidu*). [[Paper](https://arxiv.org/abs/2004.01389)][[Code (in construction)](https://github.com/yinjunbo/3DVID)]
* **Pointformer**: "3D Object Detection with Pointformer", arXiv, 2020 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2012.11409)]
* **CT3D**: "Improving 3D Object Detection with Channel-wise Transformer", ICCV, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2108.10723)][[Code (in construction)](https://github.com/hlsheng1/CT3D)]
* **Group-Free-3D**: "Group-Free 3D Object Detection via Transformers", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2104.00678)][[PyTorch](https://github.com/zeliu98/Group-Free-3D)]
* **VoTr**: "Voxel Transformer for 3D Object Detection", ICCV, 2021 (*CUHK + NUS*). [[Paper](https://arxiv.org/abs/2109.02497)]
* **3DETR**: "An End-to-End Transformer Model for 3D Object Detection", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2109.08141)][[PyTorch](https://github.com/facebookresearch/3detr)][[Website](https://facebookresearch.github.io/3detr/)]
* **DETR3D**: "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries", CoRL, 2021 (*MIT*). [[Paper](https://arxiv.org/abs/2110.06922)]
* **M3DETR**: "M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers", WACV, 2022 (*University of Maryland*). [[Paper](https://arxiv.org/abs/2104.11896)][[PyTorch](https://github.com/rayguan97/M3DETR)]

### Other Detection Tasks
* Unsupervised:
    * **LOST**: "Localizing Objects with Self-Supervised Transformers and no Labels", arXiv, 2021 (*Valeo.ai*). [[Paper](https://arxiv.org/abs/2109.14279)][[PyTorch](https://github.com/valeoai/LOST)]
    * **TokenCut**: "Self-Supervised Transformers for Unsupervised Object Discovery using Normalized Cut", arXiv, 2022 (*Univ. Grenoble Alpes, France*). [[Paper](https://arxiv.org/abs/2202.11539)][[Website](https://www.m-psi.fr/Papers/TokenCut2022/)]
* Pedestrian Detection:
    * **PED**: "DETR for Crowd Pedestrian Detection", arXiv, 2020 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2012.06785)][[PyTorch](https://github.com/Hatmm/PED-DETR-for-Pedestrian-Detection)]
    * **Pedestron**: "Pedestrian Detection: Domain Generalization, CNNs, Transformers and Beyond", arXiv, 2022 (*IIAI*). [[Paper](https://arxiv.org/abs/2201.03176)][[PyTorch](https://github.com/hasanirtiza/Pedestron)]
* Lane Detection:
    * **LSTR**: "End-to-end Lane Shape Prediction with Transformers", WACV, 2021 (*Xi'an Jiaotong*). [[Paper](https://arxiv.org/abs/2011.04233)][[PyTorch](https://github.com/liuruijin17/LSTR)]
    * **LETR**: "Line Segment Detection Using Transformers without Edges", CVPR, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2101.01909)][[PyTorch](https://github.com/mlpc-ucsd/LETR)]
* Object Localization:
    * **TS-CAM**: "TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2103.14862)]
    * **LCTR**: "LCTR: On Awakening the Local Continuity of Transformer for Weakly Supervised Object Localization", AAAI, 2022 (*Xiamen University*). [[Paper](https://arxiv.org/abs/2112.05291)]
    * **CaFT**: "CaFT: Clustering and Filter on Tokens of Transformer for Weakly Supervised Object Localization", arXiv, 2022 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2201.00475)]
* Relation Detection:
    * **PST**: "Visual Relationship Detection Using Part-and-Sum Transformers with Composite Queries", ICCV, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2105.02170)]
    * **RelTransformer**: "RelTransformer: Balancing the Visual Relationship Detection from Local Context, Scene and Memory", arXiv, 2021 (*KAUST*). [[Paper](https://arxiv.org/abs/2104.11934)][[Code (in construction)](https://github.com/Vision-CAIR/RelTransformer)]
    * **PST**: "Visual Composite Set Detection Using Part-and-Sum Transformers", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2105.02170)]
    * **TROI**: "Transformed ROIs for Capturing Visual Transformations in Videos", arXiv, 2021 (*NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.03162)]
* HOI Detection:
    * **HOI-Transformer**: "End-to-End Human Object Interaction Detection with HOI Transformer", CVPR, 2021 (*Megvii*). [[Paper](https://arxiv.org/abs/2103.04503)][[PyTorch](https://github.com/bbepoch/HoiTransformer)]
    * **HOTR**: "HOTR: End-to-End Human-Object Interaction Detection with Transformers", CVPR, 2021 (*Kakao + Korea University*). [[Paper](https://arxiv.org/abs/2104.13682)][[PyTorch](https://github.com/kakaobrain/HOTR)]
* Salient Object Detection:
    * **VST**: "Visual Saliency Transformer", ICCV, 2021 (*Northwestern Polytechincal University*). [[Paper](https://arxiv.org/abs/2104.12099)]
    * **?**: "Learning Generative Vision Transformer with Energy-Based Latent Space for Saliency Prediction", NeurIPS, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2112.13528)]
    * **SOD-Transformer**: "Transformer Transforms Salient Object Detection and Camouflaged Object Detection", arXiv, 2021 (*Northwestern Polytechnical University*). [[Paper](https://arxiv.org/abs/2104.10127)]
    * **GLSTR**: "Unifying Global-Local Representations in Salient Object Detection with Transformer", arXiv, 2021 (*South China University of Technology*). [[Paper](https://arxiv.org/abs/2108.02759)]
    * **TriTransNet**: "TriTransNet: RGB-D Salient Object Detection with a Triplet Transformer Embedding Network", arXiv, 2021 (*Anhui University*). [[Paper](https://arxiv.org/abs/2108.03990)]
    * **AbiU-Net**: "Boosting Salient Object Detection with Transformer-based Asymmetric Bilateral U-Net", arXiv, 2021 (*Nankai University*). [[Paper](https://arxiv.org/abs/2108.07851)]
    * **TranSalNet**: "TranSalNet: Visual saliency prediction using transformers", arXiv, 2021 (*Cardiff University, UK*). [[Paper](https://arxiv.org/abs/2110.03593)]
* Anomaly Detection:
    * **VT-ADL**: "VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization", ISIE, 2021 (*University of Udine, Italy*). [[Paper](https://arxiv.org/abs/2104.10036)]
    * **InTra**: "Inpainting Transformer for Anomaly Detection", arXiv, 2021 (*Fujitsu*). [[Paper](https://arxiv.org/abs/2104.13897)]
* Domain Adaptation:
    * **SSTN**: "SSTN: Self-Supervised Domain Adaptation Thermal Object Detection for Autonomous Driving", arXiv, 2021 (*Gwangju Institute of Science and Technology*). [[Paper](https://arxiv.org/abs/2103.03150)]
    * **DA-DETR**: "DA-DETR: Domain Adaptive Detection Transformer by Hybrid Attention", arXiv, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2103.17084)]
* X-Shot Object Detection:
    * **AIT**: "Adaptive Image Transformer for One-Shot Object Detection", CVPR, 2021 (*Academia Sinica*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Adaptive_Image_Transformer_for_One-Shot_Object_Detection_CVPR_2021_paper.html)]
    * **Meta-DETR**: "Meta-DETR: Few-Shot Object Detection via Unified Image-Level Meta-Learning", arXiv, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2103.11731)]
    * **CAT**: "CAT: Cross-Attention Transformer for One-Shot Object Detection", arXiv, 2021 (*Northwestern Polytechnical University*). [[Paper](https://arxiv.org/abs/2104.14984)]
* Co-Salient Object Detection:
    * **CoSformer**: "CoSformer: Detecting Co-Salient Object with Transformers", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2104.14729)]
* Oriented Object Detection:
    * **O<sup>2</sup>DETR**: "Oriented Object Detection with Transformer", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.03146)]
* Multiview Detection:
    * **MVDeTr**: "Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation)", ACMMM, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2108.05888)]
* Polygon Detection:
    * **?**: "Investigating transformers in the decomposition of polygonal shapes as point collections", ICCVW, 2021 (*Delft University of Technology, Netherlands*). [[Paper](https://arxiv.org/abs/2108.07533)]
* Drone-view:
    * **TPH**: "TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios", ICCVW, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2108.11539)]
* Infrared:
    * **?**: "Infrared Small-Dim Target Detection with Transformer under Complex Backgrounds", arXiv, 2021 (*Chongqing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2109.14379)]
* Text:
    * **?**: "Arbitrary Shape Text Detection using Transformers", arXiv, 2022 (*University of Waterloo, Canada*). [[Paper](https://arxiv.org/abs/2202.11221)]
* Change Detection:
    * **ChangeFormer**: "A Transformer-Based Siamese Network for Change Detection", arXiv, 2022 (*JHU*). [[Paper](https://arxiv.org/abs/2201.01293)][[PyTorch](https://github.com/wgcban/ChangeFormer)]


## Segmentation
### Semantic Segmentation
* **SETR**: "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers", CVPR, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2012.15840)][[PyTorch](https://github.com/fudan-zvg/SETR)][[Website](https://fudan-zvg.github.io/SETR/)]
* **TrSeg**: "TrSeg: Transformer for semantic segmentation", PRL, 2021 (*Korea University*). [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S016786552100163X)][[PyTorch](https://github.com/youngsjjn/TrSeg)]
* **CWT**: "Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer", ICCV, 2021 (*University of Surrey, UK*). [[Paper](https://arxiv.org/abs/2108.03032)][[PyTorch](https://github.com/zhiheLu/CWT-for-FSS)]
* **Segmenter**: "Segmenter: Transformer for Semantic Segmentation", ICCV, 2021 (*INRIA*). [[Paper](https://arxiv.org/abs/2105.05633)][[PyTorch](https://github.com/rstrudel/segmenter)]
* **UN-EPT**: "A Unified Efficient Pyramid Transformer for Semantic Segmentation", ICCVW, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2107.14209)]
* **SegFormer**: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2105.15203)][[PyTorch](https://github.com/NVlabs/SegFormer)]
* **FTN**: "Fully Transformer Networks for Semantic Image Segmentation", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.04108)]
* **OffRoadTranSeg**: "OffRoadTranSeg: Semi-Supervised Segmentation using Transformers on OffRoad environments", arXiv, 2021 (*IISER. India*). [[Paper](https://arxiv.org/abs/2106.13963)]
* **MaskFormer**: "Per-Pixel Classification is Not All You Need for Semantic Segmentation", arXiv, 2021 (*UIUC + Facebook*). [[Paper](https://arxiv.org/abs/2107.06278)][[Website](https://bowenc0221.github.io/maskformer/)]
* **TRFS**: "Boosting Few-shot Semantic Segmentation with Transformers", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2108.02266)]
* **Flying-Guide-Dog**: "Flying Guide Dog: Walkable Path Discovery for the Visually Impaired Utilizing Drones and Transformer-based Semantic Segmentation", arXiv, 2021 (*KIT, Germany*). [[Paper](https://arxiv.org/abs/2108.07007)][[Code (in construction)](https://github.com/EckoTan0804/flying-guide-dog)]
* **VSPW**: "Semantic Segmentation on VSPW Dataset through Aggregation of Transformer Models", arXiv, 2021 (*Xiaomi*). [[Paper](https://arxiv.org/abs/2109.01316)]
* **SDTP**: "SDTP: Semantic-aware Decoupled Transformer Pyramid for Dense Image Prediction", arXiv, 2021 (*?*). [[Paper](https://arxiv.org/abs/2109.08963)]
* **GroupViT**: "GroupViT: Semantic Segmentation Emerges from Text Supervision", arXiv, 2022 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2202.11094)][[Website](https://jerryxu.net/GroupViT/)]
* **Lawin**: "Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention", arXiv, 2022 (*Beijing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2201.01615)][[PyTorch](https://github.com/yan-hao-tian/lawin)]
* **PFT**: "Pyramid Fusion Transformer for Semantic Segmentation", arXiv, 2022 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2201.04019)]
* **DFlatFormer**: "Dual-Flattening Transformers through Decomposed Row and Column Queries for Semantic Segmentation", arXiv, 2022 (*OPPO*). [[Paper](https://arxiv.org/abs/2201.09139)]

### Object Segmentation
* **SOTR**: "SOTR: Segmenting Objects with Transformers", ICCV, 2021 (*China Agricultural University*). [[Paper](https://arxiv.org/abs/2108.06747)][[PyTorch](https://github.com/easton-cau/SOTR)]
* **Trans4Trans**: "Trans4Trans: Efficient Transformer for Transparent Object Segmentation to Help Visually Impaired People Navigate in the Real World", ICCVW, 2021 (*Karlsruhe Institute of Technology, Germany*). [[Paper](https://arxiv.org/abs/2107.03172)][[Code (in construction)](https://github.com/jamycheung/Trans4Trans)]
* **Trans2Seg**: "Segmenting Transparent Object in the Wild with Transformer", arXiv, 2021 (*HKU + SenseTime*). [[Paper](https://arxiv.org/abs/2101.08461)][[PyTorch](https://github.com/xieenze/Trans2Seg)]
* **SOIT**: "SOIT: Segmenting Objects with Instance-Aware Transformers", AAAI, 2022 (*Hikvision*). [[Paper](https://arxiv.org/abs/2112.11037)][[Code (in construction)](https://github.com/yuxiaodongHRI/SOIT)]

### Other Segmentation Tasks
* Panoptic Segmentation:
    * **MaX-DeepLab**: "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2012.00759)][[PyTorch (conradry)](https://github.com/conradry/max-deeplab)]
    * **Panoptic-SegFormer**: "Panoptic SegFormer", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2109.03814)]
    * **SIAin**: "An End-to-End Trainable Video Panoptic Segmentation Method usingTransformers", arXiv, 2021 (*SI Analytics, South Korea*). [[Paper](https://arxiv.org/abs/2110.04009)]
    * **VPS-Transformer**: "Time-Space Transformers for Video Panoptic Segmentation", WACV, 2022 (*Technical University of Cluj-Napoca, Romania*). [[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Petrovai_Time-Space_Transformers_for_Video_Panoptic_Segmentation_WACV_2022_paper.html)]
* Depth Estimation & Semantic Segmentation:
    * **DPT**: "Vision Transformers for Dense Prediction", ICCV, 2021 (*Intel*). [[Paper](https://arxiv.org/abs/2103.13413)][[PyTorch](https://github.com/intel-isl/DPT)]
* Instance Segmentation:
    * **ISTR**: "ISTR: End-to-End Instance Segmentation with Transformers", arXiv, 2021 (*Xiamen University*). [[Paper](https://arxiv.org/abs/2105.00637)][[PyTorch](https://github.com/hujiecpp/ISTR)]
* Depth Estimation:
    * **TransDepth**: "Transformer-Based Attention Networks for Continuous Pixel-Wise Prediction", ICCV, 2021 (*Haerbin Institute of Technology + University of Trento*). [[Paper](https://arxiv.org/abs/2103.12091)][[PyTorch](https://github.com/ygjwd12345/TransDepth)]
    * **MT-SfMLearner**: "Transformers in Self-Supervised Monocular Depth Estimation with Unknown Camera Intrinsics", VISAP, 2022 (*NavInfo Europe, Netherlands*). [[Paper](https://arxiv.org/abs/2202.03131)]
    * **GLPanoDepth**: "GLPanoDepth: Global-to-Local Panoramic Depth Estimation", arXiv, 2022 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2202.02796)]
* Panoramic Semantic Segmentation:
    * **Trans4PASS**: "Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation", CVPR, 2022 (*Karlsruhe Institute of Technology, Germany*). [[Paper](https://arxiv.org/abs/2203.01452)][[Code (in construction)](https://github.com/jamycheung/Trans4PASS)]
* Few-Shot:
    * **CyCTR**: "Few-Shot Segmentation via Cycle-Consistent Transformer", NeurIPS, 2021 (*University of Technology Sydney*). [[Paper](https://arxiv.org/abs/2106.02320)]
    * **LSeg**: "Language-driven Semantic Segmentation", ICLR, 2022 (*Cornell*). [[Paper](https://arxiv.org/abs/2201.03546)][[PyTorch](https://github.com/isl-org/lang-seg)]
    * **TAFT**: "Task-Adaptive Feature Transformer with Semantic Enrichment for Few-Shot Segmentation", arXiv, 2022 (*KAIST*). [[Paper](https://arxiv.org/abs/2202.06498)]
* Urban Scene:
    * **BANet**: "Transformer Meets Convolution: A Bilateral Awareness Net-work for Semantic Segmentation of Very Fine Resolution Ur-ban Scene Images", arXiv, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2106.12413)] 
* Crack Detection:
    * **CrackFormer**: "CrackFormer: Transformer Network for Fine-Grained Crack Detection", ICCV, 2021 (*Nanjing University of Science and Technology*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_CrackFormer_Transformer_Network_for_Fine-Grained_Crack_Detection_ICCV_2021_paper.html)]
* Camouflaged Object Detection:
    * **UGTR**: "Uncertainty-Guided Transformer Reasoning for Camouflaged Object Detection", ICCV, 2021 (*Group42, Abu Dhabi*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Uncertainty-Guided_Transformer_Reasoning_for_Camouflaged_Object_Detection_ICCV_2021_paper.html)][[PyTorch](https://github.com/fanyang587/UGTR)]
* Background Separation:
    * **TransBlast**: "TransBlast: Self-Supervised Learning Using Augmented Subspace With Transformer for Background/Foreground Separation", ICCVW, 2021 (*University of British Columbia*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Osman_TransBlast_Self-Supervised_Learning_Using_Augmented_Subspace_With_Transformer_for_BackgroundForeground_ICCVW_2021_paper.html)]


## Video (High-level)
### Action Recognition
* RGB mainly
    * **Action Transformer**: "Video Action Transformer Network", CVPR, 2019 (*DeepMind*). [[Paper](https://arxiv.org/abs/1812.02707)][[Code (ppriyank)](https://github.com/ppriyank/Video-Action-Transformer-Network-Pytorch-)]
    * **ViViT-Ensemble**: "Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition", CVPRW, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.05058)]
    * **TimeSformer**: "Is Space-Time Attention All You Need for Video Understanding?", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2102.05095)][[PyTorch (lucidrains)](https://github.com/lucidrains/TimeSformer-pytorch)]
    * **MViT**: "Multiscale Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.11227)][[PyTorch](https://github.com/facebookresearch/SlowFast)]
    * **VidTr**: "VidTr: Video Transformer Without Convolutions", ICCV, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2104.11746)][[PyTorch](https://github.com/amazon-research/gluonmm)]
    * **ViViT**: "ViViT: A Video Vision Transformer", ICCV, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.15691)][[PyTorch (rishikksh20)](https://github.com/rishikksh20/ViViT-pytorch)]
    * **VTN**: "Video Transformer Network", ICCVW, 2021 (*Theator*). [[Paper](https://arxiv.org/abs/2102.00719)][[PyTorch](https://github.com/bomri/SlowFast/tree/master/projects/vtn)]
    * **TokShift**: "Token Shift Transformer for Video Classification", ACMMM, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2108.02432)][[PyTorch](https://github.com/VideoNetworks/TokShift-Transformer)]
    * **Motionformer**: "Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers", NeurIPS, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.05392)][[PyTorch](https://github.com/facebookresearch/Motionformer)][[Website](https://facebookresearch.github.io/Motionformer/)]
    * **X-ViT**: "Space-time Mixing Attention for Video Transformer", NeurIPS, 2021 (*Samsung*). [[Paper](https://arxiv.org/abs/2106.05968)]
    * **SCT**: "Shifted Chunk Transformer for Spatio-Temporal Representational Learning", NeurIPS, 2021 (*Kuaishou*). [[Paper](https://arxiv.org/abs/2108.11575)]
    * **STAM**: "An Image is Worth 16x16 Words, What is a Video Worth?", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2103.13915)][[Code](https://github.com/Alibaba-MIIL/STAM)]
    * **GAT**: "Enhancing Transformer for Video Understanding Using Gated Multi-Level Attention and Temporal Adversarial Training", arXiv, 2021 (*Samsung*). [[Paper](https://arxiv.org/abs/2103.10043)]
    * **TokenLearner**: "TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.11297)]
    * **Video-Swin**: "Video Swin Transformer", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.13230)][[PyTorch](https://github.com/SwinTransformer/Video-Swin-Transformer)]
    * **VLF**: "VideoLightFormer: Lightweight Action Recognition using Transformers", arXiv, 2021 (*The University of Sheffield*). [[Paper](https://arxiv.org/abs/2107.00451)]
    * **ORViT**: "Object-Region Video Transformers", arXiv, 2021 (*Tel Aviv*). [[Paper](https://arxiv.org/abs/2110.06915)][[Website](https://roeiherz.github.io/ORViT/)]
    * **Improved-MViT**: "Improved Multiscale Vision Transformers for Classification and Detection", arXiv, 2021 (*Meta*). [[Paper](https://arxiv.org/abs/2112.01526)]
    * **UniFormer**: "UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning", ICLR, 2022 (*CAS + SenstTime*). [[Paper](https://arxiv.org/abs/2201.04676)][[PyTorch](https://github.com/Sense-X/UniFormer)]
    * **MeMViT**: "MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition", arXiv, 2022 (*Meta*). [[Paper](https://arxiv.org/abs/2201.08383)]
    * **MTV**: "Multiview Transformers for Video Recognition", arXiv, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2201.04288)]
* Depth
    * **Trear**: "Trear: Transformer-based RGB-D Egocentric Action Recognition",  IEEE Transactions on Cognitive and Developmental Systems, 2021 (*Tianjing University*). [[Paper](https://ieeexplore.ieee.org/document/9312201)]
* Pose:
    * **ST-TR**: "Spatial Temporal Transformer Network for Skeleton-based Action Recognition", ICPRW, 2020 (*Polytechnic University of Milan*). [[Paper](https://arxiv.org/abs/2012.06399)]
    * **AcT**: "Action Transformer: A Self-Attention Model for Short-Time Human Action Recognition", arXiv, 2021 (*Politecnico di Torino, Italy*). [[Paper](https://arxiv.org/abs/2107.00606)][[Code (in construction)](https://github.com/FedericoAngelini/MPOSE2021_Dataset)]
    * **STAR**: "STAR: Sparse Transformer-based Action Recognition", arXiv, 2021 (*UCLA*). [[Paper](https://arxiv.org/abs/2107.07089)]
    * **GCsT**: "GCsT: Graph Convolutional Skeleton Transformer for Action Recognition", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2109.02860)]
    * **STTFormer**: "Spatio-Temporal Tuples Transformer for Skeleton-Based Action Recognition", arXiv, 2022 (*Xidian University*). [[Paper](https://arxiv.org/abs/2201.02849)][[Code (in construction)](https://github.com/heleiqiu/STTFormer)]
    * **ProFormer**: "ProFormer: Learning Data-efficient Representations of Body Movement with Prototype-based Feature Augmentation and Visual Transformers", arXiv, 2022 (*Karlsruhe Institute of Technology, Germany*). [[Paper](https://arxiv.org/abs/2202.11423)][[PyTorch](https://github.com/KPeng9510/ProFormer)]
* Multi-modal:
    * **MBT**: "Attention Bottlenecks for Multimodal Fusion", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2107.00135)]
    * **MM-ViT**: "MM-ViT: Multi-Modal Video Transformer for Compressed Video Action Recognition", WACV, 2022 (*OPPO*). [[Paper](https://arxiv.org/abs/2108.09322)]
    * **MVFT**: "Multi-View Fusion Transformer for Sensor-Based Human Activity Recognition", arXiv, 2022 (*Alibaba*). [[Paper](https://arxiv.org/abs/2202.12949)]

### Action Detection/Localization
* **OadTR**: "OadTR: Online Action Detection with Transformers", ICCV, 2021 (*Huazhong University of Science and Technology*). [[Paper](https://arxiv.org/abs/2106.11149)][[PyTorch](https://github.com/wangxiang1230/OadTR)]
* **RTD-Net**: "Relaxed Transformer Decoders for Direct Action Proposal Generation", ICCV, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2102.01894)][[PyTorch](https://github.com/MCG-NJU/RTD-Action)]
* **LSTR**: "Long Short-Term Transformer for Online Action Detection", NeurIPS, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2107.03377)][[PyTorch](https://github.com/amazon-research/long-short-term-transformer)][[Website](https://xumingze0308.github.io/projects/lstr/)]
* **TubeR**: "TubeR: Tube-Transformer for Action Detection", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2104.00969)]
* **ATAG**: "Augmented Transformer with Adaptive Graph for Temporal Action Proposal Generation", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2103.16024)]
* **TAPG-Transformer**: "Temporal Action Proposal Generation with Transformers", arXiv, 2021 (*Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2105.12043)]
* **TadTR**: "End-to-end Temporal Action Detection with Transformer", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.10271)][[Code (in construction)](https://github.com/xlliu7/TadTR)]
* **Vidpress-Soccer**: "Feature Combination Meets Attention: Baidu Soccer Embeddings and Transformer based Temporal Detection", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.14447)][[GitHub](https://github.com/baidu-research/vidpress-sports)]
* **CoOadTR**: "Continual Transformers: Redundancy-Free Attention for Online Inference", arXiv, 2022 (*Aarhus University, Denmark*). [[Paper](https://arxiv.org/abs/2201.06268)][[PyTorch](https://github.com/LukasHedegaard/continual-transformers)]
* **ActionFormer**: "ActionFormer: Localizing Moments of Actions with Transformers", arXiv, 2022 (*UW-Madison*). [[Paper](https://arxiv.org/abs/2202.07925)][[PyTorch](https://github.com/happyharrycn/actionformer_release)]
* **Temporal-Perceiver**: "Temporal Perceiver: A General Architecture for Arbitrary Boundary Detection", arXiv, 2022 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2203.00307)]

### Action Prediction
* **AVT**: "Anticipative Video Transformer", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.02036)][[PyTorch](https://github.com/facebookresearch/AVT)][[Website](https://facebookresearch.github.io/AVT/)]
* **HORST**: "Higher Order Recurrent Space-Time Transformer", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2104.08665)][[PyTorch](https://github.com/CorcovadoMing/HORST)]
* **?**: "Action Forecasting with Feature-wise Self-Attention", arXiv, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2107.08579)]

### Video Object Segmentation
* **SSTVOS**: "SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation", CVPR, 2021 (*Modiface*). [[Paper](https://arxiv.org/abs/2101.08833)][[Code (in construction)](https://github.com/dukebw/SSTVOS)]
* **JOINT**: "Joint Inductive and Transductive Learning for Video Object Segmentation", ICCV, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2108.03679)][[PyTorch](https://github.com/maoyunyao/JOINT)]
* **AOT**: "Associating Objects with Transformers for Video Object Segmentation", NeurIPS, 2021 (*University of Technology Sydney*). [[Paper](https://arxiv.org/abs/2106.02638)][[PyTorch (yoxu515)](https://github.com/yoxu515/aot-benchmark)][[Code (in construction)](https://github.com/z-x-yang/AOT)]
* **TransVOS**: "TransVOS: Video Object Segmentation with Transformers", arXiv, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2106.00588)]
* **SITVOS**: "Siamese Network with Interactive Transformer for Video Object Segmentation", AAAI, 2022 (*JD*). [[Paper](https://arxiv.org/abs/2112.13983)] 

### Other Video Tasks
* Action Segmentation
    * **ASFormer**: "ASFormer: Transformer for Action Segmentation", BMVC, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2110.08568)][[PyTorch](https://github.com/ChinaYi/ASFormer)]
    * **?**: "Transformers in Action: Weakly Supervised Action Segmentation", arXiv, 2022 (*TUM*). [[Paper](https://arxiv.org/abs/2201.05675)]
* Video Instance Segmentation
    * **VisTR**: "End-to-End Video Instance Segmentation with Transformers", CVPR, 2021 (*Meituan*). [[Paper](https://arxiv.org/abs/2011.14503)][[PyTorch](https://github.com/Epiphqny/VisTR)]
    * **IFC**: "Video Instance Segmentation using Inter-Frame Communication Transformers", NeurIPS, 2021 (*Yonsei University*). [[Paper](https://arxiv.org/abs/2106.03299)][[PyTorch](https://github.com/sukjunhwang/IFC)]
* Video Object Detection:
    * **TransVOD**: "End-to-End Video Object Detection with Spatial-Temporal Transformers", arXiv, 2021 (*Shanghai Jiao Tong + SenseTime*). [[Paper](https://arxiv.org/abs/2105.10920)][[Code (in construction)](https://github.com/SJTU-LuHe/TransVOD)]
    * **MODETR**: "MODETR: Moving Object Detection with Transformers", arXiv, 2021 (*Valeo, Egypt*). [[Paper](https://arxiv.org/abs/2106.11422)]
    * **ST-MTL**: "Spatio-Temporal Multi-Task Learning Transformer for Joint Moving Object Detection and Segmentation", arXiv, 2021 (*Valeo, Egypt*). [[Paper](https://arxiv.org/abs/2106.11401)]
    * **ST-DETR**: "ST-DETR: Spatio-Temporal Object Traces Attention Detection Transformer", arXiv, 2021 (*Valeo, Egypt*). [[Paper](https://arxiv.org/abs/2107.05887)]
    * **TransVOD**: "TransVOD: End-to-end Video Object Detection with Spatial-Temporal Transformers", arXiv, 2022 (*Shanghai Jiao Tong + SenseTime*). [[Paper](https://arxiv.org/abs/2201.05047)]
* Video Retrieval
    * **SVRTN**: "Self-supervised Video Retrieval Transformer Network", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2104.07993)]
* Video Hashing
    * **BTH**: "Self-Supervised Video Hashing via Bidirectional Transformers", CVPR, 2021 (*Tsinghua*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Self-Supervised_Video_Hashing_via_Bidirectional_Transformers_CVPR_2021_paper.html)][[PyTorch](https://github.com/Lily1994/BTH)]
* Un/Self-supervised Learning:
    * **LSTCL**: "Long-Short Temporal Contrastive Learning of Video Transformers", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.09212)]
    * **VIMPAC**: "VIMPAC: Video Pre-Training via Masked Token Prediction and Contrastive Learning", arXiv, 2021 (*UNC*). [[Paper](https://arxiv.org/abs/2106.11250)][[PyTorch](https://github.com/airsplay/vimpac)]
* Anomaly Detection:
    * **CT-D2GAN**: "Convolutional Transformer based Dual Discriminator Generative Adversarial Networks for Video Anomaly Detection", ACMMM, 2021 (*NEC*). [[Paper](https://arxiv.org/abs/2107.13720)]
* Relation Detection:
    * **VidVRD**: "Video Relation Detection via Tracklet based Visual Transformer", ACMMMW, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2108.08669)][[PyTorch](https://github.com/Dawn-LX/VidVRD-tracklets)]
* Saliency Prediction:
    * **STSANet**: "Spatio-Temporal Self-Attention Network for Video Saliency Prediction", arXiv, 2021 (*Shanghai University*). [[Paper](https://arxiv.org/abs/2108.10696)]
* Group Activity:
    * **GroupFormer**: "GroupFormer: Group Activity Recognition with Clustered Spatial-Temporal Transformer", ICCV, 2021 (*Sensetime*). [[Paper](https://arxiv.org/abs/2108.12630)]
* Video Inpainting Detection:
    * **FAST**: "Frequency-Aware Spatiotemporal Transformers for Video Inpainting Detection", ICCV, 2021 (*Tsinghua University*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Frequency-Aware_Spatiotemporal_Transformers_for_Video_Inpainting_Detection_ICCV_2021_paper.html)]
* Driver Activity:
    * **TransDARC**: "TransDARC: Transformer-based Driver Activity Recognition with Latent Space Feature Calibration", arXiv, 2022 (*Karlsruhe Institute of Technology, Germany*). [[Paper](https://arxiv.org/abs/2203.00927)]


## Multi-Modality
### VQA / Captioning
* **MCAN**: "Deep Modular Co-Attention Networks for Visual Question Answering", CVPR, 2019 (*Hangzhou Dianzi University*). [[Paper](https://arxiv.org/abs/1906.10770)][[PyTorch](https://github.com/MILVLG/mcan-vqa)]
* **ETA-Transformer**: "Entangled Transformer for Image Captioning", ICCV, 2019 (*UTS*). [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Entangled_Transformer_for_Image_Captioning_ICCV_2019_paper.html)]
* **M4C**: "Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA", CVPR, 2020 (*Facebook*). [[Paper](https://arxiv.org/abs/1911.06258)]
* **M2-Transformer**: "Meshed-Memory Transformer for Image Captioning", CVPR, 2020 (*UniMoRE*). [[Paper](https://arxiv.org/abs/1912.08226)][[PyTorch](https://github.com/aimagelab/meshed-memory-transformer)] 
* **SA-M4C**: "Spatially Aware Multimodal Transformers for TextVQA", ECCV, 2020 (*Georgia Tech*). [[Paper](https://arxiv.org/abs/2007.12146)][[PyTorch](https://github.com/yashkant/sam-textvqa)][[Website](https://yashkant.github.io/projects/sam-textvqa.html)]
* **CATT**: "Causal Attention for Vision-Language Tasks", CVPR, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2103.03493)][[PyTorch](https://github.com/yangxuntu/lxmertcatt)]
* **?**: "Optimizing Latency for Online Video CaptioningUsing Audio-Visual Transformers", Interspeech, 2021 (*MERL*). [[Paper](https://arxiv.org/abs/2108.02147)]
* **ConClaT**: "Contrast and Classify: Training Robust VQA Models", ICCV, 2021 (*Georgia Tech*). [[Paper](https://arxiv.org/abs/2010.06087)]
* **MCCFormers**: "Describing and Localizing Multiple Changes with Transformers", ICCV, 2021 (*AIST*). [[Paper](https://arxiv.org/abs/2103.14146)][[Website](https://cvpaperchallenge.github.io/Describing-and-Localizing-Multiple-Change-with-Transformers/)]
* **TRAR**: "TRAR: Routing the Attention Spans in Transformer for Visual Question Answering", ICCV, 2021 (*Xiamen University*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_TRAR_Routing_the_Attention_Spans_in_Transformer_for_Visual_Question_ICCV_2021_paper.html)]
* **UniQer**: "Unified Questioner Transformer for Descriptive Question Generation in Goal-Oriented Visual Dialogue", ICCV, 2021 (*Keio*). [[Paper](https://arxiv.org/abs/2106.15550)]
* **SATIC**: "Semi-Autoregressive Transformer for Image Captioning", ICCVW, 2021 (*Hefei University of Technology*). [[Paper](https://arxiv.org/abs/2106.09436)][[PyTorch](https://github.com/YuanEZhou/satic)]
* **DGCN**: "Dual Graph Convolutional Networks with Transformer and Curriculum Learning for Image Captioning", ACMMM, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2108.02366)]
* **TxT**: "TxT: Crossmodal End-to-End Learning with Transformers", GCPR, 2021 (*TU Darmstadt*). [[Paper](https://arxiv.org/abs/2109.04422)]
* **ProTo**: "ProTo: Program-Guided Transformer for Program-Guided Tasks", NeurIPS, 2021 (*Georiga Tech*). [[Paper](https://arxiv.org/abs/2110.00804)]
* **CPTR**: "CPTR: Full Transformer Network for Image Captioning", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2101.10804)] 
* **VisQA**: "VisQA: X-raying Vision and Language Reasoning in Transformers", arXiv, 2021 (*INSA-Lyon*). [[Paper](https://arxiv.org/abs/2104.00926)][[PyTorch](https://github.com/Theo-Jaunet/VisQA)]
* **ReFormer**: "ReFormer: The Relational Transformer for Image Captioning", arXiv, 2021 (*Stony Brook University*). [[Paper](https://arxiv.org/abs/2107.14178)]
* **?**: "Mounting Video Metadata on Transformer-based Language Model for Open-ended Video Question Answering", arXiv, 2021 (*Seoul National University*). [[Paper](https://arxiv.org/abs/2108.05158)]
* **LAViTeR**: "LAViTeR: Learning Aligned Visual and Textual Representations Assisted by Image and Caption Generation", arXiv, 2021 (*University at Buffalo*). [[Paper](https://arxiv.org/abs/2109.04993)]
* **TPT**: "Temporal Pyramid Transformer with Multimodal Interaction for Video Question Answering", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2109.04735)]
* **LATGeO**: "Label-Attention Transformer with Geometrically Coherent Objects for Image Captioning", arXiv, 2021 (*Gwangju Institute of Science and Technology*). [[Paper](https://arxiv.org/abs/2109.07799)]
* **GEVST**: "Geometry-Entangled Visual Semantic Transformer for Image Captioning", arXiv, 2021 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2109.14137)]
* **GAT**: "Geometry Attention Transformer with Position-aware LSTMs for Image Captioning", arXiv, 2021 (*University of Electronic Science and Technology of China*). [[Paper](https://arxiv.org/abs/2110.00335)]
* **Block-Skim**: "Block-Skim: Efficient Question Answering for Transformer", AAAI, 2022 (* Shanghai Jiao Tong*). [[Paper](https://arxiv.org/abs/2112.08560)]
* **RelViT**: "RelViT: Concept-guided Vision Transformer for Visual Relational Reasoning", ICLR, 2022 (*NVIDIA*). [[Paper](https://openreview.net/forum?id=afoV8W3-IYp)]
* **X-Trans2Cap**: "X-Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning", CVPR, 2022 (*CUHK*). [[Paper](https://arxiv.org/abs/2203.00843)]
* **UCM**: "Self-Training Vision Language BERTs with a Unified Conditional Model", arXiv, 2022 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2201.02010)]
* **ViNTER**: "ViNTER: Image Narrative Generation with Emotion-Arc-Aware Transformer", arXiv, 2022 (*The University of Tokyo*). [[Paper](https://arxiv.org/abs/2202.07305)]
* **TMN**: "Transformer Module Networks for Systematic Generalization in Visual Question Answering", arXiv, 2022 (*Fujitsu*). [[Paper](https://arxiv.org/abs/2201.11316)]
* **?**: "On the Efficacy of Co-Attention Transformer Layers in Visual Question Answering", arXiv, 2022 (*Birla Institute of Technology Mesra, India*). [[Paper](https://arxiv.org/abs/2201.03965)]

### Visual Grounding
* **Multi-Stage-Transformer**: "Multi-Stage Aggregated Transformer Network for Temporal Language Localization in Videos", CVPR, 2021 (*University of Electronic Science and Technology of China*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Multi-Stage_Aggregated_Transformer_Network_for_Temporal_Language_Localization_in_Videos_CVPR_2021_paper.html)]
* **TransRefer3D**: "TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding", ACMMM, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2108.02388)]
* **?**: "Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers", EMNLP, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2109.04448)]
* **GTR**: "On Pursuit of Designing Multi-modal Transformer for Video Grounding", EMNLP, 2021 (*Peking*). [[Paper](https://arxiv.org/abs/2109.06085)]
* **MITVG**: "Multimodal Incremental Transformer with Visual Grounding for Visual Dialogue Generation", ACL Findings, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2109.08478)]
* **STVGBert**: "STVGBert: A Visual-Linguistic Transformer Based Framework for Spatio-Temporal Video Grounding", ICCV, 2021 (*Tencent*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Su_STVGBert_A_Visual-Linguistic_Transformer_Based_Framework_for_Spatio-Temporal_Video_Grounding_ICCV_2021_paper.html)]
* **TransVG**: "TransVG: End-to-End Visual Grounding with Transformers", ICCV, 2021 (*USTC*). [[Paper](https://arxiv.org/abs/2104.08541)]
* **DRFT**: "End-to-end Multi-modal Video Temporal Grounding", NeurIPS, 2021 (*UC Merced*). [[Paper](https://arxiv.org/abs/2107.05624)]
* **Referring-Transformer**: "Referring Transformer: A One-step Approach to Multi-task Visual Grounding", NeurIPS, 2021 (*UBC*). [[Paper](https://arxiv.org/abs/2106.03089)]
* **VGTR**: "Visual Grounding with Transformers", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2105.04281)]
* **Word2Pix**: "Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding", arXiv, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2108.00205)]
* **VidGTR**: "Explore and Match: End-to-End Video Grounding with Transformer", arXiv, 2022 (*KAIST*). [[Paper](https://arxiv.org/abs/2201.10168)]

### Representation Learning
* **COOT**: "COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning", NeurIPS, 2020 (*University of Freiburg*). [[Paper](https://arxiv.org/abs/2011.00597)][[PyTorch](https://github.com/gingsi/coot-videotext)]
* **Parameter-Reduction**: "Parameter Efficient Multimodal Transformers for Video Representation Learning", ICLR, 2021 (*Seoul National University*). [[Paper](https://openreview.net/forum?id=6UdQLhqJyFD)]
* **VinVL**: "VinVL: Revisiting Visual Representations in Vision-Language Models", CVPR, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2101.00529)][[Code](https://github.com/pzzhang/VinVL)]
* **ViLT**: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision", ICML, 2021 (*Kakao*). [[Paper](https://arxiv.org/abs/2102.03334)][[PyTorch](https://github.com/dandelin/vilt)]
* **VML**: "VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding", ACL Findings, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2105.09996)]
* **VATT**: "VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2104.11178)][[Tensorflow](https://github.com/google-research/google-research/tree/master/vatt)]
* **SVO-Probes**: "Probing Image-Language Transformers for Verb Understanding", arXiv, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2106.09141)]
* **CLIP-ViL**: "How Much Can CLIP Benefit Vision-and-Language Tasks?", arXiv, 2021 (*Berkeley + UCLA*). [[Paper](https://arxiv.org/abs/2107.06383)][[PyTorch](https://github.com/clip-vil/CLIP-ViL)]
* **?**: "Survey: Transformer based Video-Language Pre-training", arXiv, 2021 (*Renmin University of China*). [[Paper](https://arxiv.org/abs/2109.09920)]
* **Omnivore**: "Omnivore: A Single Model for Many Visual Modalities", arXiv, 2022 (*Meta*). [[Paper](https://arxiv.org/abs/2201.08377)][[PyTorch](https://github.com/facebookresearch/omnivore)]

### Retrieval
* **MMT**: "Multi-modal Transformer for Video Retrieval", ECCV, 2020 (*INRIA + Google*). [[Paper](https://arxiv.org/abs/2007.10639)][[Website](http://thoth.inrialpes.fr/research/MMT/)]
* **Fast-and-Slow**: "Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers", CVPR, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2103.16553)]
* **HTR**: "Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning", CVPR, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2103.13061)][[PyTorch](https://github.com/amzn/image-to-recipe-transformers)]
* **ClipBERT**: "Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling", CVPR, 2021 (*UNC + Microsoft*). [[Paper](https://arxiv.org/abs/2102.06183)][[PyTorch](https://github.com/jayleicn/ClipBERT)]
* **AYCE**: "All You Can Embed: Natural Language based Vehicle Retrieval with Spatio-Temporal Transformers", CVPRW, 2021 (*University of Modena and Reggio Emilia*). [[Paper](https://arxiv.org/abs/2106.10153)][[PyTorch](https://github.com/cscribano/AYCE_2021)]
* **TERN**: "Towards Efficient Cross-Modal Visual Textual Retrieval using Transformer-Encoder Deep Features", CBMI, 2021 (*National Research Council, Italy*). [[Paper](https://arxiv.org/abs/2106.00358)]
* **HiT**: "HiT: Hierarchical Transformer with Momentum Contrast for Video-Text Retrieval", ICCV, 2021 (*Kuaishou*). [[Paper](https://arxiv.org/abs/2103.15049)]
* **VisualSparta**: "VisualSparta: Sparse Transformer Fragment-level Matching for Large-scale Text-to-Image Search", arXiv, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2101.00265)]
* **WebVid-2M**: "Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2104.00650)]
* **CCR-CCS**: "More Than Just Attention: Learning Cross-Modal Attentions with Contrastive Constraints", arXiv, 2021 (*Rutgers + Amazon*). [[Paper](https://arxiv.org/abs/2105.09597)]
* **BridgeFormer**: "BridgeFormer: Bridging Video-text Retrieval with Multiple Choice Questions", arXiv, 2022 (*HKU*). [[Paper](https://arxiv.org/abs/2201.04850)][[Website](https://geyuying.github.io/MCQ.html)]

### Other Multi-Modal Tasks
* Detection:
    * **MDETR**: "MDETR - Modulated Detection for End-to-End Multi-Modal Understanding", ICCV, 2021 (*NYU*). [[Paper](https://arxiv.org/abs/2104.12763)][[PyTorch](https://github.com/ashkamath/mdetr)][[Website](https://ashkamath.github.io/mdetr_page/)]
    * **StrucTexT**: "StrucTexT: Structured Text Understanding with Multi-Modal Transformers", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2108.02923)]
* Segmentation:
    * **VLT**: "Vision-Language Transformer and Query Generation for Referring Segmentation", ICCV, 2021 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2108.05565)][[Tensorflow](https://github.com/henghuiding/Vision-Language-Transformer)]
* Analysis:
    * **MM-Explainability**: "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers", ICCV, 2021 (*Tel Aviv*). [[Paper](https://arxiv.org/abs/2103.15679)][[PyTorch](https://github.com/hila-chefer/Transformer-MM-Explainability)]
* Generation:
    * **DALL-E**: "Zero-Shot Text-to-Image Generation", ICML, 2021 (*OpenAI*). [[Paper](https://arxiv.org/abs/2102.12092)][[PyTorch](https://github.com/openai/DALL-E)][[PyTorch (lucidrains)](https://github.com/lucidrains/DALLE-pytorch)]
    * **CogView**: "CogView: Mastering Text-to-Image Generation via Transformers", NeurIPS, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.13290)][[PyTorch](https://github.com/THUDM/CogView)][[Website](https://lab.aminer.cn/cogview/index.html)]
    * **DALL-Eval**: "DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers", arXiv, 2022 (*UNC*). [[Paper](https://arxiv.org/abs/2202.04053)][[PyTorch](https://github.com/j-min/DallEval)]
* Speaker Localization:
    * **?**: "The Right to Talk: An Audio-Visual Transformer Approach", ICCV, 2021 (*University of Arkansas*). [[Paper](https://arxiv.org/abs/2108.03256)]
* Multi-task:
    * **UniT**: "Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2102.10772)][[PyTorch](https://github.com/facebookresearch/mmf)][[Website](https://mmf.sh/)]
* Language-based Video Editing:
    * **M<sup>3</sup>L**: "Language-based Video Editing via Multi-Modal Multi-Level Transformer", arXiv, 2021 (*UCSB*). [[Paper](https://arxiv.org/abs/2104.01122)]
* Video Summarization:
    * **GPT2MVS**: "GPT2MVS: Generative Pre-trained Transformer-2 for Multi-modal Video Summarization", ICMR, 2021 (*BBC*). [[Paper](https://arxiv.org/abs/2104.12465)]
    * **QVHighlights**: "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries", NeurIPS, 2021 (*UNC*). [[Paper](https://arxiv.org/abs/2107.09609)][[PyTorch](https://github.com/jayleicn/moment_detr)]
    * **HMT**: "Hierarchical Multimodal Transformer to Summarize Videos", arXiv, 2021 (*Xidian University*). [[Paper](https://arxiv.org/abs/2109.10559)]
* Visual Document Understanding:
    * **DocFormer**: "DocFormer: End-to-End Transformer for Document Understanding", ICCV, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2106.11539)]
    * **TableFormer**: "TableFormer: Table Structure Understanding with Transformers", arXiv, 2022 (*IBM*). [[Paper](https://arxiv.org/abs/2203.01017)]
    * **DocEnTr**: "DocEnTr: An End-to-End Document Image Enhancement Transformer", arXiv, 2022 (*UAB, Spain*). [[Paper](https://arxiv.org/abs/2201.10252)][[PyTorch](https://github.com/dali92002/DocEnTR)]
    * **DocSegTr**: "DocSegTr: An Instance-Level End-to-End Document Image Segmentation Transformer", arXiv, 2022 (*UAB, Spain*). [[Paper](https://arxiv.org/abs/2201.11438)]
* Robotics:
    * **CRT**: "Case Relation Transformer: A Crossmodal Language Generation Model for Fetching Instructions", IROS, 2021 (*Keio University*). [[Paper](https://arxiv.org/abs/2107.00789)]
    * **TraSeTR**: "TraSeTR: Track-to-Segment Transformer with Contrastive Query for Instance-level Instrument Segmentation in Robotic Surgery", ICRA, 2022 (*CUHK*). [[Paper](https://arxiv.org/abs/2202.08453)]
* Multi-modal Fusion:
    * **MICA**: "Attention Is Not Enough: Mitigating the Distribution Discrepancy in Asynchronous Multimodal Sequence Fusion", ICCV, 2021 (*Southwest Jiaotong University*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liang_Attention_Is_Not_Enough_Mitigating_the_Distribution_Discrepancy_in_Asynchronous_ICCV_2021_paper.html)]
    * **IFT**: "Image Fusion Transformer", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2107.09011)][[PyTorch](https://github.com/Vibashan/Image-Fusion-Transformer)]
    * **PPT**: "PPT Fusion: Pyramid Patch Transformerfor a Case Study in Image Fusion", arXiv, 2021 (*?*). [[Paper](https://arxiv.org/abs/2107.13967)]
    * **TransFuse**: "TransFuse: A Unified Transformer-based Image Fusion Framework using Self-supervised Learning", arXiv, 2022 (*Fudan University*). [[Paper](https://arxiv.org/abs/2201.07451)]
* Scene Graph:
    * **BGT-Net**: "BGT-Net: Bidirectional GRU Transformer Network for Scene Graph Generation", CVPRW, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2109.05346)]
    * **STTran**: "Spatial-Temporal Transformer for Dynamic Scene Graph Generation", ICCV, 2021 (*Leibniz University Hannover, Germany*). [[Paper](https://arxiv.org/abs/2107.12309)][[PyTorch](https://github.com/yrcong/STTran)]
    * **SGG-NLS**: "Learning to Generate Scene Graph from Natural Language Supervision", ICCV, 2021 (*University of Wisconsin-Madison*). [[Paper](https://arxiv.org/abs/2109.02227)][[PyTorch](https://github.com/YiwuZhong/SGG_from_NLS)]
    * **SGG-Seq2Seq**: "Context-Aware Scene Graph Generation With Seq2Seq Transformers", ICCV, 2021 (*Layer 6 AI, Canada*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Lu_Context-Aware_Scene_Graph_Generation_With_Seq2Seq_Transformers_ICCV_2021_paper.html)][[PyTorch](https://github.com/layer6ai-labs/SGG-Seq2Seq)]
    * **Relation-Transformer**: "Scenes and Surroundings: Scene Graph Generation using Relation Transformer", arXiv, 2021 (*LMU Munich*). [[Paper](https://arxiv.org/abs/2107.05448)]
    * **RelTR**: "RelTR: Relation Transformer for Scene Graph Generation", arXiv, 2022 (*Leibniz University Hannover, Germany*). [[Paper](https://arxiv.org/abs/2201.11460)][[PyTorch](https://github.com/yrcong/RelTR)]
* Human Interaction:
    * **Dyadformer**: "Dyadformer: A Multi-modal Transformer for Long-Range Modeling of Dyadic Interactions", ICCVW, 2021 (*Universitat de Barcelona*). [[Paper](https://arxiv.org/abs/2109.09487)]
* Sign Language Translation:
    * **LWTA**: "Stochastic Transformer Networks with Linear Competing Units: Application to end-to-end SL Translation", ICCV, 2021 (*Cyprus University of Technology*). [[Paper](https://arxiv.org/abs/2109.13318)]
* 3D Object Identification:
    * **3DRefTransformer**: "3DRefTransformer: Fine-Grained Object Identification in Real-World Scenes Using Natural Language", WACV, 2022 (*KAUST*). [[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Abdelreheem_3DRefTransformer_Fine-Grained_Object_Identification_in_Real-World_Scenes_Using_Natural_Language_WACV_2022_paper.html)][[Website](https://vision-cair.github.io/3dreftransformer/)]
* Speech Recognition:
    * **AV-HuBERT**: "Robust Self-Supervised Audio-Visual Speech Recognition", arXiv, 2022 (*Meta*). [[Paper](https://arxiv.org/abs/2201.01763)][[PyTorch](https://github.com/facebookresearch/av_hubert)]
    * **?**: "Transformer-Based Video Front-Ends for Audio-Visual Speech Recognition", arXiv, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2201.10439)]
* Text Spotting:
    * **TTS**: "Towards Weakly-Supervised Text Spotting using a Multi-Task Transformer", arXiv, 2022 (*Amazon*). [[Paper](https://arxiv.org/abs/2202.05508)] 
* Emotion Recognition:
    * **?**: "A Pre-trained Audio-Visual Transformer for Emotion Recognition", ICASSP, 2022 (*USC*). [[Paper](https://arxiv.org/abs/2201.09165)]

## Other High-level Vision Tasks
### Point Cloud
* **PCT**: "PCT: Point Cloud Transformer", arXiv, 2020 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2012.09688)][[Jittor](https://github.com/MenghaoGuo/PCT)][[PyTorch (uyzhang)](https://github.com/uyzhang/PCT_Pytorch)]
* **Point-Transformer**: "Point Transformer", arXiv, 2020 (*Ulm University*). [[Paper](https://arxiv.org/abs/2011.00931)]
* **NDT-Transformer**: "NDT-Transformer: Large-Scale 3D Point Cloud Localisation using the Normal Distribution Transform Representation", ICRA, 2021 (*University of Sheffield*). [[Paper](https://arxiv.org/abs/2103.12292)][[PyTorch](https://github.com/dachengxiaocheng/NDT-Transformer)]
* **P4Transformer**: "Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos", CVPR, 2021 (*NUS*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.html)]
* **PTT**: "PTT: Point-Track-Transformer Module for 3D Single Object Tracking in Point Clouds", IROS, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2108.06455)][[PyTorch (in construction)](https://github.com/shanjiayao/PTT)]
* **SnowflakeNet**: "SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer", ICCV, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2108.04444)][[PyTorch](https://github.com/AllenXiangX/SnowflakeNet)]
* **PoinTr**: "PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers", ICCV, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2108.08839)][[PyTorch](https://github.com/yuxumin/PoinTr)]
* **Point-Transformer**: "Point Transformer", ICCV, 2021 (*Oxford + CUHK*). [[Paper](https://arxiv.org/abs/2012.09164)][[PyTorch (lucidrains)](https://github.com/lucidrains/point-transformer-pytorch)]
* **CT**: "Cloud Transformers: A Universal Approach To Point Cloud Processing Tasks", ICCV, 2021 (*Samsung*). [[Paper](https://arxiv.org/abs/2007.11679)]
* **3DVG-Transformer**: "3DVG-Transformer: Relation Modeling for Visual Grounding on Point Clouds", ICCV, 2021 (*Beihang University*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_3DVG-Transformer_Relation_Modeling_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.html)]
* **PPT-Net**: "Pyramid Point Cloud Transformer for Large-Scale Place Recognition", ICCV, 2021 (*Nanjing University of Science and Technology*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Hui_Pyramid_Point_Cloud_Transformer_for_Large-Scale_Place_Recognition_ICCV_2021_paper.html)]
* **?**: "Shape registration in the time of transformers", NeurIPS, 2021 (*Sapienza University of Rome*). [[Paper](https://arxiv.org/abs/2106.13679)]
* **YOGO**: "You Only Group Once: Efficient Point-Cloud Processing with Token Representation and Relation Inference Module", arXiv, 2021 (*Berkeley*). [[Paper](https://arxiv.org/abs/2103.09975)][[PyTorch](https://github.com/chenfengxu714/YOGO)]
* **DTNet**: "Dual Transformer for Point Cloud Analysis", arXiv, 2021 (*Southwest University*). [[Paper](https://arxiv.org/abs/2104.13044)]
* **MLMSPT**: "Point Cloud Learning with Transformer", arXiv, 2021 (*Southwest University*). [[Paper](https://arxiv.org/abs/2104.13636)]
* **PQ-Transformer**: "PQ-Transformer: Jointly Parsing 3D Objects and Layouts from Point Clouds", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2109.05566)][[PyTorch](https://github.com/OPEN-AIR-SUN/PQ-Transformer)]
* **PST<sup>2</sup>**: "Spatial-Temporal Transformer for 3D Point Cloud Sequences", WACV, 2022 (*Sun Yat-sen University*). [[Paper](https://arxiv.org/abs/2110.09783)]
* **SCTN**: "SCTN: Sparse Convolution-Transformer Network for Scene Flow Estimation", AAAI, 2022 (*KAUST*). [[Paper](https://arxiv.org/abs/2105.04447)]
* **AWT-Net**: "Adaptive Wavelet Transformer Network for 3D Shape Representation Learning", ICLR, 2022 (*NYU*). [[Paper](https://openreview.net/forum?id=5MLb3cLCJY)]
* **?**: "Deep Point Cloud Reconstruction", ICLR, 2022 (*KAIST*). [[Paper](https://arxiv.org/abs/2111.11704)]
* **GeoTransformer**: "Geometric Transformer for Fast and Robust Point Cloud Registration", arXiv, 2022 (*National University of Defense Technology, China*). [[Paper](https://arxiv.org/abs/2202.06688)][[PyTorch](https://github.com/qinzheng93/GeoTransformer)]
* **LighTN**: "LighTN: Light-weight Transformer Network for Performance-overhead Tradeoff in Point Cloud Downsampling", arXiv, 2022 (*Beijing Jiaotong University*). [[Paper](https://arxiv.org/abs/2202.06263)]
* **PMP-Net++**: "PMP-Net++: Point Cloud Completion by Transformer-Enhanced Multi-step Point Moving Paths", arXiv, 2022 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2202.09507)]
* **SnowflakeNet**: "Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer", arXiv, 2022 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2202.09367)][[PyTorch](https://github.com/AllenXiangX/SnowflakeNet)]
* **ShapeFormer**: "ShapeFormer: Transformer-based Shape Completion via Sparse Representation", arXiv, 2022 (*Shenzhen University*). [[Paper](https://arxiv.org/abs/2201.10326)][[Website](https://shapeformer.github.io/)]
* **3DCTN**: "3DCTN: 3D Convolution-Transformer Network for Point Cloud Classification", arXiv, 2022 (*University of Waterloo, Canada*). [[Paper](https://arxiv.org/abs/2203.00828)]

### Pose Estimation
* Human-related: 
    * **Hand-Transformer**: "Hand-Transformer: Non-Autoregressive Structured Modeling for 3D Hand Pose Estimation", ECCV, 2020 (*Kwai*). [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4836_ECCV_2020_paper.php)]
    * **HOT-Net**: "HOT-Net: Non-Autoregressive Transformer for 3D Hand-Object Pose Estimation", ACMMM. 2020 (*Kwai*). [[Paper](https://cse.buffalo.edu/~jmeng2/publications/hotnet_mm20)]
    * **TransPose**: "TransPose: Towards Explainable Human Pose Estimation by Transformer", arXiv, 2020 (*Southeast University*). [[Paper](https://arxiv.org/abs/2012.14214)][[PyTorch](https://github.com/yangsenius/TransPose)]
    * **PTF**: "Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration", CVPR, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2104.08160)][[Code (in construction)](https://github.com/taconite/PTF)][[Website](https://taconite.github.io/PTF/website/PTF.html)]
    * **METRO**: "End-to-End Human Pose and Mesh Reconstruction with Transformers", CVPR, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2012.09760)][[PyTorch](https://github.com/microsoft/MeshTransformer)] 
    * **PRTR**: "Pose Recognition with Cascade Transformers", CVPR, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2104.06976)][[PyTorch](https://github.com/mlpc-ucsd/PRTR)]
    * **Mesh-Graphormer**: "Mesh Graphormer", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2104.00272)][[PyTorch](https://github.com/microsoft/MeshGraphormer)]
    * **THUNDR**: "THUNDR: Transformer-based 3D HUmaN Reconstruction with Markers", ICCV, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.09336)]
    * **PoseFormer**: "3D Human Pose Estimation with Spatial and Temporal Transformers", ICCV, 2021 (*UNC*). [[Paper](https://arxiv.org/abs/2103.10455)][[PyTorch](https://github.com/zczcwh/PoseFormer)]
    * **TransPose**: "TransPose: Keypoint Localization via Transformer", ICCV, 2021 (*Southeast University, China*). [[Paper](https://arxiv.org/abs/2012.14214)][[PyTorch](https://github.com/yangsenius/TransPose)]
    * **SCAT**: "SCAT: Stride Consistency With Auto-Regressive Regressor and Transformer for Hand Pose Estimation", ICCVW, 2021 (*Alibaba*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/html/Gao_SCAT_Stride_Consistency_With_Auto-Regressive_Regressor_and_Transformer_for_Hand_ICCVW_2021_paper.html)]
    * **POTR**: "Pose Transformers (POTR): Human Motion Prediction With Non-Autoregressive Transformers", ICCVW, 2021 (*Idiap*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/html/Martinez-Gonzalez_Pose_Transformers_POTR_Human_Motion_Prediction_With_Non-Autoregressive_Transformers_ICCVW_2021_paper.html)]
    * **HRT**: "HRFormer: High-Resolution Transformer for Dense Prediction", NeurIPS, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2110.09408)][[PyTorch](https://github.com/HRNet/HRFormer)]
    * **POET**: "End-to-End Trainable Multi-Instance Pose Estimation with Transformers", arXiv, 2021 (*EPFL*). [[Paper](https://arxiv.org/abs/2103.12115)]
    * **Lifting-Transformer**: "Lifting Transformer for 3D Human Pose Estimation in Video", arXiv, 2021 (*Peking*). [[Paper](https://arxiv.org/abs/2103.14304)]
    * **TFPose**: "TFPose: Direct Human Pose Estimation with Transformers", arXiv, 2021 (*The University of Adelaide*). [[Paper](https://arxiv.org/abs/2103.15320)][[PyTorch](https://github.com/aim-uofa/AdelaiDet/)]
    * **Skeletor**: "Skeletor: Skeletal Transformers for Robust Body-Pose Estimation", arXiv, 2021 (*University of Surrey*). [[Paper](https://arxiv.org/abs/2104.11712)]
    * **HandsFormer**: "HandsFormer: Keypoint Transformer for Monocular 3D Pose Estimation of Hands and Object in Interaction", arXiv, 2021 (*Graz University of Technology*). [[Paper](https://arxiv.org/abs/2104.14639)]
    * **TTP**: "Test-Time Personalization with a Transformer for Human Pose Estimation", NeurIPS, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2107.02133)][[PyTorch](https://github.com/harry11162/TTP)][[Website](https://liyz15.github.io/TTP/)]
    * **GraFormer**: "GraFormer: Graph Convolution Transformer for 3D Pose Estimation", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2109.08364)]
    * **GCT**: "Geometry-Contrastive Transformer for Generalized 3D Pose Transfer", AAAI, 2022 (*University of Oulu*). [[Paper](https://arxiv.org/abs/2112.07374)][[PyTorch](https://github.com/mikecheninoulu/CGT)]
    * **Swin-Pose**: "Swin-Pose: Swin Transformer Based Human Pose Estimation", arXiv, 2022 (*UMass Lowell*) [[Paper](https://arxiv.org/abs/2201.07384)]
    * **Poseur**: "Poseur: Direct Human Pose Regression with Transformers", arXiv, 2022 (*The University of Adelaide, Australia*). [[Paper](https://arxiv.org/abs/2201.07412)]
    * **HeadPosr**: "HeadPosr: End-to-end Trainable Head Pose Estimation using Transformer Encoders", arXiv, 2022 (*ETHZ*). [[Paper](https://arxiv.org/abs/2202.03548)]
* Others:
    * **TAPE**: "Transformer Guided Geometry Model for Flow-Based Unsupervised Visual Odometry", arXiv, 2020 (*Tianjing University*). [[Paper](https://arxiv.org/abs/2101.02143)]
    * **T6D-Direct**: "T6D-Direct: Transformers for Multi-Object 6D Pose Direct Regression", GCPR, 2021 (*University of Bonn*). [[Paper](https://arxiv.org/abs/2109.10948)]
    * **6D-ViT**: "6D-ViT: Category-Level 6D Object Pose Estimation via Transformer-based Instance Representation Learning", arXiv, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2110.04792)]

### Tracking
* **TransTrack**: "TransTrack: Multiple-Object Tracking with Transformer",arXiv, 2020 (*HKU + ByteDance*) . [[Paper](https://arxiv.org/abs/2012.15460)][[PyTorch](https://github.com/PeizeSun/TransTrack)]
* **TransformerTrack**: "Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking", CVPR, 2021 (*USTC*). [[Paper](https://arxiv.org/abs/2103.11681)][[PyTorch](https://github.com/594422814/TransformerTrack)]
* **TransT**: "Transformer Tracking", CVPR, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2103.15436)][[PyTorch](https://github.com/chenxin-dlut/TransT)]
* **STARK**: "Learning Spatio-Temporal Transformer for Visual Tracking", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.17154)][[PyTorch](https://github.com/researchmm/Stark)]
* **HiFT**: "HiFT: Hierarchical Feature Transformer for Aerial Tracking", ICCV, 2021 (*Tongji University*). [[Paper](https://arxiv.org/abs/2108.00202)][[PyTorch](https://github.com/vision4robotics/HiFT)]
* **DTT**: "High-Performance Discriminative Tracking With Transformers", ICCV, 2021 (*CAS*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_High-Performance_Discriminative_Tracking_With_Transformers_ICCV_2021_paper.html)]
* **DualTFR**: "Learning Tracking Representations via Dual-Branch Fully Transformer Networks", ICCVW, 2021 (*Microsoft*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/VOT/html/Xie_Learning_Tracking_Representations_via_Dual-Branch_Fully_Transformer_Networks_ICCVW_2021_paper.html)][[PyTorch (in construction)](https://github.com/phiphiphi31/DualTFR)]
* **TrackFormer**: "TrackFormer: Multi-Object Tracking with Transformers", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2101.02702)]
* **TransCenter**: "TransCenter: Transformers with Dense Queries for Multiple-Object Tracking", arXiv, 2021 (*INRIA + MIT*). [[Paper](https://arxiv.org/abs/2103.15145)]
* **TransMOT**: "TransMOT: Spatial-Temporal Graph Transformer for Multiple Object Tracking", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2104.00194)]
* **TREG**: "Target Transformed Regression for Accurate Tracking", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2104.00403)][[Code (in construction)](https://github.com/MCG-NJU/TREG)]
* **MOTR**: "MOTR: End-to-End Multiple-Object Tracking with TRansformer", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2105.03247)][[PyTorch](https://github.com/megvii-model/MOTR)]
* **TrTr**: "TrTr: Visual Tracking with Transformer", arXiv, 2021 (*University of Tokyo*). [[Paper](https://arxiv.org/abs/2105.03817)][[Pytorch](https://github.com/tongtybj/TrTr)]
* **RelationTrack**: "RelationTrack: Relation-aware Multiple Object Tracking with Decoupled Representation", arXiv, 2021 (*Huazhong Univerisity of Science and Technology*). [[Paper](https://arxiv.org/abs/2105.04322)]
* **SiamTPN**: "Siamese Transformer Pyramid Networks for Real-Time UAV Tracking", WACV, 2022 (*New York University*). [[Paper](https://arxiv.org/abs/2110.08822)]

### Re-ID
* **PAT**: "Diverse Part Discovery: Occluded Person Re-Identification With Part-Aware Transformer", CVPR, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2106.04095)]
* **HAT**: "HAT: Hierarchical Aggregation Transformers for Person Re-identification", ACMMM, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2107.05946)]
* **TransReID**: "TransReID: Transformer-based Object Re-Identification", ICCV, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2102.04378)][[PyTorch](https://github.com/heshuting555/TransReID)]
* **APD**: "Transformer Meets Part Model: Adaptive Part Division for Person Re-Identification", ICCVW, 2021 (*Meituan*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/HTCV/html/Lai_Transformer_Meets_Part_Model_Adaptive_Part_Division_for_Person_Re-Identification_ICCVW_2021_paper.html)]
* **Pirt**: "Pose-guided Inter- and Intra-part Relational Transformer for Occluded Person Re-Identification", ACMMM, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2109.03483)]
* **TransMatcher**: "Transformer-Based Deep Image Matching for Generalizable Person Re-identification", NeurIPS, 2021 (*IIAI*). [[Paper](https://arxiv.org/abs/2105.14432)][[PyTorch](https://github.com/ShengcaiLiao/QAConv)]
* **STT**: "Spatiotemporal Transformer for Video-based Person Re-identification", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2103.16469)] 
* **AAformer**: "AAformer: Auto-Aligned Transformer for Person Re-Identification", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2104.00921)]
* **TMT**: "A Video Is Worth Three Views: Trigeminal Transformers for Video-based Person Re-identification", arXiv, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2104.01745)]
* **LA-Transformer**: "Person Re-Identification with a Locally Aware Transformer", arXiv, 2021 (*University of Maryland Baltimore County*). [[Paper](https://arxiv.org/abs/2106.03720)]
* **DRL-Net**: "Learning Disentangled Representation Implicitly via Transformer for Occluded Person Re-Identification", arXiv, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2107.02380)]
* **GiT**: "GiT: Graph Interactive Transformer for Vehicle Re-identification", arXiv, 2021 (*Huaqiao University*). [[Paper](https://arxiv.org/abs/2107.05475)]
* **OH-Former**: "OH-Former: Omni-Relational High-Order Transformer for Person Re-Identification", arXiv, 2021 (*Shanghaitech University*). [[Paper](https://arxiv.org/abs/2109.11159)]
* **CMTR**: "CMTR: Cross-modality Transformer for Visible-infrared Person Re-identification", arXiv, 2021 (*Beijing Jiaotong University*). [[Paper](https://arxiv.org/abs/2110.08994)]
* **PFD**: "Pose-guided Feature Disentangling for Occluded Person Re-identification Based on Transformer", AAAI, 2022 (*Peking*). [[Paper](https://arxiv.org/abs/2112.02466)][[PyTorch](https://github.com/WangTaoAs/PFD_Net)]
* **PiT**: "Multi-direction and Multi-scale Pyramid in Transformer for Video-based Pedestrian Retrieval", IEEE Transactions on Industrial Informatics, 2022 (* Peking*). [[Paper](https://arxiv.org/abs/2202.06014)]
* **?**: "Motion-Aware Transformer For Occluded Person Re-identification", arXiv, 2022 (*NetEase, China*). [[Paper](https://arxiv.org/abs/2202.04243)]
* **PFT**: "Short Range Correlation Transformer for Occluded Person Re-Identification", arXiv, 2022 (*Nanjing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2201.01090)]

### Face
* General:
    * **FAU-Transformer**: "Facial Action Unit Detection With Transformers", CVPR, 2021 (*Rakuten Institute of Technology*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Jacob_Facial_Action_Unit_Detection_With_Transformers_CVPR_2021_paper.html)]
    * **Clusformer**: "Clusformer: A Transformer Based Clustering Approach to Unsupervised Large-Scale Face and Visual Landmark Recognition", CVPR, 2021 (*VinAI Research, Vietnam*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Nguyen_Clusformer_A_Transformer_Based_Clustering_Approach_to_Unsupervised_Large-Scale_Face_CVPR_2021_paper.html)][[Code (in construction)](https://github.com/VinAIResearch/Clusformer)]
    * **TransFER**: "TransFER: Learning Relation-aware Facial Expression Representations with Transformers", ICCV, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2108.11116)]
    * **Latent-Transformer**: "A Latent Transformer for Disentangled Face Editing in Images and Videos", ICCV, 2021 (*Institut Polytechnique de Paris*). [[Paper](https://arxiv.org/abs/2106.11895)][[PyTorch](https://github.com/InterDigitalInc/latent-transformer)]
    * **ViT-Face**: "Face Transformer for Recognition", arXiv, 2021 (*Beijing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2103.14803)]
    * **CVT-Face**: "Robust Facial Expression Recognition with Convolutional Visual Transformers", arXiv, 2021 (*Hunan University*). [[Paper](https://arxiv.org/abs/2103.16854)]
    * **TransRPPG**: "TransRPPG: Remote Photoplethysmography Transformer for 3D Mask Face Presentation Attack Detection", arXiv, 2021 (*University of Oulu*). [[Paper](https://arxiv.org/abs/2104.07419)]
    * **FaceT**: "Learning to Cluster Faces via Transformer", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2104.11502)]
    * **VidFace**: "VidFace: A Full-Transformer Solver for Video FaceHallucination with Unaligned Tiny Snapshots", arXiv, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2105.14954)]
    * **MViT**: "MViT: Mask Vision Transformer for Facial Expression Recognition in the wild", arXiv, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2106.04520)]
    * **FAA**: "Shuffle Transformer with Feature Alignment for Video Face Parsing", arXiv, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2106.08650)]
    * **ViT-SE**: "Learning Vision Transformer with Squeeze and Excitation for Facial Expression Recognition", arXiv, 2021 (*CentraleSupélec, France*). [[Paper](https://arxiv.org/abs/2107.03107)]
    * **EST**: "Expression Snippet Transformer for Robust Video-based Facial Expression Recognition", arXiv, 2021 (*China University of Geosciences*). [[Paper](https://arxiv.org/abs/2109.08409)][[PyTorch](https://anonymous.4open.science/r/ATSE-C58B)]
    * **LOTR**: "LOTR: Face Landmark Localization Using Localization Transformer", arXiv, 2021 (*Sertis, Thailand*). [[Paper](https://arxiv.org/abs/2109.10057)]
    * **MFEViT**: "MFEViT: A Robust Lightweight Transformer-based Network for Multimodal 2D+3D Facial Expression Recognition", arXiv, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2109.13086)]
    * **FAT**: "Facial Attribute Transformers for Precise and Robust Makeup Transfer", WACV, 2022 (*University of Rochester*). [[Paper](https://arxiv.org/abs/2104.02894)]
    * **SSAT**: "SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal", AAAI, 2022 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2112.03631)][[PyTorch](https://gitee.com/sunzhaoyang0304/ssat-msp)]
    * **ICT**: "Protecting Celebrities with Identity Consistency Transformer", CVPR, 2022 (*Microsoft*). [[Paper](https://arxiv.org/abs/2203.01318)]
    * **RestoreFormer**: "RestoreFormer: High-Quality Blind Face Restoration From Undegraded Key-Value Pairs", arXiv, 2022 (*HKU*). [[Paper](https://arxiv.org/abs/2201.06374)]
* Attack-related:
    * **?**: "Video Transformer for Deepfake Detection with Incremental Learning", ACMMM, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2108.05307)]
    * **ViTranZFAS**: "On the Effectiveness of Vision Transformers for Zero-shot Face Anti-Spoofing", International Joint Conference on Biometrics (IJCB), 2021 (*Idiap*). [[Paper](https://arxiv.org/abs/2011.08019)]
    * **MTSS**: "Multi-Teacher Single-Student Visual Transformer with Multi-Level Attention for Face Spoofing Detection", BMVC, 2021 (*National Taiwan Ocean University*). [[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0113.pdf)]
    * **CViT**: "Deepfake Video Detection Using Convolutional Vision Transformer", arXiv, 2021 (*Jimma University*). [[Paper](https://arxiv.org/abs/2102.11126)]
    * **ViT-Distill**: "Deepfake Detection Scheme Based on Vision Transformer and Distillation", arXiv, 2021 (*Sookmyung Women’s University*). [[Paper](https://arxiv.org/abs/2104.01353)]
    * **M2TR**: "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection", arXiv, 2021 (*Fudan University*). [[Paper](https://arxiv.org/abs/2104.09770)]
    * **Cross-ViT**: "Combining EfficientNet and Vision Transformers for Video Deepfake Detection", arXiv, 2021 (*University of Pisa*). [[Paper](https://arxiv.org/abs/2107.02612)][[PyTorch](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)]
    * **?**: "Self-supervised Transformer for Deepfake Detection", arXiv, 2022 (*USTC, China*). [[Paper](https://arxiv.org/abs/2203.01265)]
    * **ViTransPAD**: "ViTransPAD: Video Transformer using convolution and self-attention for Face Presentation Attack Detection", arXiv, 2022 (*University of La Rochelle, France*). [[Paper](https://arxiv.org/abs/2203.01562)]

### Neural Architecture Search
* **HR-NAS**: "HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers", CVPR, 2021 (*HKU*). [[Paper](https://arxiv.org/abs/2106.06560)][[PyTorch](https://github.com/dingmyu/HR-NAS)]
* **CATE**: "CATE: Computation-aware Neural Architecture Encoding with Transformers", ICML, 2021 (*Michigan State University*). [[Paper](https://arxiv.org/abs/2102.07108)]
* **AutoFormer**: "AutoFormer: Searching Transformers for Visual Recognition", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.00651)][[PyTorch](https://github.com/microsoft/Cream/tree/main/AutoFormer)]
* **GLiT**: "GLiT: Neural Architecture Search for Global and Local Image Transformer", ICCV, 2021 (*The University of Sydney + SenseTime*). [[Paper](https://arxiv.org/abs/2107.02960)]
* **BossNAS**: "BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search", ICCV, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2103.12424)][[PyTorch](https://github.com/changlin31/BossNAS)]
* **AutoformerV2**: "Searching the Search Space of Vision Transformer", NeurIPS, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2111.14725)][[PyTorch](https://github.com/microsoft/Cream/tree/main/AutoFormerV2)]
* **TNASP**: "TNASP: A Transformer-based NAS Predictor with a Self-evolution Framework", NeurIPS, 2021 (*CAS + Kuaishou*). [[Paper](https://proceedings.neurips.cc/paper/2021/hash/7fa1575cbd7027c9a799983a485c3c2f-Abstract.html)]
* **ViTAS**: "Vision Transformer Architecture Search", arXiv, 2021 (*The University of Sydney + SenseTime*). [[Paper](https://arxiv.org/abs/2106.13700)]
* **PSViT**: "PSViT: Better Vision Transformer via Token Pooling and Attention Sharing", arXiv, 2021 (*The University of Sydney + SenseTime*). [[Paper](https://arxiv.org/abs/2108.03428)]
* **ViT-ResNAS**: "Searching for Efficient Multi-Stage Vision Transformers", arXiv, 2021 (*MIT*). [[Paper](https://arxiv.org/abs/2109.00642)][[PyTorch](https://github.com/yilunliao/vit-search)]
* **UniNet**: "UniNet: Unified Architecture Search with Convolution, Transformer, and MLP", arXiv, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2110.04035)]
* **As-ViT**: "Auto-scaling Vision Transformers without Training", ICLR, 2022 (*UT Austin*). [[Paper](https://arxiv.org/abs/2202.11921)][[PyTorch](https://github.com/VITA-Group/AsViT)]
* **NASViT**: "NASViT: Neural Architecture Search for Efficient Vision Transformers with Gradient Conflict aware Supernet Training", ICLR, 2022 (*Facebook*). [[Paper](https://openreview.net/forum?id=Qaw16njk6L)]
* **ViT-Slim**: "Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space", arXiv, 2022 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2201.00814)]


## Low-level Vision Tasks
### Image Restoration (e.g. super-resolution, image denoising, demosaicing, compression artifacts reduction, etc.)
* **NLRN**: "Non-Local Recurrent Network for Image Restoration", NeurIPS, 2018 (*UIUC*). [[Paper](https://arxiv.org/abs/1806.02919)][[Tensorflow](https://github.com/Ding-Liu/NLRN)]
* **RNAN**: "Residual Non-local Attention Networks for Image Restoration", ICLR, 2019 (*Northeastern University*). [[Paper](https://openreview.net/forum?id=HkeGhoA5FX)][[PyTorch](https://github.com/yulunzhang/RNAN)]
* **SAN**: "Second-Order Attention Network for Single Image Super-Resolution", CVPR, 2019 (*Tsinghua*). [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)][[PyTorch](https://github.com/daitao/SAN)]
* **CS-NL**: "Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining", CVPR, 2020 (*UIUC*). [[Paper](https://arxiv.org/abs/2006.01424)][[PyTorch](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)]
* **TTSR**: "Learning Texture Transformer Network for Image Super-Resolution", CVPR, 2020 (*Microsoft*). [[Paper](https://arxiv.org/abs/2006.04139)][[PyTorch](https://github.com/researchmm/TTSR)]
* **HAN**: "Single Image Super-Resolution via a Holistic Attention Network", ECCV, 2020 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2008.08767)][[PyTorch](https://github.com/wwlCape/HAN)]
* **PANet**: "Pyramid Attention Networks for Image Restoration", arXiv, 2020 (*UIUC*). [[Paper](https://arxiv.org/abs/2004.13824)][[PyTorch](https://github.com/SHI-Labs/Pyramid-Attention-Networks)]
* **IPT**: "Pre-Trained Image Processing Transformer", CVPR, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2012.00364)][[PyTorch (in construction)](https://github.com/huawei-noah/Pretrained-IPT)]
* **NLSN**: "Image Super-Resolution With Non-Local Sparse Attention", CVPR, 2021 (*UIUC*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.html)]
* **SwinIR**: "SwinIR: Image Restoration Using Swin Transformer", ICCVW, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2108.10257)][[PyTorch](https://github.com/JingyunLiang/SwinIR)]
* **ITSRN**: "Implicit Transformer Network for Screen Content Image Continuous Super-Resolution", NeurIPS, 2021 (*Tianjin University*). [[Paper](https://arxiv.org/abs/2112.06174)][[PyTorch](https://github.com/codyshen0000/ITSRN)]
* **SDNet**: "SDNet: multi-branch for single image deraining using swin", arXiv, 2021 (*Xinjiang University*). [[Paper](https://arxiv.org/abs/2105.15077)][[Code (in construction)](https://github.com/H-tfx/SDNet)]
* **FPAN**: "Feedback Pyramid Attention Networks for Single Image Super-Resolution", arXiv, 2021 (*Nanjing University of Science and Technology*). [[Paper](https://arxiv.org/abs/2106.06966)]
* **ATTSF**: "Attention! Stay Focus!", arXiv, 2021 (*BridgeAI, Seoul*). [[Paper](https://arxiv.org/abs/2104.07925)][[Tensorflow](https://github.com/tuvovan/ATTSF)]
* **ESRT**: "Efficient Transformer for Single Image Super-Resolution", arXiv, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2108.11084)]
* **Fusformer**: "Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution", arXiv, 2021 (*University of Electronic Science and Technology of China*). [[Paper](https://arxiv.org/abs/2109.02079)]
* **HyLoG-ViT**: "Hybrid Local-Global Transformer for Image Dehazing", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2109.07100)]
* **TANet**: "TANet: A new Paradigm for Global Face Super-resolution via Transformer-CNN Aggregation Network", arXiv, 2021 (*Wuhan Institute of Technology*). [[Paper](https://arxiv.org/abs/2109.08174)]
* **DPT**: "Detail-Preserving Transformer for Light Field Image Super-Resolution", AAAI, 2022 (*Beijing Institute of Technology*). [[Paper](https://arxiv.org/abs/2201.00346)][[PyTorch](https://github.com/BITszwang/DPT)]
* **SiamTrans**: "SiamTrans: Zero-Shot Multi-Frame Image Restoration with Pre-Trained Siamese Transformers", AAAI, 2022 (*Huawei*). [[Paper](https://arxiv.org/abs/2112.09426)]
* **Uformer**: "Uformer: A General U-Shaped Transformer for Image Restoration", CVPR, 2022 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2106.03106)][[PyTorch](https://github.com/ZhendongWang6/Uformer)]
* **MAXIM**: "MAXIM: Multi-Axis MLP for Image Processing", CVPR, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2201.02973)]
* **LFT**: "Light Field Image Super-Resolution with Transformers", IEEE Signal Processing Letters, 2022 (*National University of Defense Technology, China*). [[Paper](https://arxiv.org/abs/2108.07597)][[PyTorch](https://github.com/ZhengyuLiang24/LFT)]

### Video Restoration
* **VSR-Transformer**: "Video Super-Resolution Transformer", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2106.06847)][[PyTorch](https://github.com/caojiezhang/VSR-Transformer)]
* **MANA**: "Memory-Augmented Non-Local Attention for Video Super-Resolution", arXiv, 2021 (*JD*). [[Paper](https://arxiv.org/abs/2108.11048)]
* **VRT**: "VRT: A Video Restoration Transformer", arXiv, 2022 (*ETHZ*). [[Paper](https://arxiv.org/abs/2201.12288)][[PyTorch](https://github.com/JingyunLiang/VRT)]
* **FGST**: "Flow-Guided Sparse Transformer for Video Deblurring", arXiv, 2022 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2201.01893)]

### Inpainting/Completion/Outpainting
* **Contexual-Attention**: "Generative Image Inpainting with Contextual Attention", CVPR, 2018 (*UIUC*). [[Paper](https://arxiv.org/abs/1801.07892)][[Tensorflow](https://github.com/JiahuiYu/generative_inpainting)]
* **PEN-Net**: "Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting", CVPR, 2019 (*Microsoft*). [[Paper](https://arxiv.org/abs/1904.07475)][[PyTorch](https://github.com/researchmm/PEN-Net-for-Inpainting)]
* **Copy-Paste**: "Copy-and-Paste Networks for Deep Video Inpainting", ICCV, 2019 (*Yonsei University*). [[Paper](https://arxiv.org/abs/1908.11587)][[PyTorch](https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting)]
* **Onion-Peel**: "Onion-Peel Networks for Deep Video Completion", ICCV, 2019 (*Yonsei University*). [[Paper](https://arxiv.org/abs/1908.08718)][[PyTorch](https://github.com/seoungwugoh/opn-demo)]
* **STTN**: "Learning Joint Spatial-Temporal Transformations for Video Inpainting", ECCV, 2020 (*Microsoft*). [[Paper](https://arxiv.org/abs/2007.10247)][[PyTorch](https://github.com/researchmm/STTN)]
* **FuseFormer**: "FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting", ICCV, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2109.02974)][[PyTorch](https://github.com/ruiliu-ai/FuseFormer)]
* **ICT**: "High-Fidelity Pluralistic Image Completion with Transformers", ICCV, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2103.14031)][[PyTorch](https://github.com/raywzy/ICT)][[Website](http://raywzy.com/ICT/)]
* **DSTT**: "Decoupled Spatial-Temporal Transformer for Video Inpainting", arXiv, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2104.06637)][[Code (in construction)](https://github.com/ruiliu-ai/DSTT)]
* **TFill**: "TFill: Image Completion via a Transformer-Based Architecture", arXiv, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2104.00845)][[Code (in construction)](https://github.com/lyndonzheng/TFill)]
* **BAT-Fill**: "Diverse Image Inpainting with Bidirectional and Autoregressive Transformers", arXiv, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2104.12335)]
* **?**: "Image-Adaptive Hint Generation via Vision Transformer for Outpainting", WACV, 2022 (*Sogang University, Korea*). [[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Kong_Image-Adaptive_Hint_Generation_via_Vision_Transformer_for_Outpainting_WACV_2022_paper.html)]
* **ZITS**: "Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding", CVPR, 2022 (*Fudan*). [[Paper](https://arxiv.org/abs/2203.00867)][[Code (in construction)](https://github.com/DQiaole/ZITS_inpainting)]
* **U-Transformer**: "Generalised Image Outpainting with U-Transformer", arXiv, 2022 (*Xi'an Jiaotong-Liverpool University*). [[Paper](https://arxiv.org/abs/2201.11403)]

### Image Generation
* **IT**: "Image Transformer", ICML, 2018 (*Google*). [[Paper](https://arxiv.org/abs/1802.05751)][[Tensorflow](https://github.com/tensorflow/tensor2tensor)]
* **PixelSNAIL**: "PixelSNAIL: An Improved Autoregressive Generative Model", ICML, 2018 (*Berkeley*). [[Paper](http://proceedings.mlr.press/v80/chen18h.html)][[Tensorflow](https://github.com/neocxi/pixelsnail-public)]
* **BigGAN**: "Large Scale GAN Training for High Fidelity Natural Image Synthesis", ICLR, 2019 (*DeepMind*). [[Paper](https://openreview.net/forum?id=B1xsqj09Fm)][[PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)]
* **SAGAN**: "Self-Attention Generative Adversarial Networks", ICML, 2019 (*Google*). [[Paper](http://proceedings.mlr.press/v97/zhang19d.html)][[Tensorflow](https://github.com/brain-research/self-attention-gan)]
* **VQGAN**: "Taming Transformers for High-Resolution Image Synthesis", CVPR, 2021 (*Heidelberg University*). [[Paper](https://arxiv.org/abs/2012.09841)][[PyTorch](https://github.com/CompVis/taming-transformers)][[Website](https://compvis.github.io/taming-transformers/)]
* **?**: "High-Resolution Complex Scene Synthesis with Transformers", CVPRW, 2021 (*Heidelberg University*). [[Paper](https://arxiv.org/abs/2105.06458)]
* **GANsformer**: "Generative Adversarial Transformers", ICML, 2021 (*Stanford + Facebook*). [[Paper](https://arxiv.org/abs/2103.01209)][[Tensorflow](https://github.com/dorarad/gansformer)]
* **PixelTransformer**: "PixelTransformer: Sample Conditioned Signal Generation", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2103.15813)][[Website](https://shubhtuls.github.io/PixelTransformer/)]
* **HWT**: "Handwriting Transformers", ICCV, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2104.03964)][[Code (in construction)](https://github.com/ankanbhunia/Handwriting-Transformers)]
* **Paint-Transformer**: "Paint Transformer: Feed Forward Neural Painting with Stroke Prediction", ICCV, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2108.03798)][[Paddle](https://github.com/PaddlePaddle/PaddleGAN)][[PyTorch](https://github.com/Huage001/PaintTransformer)]
* **AdaAttN**: "AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer", ICCV, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2108.03647)][[Paddle](https://github.com/PaddlePaddle/PaddleGAN)][[PyTorch](https://github.com/Huage001/AdaAttN)]
* **Geometry-Free**: "Geometry-Free View Synthesis: Transformers and no 3D Priors", ICCV, 2021 (*Heidelberg University*). [[Paper](https://arxiv.org/abs/2104.07652)][[PyTorch](https://github.com/CompVis/geometry-free-view-synthesis)]
* **VTGAN**: "VTGAN: Semi-supervised Retinal Image Synthesis and Disease Prediction using Vision Transformers", ICCVW, 2021 (*University of Nevada, Reno*). [[Paper](https://arxiv.org/abs/2104.06757)]
* **ATISS**: "ATISS: Autoregressive Transformers for Indoor Scene Synthesis", NeurIPS, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2110.03675)][[Website](https://nv-tlabs.github.io/ATISS/)]
* **GANsformer2**: "Compositional Transformers for Scene Generation", NeurIPS, 2021 (*Stanford + Facebook*). [[Paper](https://arxiv.org/abs/2111.08960)][[Tensorflow](https://github.com/dorarad/gansformer)]
* **TransGAN**: "TransGAN: Two Transformers Can Make One Strong GAN", NeurIPS, 2021 (*UT Austin*). [[Paper](https://arxiv.org/abs/2102.07074)][[PyTorch](https://github.com/VITA-Group/TransGAN)]
* **HiT**: "Improved Transformer for High-Resolution GANs", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.07631)][[Tensorflow](https://github.com/google-research/hit-gan)]
* **iLAT**: "The Image Local Autoregressive Transformer", NeurIPS, 2021 (*Fudan*). [[Paper](https://arxiv.org/abs/2106.02514)]
* **TokenGAN**: "Improving Visual Quality of Image Synthesis by A Token-based Generator with Transformers", NeurIPS, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2111.03481)]
* **SceneFormer**: "SceneFormer: Indoor Scene Generation with Transformers", arXiv, 2021 (*TUM*). [[Paper](https://arxiv.org/abs/2012.09793)]
* **SNGAN**: "Combining Transformer Generators with Convolutional Discriminators", arXiv, 2021 (*Fraunhofer ITWM*). [[Paper](https://arxiv.org/abs/2105.10189)]
* **StyTr2**: "StyTr^2: Unbiased Image Style Transfer with Transformers", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2105.14576)]
* **Styleformer**: "Styleformer: Transformer based Generative Adversarial Networks with Style Vector", arXiv, 2021 (*Seoul National University*). [[Paper](https://arxiv.org/abs/2106.07023)][[PyTorch](https://github.com/Jeeseung-Park/Styleformer)]
* **Invertible-Attention**: "Invertible Attention", arXiv, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2106.09003)]
* **GPA**: "Grid Partitioned Attention: Efficient Transformer Approximation with Inductive Bias for High Resolution Detail Generation", arXiv, 2021 (*Zalando Research, Germany*). [[Paper](https://arxiv.org/abs/2107.03742)][[PyTorch (in construction)](https://github.com/zalandoresearch/gpa)]
* **StyleSwin**: "StyleSwin: Transformer-based GAN for High-resolution Image Generation", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2112.10762)]
* **ViTGAN**: "ViTGAN: Training GANs with Vision Transformers", ICLR, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2107.04589)][[PyTorch (wilile26811249)](https://github.com/wilile26811249/ViTGAN)]
* **ViT-VQGAN**: "Vector-quantized Image Modeling with Improved VQGAN", ICLR, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2110.04627)]
* **U-Attention**: "Paying U-Attention to Textures: Multi-Stage Hourglass Vision Transformer for Universal Texture Synthesis", arXiv, 2022 (*Adobe*). [[Paper](https://arxiv.org/abs/2202.11703)]
* **MaskGIT**: "MaskGIT: Masked Generative Image Transformer", arXiv, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2202.04200)][[PyTorch (dome272)](https://github.com/dome272/MaskGIT-pytorch)]
* **Splice**: "Splicing ViT Features for Semantic Appearance Transfer", arXiv, 2022 (*Weizmann Institute of Science, Israel*). [[Paper](https://arxiv.org/abs/2201.00424)][[PyTorch](https://github.com/omerbt/Splice)][[Website](https://splice-vit.github.io/)]

### Video Generation
* **Subscale**: "Scaling Autoregressive Video Models", ICLR, 2020 (*Google*). [[Paper](https://openreview.net/forum?id=rJgsskrFwH)][[Website](https://sites.google.com/view/video-transformer-samples)]
* **ConvTransformer**: "ConvTransformer: A Convolutional Transformer Network for Video Frame Synthesis", arXiv, 2020 (*Southeast University*). [[Paper](https://arxiv.org/abs/2011.10185)]
* **OCVT**: "Generative Video Transformer: Can Objects be the Words?", ICML, 2021 (*Rutgers University*). [[Paper](http://proceedings.mlr.press/v139/wu21h.html)]
* **AIST++**: "Learn to Dance with AIST++: Music Conditioned 3D Dance Generation", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2101.08779)][[Code](https://github.com/google/aistplusplus_api)][[Website](https://google.github.io/aichoreographer/)]
* **VideoGPT**: "VideoGPT: Video Generation using VQ-VAE and Transformers", arXiv, 2021 (*Berkeley*). [[Paper](https://arxiv.org/abs/2104.10157)][[PyTorch](https://github.com/wilson1yan/VideoGPT)][[Website](https://wilson1yan.github.io/videogpt/index.html)]
* **DanceFormer**: "DanceFormer: Music Conditioned 3D Dance Generation with Parametric Motion Transformer", AAAI, 2022 (*Huiye Technology, China*). [[Paper](https://arxiv.org/abs/2103.10206)]

### Video Reconstruction
* **ET-Net**: "Event-Based Video Reconstruction Using Transformer", ICCV, 2021 (*University of Science and Technology of China*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Weng_Event-Based_Video_Reconstruction_Using_Transformer_ICCV_2021_paper.html)][[PyTorch](https://github.com/WarranWeng/ET-Net)]

### Other Low-Level Tasks
* Colorization: 
    * **ColTran**: "Colorization Transformer", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=5NA1PinlGFu)][[Tensorflow](https://github.com/google-research/google-research/tree/master/coltran)]
    * **ViT-I-GAN**: "ViT-Inception-GAN for Image Colourising", arXiv, 2021 (*D.Y Patil College of Engineering, India*). [[Paper](https://arxiv.org/abs/2106.06321)]
* Enhancement:
    * **STAR**: "STAR: A Structure-Aware Lightweight Transformer for Real-Time Image Enhancement", ICCV, 2021 (*CUHK + SenseBrain*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_STAR_A_Structure-Aware_Lightweight_Transformer_for_Real-Time_Image_Enhancement_ICCV_2021_paper.html)]
* Harmonization:
    * **HT**: "Image Harmonization With Transformer", ICCV, 2021 (*Ocean University of China*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Harmonization_With_Transformer_ICCV_2021_paper.html)]
* Others:
    * **MS-Unet**: "Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2109.08024)]
    * **TransMEF**: "TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning", AAAI, 2022 (*Fudan*). [[Paper](https://arxiv.org/abs/2112.01030)]
    * **?**: "Towards End-to-End Image Compression and Analysis with Transformers", AAAI, 2022 (*1Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2112.09300)][[PyTorch](https://github.com/BYchao100/Towards-Image-Compression-and-Analysis-with-Transformers)]
    * **Entroformer**: "Entroformer: A Transformer-based Entropy Model for Learned Image Compression", ICLR, 2022 (*Alibaba*). [[Paper](https://arxiv.org/abs/2202.05492)]
    * **GAP-CSCoT**: "Spectral Compressive Imaging Reconstruction Using Convolution and Spectral Contextual Transformer", arXiv, 2022 (*CAS*). [[Paper](https://arxiv.org/abs/2201.05768)]


## Reinforcement Learning
### Navigation
* **VTNet**: "VTNet: Visual Transformer Network for Object Goal Navigation", ICLR, 2021 (*ANU*). [[Paper](https://openreview.net/forum?id=DILxQP08O3B)]
* **MaAST**: "MaAST: Map Attention with Semantic Transformersfor Efficient Visual Navigation", ICRA, 2021 (*SRI*). [[Paper](https://arxiv.org/abs/2103.11374)]
* **TransFuser**: "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving", CVPR, 2021 (*MPI*). [[Paper](https://arxiv.org/abs/2104.09224)][[PyTorch](https://github.com/autonomousvision/transfuser)]
* **CMTP**: "Topological Planning With Transformers for Vision-and-Language Navigation", CVPR, 2021 (*Stanford*). [[Paper](https://arxiv.org/abs/2012.05292)]
* **VLN-BERT**: "VLN-BERT: A Recurrent Vision-and-Language BERT for Navigation", CVPR, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2011.13922)][[PyTorch](https://github.com/YicongHong/Recurrent-VLN-BERT)]
* **E.T.**: "Episodic Transformer for Vision-and-Language Navigation", ICCV, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.06453)][[PyTorch](https://github.com/alexpashevich/E.T.)]
* **HAMT**: "History Aware Multimodal Transformer for Vision-and-Language Navigation", NeurIPS, 2021 (*INRIA*). [[Paper](https://arxiv.org/abs/2110.13309)][[PyTorch](https://github.com/cshizhe/VLN-HAMT)][[Website](https://cshizhe.github.io/projects/vln_hamt.html)]
* **SOAT**: "SOAT: A Scene- and Object-Aware Transformer for Vision-and-Language Navigation", NeurIPS, 2021 (*Georgia Tech*). [[Paper](https://arxiv.org/abs/2110.14143)]
* **DUET**: "Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation", arXiv, 2022 (*INRIA*). [[Paper](https://arxiv.org/abs/2202.11742)][[Website](https://cshizhe.github.io/projects/vln_duet.html)]

### Other RL Tasks
* **SVEA**: "Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation", arXiv, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2107.00644)][[GitHub](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)][[Website](https://nicklashansen.github.io/SVEA/)]
* **LocoTransformer**: "Learning Vision-Guided Quadrupedal Locomotion End-to-End with Cross-Modal Transformers", ICLR, 2022 (*UCSD*). [[Paper](https://arxiv.org/abs/2107.03996)][[Website](https://rchalyang.github.io/LocoTransformer/)]


## Transfer Learning / Self-Supervised Learning / Few-Shot
* **CrossTransformer**: "CrossTransformers: spatially-aware few-shot transfer", NeurIPS, 2020 (*DeepMind*). [[Paper](https://arxiv.org/abs/2007.11498)][[Tensorflow](https://github.com/google-research/meta-dataset)]
* **URT**: "A Universal Representation Transformer Layer for Few-Shot Image Classification", ICLR, 2021 (*Mila*). [[Paper](https://openreview.net/forum?id=04cII6MumYV)][[PyTorch](https://github.com/liulu112601/URT)]
* **TRX**: "Temporal-Relational CrossTransformers for Few-Shot Action Recognition", CVPR, 2021 (*University of Bristol*). [[Paper](https://arxiv.org/abs/2101.06184)][[PyTorch](https://github.com/tobyperrett/TRX)]
* **Few-shot-Transformer**: "Few-Shot Transformation of Common Actions into Time and Space", arXiv, 2021 (*University of Amsterdam*). [[Paper](https://arxiv.org/abs/2104.02439)]
* **HyperTransformer**: "HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning", arXiv, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2201.04182)]


## Medical
* Segmentation:
    * **Cross-Transformer**: "The entire network structure of Crossmodal Transformer", ICBSIP, 2021 (*Capital Medical University*). [[Paper](https://arxiv.org/abs/2104.14273)]
    * **Segtran**: "Medical Image Segmentation using Squeeze-and-Expansion Transformers", IJCAI, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2105.09511)]
    * **i-ViT**: "Instance-based Vision Transformer for Subtyping of Papillary Renal Cell Carcinoma in Histopathological Image", MICCAI, 2021 (*Xi'an Jiaotong University*). [[Paper](https://arxiv.org/abs/2106.12265)][[PyTorch](https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer)][[Website](https://dataset.chenli.group/home/prcc-subtyping)]
    * **UTNet**: "UTNet: A Hybrid Transformer Architecture for Medical Image Segmentation", MICCAI, 2021 (*Rutgers*). [[Paper](https://arxiv.org/abs/2107.00781)]
    * **MCTrans**: "Multi-Compound Transformer for Accurate Biomedical Image Segmentation", MICCAI, 2021 (*HKU + CUHK*). [[Paper](https://arxiv.org/abs/2106.14385)][[Code (in construction)](https://github.com/JiYuanFeng/MCTrans)]
    * **Polyformer**: "Few-Shot Domain Adaptation with Polymorphic Transformers", MICCAI, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2107.04805)][[PyTorch](https://github.com/askerlee/segtran)]
    * **BA-Transformer**: "Boundary-aware Transformers for Skin Lesion Segmentation". MICCAI, 2021 (*Xiamen University*). [[Paper](https://arxiv.org/abs/2110.03864)][[PyTorch](https://github.com/jcwang123/BA-Transformer)]
    * **GT-U-Net**: "GT U-Net: A U-Net Like Group Transformer Network for Tooth Root Segmentation", MICCAIW, 2021 (*Hangzhou Dianzi University*). [[Paper](https://arxiv.org/abs/2109.14813)][[PyTorch](https://github.com/Kent0n-Li/GT-U-Net)]
    * **STN**: "Automatic size and pose homogenization with spatial transformer network to improve and accelerate pediatric segmentation", ISBI, 2021 (*Institut Polytechnique de Paris*). [[Paper](https://arxiv.org/abs/2107.02655)]
    * **T-AutoML**: "T-AutoML: Automated Machine Learning for Lesion Segmentation Using Transformers in 3D Medical Imaging", ICCV, 2021 (*NVIDIA*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_T-AutoML_Automated_Machine_Learning_for_Lesion_Segmentation_Using_Transformers_in_ICCV_2021_paper.html)]
    * **MedT**: "Medical Transformer: Gated Axial-Attention for Medical Image Segmentation", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2102.10662)][[PyTorch](https://github.com/jeya-maria-jose/Medical-Transformer)]
    * **Convolution-Free**: "Convolution-Free Medical Image Segmentation using Transformers", arXiv, 2021 (*Harvard*). [[Paper](https://arxiv.org/abs/2102.13645)]
    * **CoTR**: "CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation", arXiv, 2021 (*Northwestern Polytechnical University*). [[Paper](https://arxiv.org/abs/2103.03024)][[PyTorch](https://github.com/YtongXie/CoTr)]
    * **TransBTS**: "TransBTS: Multimodal Brain Tumor Segmentation Using Transformer", arXiv, 2021 (*University of Science and Technology Beijing*). [[Paper](https://arxiv.org/abs/2103.04430)][[PyTorch](https://github.com/Wenxuan-1119/TransBTS)]
    * **SpecTr**: "SpecTr: Spectral Transformer for Hyperspectral Pathology Image Segmentation", arXiv, 2021 (*East China Normal University*). [[Paper](https://arxiv.org/abs/2103.03604)][[Code (in construction)](https://github.com/hfut-xc-yun/SpecTr)]
    * **U-Transformer**: "U-Net Transformer: Self and Cross Attention for Medical Image Segmentation", arXiv, 2021 (*CEDRIC*). [[Paper](https://arxiv.org/abs/2103.06104)]
    * **TransUNet**: "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2102.04306)][[PyTorch](https://github.com/Beckschen/TransUNet)]
    * **PMTrans**: "Pyramid Medical Transformer for Medical Image Segmentation", arXiv, 2021 (*Washington University in St. Louis*). [[Paper](https://arxiv.org/abs/2104.14702)]
    * **PBT-Net**: "Anatomy-Guided Parallel Bottleneck Transformer Network for Automated Evaluation of Root Canal Therapy", arXiv, 2021 (*Hangzhou Dianzi University*). [[Paper](https://arxiv.org/abs/2105.00381)]
    * **Swin-Unet**: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2105.05537)][[Code (in construction)](https://github.com/HuCaoFighting/Swin-Unet)]
    * **MBT-Net**: "A Multi-Branch Hybrid Transformer Networkfor Corneal Endothelial Cell Segmentation", arXiv, 2021 (*Southern University of Science and Technology*). [[Paper](https://arxiv.org/abs/2106.07557)]
    * **WAD**: "More than Encoder: Introducing Transformer Decoder to Upsample", arXiv, 2021 (*South China University of Technology*). [[Paper](https://arxiv.org/abs/2106.10637)]
    * **LeViT-UNet**: "LeViT-UNet: Make Faster Encoders with Transformer for Medical Image Segmentation", arXiv, 2021 (*Wuhan Institute of Technology*). [[Paper](https://arxiv.org/abs/2107.08623)]
    * **Polyp-PVT**: "Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers", arXiv, 2021 (*IIAI*). [[Paper](https://arxiv.org/abs/2108.06932)][[PyTorch](https://github.com/DengPingFan/Polyp-PVT)]
    * **?**: "Evaluating Transformer based Semantic Segmentation Networks for Pathological Image Segmentation", arXiv, 2021 (*Vanderbilt University*). [[Paper](https://arxiv.org/abs/2108.11993)]
    * **nnFormer**: "nnFormer: Interleaved Transformer for Volumetric Segmentation", arXiv, 2021 (*HKU + Xiamen University*). [[Paper](https://arxiv.org/abs/2109.03201)][[PyTorch](https://github.com/282857341/nnFormer)]
    * **MISSFormer**: "MISSFormer: An Effective Medical Image Segmentation Transformer", arXiv, 2021 (*Beijing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2109.07162)]
    * **TUnet**: "Transformer-Unet: Raw Image Processing with Unet", arXiv, 2021 (*Beijing Zoezen Robot + Beihang University*). [[Paper](https://arxiv.org/abs/2109.08417)]
    * **BiTr-Unet**: "BiTr-Unet: a CNN-Transformer Combined Network for MRI Brain Tumor Segmentation", arXiv, 2021 (*New York University*). [[Paper](https://arxiv.org/abs/2109.12271)]
    * **?**: "Transformer Assisted Convolutional Network for Cell Instance Segmentation", arXiv, 2021 (*IIT Dhanbad*). [[Paper](https://arxiv.org/abs/2110.02270)]
    * **?**: "Combining CNNs With Transformer for Multimodal 3D MRI Brain Tumor Segmentation With Self-Supervised Pretraining", arXiv, 2021 (*Ukrainian Catholic University*). [[Paper](https://arxiv.org/abs/2110.07919)]
    * **UNETR**: "UNETR: Transformers for 3D Medical Image Segmentation", WACV, 2022 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2103.10504)][[PyTorch](https://github.com/Project-MONAI/research-contributions/tree/master/UNETR/BTCV)]
    * **AFTer-UNet**: "AFTer-UNet: Axial Fusion Transformer UNet for Medical Image Segmentation", WACV, 2022 (*UC Irvine*). [[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Yan_AFTer-UNet_Axial_Fusion_Transformer_UNet_for_Medical_Image_Segmentation_WACV_2022_paper.html)]
    * **UCTransNet**: "UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer", AAAI, 2022 (*Northeastern University, China*). [[Paper](https://arxiv.org/abs/2109.04335)][[PyTorch](https://github.com/McGregorWwww/UCTransNet)]
    * **Tempera**: "Tempera: Spatial Transformer Feature Pyramid Network for Cardiac MRI Segmentation", arXiv, 2022 (*ICL*). [[Paper](https://arxiv.org/abs/2203.00355)]
    * **UTNetV2**: "A Multi-scale Transformer for Medical Image Segmentation: Architectures, Model Efficiency, and Benchmarks", arXiv, 2022 (*Rutgers*). [[Paper](https://arxiv.org/abs/2203.00131)]
* Classification:
    * **COVID19T**: "A Transformer-Based Framework for Automatic COVID19 Diagnosis in Chest CTs", ICCVW, 2021 (*?*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/MIA-COV19D/html/Zhang_A_Transformer-Based_Framework_for_Automatic_COVID19_Diagnosis_in_Chest_CTs_ICCVW_2021_paper.html)][[PyTorch](https://github.com/leizhangtech/COVID19T)]
    * **TransMIL**: "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classication", NeurIPS, 2021 (*Tsinghua University*). [[Paper](https://arxiv.org/abs/2106.00908)][[PyTorch](https://github.com/szc19990412/TransMIL)]
    * **TransMed**: "TransMed: Transformers Advance Multi-modal Medical Image Classification", arXiv, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2103.05940)]
    * **CXR-ViT**: "Vision Transformer using Low-level Chest X-ray Feature Corpus for COVID-19 Diagnosis and Severity Quantification", arXiv, 2021 (*KAIST*). [[Paper](https://arxiv.org/abs/2104.07235)]
    * **ViT-TSA**: "Shoulder Implant X-Ray Manufacturer Classification: Exploring with Vision Transformer", arXiv, 2021 (*Queen’s University*). [[Paper](https://arxiv.org/abs/2104.07667)]
    * **GasHis-Transformer**: "GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification", arXiv, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2104.14528)]
    * **POCFormer**: "POCFormer: A Lightweight Transformer Architecture for Detection of COVID-19 Using Point of Care Ultrasound", arXiv, 2021 (*The Ohio State University*). [[Paper](https://arxiv.org/abs/2105.09913)]
    * **COVID-ViT**: "COVID-VIT: Classification of COVID-19 from CT chest images based on vision transformer models", arXiv, 2021 (*Middlesex University, UK*). [[Paper](https://arxiv.org/abs/2107.01682)][[PyTorch](https://github.com/xiaohong1/COVID-ViT)]
    * **EEG-ConvTransformer**: "EEG-ConvTransformer for Single-Trial EEG based Visual Stimuli Classification", arXiv, 2021 (*IIT Ropar*). [[Paper](https://arxiv.org/abs/2107.03983)]
    * **CCAT**: "Visual Transformer with Statistical Test for COVID-19 Classification", arXiv, 2021 (*NCKU*). [[Paper](https://arxiv.org/abs/2107.05334)]
    * **ScoreNet**: "ScoreNet: Learning Non-Uniform Attention and Augmentation for Transformer-Based Histopathological Image Classification", arXiv, 2022 (*EPFL*). [[Paper](https://arxiv.org/abs/2202.07570)]
    * **RadioTransformer**: "RadioTransformer: A Cascaded Global-Focal Transformer for Visual Attention-guided Disease Classification", arXiv, 2022 (*Stony Brook*). [[Paper](https://arxiv.org/abs/2202.11781)]
* Detection:
    * **COTR**: "COTR: Convolution in Transformer Network for End to End Polyp Detection", arXiv, 2021 (*Fuzhou University*). [[Paper](https://arxiv.org/abs/2105.10925)]
    * **TR-Net**: "Transformer Network for Significant Stenosis Detection in CCTA of Coronary Arteries", arXiv, 2021 (*Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2107.03035)]
    * **CAE-Transformer**: "CAE-Transformer: Transformer-based Model to Predict Invasiveness of Lung Adenocarcinoma Subsolid Nodules from Non-thin Section 3D CT Scans", arXiv, 2021 (*Concordia University, Canada*). [[Paper](https://arxiv.org/abs/2110.08721)]
* Reconstruction:
    * **T<sup>2</sup>Net**: "Task Transformer Network for Joint MRI Reconstruction and Super-Resolution", MICCAI, 2021 (*Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2106.06742)][[PyTorch](https://github.com/chunmeifeng/T2Net)]
    * **FIT**: "Fourier Image Transformer", arXiv, 2021 (*MPI*). [[Paper](https://arxiv.org/abs/2104.02555)][[PyTorch](https://github.com/juglab/FourierImageTransformer)]
    * **SLATER**: "Unsupervised MRI Reconstruction via Zero-Shot Learned Adversarial Transformers", arXiv, 2021 (*Bilkent University*). [[Paper](https://arxiv.org/abs/2105.08059)]
    * **MTrans**: "MTrans: Multi-Modal Transformer for Accelerated MR Imaging", arXiv, 2021 (*Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2106.14248)][[PyTorch](https://github.com/chunmeifeng/MTrans)]
* Temporal Localization:
    * **CE-TFE**: "Deep Transformers for Fast Small Intestine Grounding in Capsule Endoscope Video", arXiv, 2021 (*Sun Yat-Sen University*). [[Paper](https://arxiv.org/abs/2104.02866)]
* Registration:
    * **ViT-V-Net**: "ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration", arXiv, 2021 (*JHU*). [[Paper](https://arxiv.org/abs/2104.06468)][[PyTorch](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)]
* Others:
    * **LAT**: "Lesion-Aware Transformers for Diabetic Retinopathy Grading", CVPR, 2021 (*USTC*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html)]
    * **UVT**: "Ultrasound Video Transformers for Cardiac Ejection Fraction Estimation", MICCAI, 2021 (*ICL*). [[Paper](https://arxiv.org/abs/2107.00977)][[PyTorch](https://github.com/HReynaud/UVT)]
    * **?**: "Surgical Instruction Generation with Transformers", MICCAI, 2021 (*Bournemouth University, UK*). [[Paper](https://arxiv.org/abs/2107.06964)]
    * **Eformer**: "Eformer: Edge Enhancement based Transformer for Medical Image Denoising", ICCV, 2021 (*BITS Pilani, India*). [[Paper](https://arxiv.org/abs/2109.08044)]
    * **MCAT**: "Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images", ICCV, 2021 (*Harvard*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html)][[PyTorch](https://github.com/mahmoodlab/MCAT)]
    * **?**: "Is it Time to Replace CNNs with Transformers for Medical Images?", ICCVW, 2021 (*KTH, Sweden*). [[Paper](https://arxiv.org/abs/2108.09038)]
    * **Global-Local-Transformer**: "Global-Local Transformer for Brain Age Estimation", IEEE Transactions on Medical Imaging, 2021 (*Harvard*). [[Paper](https://arxiv.org/abs/2109.01663)][[PyTorch](https://github.com/shengfly/global-local-transformer)]
    * **?**: "Federated Split Vision Transformer for COVID-19 CXR Diagnosis using Task-Agnostic Training", NeurIPS, 2021 (*KAIST*). [[Paper](https://arxiv.org/abs/2111.01338)]
    * **ViT-Path**: "Self-Supervised Vision Transformers Learn Visual Concepts in Histopathology", NeurIPSW, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2203.00585)]
    * **DeepProg**: "DeepProg: A Transformer-based Framework for Predicting Disease Prognosis", arXiv, 2021 (*University of Oulu*). [[Paper](https://arxiv.org/abs/2104.03642)]
    * **Medical-Transformer**: "Medical Transformer: Universal Brain Encoder for 3D MRI Analysis", arXiv, 2021 (*Korea University*). [[Paper](https://arxiv.org/abs/2104.13633)]
    * **PTNet**: "PTNet: A High-Resolution Infant MRI Synthesizer Based on Transformer", arXiv, 2021 (* Columbia *). [[Paper](https://arxiv.org/ftp/arxiv/papers/2105/2105.13993.pdf)]
    * **ResViT**: "ResViT: Residual vision transformers for multi-modal medical image synthesis", arXiv, 2021 (*Bilkent University, Turkey*). [[Paper](https://arxiv.org/abs/2106.16031)]
    * **RATCHET**: "RATCHET: Medical Transformer for Chest X-ray Diagnosis and Reporting", arXiv, 2021 (*ICL*). [[Paper](https://arxiv.org/abs/2107.02104)]
    * **CyTran**: "CyTran: Cycle-Consistent Transformers for Non-Contrast to Contrast CT Translation", arXiv, 2021 (*University Politehnica of Bucharest, Romania*). [[Paper](https://arxiv.org/abs/2110.06400)][[PyTorch](https://github.com/ristea/cycle-transformer)]
    * **RFormer**: "RFormer: Transformer-based Generative Adversarial Network for Real Fundus Image Restoration on A New Clinical Benchmark", arXiv, 2022 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2201.00466)]
    * **CTformer**: "CTformer: Convolution-free Token2Token Dilated Vision Transformer for Low-dose CT Denoising", arXiv, 2022 (*UMass Lowell*). [[Paper](https://arxiv.org/abs/2202.13517)][[PyTorch](https://github.com/wdayang/CTformer)]


## Other Tasks
* Feature Matching:
    * **LoFTR**: "LoFTR: Detector-Free Local Feature Matching with Transformers", CVPR, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2104.00680)][[PyTorch](https://github.com/zju3dv/LoFTR)][[Website](https://zju3dv.github.io/loftr/)]
    * **COTR**: "COTR: Correspondence Transformer for Matching Across Images", ICCV, 2021 (*UBC*). [[Paper](https://arxiv.org/abs/2103.14167)]
    * **CATs**: "CATs: Cost Aggregation Transformers for Visual Correspondence", NeurIPS, 2021 (*Yonsei University + Korea University*). [[Paper](https://arxiv.org/abs/2106.02520)][[PyTorch](https://github.com/SunghwanHong/Cost-Aggregation-transformers)][[Website](https://sunghwanhong.github.io/CATs/)]
    * **CATs++**: "CATs++: Boosting Cost Aggregation with Convolutions and Transformers", arXiv, 2022 (*Korea University*). [[Paper](https://arxiv.org/abs/2202.06817)]
    * **LoFTR-TensorRT**: "Local Feature Matching with Transformers for low-end devices", arXiv, 2022 (*?*). [[Paper](https://arxiv.org/abs/2202.00770)][[PyTorch](https://github.com/Kolkir/Coarse_LoFTR_TRT)]
* Multi-label:
    * **C-Tran**: "General Multi-label Image Classification with Transformers", CVPR, 2021 (*University of Virginia*). [[Paper](https://arxiv.org/abs/2011.14027)]
    * **TDRG**: "Transformer-Based Dual Relation Graph for Multi-Label Image Recognition", ICCV, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2110.04722)]
    * **MlTr**: "MlTr: Multi-label Classification with Transformer", arXiv, 2021 (*KuaiShou*). [[Paper](https://arxiv.org/abs/2106.06195)]
* Fine-grained:
    * **ViT-FGVC**: "Exploring Vision Transformers for Fine-grained Classification", CVPRW, 2021 (*Universidad de Valladolid*). [[Paper](https://arxiv.org/abs/2106.10587)]
    * **FFVT**: "Feature Fusion Vision Transformer Fine-Grained Visual Categorization", arXiv, 2021 (*Griffith University, Australia*). [[Paper](https://arxiv.org/abs/2107.02341)]
    * **TPSKG**: "Transformer with Peak Suppression and Knowledge Guidance for Fine-grained Image Recognition", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2107.06538)]
    * **AFTrans**: "A free lunch from ViT: Adaptive Attention Multi-scale Fusion Transformer for Fine-grained Visual Recognition", arXiv, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2110.01240)]
    * **TransFG**: "TransFG: A Transformer Architecture for Fine-grained Recognition", AAAI, 2022 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2103.07976)][[PyTorch](https://github.com/TACJu/TransFG)]
* Image Quality Assessment:
    * **TRIQ**: "Transformer for Image Quality Assessment", arXiv, 2020 (*NORCE*). [[Paper](https://arxiv.org/abs/2101.01097)][[Tensorflow-Keras](https://github.com/junyongyou/triq)]
    * **IQT**: "Perceptual Image Quality Assessment with Transformers", CVPRW, 2021 (*LG*). [[Paper](https://arxiv.org/abs/2104.14730)][[Code (in construction)](https://github.com/manricheon/IQT)]
    * **MUSIQ**: "MUSIQ: Multi-scale Image Quality Transformer", ICCV, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2108.05997)]
    * **TranSLA**: "Saliency-Guided Transformer Network Combined With Local Embedding for No-Reference Image Quality Assessment", ICCVW, 2021 (*Hikvision*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Zhu_Saliency-Guided_Transformer_Network_Combined_With_Local_Embedding_for_No-Reference_Image_ICCVW_2021_paper.html)]
    * **TReS**: "No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency", WACV, 2022 (*CMU*). [[Paper](https://arxiv.org/abs/2108.06858)]
* Image Retrieval:
    * **RRT**: "Instance-level Image Retrieval using Reranking Transformers", ICCV, 2021 (*University of Virginia*). [[Paper](https://arxiv.org/abs/2103.12236)][[PyTorch](https://github.com/uvavision/RerankingTransformer)]
    * **ViT-Retrieval**: "Investigating the Vision Transformer Model for Image Retrieval Tasks", arXiv, 2021 (*Democritus University of Thrace*). [[Paper](https://arxiv.org/abs/2101.03771)]
    * **IRT**: "Training Vision Transformers for Image Retrieval", arXiv, 2021 (*Facebook + INRIA*). [[Paper](https://arxiv.org/abs/2102.05644)]
    * **TransHash**: "TransHash: Transformer-based Hamming Hashing for Efficient Image Retrieval", arXiv, 2021 (*Shanghai Jiao Tong University*). [[Paper](https://arxiv.org/abs/2105.01823)]
    * **VTS**: "Vision Transformer Hashing for Image Retrieval", arXiv, 2021 (*IIIT-Allahabad*). [[Paper](https://arxiv.org/abs/2109.12564)]
    * **GTZSR**: "Zero-Shot Sketch Based Image Retrieval using Graph Transformer", arXiv, 2022 (*IIT Bombay*). [[Paper](https://arxiv.org/abs/2201.10185)]
* 3D Reconstruction:
    * **PlaneTR**: "PlaneTR: Structure-Guided Transformers for 3D Plane Recovery", ICCV, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2107.13108)][[PyTorch](https://github.com/IceTTTb/PlaneTR3D)]
    * **CO3D**: "CommonObjects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2109.00512)][[PyTorch](https://github.com/facebookresearch/co3d)]
    * **VolT**: "Multi-view 3D Reconstruction with Transformer", ICCV, 2021 (*University of British Columbia*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Multi-View_3D_Reconstruction_With_Transformers_ICCV_2021_paper.html)]
    * **3D-RETR**: "3D-RETR: End-to-End Single and Multi-View 3D Reconstruction with Transformers", BMVC, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2110.08861)]
    * **TransformerFusion**: "TransformerFusion: Monocular RGB Scene Reconstruction using Transformers", NeurIPS, 2021 (*TUM*). [[Paper](https://arxiv.org/abs/2107.02191)][[Website](https://aljazbozic.github.io/transformerfusion/)]
    * **LegoFormer**: "LegoFormer: Transformers for Block-by-Block Multi-view 3D Reconstruction", arXiv, 2021 (*TUM + Google*). [[Paper](https://arxiv.org/abs/2106.12102)]
* Trajectory Prediction:
    * **mmTransformer**: "Multimodal Motion Prediction with Stacked Transformers", CVPR, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2103.11624)][[Code (in construction)](https://github.com/decisionforce/mmTransformer)][[Website](https://decisionforce.github.io/mmTransformer/)]
    * **AgentFormer**: "AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting", ICCV, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2103.14023)][[Code (in construction)](https://github.com/Khrylx/AgentFormer)][[Website](https://www.ye-yuan.com/agentformer/)]
    * **MRT**: "Multi-Person 3D Motion Prediction with Multi-Range Transformers", NeurIPS, 2021 (*UCSD + Berkeley*). [[Paper](https://arxiv.org/abs/2111.12073)][[PyTorch](https://github.com/jiashunwang/MRT)][[Website](https://jiashunwang.github.io/MRT/)]
    * **?**: "Latent Variable Sequential Set Transformers for Joint Multi-Agent Motion Prediction", ICLR, 2022 (*MILA*). [[Paper](https://openreview.net/forum?id=Dup_dDqkZC5)]
    * **Scene-Transformer**: "Scene Transformer: A unified architecture for predicting multiple agent trajectories", ICLR, 2022 (*Google*). [[Paper](https://arxiv.org/abs/2106.08417)]
    * **LatentFormer**: "LatentFormer: Multi-Agent Transformer-Based Interaction Modeling and Trajectory Prediction", arXiv, 2022 (*Huawei*). [[Paper](https://arxiv.org/abs/2203.01880)]
* 3D Motion Synthesis:
    * **ACTOR**: "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE", ICCV, 2021 (*Univ Gustave Eiffel*). [[Paper](https://arxiv.org/abs/2104.05670)][[PyTorch](https://github.com/Mathux/ACTOR)][[Website](https://imagine.enpc.fr/~petrovim/actor/)]
* Fashion:
    * **Kaleido-BERT**: "Kaleido-BERT: Vision-Language Pre-training on Fashion Domain", CVPR, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2103.16110)][[Tensorflow](https://github.com/mczhuge/Kaleido-BERT)]
    * **CIT**: "Cloth Interactive Transformer for Virtual Try-On", arXiv, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2104.05519)][[Code (in construction)](https://github.com/Amazingren/CIT)]
* Crowd Counting:
    * **CC-AV**: "Audio-Visual Transformer Based Crowd Counting", ICCVW, 2021 (*University of Kansas*). [[Paper](https://arxiv.org/abs/2109.01926)]
    * **TransCrowd**: "TransCrowd: Weakly-Supervised Crowd Counting with Transformer", arXiv, 2021 (*Huazhong University of Science and Technology*). [[Paper](https://arxiv.org/abs/2104.09116)][[PyTorch](https://github.com/dk-liang/TransCrowd)]
    * **TAM-RTM**: "Boosting Crowd Counting with Transformers", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2105.10926)]
    * **CCTrans**: "CCTrans: Simplifying and Improving Crowd Counting with Transformer", arXiv, 2021 (*Meituan*). [[Paper](https://arxiv.org/abs/2109.14483)]
    * **SAANet**: "Scene-Adaptive Attention Network for Crowd Counting", arXiv, 2022 (*Xi'an Jiaotong*). [[Paper](https://arxiv.org/abs/2112.15509)]
* Remote Sensing:
    * **DCFAM**: "Transformer Meets DCFAM: A Novel Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images", arXiv, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2104.12137)]
    * **WiCNet**: "Looking Outside the Window: Wider-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images", arXiv, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2106.15754)]
    * **?**: "Vision Transformers For Weeds and Crops Classification Of High Resolution UAV Images", arXiv, 2021 (*University of Orleans, France*). [[Paper](https://arxiv.org/abs/2109.02716)]
    * **RNGDet**: "RNGDet: Road Network Graph Detection by Transformer in Aerial Images", arXiv, 2022 (*HKUST*). [[Paper](https://arxiv.org/abs/2202.07824)]
    * **FSRA**: "A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization", arXiv, 2022 (*China Jiliang University*). [[Paper](https://arxiv.org/abs/2201.09206)][[PyTorch](https://github.com/Dmmm1997/FSRA)]
* Place Recognition:
    * **SVT-Net**: "SVT-Net: A Super Light-Weight Network for Large Scale Place Recognition using Sparse Voxel Transformers", AAAI, 2022 (*Renmin University of China*). [[Paper](https://arxiv.org/abs/2105.00149)]
    * **TransVPR**: "TransVPR: Transformer-based place recognition with multi-level attention aggregation", arXiv, 2022 (*Xi'an Jiaotong*). [[Paper](https://arxiv.org/abs/2201.02001)]
* Traffic:
    * **ViTAL**: "Novelty Detection and Analysis of Traffic Scenario Infrastructures in the Latent Space of a Vision Transformer-Based Triplet Autoencoder", IV, 2021 (*Technische Hochschule Ingolstadt*). [[Paper](https://arxiv.org/abs/2105.01924)]
    * **?**: "Predicting Vehicles Trajectories in Urban Scenarios with Transformer Networks and Augmented Information", IVS, 2021 (*Universidad de Alcala*). [[Paper](https://arxiv.org/abs/2106.00559)]
* Character Recognition:
    * **BTTR**: "Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer", arXiv, 2021 (*Peking*). [[Paper](https://arxiv.org/abs/2105.02412)]
    * **TrOCR**: "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2109.10282)]
    * **?**: "Robustness Evaluation of Transformer-based Form Field Extractors via Form Attacks", arXiv, 2021 (*Salesforce*). [[Paper](https://arxiv.org/abs/2110.04413)]
* Image Registration:
    * **AiR**: "Attention for Image Registration (AiR): an unsupervised Transformer approach", arXiv, 2021 (*INRIA*). [[Paper](https://arxiv.org/abs/2105.02282)]
* Biology:
    * **?**: "A State-of-the-art Survey of Object Detection Techniques in Microorganism Image Analysis: from Traditional Image Processing and Classical Machine Learning to Current Deep Convolutional Neural Networks and Potential Visual Transformers", arXiv, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2105.03148)]
* Satellite:
    * **Satellite-ViT**: "Manipulation Detection in Satellite Images Using Vision Transformer", arXiv, 2021 (*Purdue*). [[Paper](https://arxiv.org/abs/2105.06373)]
* Pedestrian Intention:
    * **IntFormer**: "IntFormer: Predicting pedestrian intention with the aid of the Transformer architecture", arXiv, 2021 (*Universidad de Alcala*). [[Paper](https://arxiv.org/abs/2105.08647)]
* Scene Text Recognition:
    * **ViTSTR**: "Vision Transformer for Fast and Efficient Scene Text Recognition", ICDAR, 2021 (*University of the Philippines*). [[Paper](https://arxiv.org/abs/2105.08582)]
    * **STKM**: "Self-attention based Text Knowledge Mining for Text Detection", CVPR, 2021 (*?*). [[Paper]()][[Code (in construction)](https://github.com/CVI-SZU/STKM)]
    * **I2C2W**: "I2C2W: Image-to-Character-to-Word Transformers for Accurate Scene Text Recognition", arXiv, 2021 (*NTU Singapoer*). [[Paper](https://arxiv.org/abs/2105.08383)]
* Gaze:
    * **GazeTR**: "Gaze Estimation using Transformer", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2105.14424)][[PyTorch](https://github.com/yihuacheng/GazeTR)]
* Active Learning:
    * **TJLS**: "Visual Transformer for Task-aware Active Learning", arXiv, 2021 (*ICL*). [[Paper](https://arxiv.org/abs/2106.03801)][[PyTorch](https://github.com/razvancaramalau/Visual-Transformer-for-Task-aware-Active-Learning)]
* Homography Estimation:
    * **LocalTrans**: "LocalTrans: A Multiscale Local Transformer Network for Cross-Resolution Homography Estimation", ICCV, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2106.04067)]
* Time Series:
    * **MissFormer**: "MissFormer: (In-)attention-based handling of missing observations for trajectory filtering and prediction", arXiv, 2021 (*Fraunhofer IOSB, Germany*). [[Paper](https://arxiv.org/abs/2106.16009)]
* Geo-Localization:
    * **EgoTR**: "Cross-view Geo-localization with Evolving Transformer", arXiv, 2021 (*Shenzhen University*). [[Paper](https://arxiv.org/abs/2107.00842)]
* Out-Of-Distribution:
    * **OODformer**: "OODformer: Out-Of-Distribution Detection Transformer", arXiv, 2021 (*LMU Munich*). [[Paper](https://arxiv.org/abs/2107.08976)]
* Layout Generation:
    * **VTN**: "Variational Transformer Networks for Layout Generation", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2104.02416)]
    * **LayoutTransformer**: "LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity", CVPR, 2021 (*NTU*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html)][[PyTorch](https://github.com/davidhalladay/LayoutTransformer)]
    * **LayoutTransformer**: "LayoutTransformer: Layout Generation and Completion with Self-attention", ICCV, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2006.14615)][[Website](https://kampta.github.io/layout/)]
    * **LGT-Net**: "LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network", CVPR, 2022 (*East China Normal University*). [[Paper](https://arxiv.org/abs/2203.01824)][[PyTorch](https://github.com/zhigangjiang/LGT-Net)]
    * **ATEK**: "ATEK: Augmenting Transformers with Expert Knowledge for Indoor Layout Synthesis", arXiv, 2022 (*New Jersey Institute of Technology*). [[Paper](https://arxiv.org/abs/2202.00185)]
* Zero-Shot:
    * **ViT-ZSL**: "Multi-Head Self-Attention via Vision Transformer for Zero-Shot Learning", IMVIP, 2021 (*University of Exeter, UK*). [[Paper](https://arxiv.org/abs/2108.00045)]
    * **TransZero**: "TransZero: Attribute-guided Transformer for Zero-Shot Learning", AAAI, 2022 (*Huazhong University of Science and Technology*). [[Paper](https://arxiv.org/abs/2112.01683)][[PyTorch](https://github.com/shiming-chen/TransZero)]
* Domain Adaptation/Generalization:
    * **TransDA**: "Transformer-Based Source-Free Domain Adaptation", arXiv, 2021 (*Haerbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2105.14138)][[PyTorch](https://github.com/ygjwd12345/TransDA)]
    * **TVT**: "TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation", arXiv, 2021 (*UT Arlington + Kuaishou*). [[Paper](https://arxiv.org/abs/2108.05988)]
    * **ResTran**: "Discovering Spatial Relationships by Transformers for Domain Generalization", arXiv, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2108.10046)]
    * **WinTR**: "Exploiting Both Domain-specific and Invariant Knowledge via a Win-win Transformer for Unsupervised Domain Adaptation", arXiv, 2021 (*Beijing Institute of Technology*). [[Paper](https://arxiv.org/abs/2111.12941)]
    * **CDTrans**: "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation", ICLR, 2022 (*Alibaba*). [[Paper](https://arxiv.org/abs/2109.06165)][[PyTorch](https://github.com/CDTrans/CDTrans)]
    * **BCAT**: "Domain Adaptation via Bidirectional Cross-Attention Transformer", arXiv, 2022 (*Southern University of Science and Technology*). [[Paper](https://arxiv.org/abs/2201.05887)]
    * **DoTNet**: "Towards Unsupervised Domain Adaptation via Domain-Transformer", arXiv, 2022 (*Sun Yat-Sen University*). [[Paper](https://arxiv.org/abs/2202.13777)]
* Digital Holography:
    * **?**: "Convolutional Neural Network (CNN) vs Visual Transformer (ViT) for Digital Holography", ICCCR, 2022 (*UBFC, France*). [[Paper](https://arxiv.org/abs/2108.09147)]
* Curriculum Learning:
    * **SSTN**: "Spatial Transformer Networks for Curriculum Learning", arXiv, 2021 (*TU Kaiserslautern, Germany*). [[Paper](https://arxiv.org/abs/2108.09696)]
* 3D Human Texture Estimation:
    * **Texformer**: "3D Human Texture Estimation from a Single Image with Transformers", ICCV, 2021 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2109.02563)][[PyTorch](https://github.com/xuxy09/Texformer)][[Website](https://www.mmlab-ntu.com/project/texformer/)]
* Camera-related:
    * **CTRL-C**: "CTRL-C: Camera calibration TRansformer with Line-Classification", ICCV, 2021 (*Kakao + Kookmin University*). [[Paper](https://arxiv.org/abs/2109.02259)][[PyTorch](https://github.com/jwlee-vcl/CTRL-C)]
    * **MS-Transformer**: "Learning Multi-Scene Absolute Pose Regression with Transformers", ICCV, 2021 (*Bar-Ilan University, Israel*). [[Paper](https://arxiv.org/abs/2103.11468)][[PyTorch](https://github.com/yolish/multi-scene-pose-transformer)]
* Animation-related:
    * **AnT**: "The Animation Transformer: Visual Correspondence via Segment Matching", ICCV, 2021 (*Cadmium*). [[Paper](https://arxiv.org/abs/2109.02614)]
* Stereo:
    * **STTR**: "Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers", ICCV, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2011.02910)][[PyTorch](https://github.com/mli0603/stereo-transformer)]
* 360 Scene:
    * **?**: "Improving 360 Monocular Depth Estimation via Non-local Dense Prediction Transformer and Joint Supervised and Self-supervised Learning", AAAI, 2022 (*Seoul National University*). [[Paper](https://arxiv.org/abs/2109.10563)][[PyTorch](https://github.com/yuniw18/Joint_360depth)]
    * **SPH**: "Spherical Transformer", arXiv, 2022 (*Chung-Ang University, Korea*). [[Paper](https://arxiv.org/abs/2202.04942)]
* Robotics:
    * **TF-Grasp**: "When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection", arXiv, 2022 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2202.11911)][[Code (in construction)](https://github.com/WangShaoSUN/grasp-transformer)]
* Continual Learning:
    * **COLT**: "Transformers Are Better Continual Learners", arXiv, 2022 (*Hikvision*). [[Paper](https://arxiv.org/abs/2201.04924)]
* Long-tail:
    * **BatchFormer**: "BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning", CVPR, 2022 (*The University of Sydney*). [[Paper](https://arxiv.org/abs/2203.01522)]

---

## Attention Mechanisms in Vision/NLP
### Vision
* **AA**: "Attention Augmented Convolutional Networks", ICCV, 2019 (*Google*). [[Paper](https://arxiv.org/abs/1904.09925)][[PyTorch (Unofficial)](https://github.com/leaderj1001/Attention-Augmented-Conv2d)][[Tensorflow (Unofficial)](https://github.com/titu1994/keras-attention-augmented-convs)]
* **LR-Net**: "Local Relation Networks for Image Recognition", ICCV, 2019 (*Microsoft*). [[Paper](https://arxiv.org/abs/1904.11491)][[PyTorch (Unofficial)](https://github.com/gan3sh500/local-relational-nets)]
* **CCNet**: "CCNet: Criss-Cross Attention for Semantic Segmentation", ICCV, 2019 (& TPAMI 2020) (*Horizon*). [[Paper](https://arxiv.org/abs/1811.11721)][[PyTorch](https://github.com/speedinghzl/CCNet)]
* **GCNet**: "Global Context Networks", ICCVW, 2019 (& TPAMI 2020) (*Microsoft*). [[Paper](https://arxiv.org/abs/2012.13375)][[PyTorch](https://github.com/xvjiarui/GCNet)]
* **SASA**: "Stand-Alone Self-Attention in Vision Models", NeurIPS, 2019 (*Google*). [[Paper](https://arxiv.org/abs/1906.05909)][[PyTorch-1 (Unofficial)](https://github.com/leaderj1001/Stand-Alone-Self-Attention)][[PyTorch-2 (Unofficial)](https://github.com/MerHS/SASA-pytorch)]
    * key message: attention module is more efficient than conv & provide comparable accuracy
* **Axial-Transformer**: "Axial Attention in Multidimensional Transformers", arXiv, 2019 (rejected by ICLR 2020) (*Google*). [[Paper](https://openreview.net/forum?id=H1e5GJBtDr)][[PyTorch (Unofficial)](https://github.com/lucidrains/axial-attention)]
* **Attention-CNN**: "On the Relationship between Self-Attention and Convolutional Layers", ICLR, 2020 (*EPFL*). [[Paper](https://openreview.net/forum?id=HJlnC1rKPB)][[PyTorch](https://github.com/epfml/attention-cnn)][[Website](https://epfml.github.io/attention-cnn/)]
* **SAN**: "Exploring Self-attention for Image Recognition", CVPR, 2020 (*CUHK + Intel*). [[Paper](https://arxiv.org/abs/2004.13621)][[PyTorch](https://github.com/hszhao/SAN)]
* **BA-Transform**: "Non-Local Neural Networks With Grouped Bilinear Attentional Transforms", CVPR, 2020 (*ByteDance*). [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html)]
* **Axial-DeepLab**: "Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation", ECCV, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2003.07853)][[PyTorch](https://github.com/csrhddlam/axial-deeplab)]
* **GSA**: "Global Self-Attention Networks for Image Recognition", arXiv, 2020 (rejected by ICLR 2021) (*Google*). [[Paper](https://openreview.net/forum?id=KiFeuZu24k)][[PyTorch (Unofficial)](https://github.com/lucidrains/global-self-attention-network)]
* **EA**: "Efficient Attention: Attention with Linear Complexities", WACV, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/1812.01243)][[PyTorch](https://github.com/cmsflash/efficient-attention)]
* **LambdaNetworks**: "LambdaNetworks: Modeling long-range Interactions without Attention", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=xTJEN-ggl1b)][[PyTorch-1 (Unofficial)](https://github.com/lucidrains/lambda-networks)][[PyTorch-2 (Unofficial)](https://github.com/leaderj1001/LambdaNetworks)]
* **GSA-Nets**: "Group Equivariant Stand-Alone Self-Attention For Vision", ICLR, 2021 (*EPFL*). [[Paper](https://openreview.net/forum?id=JkfYjnOEo6M)]
* **Hamburger**: "Is Attention Better Than Matrix Decomposition?", ICLR, 2021 (*Peking*). [[Paper](https://openreview.net/forum?id=1FvkSpWosOl)][[PyTorch (Unofficial)](https://github.com/lucidrains/hamburger-pytorch)]
* **HaloNet**: "Scaling Local Self-Attention For Parameter Efficient Visual Backbones", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.12731)]
* **BoTNet**: "Bottleneck Transformers for Visual Recognition", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2101.11605)]
* **SSAN**: "SSAN: Separable Self-Attention Network for Video Representation Learning", CVPR, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2105.13033)]
* **CoTNet**: "Contextual Transformer Networks for Visual Recognition", CVPRW, 2021 (*JD*). [[Paper](https://arxiv.org/abs/2107.12292)][[PyTorch](https://github.com/JDAI-CV/CoTNet)]
* **Involution**: "Involution: Inverting the Inherence of Convolution for Visual Recognition", CVPR, 2021 (*HKUST*). [[Paper](https://arxiv.org/abs/2103.06255)][[PyTorch](https://github.com/d-li14/involution)]
* **Perceiver**: "Perceiver: General Perception with Iterative Attention", ICML, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2103.03206)][[PyTorch (lucidrains)](https://github.com/lucidrains/perceiver-pytorch)]
* **SNL**: "Unifying Nonlocal Blocks for Neural Networks", ICCV, 2021 (*Peking + Bytedance*). [[Paper](https://arxiv.org/abs/2108.02451)]
* **External-Attention**: "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.02358)]
* **KVT**: "KVT: k-NN Attention for Boosting Vision Transformers", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.00515)]
* **Container**: "Container: Context Aggregation Network", arXiv, 2021 (*AI2*). [[Paper](https://arxiv.org/abs/2106.01401)]
* **X-volution**: "X-volution: On the unification of convolution and self-attention", arXiv, 2021 (*Huawei Hisilicon*). [[Paper](https://arxiv.org/abs/2106.02253)]
* **Invertible-Attention**: "Invertible Attention", arXiv, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2106.09003)]
* **VOLO**: "VOLO: Vision Outlooker for Visual Recognition", arXiv, 2021 (*Sea AI Lab + NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.13112)][[PyTorch](https://github.com/sail-sg/volo)]
* **LESA**: "Locally Enhanced Self-Attention: Rethinking Self-Attention as Local and Context Terms", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2107.05637)]
* **PS-Attention**: "Pale Transformer: A General Vision Transformer Backbone with Pale-Shaped Attention", AAAI, 2022 (*Baidu*). [[Paper](https://arxiv.org/abs/2112.14000)][[Paddle](https://github.com/BR-IDL/PaddleViT)]
* **QuadTree**: "QuadTree Attention for Vision Transformers", ICLR, 2022 (*Simon Fraser + Alibaba*). [[Paper](https://arxiv.org/abs/2201.02767)][[PyTorch](https://github.com/Tangshitao/QuadtreeAttention)]
* **HiP**: "Hierarchical Perceiver", arXiv, 2022 (*DeepMind*). [[Paper](https://arxiv.org/abs/2202.10890)]

### NLP
* **T-DMCA**: "Generating Wikipedia by Summarizing Long Sequences", ICLR, 2018 (*Google*). [[Paper](https://openreview.net/forum?id=Hyg0vbWC-)]
* **LSRA**: "Lite Transformer with Long-Short Range Attention", ICLR, 2020 (*MIT*). [[Paper](https://openreview.net/forum?id=ByeMPlHKPH)][[PyTorch](https://github.com/mit-han-lab/lite-transformer)]
* **ETC**: "ETC: Encoding Long and Structured Inputs in Transformers", EMNLP, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2004.08483)][[Tensorflow](https://github.com/google-research/google-research/tree/master/etcmodel)]
* **BlockBERT**: "Blockwise Self-Attention for Long Document Understanding", EMNLP Findings, 2020 (*Facebook*). [[Paper](https://arxiv.org/abs/1911.02972)][[GitHub](https://github.com/xptree/BlockBERT)]
* **Clustered-Attention**: "Fast Transformers with Clustered Attention", NeurIPS, 2020 (*Idiap*). [[Paper](https://arxiv.org/abs/2007.04825)][[PyTorch](https://github.com/idiap/fast-transformers)][[Website](https://clustered-transformers.github.io/)]
* **BigBird**: "Big Bird: Transformers for Longer Sequences", NeurIPS, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2007.14062)][[Tensorflow](https://github.com/google-research/bigbird)]
* **Longformer**: "Longformer: The Long-Document Transformer", arXiv, 2020 (*AI2*). [[Paper](https://arxiv.org/abs/2004.05150)][[PyTorch](https://github.com/allenai/longformer)]
* **Linformer**: "Linformer: Self-Attention with Linear Complexity", arXiv, 2020 (*Facebook*). [[Paper](https://arxiv.org/abs/2006.04768)][[PyTorch (Unofficial)](https://github.com/tatp22/linformer-pytorch)]
* **Nystromformer**: "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention", AAAI, 2021 (*UW-Madison*). [[Paper](https://arxiv.org/abs/2102.03902)][[PyTorch](https://github.com/mlpen/Nystromformer)]
* **RFA**: "Random Feature Attention", ICLR, 2021 (*DeepMind*). [[Paper](https://openreview.net/forum?id=QtTKTdVrFBB)]
* **Performer**: "Rethinking Attention with Performers", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=Ua6zuk0WRH)][[Code](https://github.com/google-research/google-research/tree/master/performer)][[Blog](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)]
* **DeLight**: "DeLighT: Deep and Light-weight Transformer", ICLR, 2021 (*UW*). [[Paper](https://openreview.net/forum?id=ujmgfuxSLrO)]
* **Synthesizer**: "Synthesizer: Rethinking Self-Attention for Transformer Models", ICML, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2005.00743)][[Tensorflow](https://github.com/tensorflow/mesh)][[Pytorch (leaderj1001)](https://github.com/leaderj1001/Synthesizer-Rethinking-Self-Attention-Transformer-Models)]
* **Poolingformer**: "Poolingformer: Long Document Modeling with Pooling Attention", ICML, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2105.04371)]
* **Hi-Transformer**: "Hi-Transformer: Hierarchical Interactive Transformer for Efficient and Effective Long Document Modeling", ACL, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2106.01040)]
* **Smart-Bird**: "Smart Bird: Learnable Sparse Attention for Efficient and Effective Transformer", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2108.09193)]
* **Fastformer**: "Fastformer: Additive Attention is All You Need", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2108.09084)]
* **∞-former**: "∞-former: Infinite Memory Transformer", arXiv, 2021 (*Instituto de Telecomunicações, Portugal*). [[Paper](https://arxiv.org/abs/2109.00301)]
* **cosFormer**: "cosFormer: Rethinking Softmax In Attention", ICLR, 2022 (*SenseTime*). [[Paper](https://openreview.net/forum?id=Bl8CQrx2Up4)][[PyTorch (davidsvy)](https://github.com/davidsvy/cosformer-pytorch)]

### Both
* **Sparse-Transformer**: "Generating Long Sequences with Sparse Transformers", arXiv, 2019 (rejected by ICLR 2020) (*OpenAI*). [[Paper](https://arxiv.org/abs/1904.10509)][[Tensorflow](https://github.com/openai/sparse_attention)][[Blog](https://openai.com/blog/sparse-transformer/)]
* **Reformer**: "Reformer: The Efficient Transformer", ICLR, 2020 (*Google*). [[Paper](https://openreview.net/forum?id=rkgNKkHtvB)][[Tensorflow](https://github.com/google/trax/tree/master/trax/models/reformer)][[Blog](https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html)] 
* **Sinkhorn-Transformer**: "Sparse Sinkhorn Attention", ICML, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2002.11296)][[PyTorch (Unofficial)](https://github.com/lucidrains/sinkhorn-transformer)]
* **Linear-Transformer**: "Transformers are rnns: Fast autoregressive transformers with linear attention", ICML, 2020 (*Idiap*). [[Paper](https://arxiv.org/abs/2006.16236)][[PyTorch](https://github.com/idiap/fast-transformers)][[Website](https://linear-transformers.com/)]
* **SMYRF**: "SMYRF: Efficient Attention using Asymmetric Clustering", NeurIPS, 2020 (*UT Austin + Google*). [[Paper](https://arxiv.org/abs/2010.05315)][[PyTorch](https://github.com/giannisdaras/smyrf)]
* **Routing-Transformer**: "Efficient Content-Based Sparse Attention with Routing Transformers", TACL, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2003.05997)][[Tensorflow](https://github.com/google-research/google-research/tree/master/routing_transformer)][[PyTorch (Unofficial)](https://github.com/lucidrains/routing-transformer)][[Slides](https://drive.google.com/file/d/1maX-UQbtnVtxQqLmHvWVN6LNYtnBaTd9/view)]
* **LRA**: "Long Range Arena: A Benchmark for Efficient Transformers", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=qVyeW-grC2k)][[Tensorflow](https://github.com/google-research/long-range-arena)]
* **OmniNet**: "OmniNet: Omnidirectional Representations from Transformers", ICML, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.01075)]
* **Evolving-Attention**: "Evolving Attention with Residual Convolutions", ICML, 2021 (*Peking + Microsoft*). [[Paper](https://arxiv.org/abs/2102.12895)]
* **H-Transformer-1D**: "H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences", ACL, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2107.11906)]
* **Combiner**: "Combiner: Full Attention Transformer with Sparse Computation Cost", NeurIPS, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2107.05768)]
* **Centroid-Transformer**: "Centroid Transformers: Learning to Abstract with Attention", arXiv, 2021 (*UT Austin*). [[Paper](https://arxiv.org/abs/2102.08606)]
* **AFT**: "An Attention Free Transformer", arXiv, 2021 (*Apple*). [[Paper](https://arxiv.org/abs/2105.14103)]
* **Luna**: "Luna: Linear Unified Nested Attention", arXiv, 2021 (*USC + CMU + Facebook*). [[Paper](https://arxiv.org/abs/2106.01540)]
* **Transformer-LS**: "Long-Short Transformer: Efficient Transformers for Language and Vision", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2107.02192)]
* **PoNet**: "PoNet: Pooling Network for Efficient Token Mixing in Long Sequences", ICLR, 2022 (*Alibaba*). [[Paper](https://arxiv.org/abs/2110.02442)]

### Others
* **Informer**: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2012.07436)][[PyTorch](https://github.com/zhouhaoyi/Informer2020)]
* **Attention-Rank-Collapse**: "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth", ICML, 2021 (*Google + EPFL*). [[Paper](https://arxiv.org/abs/2103.03404)][[PyTorch](https://github.com/twistedcubic/attention-rank-collapse)]
* **NPT**: "Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2106.02584)]

---

## References

* References:
    * Survey:
        * "A Comprehensive Study of Vision Transformers on Dense Prediction Tasks", VISAP, 2022 (*NavInfo Europe, Netherlands*). [[Paper](https://arxiv.org/abs/2201.08683)]
        * "Video Transformers: A Survey", arXiv, 2022 (*Universitat de Barcelona, Spain*). [[Paper](https://arxiv.org/abs/2201.05991)] 
        * "Transformers in Medical Image Analysis: A Review", arXiv, 2022 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2202.12165)]
        * "Recent Advances in Vision Transformer: A Survey and Outlook of Recent Work", arXiv, 2022 (*?*). [[Paper](https://arxiv.org/abs/2203.01536)]
        * "Transformers in Vision: A Survey", ACM Computing Surveys, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2101.01169)]
        * "Survey: Transformer based Video-Language Pre-training", arXiv, 2021 (*Renmin University of China*). [[Paper](https://arxiv.org/abs/2109.09920)]
        * "A Survey of Transformers", arXiv, 2021 (*Fudan*). [[Paper](https://arxiv.org/abs/2106.04554)]
        * "A Survey on Visual Transformer", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2012.12556)]
        * "A Survey of Visual Transformers", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2111.06091)]
        * "Efficient Transformers: A Survey", arXiv, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2009.06732)]
        * "Attention mechanisms and deep learning for machine vision: A survey of the state of the art", arXiv, 2021 (*University of Kashmir*). [[Paper](https://arxiv.org/abs/2106.07550)]
    * Online Resources:
        * [Transformer-in-Vision (GitHub)](https://github.com/DirtyHarryLYL/Transformer-in-Vision)   
        * [Awesome Visual-Transformer (GitHub)](https://github.com/dk-liang/Awesome-Visual-Transformer)
        * [Awesome Transformer for Vision Resources List (GitHub)](https://github.com/lijiaman/awesome-transformer-for-vision)
        * [Transformer-in-Computer-Vision (GitHub)](https://github.com/Yangzhangcst/Transformer-in-Computer-Vision)
