# Awesome-Transformer-Attention
A comprehensive paper list of Vision Transformer/Attention

## Content 
(TODO)

### Image Classification / Backbone
##### Replace Conv w/ Attention
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
    * **CoAtNet**: "CoAtNet: Marrying Convolution and Attention for All Data Sizes", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.04803)]
    
##### Vision Transformer
* For general performance:
    * **ViT**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=YicbFdNTTy)][[Tensorflow](https://github.com/google-research/vision_transformer)][[PyTorch (lucidrains)](https://github.com/lucidrains/vit-pytorch)]
    * **Perceiver**: "Perceiver: General Perception with Iterative Attention", ICML, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2103.03206)][[PyTorch (lucidrains)](https://github.com/lucidrains/perceiver-pytorch)]
    * **PiT**: "Rethinking Spatial Dimensions of Vision Transformers", ICCV, 2021 (*NAVER*). [[Paper](https://arxiv.org/abs/2103.16302)][[PyTorch](https://github.com/naver-ai/pit)]
    * **VT**: "Visual Transformers: Token-based Image Representation and Processing for Computer Vision", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2006.03677)][[PyTorch (tahmid0007)](https://github.com/tahmid0007/VisualTransformers)] 
    * **MViT**: "Multiscale Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.11227)][[PyTorch](https://github.com/facebookresearch/SlowFast)]
    * **PVT**: "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions", ICCV, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2102.12122)][[PyTorch](https://github.com/whai362/PVT)] 
    * **iRPE**: "Rethinking and Improving Relative Position Encoding for Vision Transformer", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.14222)][[PyTorch](https://github.com/microsoft/Cream/tree/main/iRPE)]
    * **CaiT**: "Going deeper with Image Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2103.17239)][[PyTorch](https://github.com/facebookresearch/deit)]
    * **Swin-Transformer**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.14030)][[PyTorch (berniwal)](https://github.com/berniwal/swin-transformer-pytorch)][[Code](https://github.com/microsoft/Swin-Transformer)]
    * **T2T-ViT**: "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet", ICCV, 2021 (*Yitu*). [[Paper](https://arxiv.org/abs/2101.11986)][[PyTorch](https://github.com/yitu-opensource/T2T-ViT)]
    * **DPT**: "DPT: Deformable Patch-based Transformer for Visual Recognition", ACMMM, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2107.14467)][[PyTorch](https://github.com/CASIA-IVA-Lab/DPT)]
    * **TNT**: "Transformer in Transformer", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2103.00112)][[PyTorch (lucidrains)](https://github.com/lucidrains/transformer-in-transformer)][[Code](https://github.com/huawei-noah/noah-research/tree/master/TNT)]
    * **DeepViT**: "DeepViT: Towards Deeper Vision Transformer", arXiv, 2021 (*NUS + ByteDance*). [[Paper](https://arxiv.org/abs/2103.11886)][[Code](https://github.com/zhoudaquan/dvit_repo)]
    * **So-ViT**: "So-ViT: Mind Visual Tokens for Vision Transformer", arXiv, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2104.10935)][[PyTorch](https://github.com/jiangtaoxie/So-ViT)]
    * **LV-ViT**: "Token Labeling: Training a 85.5% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet", arXiv, 2021 (*ByteDance*). [[Paper](https://arxiv.org/abs/2104.10858)][[Code (in construction)](https://github.com/zihangJiang/TokenLabeling)]
    * **Twins**: "Twins: Revisiting Spatial Attention Design in Vision Transformers", arXiv, 2021 (*Meituan*). [[Paper](https://arxiv.org/abs/2104.13840)][[Code (in construction)](https://github.com/Meituan-AutoML/Twins)]
    * **NesT**: "Aggregating Nested Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.12723)]
    * **LIT**: "Less is More: Pay Less Attention in Vision Transformers", arXiv, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2105.14217)]
    * **MSG-Transformer**: "MSG-Transformer: Exchanging Local Spatial Information by Manipulating Messenger Tokens", arXiv, 2021 (*Huazhong University of Science & Technology*). [[Paper](https://arxiv.org/abs/2105.15168)][[Code (in construction)](https://github.com/hustvl/MSG-Transformer)]
    * **DVT**: "Not All Images are Worth 16x16 Words: Dynamic Vision Transformers with Adaptive Sequence Length", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.15075)]
    * **KVT**: "KVT: k-NN Attention for Boosting Vision Transformers", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.00515)]
    * **ViTAE**: "ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias", arXiv, 2021 (*The University of Sydney*). [[Paper](https://arxiv.org/abs/2106.03348)][[Code (in construction)](https://github.com/Annbless/ViTAE)]
    * **RegionViT**: "RegionViT: Regional-to-Local Attention for Vision Transformers", arXiv, 2021 (*MIT-IBM Watson*). [[Paper](https://arxiv.org/abs/2106.02689)]
    * **Refined-ViT**: "Refiner: Refining Self-attention for Vision Transformers", arXiv, 2021 (*NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.03714)][[PyTorch](https://github.com/zhoudaquan/Refiner_ViT)]
    * **Shuffle-Transformer**: "Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer", arXiv, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2106.03650)]
    * **ViT-G**: "Scaling Vision Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.04560)]
    * **CAT**: "CAT: Cross Attention in Vision Transformer", arXiv, 2021 (*KuaiShou*). [[Paper](https://arxiv.org/abs/2106.05786)][[PyTorch](https://github.com/linhezheng19/CAT)]
    * **V-MoE**: "Scaling Vision with Sparse Mixture of Experts", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.05974)]
    * **XCiT**: "XCiT: Cross-Covariance Image Transformers", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.09681)]
    * **P2T**: "P2T: Pyramid Pooling Transformer for Scene Understanding", arXiv, 2021 (*Nankai University*). [[Paper](https://arxiv.org/abs/2106.12011)]
    * **PvTv2**: "PVTv2: Improved Baselines with Pyramid Vision Transformer", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2106.13797)][[PyTorch](https://github.com/whai362/PVT)]
    * **Aug-S**: "Augmented Shortcuts for Vision Transformers", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2106.15941)]
    * **CSWin**: "CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.00652)]
    * **Focal**: "Focal Self-attention for Local-Global Interactions in Vision Transformers", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.00641)]
    * **LG-Transformer**: "Local-to-Global Self-Attention in Vision Transformers", arXiv, 2021 (*IIAI, UAE*). [[Paper](https://arxiv.org/abs/2107.04735)]
    * **ViP**: "Visual Parser: Representing Part-whole Hierarchies with Transformers", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2107.05790)]
    * **CrossFormer**: "CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention", arXiv, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2108.00154)][[PyTorch](https://github.com/cheerss/CrossFormer)]
    * **Scaled-ReLU**: "Scaled ReLU Matters for Training Vision Transformers", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2109.03810)]
* For efficiency:
    * **DeiT**: "Training data-efficient image transformers & distillation through attention", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2012.12877)][[PyTorch](https://github.com/facebookresearch/deit)]
    * **ConViT**: "ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2103.10697)][[Code](https://github.com/facebookresearch/convit)]
    * **?**: "Improving the Efficiency of Transformers for Resource-Constrained Devices", DSD, 2021 (*NavInfo Europe, Netherlands*). [[Paper](https://arxiv.org/abs/2106.16006)]
    * **PS-ViT**: "Vision Transformer with Progressive Sampling", ICCV, 2021 (*CPII*). [[Paper](https://arxiv.org/abs/2108.01684)]
    * **HVT**: "Scalable Visual Transformers with Hierarchical Pooling", arXiv, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2103.10619)]
    * **CrossViT**: "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification", arXiv, 2021 (*MIT-IBM*). [[Paper](https://arxiv.org/abs/2103.14899)]
    * **ViL**: "Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.15358)]
    * **LocalViT**: "LocalViT: Bringing Locality to Vision Transformers", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2104.05707)][[Code (in construction)](https://github.com/ofsoundof/LocalViT)]
    * **CCT**: "Escaping the Big Data Paradigm with Compact Transformers", arXiv, 2021 (*University of Oregon*). [[Paper](https://arxiv.org/abs/2104.05704)][[PyTorch](https://github.com/SHI-Labs/Compact-Transformers)]
    * **Over-Smoothing**: "Improve Vision Transformers Training by Suppressing Over-smoothing", arXiv, 2021 (*UT Austin + Facebook*). [[Paper](https://arxiv.org/abs/2104.12753)][[PyTorch](https://github.com/ChengyueGongR/PatchVisionTransformer)] 
    * **Visformer**: "Visformer: The Vision-friendly Transformer", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2104.12533)][[PyTorch](https://github.com/danczs/Visformer)]
    * **SL-ViT**: "Single-Layer Vision Transformers for More Accurate Early Exits with Less Overhead", arXiv, 2021 (*Aarhus University*). [[Paper](https://arxiv.org/abs/2105.09121)]
    * **ResT**: "ResT: An Efficient Transformer for Visual Recognition", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2105.13677)][[PyTorch](https://github.com/wofmanaf/ResT)]
    * **DynamicViT**: "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2106.02034)][[PyTorch](https://github.com/raoyongming/DynamicViT)][[Website](https://dynamicvit.ivg-research.xyz/)]
    * **GG-Transformer**: "Glance-and-Gaze Vision Transformer", arXiv, 2021 (*JHU*). [[Paper](https://arxiv.org/abs/2106.02277)][[Code (in construction)](https://github.com/yucornetto/GG-Transformer)]
    * **PS-ViT**: "Patch Slimming for Efficient Vision Transformers", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2106.02852)]
    * **SViTE**: "Chasing Sparsity in Vision Transformers:An End-to-End Exploration", arXiv, 2021 (*UT Austin*). [[Paper](https://arxiv.org/abs/2106.04533)][[Code (in construction)](https://github.com/VITA-Group/SViTE)]
    * **ViT-quant**: "Post-Training Quantization for Vision Transformer", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2106.14156)]
    * **?**: "Multi-Exit Vision Transformer for Dynamic Inference", arXiv, 2021 (*Aarhus University, Denmark*). [[Paper](https://arxiv.org/abs/2106.15183)]
    * **DeiT-Manifold**: "Efficient Vision Transformers via Fine-Grained Manifold Distillation", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2107.01378)]
    * **ViX**: "Vision Xformers: Efficient Attention for Image Classification", arXiv, 2021 (*Indian Institute of Technology Bombay*). [[Paper](https://arxiv.org/abs/2107.02239)]
    * **Transformer-LS**: "Long-Short Transformer: Efficient Transformers for Language and Vision", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2107.02192)]
    * **WideNet**: "Go Wider Instead of Deeper", arXiv, 2021 (*NUS*). [[Paper](https://arxiv.org/abs/2107.11817)]
    * **Evo-ViT**: "Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer", arXiv, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2108.01390)]
    * **Armour**: "Armour: Generalizable Compact Self-Attention for Vision Transformers", arXiv, 2021 (*Arm*). [[Paper](https://arxiv.org/abs/2108.01778)]
    * **Mobile-Former**: "Mobile-Former: Bridging MobileNet and Transformer", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2108.05895)]
    * **IPE**: "Exploring and Improving Mobile Level Vision Transformers", arXiv, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2108.13015)]
* Conv + Transformer:
    * **LeViT**: "LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.01136)][[PyTorch](https://github.com/facebookresearch/LeViT)]
    * **CeiT**: "Incorporating Convolution Designs into Visual Transformers", ICCV, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/2103.11816)][[PyTorch (rishikksh20)](https://github.com/rishikksh20/CeiT)]
    * **Conformer**: "Conformer: Local Features Coupling Global Representations for Visual Recognition", ICCV, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2105.03889)][[PyTorch](https://github.com/pengzhiliang/Conformer)]
    * **CvT**: "CvT: Introducing Convolutions to Vision Transformers", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.15808)][[Code](https://github.com/leoxiaobin/CvT)]
    * **CoaT**: "Co-Scale Conv-Attentional Image Transformers", arXiv, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2104.06399)][[Code (in construction)](https://github.com/mlpc-ucsd/CoaT)]
    * **ConTNet**: "ConTNet: Why not use convolution and transformer at the same time?", arXiv, 2021 (*ByteDance*). [[Paper](https://arxiv.org/abs/2104.13497)][[Code (in construction)](https://github.com/yan-hao-tian/ConTNet)]
    * **ViTc**: "Early Convolutions Help Transformers See Better", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.14881)]
    * **CMT**: "CMT: Convolutional Neural Networks Meet Vision Transformers", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2107.06263)]
    * **SPACH**: "A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2108.13002)]
* For pre-training:
    * **iGPT**: "Generative Pretraining From Pixels", ICML, 2020 (*OpenAI*). [[Paper](http://proceedings.mlr.press/v119/chen20s.html)][[Tensorflow](https://github.com/openai/image-gpt)]
    * **MoCo-V3**: "An Empirical Study of Training Self-Supervised Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.02057)]
    * **DINO**: "Emerging Properties in Self-Supervised Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.14294)][[PyTorch](https://github.com/facebookresearch/dino)]
    * **Annotations-1.3B**: "Billion-Scale Pretraining with Vision Transformers for Multi-Task Visual Representations", WACV, 2022 (*Pinterest*). [[Paper](https://arxiv.org/abs/2108.05887)]
    * **SiT**: "SiT: Self-supervised Vision Transformer", arXiv, 2021 (*University of Surrey*). [[Paper](https://arxiv.org/abs/2104.03602)][[Code (in construction)](https://github.com/Sara-Ahmed/SiT)]
    * **MoBY**: "Self-Supervised Learning with Swin Transformers", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2105.04553)][[Pytorch](https://github.com/SwinTransformer/Transformer-SSL)]
    * **?**: "Efficient Training of Visual Transformers with Small-Size Datasets", arXiv, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2106.03746)]
    * **MST**: "MST: Masked Self-Supervised Transformer for Visual Representation", arXiv, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/2106.05656)]
    * **BEiT**: "BEiT: BERT Pre-Training of Image Transformers", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.08254)][[Code (in construction)](https://github.com/microsoft/unilm/tree/master/beit)]
    * **EsViT**: "Efficient Self-supervised Vision Transformers for Representation Learning", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.09785)]
* For robustness:
    * **RVT**: "Rethinking the Design Principles of Robust Vision Transformer", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2105.07926)][[PyTorch](https://github.com/vtddggg/Robust-Vision-Transformer)]
    * **T-CNN**: "Transformed CNNs: recasting pre-trained convolutional layers with self-attention", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.05795)]
    * **Generalization-Enhanced-ViT**: "Delving Deep into the Generalization of Vision Transformers under Distribution Shifts", arXiv, 2021 (*Beihang University + NTU, Singapore*). [[Paper](https://arxiv.org/abs/2106.07617)]

##### MLP
* **MLP-Mixer**: "MLP-Mixer: An all-MLP Architecture for Vision", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.01601)][[PyTorch-1 (lucidrains)](https://github.com/lucidrains/mlp-mixer-pytorch)][[PyTorch-2 (rishikksh20)](https://github.com/rishikksh20/MLP-Mixer-pytorch)]
* **RepMLP**: "RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2105.01883)][[PyTorch](https://github.com/DingXiaoH/RepMLP)]
* **EAMLP**: "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks", arXiv, 2021 (*Tsinghua University*). [[Paper](https://arxiv.org/abs/2105.02358)]
* **Forward-Only**: "Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2105.02723)][[PyTorch](https://github.com/lukemelas/do-you-even-need-attention)]
* **ResMLP**: "ResMLP: Feedforward networks for image classification with data-efficient training", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2105.03404)]
* **gMLP**: "Pay Attention to MLPs", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.08050)]
* **?**: "Can Attention Enable MLPs To Catch Up With CNNs?", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.15078)]
* **S<sup>2</sup>-MLP**: "S<sup>2</sup>-MLP: Spatial-Shift MLP Architecture for Vision", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.07477)]
* **ViP**: "Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition", arXiv, 2021 (*NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.12368)][[PyTorch](https://github.com/Andrew-Qibin/VisionPermutator)]
* **CCS**: "Rethinking Token-Mixing MLP for MLP-based Vision Backbone", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.14882)]
* **AS-MLP**: "AS-MLP: An Axial Shifted MLP Architecture for Vision", arXiv, 2021 (*ShanghaiTech University*). [[Paper](https://arxiv.org/abs/2107.08391)][[PyTorch](https://github.com/svip-lab/AS-MLP)]
* **CycleMLP**: "CycleMLP: A MLP-like Architecture for Dense Prediction", arXiv, 2021 (*HKU*). [[Paper](https://arxiv.org/abs/2107.10224)][[PyTorch](https://github.com/ShoufaChen/CycleMLP)]
* **S<sup>2</sup>-MLPv2**: "S<sup>2</sup>-MLPv2: Improved Spatial-Shift MLP Architecture for Vision", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2108.01072)]
* **RaftMLP**: "RaftMLP: Do MLP-based Models Dream of Winning Over Computer Vision?", arXiv, 2021 (*Rikkyo University, Japan*). [[Paper](https://arxiv.org/abs/2108.04384)][[PyTorch](https://github.com/okojoalg/raft-mlp)]
* **Hire-MLP**: "Hire-MLP: Vision MLP via Hierarchical Rearrangement", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2108.13341)]
* **Sparse-MLP**: "Sparse-MLP: A Fully-MLP Architecture with Conditional Computation", arXiv, 2021 (*NUS*). [[Paper](https://arxiv.org/abs/2109.02008)]
* **ConvMLP**: "ConvMLP: Hierarchical Convolutional MLPs for Vision", arXiv, 2021 (*University of Oregon*). [[Paper](https://arxiv.org/abs/2109.04454)][[PyTorch](https://github.com/SHI-Labs/Convolutional-MLPs)]
* **sMLP**: "Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2109.05422)]

##### Analysis
* [**Attention-CNN**](approaches/attention/vision/Attention-CNN.md): "On the Relationship between Self-Attention and Convolutional Layers", ICLR, 2020 (*EPFL*). [[Paper](https://openreview.net/forum?id=HJlnC1rKPB)][[PyTorch](https://github.com/epfml/attention-cnn)][[Website](https://epfml.github.io/attention-cnn/)]
* **Transformer-Explainability**: "Transformer Interpretability Beyond Attention Visualization", CVPR, 2021 (*Tel Aviv*). [[Paper](https://arxiv.org/abs/2012.09838)][[PyTorch](https://github.com/hila-chefer/Transformer-Explainability)]
* **?**: "Are Convolutional Neural Networks or Transformers more like human vision?", CogSci, 2021 (*Princeton*). [[Paper](https://arxiv.org/abs/2105.07197)]
* **?**: "ConvNets vs. Transformers: Whose Visual Representations are More Transferable?", ICCVW, 2021 (*HKU*). [[Paper](https://arxiv.org/abs/2108.05305)]
* **FDSL**: "Can Vision Transformers Learn without Natural Images?", arXiv, 2021 (*AIST*). [[Paper](https://arxiv.org/abs/2103.13023)][[Website](https://hirokatsukataoka16.github.io/Vision-Transformers-without-Natural-Images/)]
* **ViT-Robustness**: "Understanding Robustness of Transformers for Image Classification", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.14586)] 
* **Transformer-Attack**: "On the Adversarial Robustness of Visual Transformers", arXiv, 2021 (*Xi'an Jiaotong*). [[Paper](https://arxiv.org/abs/2103.15670)]
* **SAGA**: "On the Robustness of Vision Transformers to Adversarial Examples", arXiv, 2021 (*University of Connecticut*). [[Paper](https://arxiv.org/abs/2104.02610)]
* **?**: "Vision Transformers are Robust Learners", arXiv, 2021 (*PyImageSearch + IBM*). [[Paper](https://arxiv.org/abs/2105.07581)][[Tensorflow](https://github.com/sayakpaul/robustness-vit)]
* **?**: "Intriguing Properties of Vision Transformers", arXiv, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2105.10497)][[Code (in construction)](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers)]
* **FoveaTer**: "FoveaTer: Foveated Transformer for Image Classification", arXiv, 2021 (*UCSB*). [[Paper](https://arxiv.org/abs/2105.14173)]
* **?**: "When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.01548)]
* **?**: "Reveal of Vision Transformers Robustness against Adversarial Attacks", arXiv, 2021 (*University of Rennes*). [[Paper](https://arxiv.org/abs/2106.03734)]
* **?**: "Demystifying Local Vision Transformer: Sparse Connectivity, Weight Sharing, and Dynamic Weight", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.04263)]
* **?**: "On Improving Adversarial Transferability of Vision Transformers", arXiv, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2106.04169)][[Code (in construction)](https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers)]
* **?**: "Revisiting the Calibration of Modern Neural Networks", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.07998)]
* **?**: "How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.10270)][[Tensorflow](https://github.com/google-research/vision_transformer)][[PyTorch (rwightman)](https://github.com/rwightman/pytorch-image-models)]
* **?**: "Exploring Corruption Robustness: Inductive Biases in Vision Transformers and MLP-Mixers", arXiv, 2021 (*University of Pittsburgh*). [[Paper](https://arxiv.org/abs/2106.13122)]
* **?**: "What Makes for Hierarchical Vision Transformer?", arXiv, 2021 (*Horizon Robotic*). [[Paper](https://arxiv.org/abs/2107.02174)]
* **?**: "Do Vision Transformers See Like Convolutional Neural Networks?", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2108.08810)]
* **PNA**: "Towards Transferable Adversarial Attacks on Vision Transformers", arXiv, 2021 (*Fudan + Maryland*). [[Paper](https://arxiv.org/abs/2109.04176)]


### Detection
##### Object Detection
* CNN-based backbone:
    * **DETR**: "End-to-End Object Detection with Transformers", ECCV, 2020 (*Facebook*). [[Paper](https://arxiv.org/abs/2005.12872)][[PyTorch](https://github.com/facebookresearch/detr)]
    * **Deformable DETR**: "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR, 2021 (*SenseTime*). [[Paper](https://arxiv.org/abs/2010.04159)][[PyTorch](https://github.com/fundamentalvision/Deformable-DETR)]
    * **UP-DETR**: "UP-DETR: Unsupervised Pre-training for Object Detection with Transformers", CVPR, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2011.09094)][[PyTorch](https://github.com/dddzg/up-detr)]
    * **SMCA**: "Fast Convergence of DETR with Spatially Modulated Co-Attention", ICCV, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2108.02404)][[PyTorch](https://github.com/gaopengcuhk/SMCA-DETR)]
    * **Conditional-DETR**: "Conditional DETR for Fast Training Convergence", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2108.06152)]
    * **PnP-DETR**: "PnP-DETR: Towards Efficient Visual Analysis with Transformers", ICCV, 2021 (*Yitu*). [[Paper](https://arxiv.org/abs/2109.07036)][[Code (in construction)](https://github.com/twangnh/pnp-detr)]
    * **ACT**: "End-to-End Object Detection with Adaptive Clustering Transformer", arXiv, 2021 (*Peking + CUHK*). [[Paper](https://arxiv.org/abs/2011.09315)]
    * **TSP**: "Rethinking Transformer-based Set Prediction for Object Detection", arXiv, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2011.10881)]
    * **Efficient-DETR**: "Efficient DETR: Improving End-to-End Object Detector with Dense Prior", arXiv, 2021 (*Megvii*). [[Paper](https://arxiv.org/abs/2104.01318)]
    * **CA-FPN**: "Content-Augmented Feature Pyramid Network with Light Linear Transformers", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2105.09464)]
    * **DETReg**: "DETReg: Unsupervised Pretraining with Region Priors for Object Detection", arXiv, 2021 (*Tel-Aviv + Berkeley*). [[Paper](https://arxiv.org/abs/2106.04550)][[Website](https://www.amirbar.net/detreg/)]
    * **GQPos**: "Guiding Query Position and Performing Similar Attention for Transformer-Based Detection Heads", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2108.09691)]
    * **Anchor-DETR**: "Anchor DETR: Query Design for Transformer-Based Detector", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2109.07107)]
* Transformer-based backbone:
    * **ViT-FRCNN**: "Toward Transformer-Based Object Detection", arXiv, 2020 (*Pinterest*). [[Paper](https://arxiv.org/abs/2012.09958)]
    * **YOLOS**: "You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection", arXiv, 2021 (*Horizon Robotics*). [[Paper](https://arxiv.org/abs/2106.00666)][[PyTorch](https://github.com/hustvl/YOLOS)]

##### Other Detection Tasks
* Pedestrian Detection:
    * **PED**: "DETR for Crowd Pedestrian Detection", arXiv, 2020 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2012.06785)][[PyTorch](https://github.com/Hatmm/PED-DETR-for-Pedestrian-Detection)]
* Lane Detection:
    * **LSTR**: "End-to-end Lane Shape Prediction with Transformers", WACV, 2021 (*Xi'an Jiaotong*). [[Paper](https://arxiv.org/abs/2011.04233)][[PyTorch](https://github.com/liuruijin17/LSTR)]
    * **LETR**: "Line Segment Detection Using Transformers without Edges", CVPR, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2101.01909)][[PyTorch](https://github.com/mlpc-ucsd/LETR)]
* 3D Object Detection:
    * **AST-GRU**: "LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention", CVPR, 2020 (*Baidu*). [[Paper](https://arxiv.org/abs/2004.01389)][[Code (in construction)](https://github.com/yinjunbo/3DVID)]
    * **Pointformer**: "3D Object Detection with Pointformer", arXiv, 2020 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2012.11409)]
    * **CT3D**: "Improving 3D Object Detection with Channel-wise Transformer", ICCV, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2108.10723)][[Code (in construction)](https://github.com/hlsheng1/CT3D)]
    * **Group-Free-3D**: "Group-Free 3D Object Detection via Transformers", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2104.00678)][[PyTorch](https://github.com/zeliu98/Group-Free-3D)]
    * **VoTr**: "Voxel Transformer for 3D Object Detection", ICCV, 2021 (*CUHK + NUS*). [[Paper](https://arxiv.org/abs/2109.02497)]
    * **3DETR**: "An End-to-End Transformer Model for 3D Object Detection", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2109.08141)][[PyTorch](https://github.com/facebookresearch/3detr)][[Website](https://facebookresearch.github.io/3detr/)]
    * **M3DeTR**: "M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers", arXiv, 2021 (*University of Maryland*). [[Paper](https://arxiv.org/abs/2104.11896)]
* Object Localization:
    * **TS-CAM**: "TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2103.14862)]
* Relation Detection:
    * **RelTransformer**: "RelTransformer: Balancing the Visual Relationship Detection from Local Context, Scene and Memory", arXiv, 2021 (*KAUST*). [[Paper](https://arxiv.org/abs/2104.11934)][[Code (in construction)](https://github.com/Vision-CAIR/RelTransformer)]
    * **PST**: "Visual Composite Set Detection Using Part-and-Sum Transformers", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2105.02170)]
    * **TROI**: "Transformed ROIs for Capturing Visual Transformations in Videos", arXiv, 2021 (*NUS, Singapore*). [[Paper](https://arxiv.org/abs/2106.03162)]
    * **Relation-Transformer**: "Scenes and Surroundings: Scene Graph Generation using Relation Transformer", arXiv, 2021 (*LMU Munich*). [[Paper](https://arxiv.org/abs/2107.05448)]
* HOI Detection:
    * **HOI-Transformer**: "End-to-End Human Object Interaction Detection with HOI Transformer", CVPR, 2021 (*Megvii*). [[Paper](https://arxiv.org/abs/2103.04503)][[PyTorch](https://github.com/bbepoch/HoiTransformer)]
    * **HOTR**: "HOTR: End-to-End Human-Object Interaction Detection with Transformers", CVPR, 2021 (*Kakao + Korea University*). [[Paper](https://arxiv.org/abs/2104.13682)][[PyTorch](https://github.com/kakaobrain/HOTR)]
* Salient Object Detection:
    * **SOD-Transformer**: "Transformer Transforms Salient Object Detection and Camouflaged Object Detection", arXiv, 2021 (*Northwestern Polytechnical University*). [[Paper](https://arxiv.org/abs/2104.10127)]
    * **VST**: "Visual Saliency Transformer", arXiv, 2021 (*Northwestern Polytechincal University*). [[Paper](https://arxiv.org/abs/2104.12099)]
    * **GLSTR**: "Unifying Global-Local Representations in Salient Object Detection with Transformer", arXiv, 2021 (*South China University of Technology*). [[Paper](https://arxiv.org/abs/2108.02759)]
    * **TriTransNet**: "TriTransNet: RGB-D Salient Object Detection with a Triplet Transformer Embedding Network", arXiv, 2021 (*Anhui University*). [[Paper](https://arxiv.org/abs/2108.03990)]
    * **AbiU-Net**: "Boosting Salient Object Detection with Transformer-based Asymmetric Bilateral U-Net", arXiv, 2021 (*Nankai University*). [[Paper](https://arxiv.org/abs/2108.07851)]
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


### Segmentation
* Semantic Segmentation:
    * **SETR**: "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers", CVPR, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2012.15840)][[PyTorch](https://github.com/fudan-zvg/SETR)][[Website](https://fudan-zvg.github.io/SETR/)]
    * **TrSeg**: "TrSeg: Transformer for semantic segmentation", PRL, 2021 (*Korea University*). [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S016786552100163X)][[PyTorch](https://github.com/youngsjjn/TrSeg)]
    * **CWT**: "Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer", ICCV, 2021 (*University of Surrey, UK*). [[Paper](https://arxiv.org/abs/2108.03032)][[PyTorch](https://github.com/zhiheLu/CWT-for-FSS)]
    * **Segmenter**: "Segmenter: Transformer for Semantic Segmentation", arXiv, 2021 (*INRIA*). [[Paper](https://arxiv.org/abs/2105.05633)][[Code (in construction)](https://github.com/rstrudel/segmenter)]
    * **SegFormer**: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2105.15203)]
    * **FTN**: "Fully Transformer Networks for Semantic Image Segmentation", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.04108)]
    * **OffRoadTranSeg**: "OffRoadTranSeg: Semi-Supervised Segmentation using Transformers on OffRoad environments", arXiv, 2021 (*IISER. India*). [[Paper](https://arxiv.org/abs/2106.13963)]
    * **MaskFormer**: "Per-Pixel Classification is Not All You Need for Semantic Segmentation", arXiv, 2021 (*UIUC + Facebook*). [[Paper](https://arxiv.org/abs/2107.06278)][[Website](https://bowenc0221.github.io/maskformer/)]
    * **UN-EPT**: "A Unified Efficient Pyramid Transformer for Semantic Segmentation", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2107.14209)]
    * **TRFS**: "Boosting Few-shot Semantic Segmentation with Transformers", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2108.02266)]
    * **Flying-Guide-Dog**: "Flying Guide Dog: Walkable Path Discovery for the Visually Impaired Utilizing Drones and Transformer-based Semantic Segmentation", arXiv, 2021 (*KIT, Germany*). [[Paper](https://arxiv.org/abs/2108.07007)][[Code (in construction)](https://github.com/EckoTan0804/flying-guide-dog)]
    * **VSPW**: "Semantic Segmentation on VSPW Dataset through Aggregation of Transformer Models", arXiv, 2021 (*Xiaomi*). [[Paper](https://arxiv.org/abs/2109.01316)]
    * **SDTP**: "SDTP: Semantic-aware Decoupled Transformer Pyramid for Dense Image Prediction", arXiv, 2021 (*?*). [[Paper](https://arxiv.org/abs/2109.08963)]
* Panoptic Segmentation:
    * **MaX-DeepLab**: "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2012.00759)][[PyTorch (conradry)](https://github.com/conradry/max-deeplab)]
    * **Panoptic-SegFormer**: "Panoptic SegFormer", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2109.03814)]
* Depth Estimation & Semantic Segmentation:
    * **DPT**: "Vision Transformers for Dense Prediction", ICCV, 2021 (*Intel*). [[Paper](https://arxiv.org/abs/2103.13413)][[PyTorch](https://github.com/intel-isl/DPT)]
* Instance Segmentation:
    * **ISTR**: "ISTR: End-to-End Instance Segmentation with Transformers", arXiv, 2021 (*Xiamen University*). [[Paper](https://arxiv.org/abs/2105.00637)][[PyTorch](https://github.com/hujiecpp/ISTR)]
* Depth Estimation:
    * **TransDepth**: "Transformers Solve the Limited Receptive Field for Monocular Depth Prediction", arXiv, 2021 (*Haerbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2103.12091)][[PyTorch](https://github.com/ygjwd12345/TransDepth)]
* Object Segmentation:
    * **SOTR**: "SOTR: Segmenting Objects with Transformers", ICCV, 2021 (*China Agricultural University*). [[Paper](https://arxiv.org/abs/2108.06747)][[PyTorch](https://github.com/easton-cau/SOTR)]
    * **Trans4Trans**: "Trans4Trans: Efficient Transformer for Transparent Object Segmentation to Help Visually Impaired People Navigate in the Real World", ICCVW, 2021 (*Karlsruhe Institute of Technology, Germany*). [[Paper](https://arxiv.org/abs/2107.03172)][[Code (in construction)](https://github.com/jamycheung/Trans4Trans)]
    * **Trans2Seg**: "Segmenting Transparent Object in the Wild with Transformer", arXiv, 2021 (*HKU + SenseTime*). [[Paper](https://arxiv.org/abs/2101.08461)][[PyTorch](https://github.com/xieenze/Trans2Seg)]
* Few-Shot:
    * **CyCTR**: "Few-Shot Segmentation via Cycle-Consistent Transformer", arXiv, 2021 (*University of Technology Sydney*). [[Paper](https://arxiv.org/abs/2106.02320)]
* Urban Scene:
    * **BANet**: "Transformer Meets Convolution: A Bilateral Awareness Net-work for Semantic Segmentation of Very Fine Resolution Ur-ban Scene Images", arXiv, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2106.12413)] 


### Video (High-level)
##### Action Recognition
* RGB mainly
    * **Action Transformer**: "Video Action Transformer Network", CVPR, 2019 (*DeepMind*). [[Paper](https://arxiv.org/abs/1812.02707)][[Code (ppriyank)](https://github.com/ppriyank/Video-Action-Transformer-Network-Pytorch-)]
    * **ViViT-Ensemble**: "Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition", CVPRW, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.05058)]
    * **TimeSformer**: "Is Space-Time Attention All You Need for Video Understanding?", ICML, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2102.05095)][[PyTorch (lucidrains)](https://github.com/lucidrains/TimeSformer-pytorch)]
    * **MViT**: "Multiscale Vision Transformers", ICCV, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2104.11227)][[PyTorch](https://github.com/facebookresearch/SlowFast)]
    * **TokShift**: "Token Shift Transformer for Video Classification", ACMMM, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2108.02432)][[PyTorch](https://github.com/VideoNetworks/TokShift-Transformer)]
    * **VTN**: "Video Transformer Network", arXiv, 2021 (*Theator*). [[Paper](https://arxiv.org/abs/2102.00719)]
    * **STAM**: "An Image is Worth 16x16 Words, What is a Video Worth?", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2103.13915)][[Code](https://github.com/Alibaba-MIIL/STAM)]
    * **GAT**: "Enhancing Transformer for Video Understanding Using Gated Multi-Level Attention and Temporal Adversarial Training", arXiv, 2021 (*Samsung*). [[Paper](https://arxiv.org/abs/2103.10043)]
    * **ViViT**: "ViViT: A Video Vision Transformer", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2103.15691)]
    * **VidTr**: "VidTr: Video Transformer Without Convolutions", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2104.11746)]
    * **Motionformer**: "Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.05392)][[PyTorch](https://github.com/facebookresearch/Motionformer)]
    * **X-ViT**: "Space-time Mixing Attention for Video Transformer", arXiv, 2021 (*Samsung*). [[Paper](https://arxiv.org/abs/2106.05968)]
    * **TokenLearner**: "TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.11297)]
    * **Video-Swin**: "Video Swin Transformer", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2106.13230)][[PyTorch](https://github.com/SwinTransformer/Video-Swin-Transformer)]
    * **VLF**: "VideoLightFormer: Lightweight Action Recognition using Transformers", arXiv, 2021 (*The University of Sheffield*). [[Paper](https://arxiv.org/abs/2107.00451)]
    * **SCT**: "Shifted Chunk Transformer for Spatio-Temporal Representational Learning", arXiv, 2021 (*Kuaishou*). [[Paper](https://arxiv.org/abs/2108.11575)]
* Depth
    * **Trear**: "Trear: Transformer-based RGB-D Egocentric Action Recognition",  IEEE Transactions on Cognitive and Developmental Systems, 2021 (*Tianjing University*). [[Paper](https://ieeexplore.ieee.org/document/9312201)]
* Pose:
    * **AcT**: "Action Transformer: A Self-Attention Model for Short-Time Human Action Recognition", arXiv, 2021 (*Politecnico di Torino, Italy*). [[Paper](https://arxiv.org/abs/2107.00606)][[Code (in construction)](https://github.com/FedericoAngelini/MPOSE2021_Dataset)]
    * **STAR**: "STAR: Sparse Transformer-based Action Recognition", arXiv, 2021 (*UCLA*). [[Paper](https://arxiv.org/abs/2107.07089)]
    * **GCsT**: "GCsT: Graph Convolutional Skeleton Transformer for Action Recognition", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2109.02860)]
* Multi-modal:
    * **POTR**: "Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive Transformers", ICCVW, 2021 (*Idiap*). [[Paper](https://arxiv.org/abs/2109.07531)]
    * **MBT**: "Attention Bottlenecks for Multimodal Fusion", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2107.00135)]
    * **MM-ViT**: "MM-ViT: Multi-Modal Video Transformer for Compressed Video Action Recognition", arXiv, 2021 (*OPPO*). [[Paper](https://arxiv.org/abs/2108.09322)]

##### Action Detection
* **OadTR**: "OadTR: Online Action Detection with Transformers", ICCV, 2021 (*Huazhong University of Science and Technology*). [[Paper](https://arxiv.org/abs/2106.11149)][[PyTorch](https://github.com/wangxiang1230/OadTR)]
* **TubeR**: "TubeR: Tube-Transformer for Action Detection", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2104.00969)]
* **ATAG**: "Augmented Transformer with Adaptive Graph for Temporal Action Proposal Generation", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2103.16024)]
* **TAPG-Transformer**: "Temporal Action Proposal Generation with Transformers", arXiv, 2021 (*Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2105.12043)]
* **TadTR**: "End-to-end Temporal Action Detection with Transformer", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2106.10271)][[Code (in construction)](https://github.com/xlliu7/TadTR)]
* **Vidpress-Soccer**: "Feature Combination Meets Attention: Baidu Soccer Embeddings and Transformer based Temporal Detection", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2106.14447)][[GitHub](https://github.com/baidu-research/vidpress-sports)]
* **LSTR**: "Long Short-Term Transformer for Online Action Detection", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2107.03377)]

##### Action Prediction
* **AVT**: "Anticipative Video Transformer", CVPRW, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2106.02036)]
* **HORST**: "Higher Order Recurrent Space-Time Transformer", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2104.08665)][[PyTorch](https://github.com/CorcovadoMing/HORST)]
* **?**: "Action Forecasting with Feature-wise Self-Attention", arXiv, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2107.08579)]

##### Other Video Tasks
* Video Instance Segmentation
    * **VisTR**: "End-to-End Video Instance Segmentation with Transformers", CVPR, 2021 (*Meituan*). [[Paper](https://arxiv.org/abs/2011.14503)][[PyTorch](https://github.com/Epiphqny/VisTR)]
    * **IFC**: "Video Instance Segmentation using Inter-Frame Communication Transformers", arXiv, 2021 (*Yonsei University*). [[Paper](https://arxiv.org/abs/2106.03299)]
* Video Object Segmentation
    * **SSTVOS**: "SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation", CVPR, 2021 (*Modiface*). [[Paper](https://arxiv.org/abs/2101.08833)][[Code (in construction)](https://github.com/dukebw/SSTVOS)]
    * **TransVOS**: "TransVOS: Video Object Segmentation with Transformers", arXiv, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2106.00588)]
    * **AOT**: "Associating Objects with Transformers for Video Object Segmentation", arXiv, 2021 (*University of Technology Sydney*). [[Paper](https://arxiv.org/abs/2106.02638)][[Code (in construction)](https://github.com/z-x-yang/AOT)]
* Video Object Detection:
    * **TransVOD**: "End-to-End Video Object Detection with Spatial-Temporal Transformers", arXiv, 2021 (*Shanghai Jiao Tong + SenseTime*). [[Paper](https://arxiv.org/abs/2105.10920)][[Code (in construction)](https://github.com/SJTU-LuHe/TransVOD)]
    * **MODETR**: "MODETR: Moving Object Detection with Transformers", arXiv, 2021 (*Valeo, Egypt*). [[Paper](https://arxiv.org/abs/2106.11422)]
    * **ST-MTL**: "Spatio-Temporal Multi-Task Learning Transformer for Joint Moving Object Detection and Segmentation", arXiv, 2021 (*Valeo, Egypt*). [[Paper](https://arxiv.org/abs/2106.11401)]
    * **ST-DETR**: "ST-DETR: Spatio-Temporal Object Traces Attention Detection Transformer", arXiv, 2021 (*Valeo, Egypt*). [[Paper](https://arxiv.org/abs/2107.05887)]
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


### Multi-Modality
##### VQA / Captioning
* **ETA-Transformer**: "Entangled Transformer for Image Captioning", ICCV, 2019 (*UTS*). [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Entangled_Transformer_for_Image_Captioning_ICCV_2019_paper.html)]
* **M4C**: "Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA", CVPR, 2020 (*Facebook*). [[Paper](https://arxiv.org/abs/1911.06258)]
* **M2-Transformer**: "Meshed-Memory Transformer for Image Captioning", CVPR, 2020 (*UniMoRE*). [[Paper](https://arxiv.org/abs/1912.08226)][[PyTorch](https://github.com/aimagelab/meshed-memory-transformer)] 
* **SA-M4C**: "Spatially Aware Multimodal Transformers for TextVQA", ECCV, 2020 (*Georgia Tech*). [[Paper](https://arxiv.org/abs/2007.12146)][[PyTorch](https://github.com/yashkant/sam-textvqa)][[Website](https://yashkant.github.io/projects/sam-textvqa.html)]
* **CATT**: "Causal Attention for Vision-Language Tasks", CVPR, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2103.03493)][[PyTorch](https://github.com/yangxuntu/lxmertcatt)]
* **?**: "Optimizing Latency for Online Video CaptioningUsing Audio-Visual Transformers", Interspeech, 2021 (*MERL*). [[Paper](https://arxiv.org/abs/2108.02147)]
* **ConClaT**: "Contrast and Classify: Training Robust VQA Models", ICCV, 2021 (*Georgia Tech*). [[Paper](https://arxiv.org/abs/2010.06087)]
* **DGCN**: "Dual Graph Convolutional Networks with Transformer and Curriculum Learning for Image Captioning", ACMMM, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2108.02366)]
* **TxT**: "TxT: Crossmodal End-to-End Learning with Transformers", GCPR, 2021 (*TU Darmstadt*). [[Paper](https://arxiv.org/abs/2109.04422)]
* **CPTR**: "CPTR: Full Transformer Network for Image Captioning", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2101.10804)] 
* **VisQA**: "VisQA: X-raying Vision and Language Reasoning in Transformers", arXiv, 2021 (*INSA-Lyon*). [[Paper](https://arxiv.org/abs/2104.00926)][[PyTorch](https://github.com/Theo-Jaunet/VisQA)]
* **SATIC**: "Semi-Autoregressive Transformer for Image Captioning", arXiv, 2021 (*Hefei University of Technology*). [[Paper](https://arxiv.org/abs/2106.09436)][[PyTorch](https://github.com/YuanEZhou/satic)]
* **UniQer**: "Unified Questioner Transformer for Descriptive Question Generation in Goal-Oriented Visual Dialogue", arXiv, 2021 (*Keio*). [[Paper](https://arxiv.org/abs/2106.15550)]
* **ReFormer**: "ReFormer: The Relational Transformer for Image Captioning", arXiv, 2021 (*Stony Brook University*). [[Paper](https://arxiv.org/abs/2107.14178)]
* **?**: "Mounting Video Metadata on Transformer-based Language Model for Open-ended Video Question Answering", arXiv, 2021 (*Seoul National University*). [[Paper](https://arxiv.org/abs/2108.05158)]
* **LAViTeR**: "LAViTeR: Learning Aligned Visual and Textual Representations Assisted by Image and Caption Generation", arXiv, 2021 (*University at Buffalo*). [[Paper](https://arxiv.org/abs/2109.04993)]
* **TPT**: "Temporal Pyramid Transformer with Multimodal Interaction for Video Question Answering", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2109.04735)]
* **LATGeO**: "Label-Attention Transformer with Geometrically Coherent Objects for Image Captioning", arXiv, 2021 (*Gwangju Institute of Science and Technology*). [[Paper](https://arxiv.org/abs/2109.07799)]

##### Visual Grounding
* **Multi-Stage-Transformer**: "Multi-Stage Aggregated Transformer Network for Temporal Language Localization in Videos", CVPR, 2021 (*University of Electronic Science and Technology of China*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Multi-Stage_Aggregated_Transformer_Network_for_Temporal_Language_Localization_in_Videos_CVPR_2021_paper.html)]
* **TransRefer3D**: "TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding", ACMMM, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2108.02388)]
* **?**: "Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers", EMNLP, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2109.04448)]
* **GTR**: "On Pursuit of Designing Multi-modal Transformer for Video Grounding", EMNLP, 2021 (*Peking*). [[Paper](https://arxiv.org/abs/2109.06085)]
* **MITVG**: "Multimodal Incremental Transformer with Visual Grounding for Visual Dialogue Generation", ACL Findings, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2109.08478)]
* **TransVG**: "TransVG: End-to-End Visual Grounding with Transformers", arXiv, 2021 (*USTC*). [[Paper](https://arxiv.org/abs/2104.08541)]
* **VGTR**: "Visual Grounding with Transformers", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2105.04281)]
* **Referring-Transformer**: "Referring Transformer: A One-step Approach to Multi-task Visual Grounding", arXiv, 2021 (*UBC*). [[Paper](https://arxiv.org/abs/2106.03089)]
* **Word2Pix**: "Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding", arXiv, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2108.00205)]

##### Representation Learning
* **COOT**: "COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning", NeurIPS, 2020 (*University of Freiburg*). [[Paper](https://arxiv.org/abs/2011.00597)][[PyTorch](https://github.com/gingsi/coot-videotext)]
* **Parameter-Reduction**: "Parameter Efficient Multimodal Transformers for Video Representation Learning", ICLR, 2021 (*Seoul National University*). [[Paper](https://openreview.net/forum?id=6UdQLhqJyFD)]
* **VinVL**: "VinVL: Revisiting Visual Representations in Vision-Language Models", CVPR, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2101.00529)][[Code](https://github.com/pzzhang/VinVL)]
* **ViLT**: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision", ICML, 2021 (*Kakao*). [[Paper](https://arxiv.org/abs/2102.03334)][[PyTorch](https://github.com/dandelin/vilt)]
* **VML**: "VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding", ACL Findings, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2105.09996)]
* **VATT**: "VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2104.11178)]
* **SVO-Probes**: "Probing Image-Language Transformers for Verb Understanding", arXiv, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2106.09141)]
* **CLIP-ViL**: "How Much Can CLIP Benefit Vision-and-Language Tasks?", arXiv, 2021 (*Berkeley + UCLA*). [[Paper](https://arxiv.org/abs/2107.06383)][[PyTorch](https://github.com/clip-vil/CLIP-ViL)]

##### Retrieval
* **MMT**: "Multi-modal Transformer for Video Retrieval", ECCV, 2020 (*INRIA + Google*). [[Paper](https://arxiv.org/abs/2007.10639)][[Website](http://thoth.inrialpes.fr/research/MMT/)]
* **Fast-and-Slow**: "Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers", CVPR, 2021 (*DeepMind*). [[Paper](https://arxiv.org/abs/2103.16553)]
* **HTR**: "Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning", CVPR, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2103.13061)][[PyTorch](https://github.com/amzn/image-to-recipe-transformers)]
* **ClipBERT**: "Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling", CVPR, 2021 (*UNC + Microsoft*). [[Paper](https://arxiv.org/abs/2102.06183)][[PyTorch](https://github.com/jayleicn/ClipBERT)]
* **AYCE**: "All You Can Embed: Natural Language based Vehicle Retrieval with Spatio-Temporal Transformers", CVPRW, 2021 (*University of Modena and Reggio Emilia*). [[Paper](https://arxiv.org/abs/2106.10153)][[PyTorch](https://github.com/cscribano/AYCE_2021)]
* **TERN**: "Towards Efficient Cross-Modal Visual Textual Retrieval using Transformer-Encoder Deep Features", CBMI, 2021 (*National Research Council, Italy*). [[Paper](https://arxiv.org/abs/2106.00358)]
* **VisualSparta**: "VisualSparta: Sparse Transformer Fragment-level Matching for Large-scale Text-to-Image Search", arXiv, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2101.00265)]
* **WebVid-2M**: "Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval", arXiv, 2021 (*Oxford*). [[Paper](https://arxiv.org/abs/2104.00650)]
* **CCR-CCS**: "More Than Just Attention: Learning Cross-Modal Attentions with Contrastive Constraints", arXiv, 2021 (*Rutgers + Amazon*). [[Paper](https://arxiv.org/abs/2105.09597)]

##### Others
* Detection:
    * **MDETR**: "MDETR - Modulated Detection for End-to-End Multi-Modal Understanding", ICCV, 2021 (*NYU*). [[Paper](https://arxiv.org/abs/2104.12763)][[PyTorch](https://github.com/ashkamath/mdetr)][[Website](https://ashkamath.github.io/mdetr_page/)]
    * **StrucTexT**: "StrucTexT: Structured Text Understanding with Multi-Modal Transformers", arXiv, 2021 (*Baidu*). [[Paper](https://arxiv.org/abs/2108.02923)]
* Segmentation:
    * **VLT**: "Vision-Language Transformer and Query Generation for Referring Segmentation", ICCV, 2021 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2108.05565)][[Tensorflow](https://github.com/henghuiding/Vision-Language-Transformer)]
* Analysis:
    * **MM-Explainability**: "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers", ICCV, 2021 (*Tel Aviv*). [[Paper](https://arxiv.org/abs/2103.15679)][[PyTorch](https://github.com/hila-chefer/Transformer-MM-Explainability)]
* Generation:
    * **DALL-E**: "Zero-Shot Text-to-Image Generation", arXiv, 2021 (*OpenAI*). [[Paper](https://arxiv.org/abs/2102.12092)][[PyTorch](https://github.com/openai/DALL-E)][[PyTorch (lucidrains)](https://github.com/lucidrains/DALLE-pytorch)]
    * **CogView**: "CogView: Mastering Text-to-Image Generation via Transformers", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2105.13290)][[Code (in construction)](https://github.com/THUDM/CogView)][[Website](https://lab.aminer.cn/cogview/index.html)]
* Speaker Localization:
    * **?**: "The Right to Talk: An Audio-Visual Transformer Approach", ICCV, 2021 (*University of Arkansas*). [[Paper](https://arxiv.org/abs/2108.03256)]
* Multi-task:
    * **UniT**: "Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2102.10772)][[PyTorch](https://github.com/facebookresearch/mmf)][[Website](https://mmf.sh/)]
* Language-based Video Editing:
    * **M<sup>3</sup>L**: "Language-based Video Editing via Multi-Modal Multi-Level Transformer", arXiv, 2021 (*UCSB*). [[Paper](https://arxiv.org/abs/2104.01122)]
* Video Summarization:
    * **GPT2MVS**: "GPT2MVS: Generative Pre-trained Transformer-2 for Multi-modal Video Summarization", ICMR, 2021 (*BBC*). [[Paper](https://arxiv.org/abs/2104.12465)]
* Visual Document Understanding:
    * **DocFormer**: "DocFormer: End-to-End Transformer for Document Understanding", arXiv, 2021 (*Amazon*). [[Paper](https://arxiv.org/abs/2106.11539)]
* Robotics:
    * **CRT**: "Case Relation Transformer: A Crossmodal Language Generation Model for Fetching Instructions", IROS, 2021 (*Keio University*). [[Paper](https://arxiv.org/abs/2107.00789)]
* Image Fusion:
    * **IFT**: "Image Fusion Transformer", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2107.09011)][[PyTorch](https://github.com/Vibashan/Image-Fusion-Transformer)]
    * **PPT**: "PPT Fusion: Pyramid Patch Transformerfor a Case Study in Image Fusion", arXiv, 2021 (*?*). [[Paper](https://arxiv.org/abs/2107.13967)]
* Scene Graph:
    * **SGG-NLS**: "Learning to Generate Scene Graph from Natural Language Supervision", ICCV, 2021 (*University of Wisconsin-Madison*). [[Paper](https://arxiv.org/abs/2109.02227)][[PyTorch](https://github.com/YiwuZhong/SGG_from_NLS)]
* Human Interaction:
    * **Dyadformer**: "Dyadformer: A Multi-modal Transformer for Long-Range Modeling of Dyadic Interactions", ICCVW, 2021 (*Universitat de Barcelona*). [[Paper](https://arxiv.org/abs/2109.09487)]


### Other High-level Vision Tasks
##### Point Cloud
* **PCT**: "PCT: Point Cloud Transformer", arXiv, 2020 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2012.09688)][[Jittor](https://github.com/MenghaoGuo/PCT)][[PyTorch (uyzhang)](https://github.com/uyzhang/PCT_Pytorch)]
* **Point-Transformer**: "Point Transformer", arXiv, 2020 (*Ulm University*). [[Paper](https://arxiv.org/abs/2011.00931)]
* **NDT-Transformer**: "NDT-Transformer: Large-Scale 3D Point Cloud Localisation using the Normal Distribution Transform Representation", ICRA, 2021 (*University of Sheffield*). [[Paper](https://arxiv.org/abs/2103.12292)][[PyTorch](https://github.com/dachengxiaocheng/NDT-Transformer)]
* **P4Transformer**: "Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos", CVPR, 2021 (*NUS*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.html)]
* **PTT**: "PTT: Point-Track-Transformer Module for 3D Single Object Tracking in Point Clouds", IROS, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2108.06455)][[PyTorch (in construction)](https://github.com/shanjiayao/PTT)]
* **SnowflakeNet**: "SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer", ICCV, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2108.04444)]
* **PoinTr**: "PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers", ICCV, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2108.08839)][[PyTorch](https://github.com/yuxumin/PoinTr)]
* **Point-Transformer**: "Point Transformer", ICCV, 2021 (*Oxford + CUHK*). [[Paper](https://arxiv.org/abs/2012.09164)][[PyTorch (lucidrains)](https://github.com/lucidrains/point-transformer-pytorch)]
* **YOGO**: "You Only Group Once: Efficient Point-Cloud Processing with Token Representation and Relation Inference Module", arXiv, 2021 (*Berkeley*). [[Paper](https://arxiv.org/abs/2103.09975)][[PyTorch](https://github.com/chenfengxu714/YOGO)]
* **DTNet**: "Dual Transformer for Point Cloud Analysis", arXiv, 2021 (*Southwest University*). [[Paper](https://arxiv.org/abs/2104.13044)]
* **MLMSPT**: "Point Cloud Learning with Transformer", arXiv, 2021 (*Southwest University*). [[Paper](https://arxiv.org/abs/2104.13636)]
* **SCTN**: "SCTN: Sparse Convolution-Transformer Network for Scene Flow Estimation", arXiv, 2021 (*KAUST*). [[Paper](https://arxiv.org/abs/2105.04447)]
* **?**: "Shape registration in the time of transformers", arXiv, 2021 (*Sapienza University of Rome*). [[Paper](https://arxiv.org/abs/2106.13679)]
* **PQ-Transformer**: "PQ-Transformer: Jointly Parsing 3D Objects and Layouts from Point Clouds", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2109.05566)][[PyTorch](https://github.com/OPEN-AIR-SUN/PQ-Transformer)]

##### Pose Estimation
* Human-related: 
    * **Hand-Transformer**: "Hand-Transformer: Non-Autoregressive Structured Modeling for 3D Hand Pose Estimation", ECCV, 2020 (*Kwai*). [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4836_ECCV_2020_paper.php)]
    * **HOT-Net**: "HOT-Net: Non-Autoregressive Transformer for 3D Hand-Object Pose Estimation", ACMMM. 2020 (*Kwai*). [[Paper](https://cse.buffalo.edu/~jmeng2/publications/hotnet_mm20)]
    * **TransPose**: "TransPose: Towards Explainable Human Pose Estimation by Transformer", arXiv, 2020 (*Southeast University*). [[Paper](https://arxiv.org/abs/2012.14214)][[PyTorch](https://github.com/yangsenius/TransPose)]
    * **PTF**: "Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration", CVPR, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2104.08160)][[Code (in construction)](https://github.com/taconite/PTF)][[Website](https://taconite.github.io/PTF/website/PTF.html)]
    * **METRO**: "End-to-End Human Pose and Mesh Reconstruction with Transformers", CVPR, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2012.09760)][[PyTorch](https://github.com/microsoft/MeshTransformer)] 
    * **PRTR**: "Pose Recognition with Cascade Transformers", CVPR, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2104.06976)][[PyTorch](https://github.com/mlpc-ucsd/PRTR)]
    * **PoseFormer**: "3D Human Pose Estimation with Spatial and Temporal Transformers", arXiv, 2021 (*UNC*). [[Paper](https://arxiv.org/abs/2103.10455)][[PyTorch](https://github.com/zczcwh/PoseFormer)]
    * **POET**: "End-to-End Trainable Multi-Instance Pose Estimation with Transformers", arXiv, 2021 (*EPFL*). [[Paper](https://arxiv.org/abs/2103.12115)]
    * **Lifting-Transformer**: "Lifting Transformer for 3D Human Pose Estimation in Video", arXiv, 2021 (*Peking*). [[Paper](https://arxiv.org/abs/2103.14304)]
    * **TFPose**: "TFPose: Direct Human Pose Estimation with Transformers", arXiv, 2021 (*The University of Adelaide*). [[Paper](https://arxiv.org/abs/2103.15320)][[PyTorch](https://github.com/aim-uofa/AdelaiDet/)]
    * **Skeletor**: "Skeletor: Skeletal Transformers for Robust Body-Pose Estimation", arXiv, 2021 (*University of Surrey*). [[Paper](https://arxiv.org/abs/2104.11712)]
    * **HandsFormer**: "HandsFormer: Keypoint Transformer for Monocular 3D Pose Estimation of Hands and Object in Interaction", arXiv, 2021 (*Graz University of Technology*). [[Paper](https://arxiv.org/abs/2104.14639)]
    * **Mesh-Graphormer**: "Mesh Graphormer", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2104.00272)]
    * **TTP**: "Test-Time Personalization with a Transformer for Human Pose Estimation", arXiv, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2107.02133)][[Website](https://liyz15.github.io/TTP/)]
    * **GraFormer**: "GraFormer: Graph Convolution Transformer for 3D Pose Estimation", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2109.08364)]
* Others:
    * **TAPE**: "Transformer Guided Geometry Model for Flow-Based Unsupervised Visual Odometry", arXiv, 2020 (*Tianjing University*). [[Paper](https://arxiv.org/abs/2101.02143)]

##### Tracking
* **TransTrack**: "TransTrack: Multiple-Object Tracking with Transformer",arXiv, 2020 (*HKU + ByteDance*) . [[Paper](https://arxiv.org/abs/2012.15460)][[PyTorch](https://github.com/PeizeSun/TransTrack)]
* **TransformerTrack**: "Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking", CVPR, 2021 (*USTC*). [[Paper](https://arxiv.org/abs/2103.11681)][[PyTorch](https://github.com/594422814/TransformerTrack)]
* **TransT**: "Transformer Tracking", CVPR, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2103.15436)][[PyTorch](https://github.com/chenxin-dlut/TransT)]
* **STARK**: "Learning Spatio-Temporal Transformer for Visual Tracking", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2103.17154)][[PyTorch](https://github.com/researchmm/Stark)]
* **HiFT**: "HiFT: Hierarchical Feature Transformer for Aerial Tracking", ICCV, 2021 (*Tongji University*). [[Paper](https://arxiv.org/abs/2108.00202)][[PyTorch](https://github.com/vision4robotics/HiFT)]
* **TrackFormer**: "TrackFormer: Multi-Object Tracking with Transformers", arXiv, 2021 (*Facebook*). [[Paper](https://arxiv.org/abs/2101.02702)]
* **TransCenter**: "TransCenter: Transformers with Dense Queries for Multiple-Object Tracking", arXiv, 2021 (*INRIA + MIT*). [[Paper](https://arxiv.org/abs/2103.15145)]
* **TransMOT**: "TransMOT: Spatial-Temporal Graph Transformer for Multiple Object Tracking", arXiv, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2104.00194)]
* **TREG**: "Target Transformed Regression for Accurate Tracking", arXiv, 2021 (*Nanjing University*). [[Paper](https://arxiv.org/abs/2104.00403)][[Code (in construction)](https://github.com/MCG-NJU/TREG)]
* **MOTR**: "MOTR: End-to-End Multiple-Object Tracking with TRansformer", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2105.03247)][[PyTorch](https://github.com/megvii-model/MOTR)]
* **TrTr**: "TrTr: Visual Tracking with Transformer", arXiv, 2021 (*University of Tokyo*). [[Paper](https://arxiv.org/abs/2105.03817)][[Pytorch](https://github.com/tongtybj/TrTr)]
* **RelationTrack**: "RelationTrack: Relation-aware Multiple Object Tracking with Decoupled Representation", arXiv, 2021 (*Huazhong Univerisity of Science and Technology*). [[Paper](https://arxiv.org/abs/2105.04322)]

##### Re-ID
* **PAT**: "Diverse Part Discovery: Occluded Person Re-Identification With Part-Aware Transformer", CVPR, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2106.04095)]
* **HAT**: "HAT: Hierarchical Aggregation Transformers for Person Re-identification", ACMMM, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2107.05946)]
* **TransReID**: "TransReID: Transformer-based Object Re-Identification", ICCV, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2102.04378)][[PyTorch](https://github.com/heshuting555/TransReID)]
* **Pirt**: "Pose-guided Inter- and Intra-part Relational Transformer for Occluded Person Re-Identification", ACMMM, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2109.03483)]
* **STT**: "Spatiotemporal Transformer for Video-based Person Re-identification", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2103.16469)] 
* **AAformer**: "AAformer: Auto-Aligned Transformer for Person Re-Identification", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2104.00921)]
* **TMT**: "A Video Is Worth Three Views: Trigeminal Transformers for Video-based Person Re-identification", arXiv, 2021 (*Dalian University of Technology*). [[Paper](https://arxiv.org/abs/2104.01745)]
* **TransMatcher**: "Transformer-Based Deep Image Matching for Generalizable Person Re-identification", arXiv, 2021 (*IIAI*). [[Paper](https://arxiv.org/abs/2105.14432)]
* **LA-Transformer**: "Person Re-Identification with a Locally Aware Transformer", arXiv, 2021 (*University of Maryland Baltimore County*). [[Paper](https://arxiv.org/abs/2106.03720)]
* **DRL-Net**: "Learning Disentangled Representation Implicitly via Transformer for Occluded Person Re-Identification", arXiv, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2107.02380)]
* **GiT**: "GiT: Graph Interactive Transformer for Vehicle Re-identification", arXiv, 2021 (*Huaqiao University*). [[Paper](https://arxiv.org/abs/2107.05475)]

##### Face
* **FAU-Transformer**: "Facial Action Unit Detection With Transformers", CVPR, 2021 (*Rakuten Institute of Technology*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Jacob_Facial_Action_Unit_Detection_With_Transformers_CVPR_2021_paper.html)]
* **Clusformer**: "Clusformer: A Transformer Based Clustering Approach to Unsupervised Large-Scale Face and Visual Landmark Recognition", CVPR, 2021 (*VinAI Research, Vietnam*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Nguyen_Clusformer_A_Transformer_Based_Clustering_Approach_to_Unsupervised_Large-Scale_Face_CVPR_2021_paper.html)][[Code (in construction)](https://github.com/VinAIResearch/Clusformer)]
* **TransFER**: "TransFER: Learning Relation-aware Facial Expression Representations with Transformers", ICCV, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2108.11116)]
* **ViT-Face**: "Face Transformer for Recognition", arXiv, 2021 (*Beijing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2103.14803)]
* **CVT-Face**: "Robust Facial Expression Recognition with Convolutional Visual Transformers", arXiv, 2021 (*Hunan University*). [[Paper](https://arxiv.org/abs/2103.16854)]
* **FAT**: "Facial Attribute Transformers for Precise and Robust Makeup Transfer", arXiv, 2021 (*University of Rochester*). [[Paper](https://arxiv.org/abs/2104.02894)]
* **TransRPPG**: "TransRPPG: Remote Photoplethysmography Transformer for 3D Mask Face Presentation Attack Detection", arXiv, 2021 (*University of Oulu*). [[Paper](https://arxiv.org/abs/2104.07419)]
* **FaceT**: "Learning to Cluster Faces via Transformer", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2104.11502)]
* **VidFace**: "VidFace: A Full-Transformer Solver for Video FaceHallucination with Unaligned Tiny Snapshots", arXiv, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2105.14954)]
* **MViT**: "MViT: Mask Vision Transformer for Facial Expression Recognition in the wild", arXiv, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2106.04520)]
* **FAA**: "Shuffle Transformer with Feature Alignment for Video Face Parsing", arXiv, 2021 (*Tencent*). [[Paper](https://arxiv.org/abs/2106.08650)]
* **ViT-SE**: "Learning Vision Transformer with Squeeze and Excitation for Facial Expression Recognition", arXiv, 2021 (*CentraleSupélec, France*). [[Paper](https://arxiv.org/abs/2107.03107)]
* **EST**: "Expression Snippet Transformer for Robust Video-based Facial Expression Recognition", arXiv, 2021 (*China University of Geosciences*). [[Paper](https://arxiv.org/abs/2109.08409)][[PyTorch](https://anonymous.4open.science/r/ATSE-C58B)]


### Low-level Vision Tasks
##### Image Restoration (e.g. super-resolution, image denoising, demosaicing, compression artifacts reduction, etc.):
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
* **SDNet**: "SDNet: multi-branch for single image deraining using swin", arXiv, 2021 (*Xinjiang University*). [[Paper](https://arxiv.org/abs/2105.15077)][[Code (in construction)](https://github.com/H-tfx/SDNet)]
* **Uformer**: "Uformer: A General U-Shaped Transformer for Image Restoration", arXiv, 2021 (*University of Science and Technology of China*). [[Paper](https://arxiv.org/abs/2106.03106)]
* **FPAN**: "Feedback Pyramid Attention Networks for Single Image Super-Resolution", arXiv, 2021 (*Nanjing University of Science and Technology*). [[Paper](https://arxiv.org/abs/2106.06966)]
* **VSR-Transformer**: "Video Super-Resolution Transformer", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2106.06847)]
* **ATTSF**: "Attention! Stay Focus!", arXiv, 2021 (*BridgeAI, Seoul*). [[Paper](https://arxiv.org/abs/2104.07925)][[Tensorflow](https://github.com/tuvovan/ATTSF)]
* **LFT**: "Light Field Image Super-Resolution with Transformers", arXiv, 2021 (*National University of Defense Technology, China*). [[Paper](https://arxiv.org/abs/2108.07597)][[PyTorch (in construction)](https://github.com/ZhengyuLiang24/LFT)]
* **MANA**: "Memory-Augmented Non-Local Attention for Video Super-Resolution", arXiv, 2021 (*JD*). [[Paper](https://arxiv.org/abs/2108.11048)]
* **ESRT**: "Efficient Transformer for Single Image Super-Resolution", arXiv, 2021 (*Peking University*). [[Paper](https://arxiv.org/abs/2108.11084)]
* **Fusformer**: "Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution", arXiv, 2021 (*University of Electronic Science and Technology of China*). [[Paper](https://arxiv.org/abs/2109.02079)]
* **HyLoG-ViT**: "Hybrid Local-Global Transformer for Image Dehazing", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2109.07100)]
* **TANet**: "TANet: A new Paradigm for Global Face Super-resolution via Transformer-CNN Aggregation Network", arXiv, 2021 (*Wuhan Institute of Technology*). [[Paper](https://arxiv.org/abs/2109.08174)]

##### Colorization:
* **ColTran**: "Colorization Transformer", ICLR, 2021 (*Google*). [[Paper](https://openreview.net/forum?id=5NA1PinlGFu)][[Tensorflow](https://github.com/google-research/google-research/tree/master/coltran)]
* **ViT-I-GAN**: "ViT-Inception-GAN for Image Colourising", arXiv, 2021 (*D.Y Patil College of Engineering, India*). [[Paper](https://arxiv.org/abs/2106.06321)]

##### Inpainting:
* **Contexual-Attention**: "Generative Image Inpainting with Contextual Attention", CVPR, 2018 (*UIUC*). [[Paper](https://arxiv.org/abs/1801.07892)][[Tensorflow](https://github.com/JiahuiYu/generative_inpainting)]
* **PEN-Net**: "Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting", CVPR, 2019 (*Microsoft*). [[Paper](https://arxiv.org/abs/1904.07475)][[PyTorch](https://github.com/researchmm/PEN-Net-for-Inpainting)]
* **Copy-Paste**: "Copy-and-Paste Networks for Deep Video Inpainting", ICCV, 2019 (*Yonsei University*). [[Paper](https://arxiv.org/abs/1908.11587)][[PyTorch](https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting)]
* **Onion-Peel**: "Onion-Peel Networks for Deep Video Completion", ICCV, 2019 (*Yonsei University*). [[Paper](https://arxiv.org/abs/1908.08718)][[PyTorch](https://github.com/seoungwugoh/opn-demo)]
* **STTN**: "Learning Joint Spatial-Temporal Transformations for Video Inpainting", ECCV, 2020 (*Microsoft*). [[Paper](https://arxiv.org/abs/2007.10247)][[PyTorch](https://github.com/researchmm/STTN)]
* **FuseFormer**: "FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting", ICCV, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2109.02974)][[Code (in construction)](https://github.com/ruiliu-ai/FuseFormer)]
* **DSTT**: "Decoupled Spatial-Temporal Transformer for Video Inpainting", arXiv, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2104.06637)][[Code (in construction)](https://github.com/ruiliu-ai/DSTT)]

##### Completion:
* **ICT**: "High-Fidelity Pluralistic Image Completion with Transformers", ICCV, 2021 (*CUHK*). [[Paper](https://arxiv.org/abs/2103.14031)][[PyTorch](https://github.com/raywzy/ICT)][[Website](http://raywzy.com/ICT/)]
* **TFill**: "TFill: Image Completion via a Transformer-Based Architecture", arXiv, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2104.00845)][[Code (in construction)](https://github.com/lyndonzheng/TFill)]
* **BAT-Fill**: "Diverse Image Inpainting with Bidirectional and Autoregressive Transformers", arXiv, 2021 (*NTU Singapore*). [[Paper](https://arxiv.org/abs/2104.12335)]

##### Image Generation:
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
* **TransGAN**: "TransGAN: Two Transformers Can Make One Strong GAN", arXiv, 2021 (*UT Austin*). [[Paper](https://arxiv.org/abs/2102.07074)][[PyTorch](https://github.com/VITA-Group/TransGAN)]
* **SceneFormer**: "SceneFormer: Indoor Scene Generation with Transformers", arXiv, 2021 (*TUM*). [[Paper](https://arxiv.org/abs/2012.09793)]
* **VTGAN**: "VTGAN: Semi-supervised Retinal Image Synthesis and Disease Prediction using Vision Transformers", arXiv, 2021 (*University of Nevada, Reno*). [[Paper](https://arxiv.org/abs/2104.06757)]
* **Geometry-Free**: "Geometry-Free View Synthesis: Transformers and no 3D Priors", arXiv, 2021 (*Heidelberg University*). [[Paper](https://arxiv.org/abs/2104.07652)][[Code (in construction)](https://github.com/CompVis/geometry-free-view-synthesis)]
* **SNGAN**: "Combining Transformer Generators with Convolutional Discriminators", arXiv, 2021 (*Fraunhofer ITWM*). [[Paper](https://arxiv.org/abs/2105.10189)]
* **StyTr2**: "StyTr^2: Unbiased Image Style Transfer with Transformers", arXiv, 2021 (*CAS*). [[Paper](https://arxiv.org/abs/2105.14576)]
* **iLAT**: "The Image Local Autoregressive Transformer", arXiv, 2021 (*Fudan*). [[Paper](https://arxiv.org/abs/2106.02514)]
* **HiT**: "Improved Transformer for High-Resolution GANs", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2106.07631)]
* **Styleformer**: "Styleformer: Transformer based Generative Adversarial Networks with Style Vector", arXiv, 2021 (*Seoul National University*). [[Paper](https://arxiv.org/abs/2106.07023)][[PyTorch](https://github.com/Jeeseung-Park/Styleformer)]
* **Invertible Attention**: "Invertible Attention", arXiv, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2106.09003)]
* **GPA**: "Grid Partitioned Attention: Efficient Transformer Approximation with Inductive Bias for High Resolution Detail Generation", arXiv, 2021 (*Zalando Research, Germany*). [[Paper](https://arxiv.org/abs/2107.03742)][[PyTorch (in construction)](https://github.com/zalandoresearch/gpa)]
* **ViTGAN**: "ViTGAN: Training GANs with Vision Transformers", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2107.04589)]

##### Video Generation:
* **Subscale**: "Scaling Autoregressive Video Models", ICLR, 2020 (*Google*). [[Paper](https://openreview.net/forum?id=rJgsskrFwH)][[Website](https://sites.google.com/view/video-transformer-samples)]
* **ConvTransformer**: "ConvTransformer: A Convolutional Transformer Network for Video Frame Synthesis", arXiv, 2020 (*Southeast University*). [[Paper](https://arxiv.org/abs/2011.10185)]
* **OCVT**: "Generative Video Transformer: Can Objects be the Words?", ICML, 2021 (*Rutgers University*). [[Paper](http://proceedings.mlr.press/v139/wu21h.html)]
* **AIST++**: "Learn to Dance with AIST++: Music Conditioned 3D Dance Generation", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2101.08779)][[Code](https://github.com/google/aistplusplus_api)][[Website](https://google.github.io/aichoreographer/)]
* **VideoGPT**: "VideoGPT: Video Generation using VQ-VAE and Transformers", arXiv, 2021 (*Berkeley*). [[Paper](https://arxiv.org/abs/2104.10157)][[PyTorch](https://github.com/wilson1yan/VideoGPT)][[Website](https://wilson1yan.github.io/videogpt/index.html)]

##### Others:
* **MS-Unet**: "Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer", arXiv, 2021 (*MEGVII*). [[Paper](https://arxiv.org/abs/2109.08024)]


### Reinforcement Learning
##### Navigation
* **VTNet**: "VTNet: Visual Transformer Network for Object Goal Navigation", ICLR, 2021 (*ANU*). [[Paper](https://openreview.net/forum?id=DILxQP08O3B)]
* **TransFuser**: "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving", CVPR, 2021 (*MPI*). [[Paper](https://arxiv.org/abs/2104.09224)][[PyTorch](https://github.com/autonomousvision/transfuser)]
* **CMTP**: "Topological Planning With Transformers for Vision-and-Language Navigation", CVPR, 2021 (*Stanford*). [[Paper](https://arxiv.org/abs/2012.05292)]
* **VLN-BERT**: "VLN-BERT: A Recurrent Vision-and-Language BERT for Navigation", CVPR, 2021 (*ANU*). [[Paper](https://arxiv.org/abs/2011.13922)][[PyTorch](https://github.com/YicongHong/Recurrent-VLN-BERT)]
* **E.T.**: "Episodic Transformer for Vision-and-Language Navigation", arXiv, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2105.06453)][[PyTorch](https://github.com/alexpashevich/E.T.)]

##### Others
* **SVEA**: "Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation", arXiv, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2107.00644)][[GitHub](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)][[Website](https://nicklashansen.github.io/SVEA/)]
* **LocoTransformer**: "Learning Vision-Guided Quadrupedal Locomotion End-to-End with Cross-Modal Transformers", arXiv, 2021 (*UCSD*). [[Paper](https://arxiv.org/abs/2107.03996)][[Website](https://rchalyang.github.io/LocoTransformer/)]


### Transfer Learning / Self-Supervised Learning / Few-Shot
* **CrossTransformer**: "CrossTransformers: spatially-aware few-shot transfer", NeurIPS, 2020 (*DeepMind*). [[Paper](https://arxiv.org/abs/2007.11498)][[Tensorflow](https://github.com/google-research/meta-dataset)]
* **URT**: "A Universal Representation Transformer Layer for Few-Shot Image Classification", ICLR, 2021 (*Mila*). [[Paper](https://openreview.net/forum?id=04cII6MumYV)][[PyTorch](https://github.com/liulu112601/URT)]
* **TRX**: "Temporal-Relational CrossTransformers for Few-Shot Action Recognition", CVPR, 2021 (*University of Bristol*). [[Paper](https://arxiv.org/abs/2101.06184)][[PyTorch](https://github.com/tobyperrett/TRX)]
* **Few-shot-Transformer**: "Few-Shot Transformation of Common Actions into Time and Space", arXiv, 2021 (*University of Amsterdam*). [[Paper](https://arxiv.org/abs/2104.02439)]


### Medical
* Segmentation:
    * **Cross-Transformer**: "The entire network structure of Crossmodal Transformer", ICBSIP, 2021 (*Capital Medical University*). [[Paper](https://arxiv.org/abs/2104.14273)]
    * **Segtran**: "Medical Image Segmentation using Squeeze-and-Expansion Transformers", IJCAI, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2105.09511)]
    * **i-ViT**: "Instance-based Vision Transformer for Subtyping of Papillary Renal Cell Carcinoma in Histopathological Image", MICCAI, 2021 (*Xi'an Jiaotong University*). [[Paper](https://arxiv.org/abs/2106.12265)][[PyTorch](https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer)][[Website](https://dataset.chenli.group/home/prcc-subtyping)]
    * **UTNet**: "UTNet: A Hybrid Transformer Architecture for Medical Image Segmentation", MICCAI, 2021 (*Rutgers*). [[Paper](https://arxiv.org/abs/2107.00781)]
    * **MCTrans**: "Multi-Compound Transformer for Accurate Biomedical Image Segmentation", MICCAI, 2021 (*HKU + CUHK*). [[Paper](https://arxiv.org/abs/2106.14385)][[Code (in construction)](https://github.com/JiYuanFeng/MCTrans)]
    * **Polyformer**: "Few-Shot Domain Adaptation with Polymorphic Transformers", MICCAI, 2021 (*A\*STAR*). [[Paper](https://arxiv.org/abs/2107.04805)][[PyTorch](https://github.com/askerlee/segtran)]
    * **STN**: "Automatic size and pose homogenization with spatial transformer network to improve and accelerate pediatric segmentation", ISBI, 2021 (*Institut Polytechnique de Paris*). [[Paper](https://arxiv.org/abs/2107.02655)]
    * **MedT**: "Medical Transformer: Gated Axial-Attention for Medical Image Segmentation", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2102.10662)][[PyTorch](https://github.com/jeya-maria-jose/Medical-Transformer)]
    * **Convolution-Free**: "Convolution-Free Medical Image Segmentation using Transformers", arXiv, 2021 (*Harvard*). [[Paper](https://arxiv.org/abs/2102.13645)]
    * **CoTR**: "CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation", arXiv, 2021 (*Northwestern Polytechnical University*). [[Paper](https://arxiv.org/abs/2103.03024)][[PyTorch](https://github.com/YtongXie/CoTr)]
    * **TransBTS**: "TransBTS: Multimodal Brain Tumor Segmentation Using Transformer", arXiv, 2021 (*University of Science and Technology Beijing*). [[Paper](https://arxiv.org/abs/2103.04430)][[PyTorch](https://github.com/Wenxuan-1119/TransBTS)]
    * **SpecTr**: "SpecTr: Spectral Transformer for Hyperspectral Pathology Image Segmentation", arXiv, 2021 (*East China Normal University*). [[Paper](https://arxiv.org/abs/2103.03604)][[Code (in construction)](https://github.com/hfut-xc-yun/SpecTr)]
    * **U-Transformer**: "U-Net Transformer: Self and Cross Attention for Medical Image Segmentation", arXiv, 2021 (*CEDRIC*). [[Paper](https://arxiv.org/abs/2103.06104)]
    * **UNETR**: "UNETR: Transformers for 3D Medical Image Segmentation", arXiv, 2021 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2103.10504)]
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
    * **UCTransNet**: "UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer", arXiv, 2021 (*Northeastern University, China*). [[Paper](https://arxiv.org/abs/2109.04335)][[PyTorch](https://github.com/McGregorWwww/UCTransNet)]
    * **MISSFormer**: "MISSFormer: An Effective Medical Image Segmentation Transformer", arXiv, 2021 (*Beijing University of Posts and Telecommunications*). [[Paper](https://arxiv.org/abs/2109.07162)]
    * **TUnet**: "Transformer-Unet: Raw Image Processing with Unet", arXiv, 2021 (*Beijing Zoezen Robot + Beihang University*). [[Paper](https://arxiv.org/abs/2109.08417)]
* Classification:
    * **TransMed**: "TransMed: Transformers Advance Multi-modal Medical Image Classification", arXiv, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2103.05940)]
    * **CXR-ViT**: "Vision Transformer using Low-level Chest X-ray Feature Corpus for COVID-19 Diagnosis and Severity Quantification", arXiv, 2021 (*KAIST*). [[Paper](https://arxiv.org/abs/2104.07235)]
    * **ViT-TSA**: "Shoulder Implant X-Ray Manufacturer Classification: Exploring with Vision Transformer", arXiv, 2021 (*Queen’s University*). [[Paper](https://arxiv.org/abs/2104.07667)]
    * **GasHis-Transformer**: "GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification", arXiv, 2021 (*Northeastern University*). [[Paper](https://arxiv.org/abs/2104.14528)]
    * **POCFormer**: "POCFormer: A Lightweight Transformer Architecture for Detection of COVID-19 Using Point of Care Ultrasound", arXiv, 2021 (*The Ohio State University*). [[Paper](https://arxiv.org/abs/2105.09913)]
    * **TransMIL**: "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classication", arXiv, 2021 (*Tsinghua University*). [[Paper](https://arxiv.org/abs/2106.00908)]
    * **COVID-ViT**: "COVID-VIT: Classification of COVID-19 from CT chest images based on vision transformer models", arXiv, 2021 (*Middlesex University, UK*). [[Paper](https://arxiv.org/abs/2107.01682)][[PyTorch](https://github.com/xiaohong1/COVID-ViT)]
    * **EEG-ConvTransformer**: "EEG-ConvTransformer for Single-Trial EEG based Visual Stimuli Classification", arXiv, 2021 (*IIT Ropar*). [[Paper](https://arxiv.org/abs/2107.03983)]
    * **CCAT**: "Visual Transformer with Statistical Test for COVID-19 Classification", arXiv, 2021 (*NCKU*). [[Paper](https://arxiv.org/abs/2107.05334)]
* Detection:
    * **COTR**: "COTR: Convolution in Transformer Network for End to End Polyp Detection", arXiv, 2021 (*Fuzhou University*). [[Paper](https://arxiv.org/abs/2105.10925)]
    * **TR-Net**: "Transformer Network for Significant Stenosis Detection in CCTA of Coronary Arteries", arXiv, 2021 (*Harbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2107.03035)]
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
    * **?**: "Is it Time to Replace CNNs with Transformers for Medical Images?", ICCVW, 2021 (*KTH, Sweden*). [[Paper](https://arxiv.org/abs/2108.09038)]
    * **Global-Local-Transformer**: "Global-Local Transformer for Brain Age Estimation", IEEE Transactions on Medical Imaging, 2021 (*Harvard*). [[Paper](https://arxiv.org/abs/2109.01663)][[PyTorch](https://github.com/shengfly/global-local-transformer)]
    * **DeepProg**: "DeepProg: A Transformer-based Framework for Predicting Disease Prognosis", arXiv, 2021 (*University of Oulu*). [[Paper](https://arxiv.org/abs/2104.03642)]
    * **Medical-Transformer**: "Medical Transformer: Universal Brain Encoder for 3D MRI Analysis", arXiv, 2021 (*Korea University*). [[Paper](https://arxiv.org/abs/2104.13633)]
    * **PTNet**: "PTNet: A High-Resolution Infant MRI Synthesizer Based on Transformer", arXiv, 2021 (* Columbia *). [[Paper](https://arxiv.org/ftp/arxiv/papers/2105/2105.13993.pdf)]
    * **ResViT**: "ResViT: Residual vision transformers for multi-modal medical image synthesis", arXiv, 2021 (*Bilkent University, Turkey*). [[Paper](https://arxiv.org/abs/2106.16031)]
    * **RATCHET**: "RATCHET: Medical Transformer for Chest X-ray Diagnosis and Reporting", arXiv, 2021 (*ICL*). [[Paper](https://arxiv.org/abs/2107.02104)]
    * **Eformer**: "Eformer: Edge Enhancement based Transformer for Medical Image Denoising", arXiv, 2021 (*BITS Pilani, India*). [[Paper](https://arxiv.org/abs/2109.08044)]


### Others
* Feature Matching:
    * **LoFTR**: "LoFTR: Detector-Free Local Feature Matching with Transformers", CVPR, 2021 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2104.00680)][[PyTorch](https://github.com/zju3dv/LoFTR)][[Website](https://zju3dv.github.io/loftr/)]
    * **COTR**: "COTR: Correspondence Transformer for Matching Across Images", ICCV, 2021 (*UBC*). [[Paper](https://arxiv.org/abs/2103.14167)]
    * **CATs**: "Semantic Correspondence with Transformers", arXiv, 2021 (*Yonsei University + Korea University*). [[Paper](https://arxiv.org/abs/2106.02520)][[Code (in construction)](https://github.com/SunghwanHong/CATs)]
* Multi-label:
    * **C-Tran**: "General Multi-label Image Classification with Transformers", CVPR, 2021 (*University of Virginia*). [[Paper](https://arxiv.org/abs/2011.14027)]
    * **MlTr**: "MlTr: Multi-label Classification with Transformer", arXiv, 2021 (*KuaiShou*). [[Paper](https://arxiv.org/abs/2106.06195)]
* Fine-grained:
    * **ViT-FGVC**: "Exploring Vision Transformers for Fine-grained Classification", CVPRW, 2021 (*Universidad de Valladolid*). [[Paper](https://arxiv.org/abs/2106.10587)]
    * **TransFG**: "TransFG: A Transformer Architecture for Fine-grained Recognition", arXiv, 2021 (*Johns Hopkins*). [[Paper](https://arxiv.org/abs/2103.07976)][[PyTorch](https://github.com/TACJu/TransFG)]
    * **FFVT**: "Feature Fusion Vision Transformer Fine-Grained Visual Categorization", arXiv, 2021 (*Griffith University, Australia*). [[Paper](https://arxiv.org/abs/2107.02341)]
    * **TPSKG**: "Transformer with Peak Suppression and Knowledge Guidance for Fine-grained Image Recognition", arXiv, 2021 (*Beihang University*). [[Paper](https://arxiv.org/abs/2107.06538)]
* Image Quality Assessment:
    * **TRIQ**: "Transformer for Image Quality Assessment", arXiv, 2020 (*NORCE*). [[Paper](https://arxiv.org/abs/2101.01097)][[Tensorflow-Keras](https://github.com/junyongyou/triq)]
    * **IQT**: "Perceptual Image Quality Assessment with Transformers", CVPRW, 2021 (*LG*). [[Paper](https://arxiv.org/abs/2104.14730)][[Code (in construction)](https://github.com/manricheon/IQT)]
    * **MUSIQ**: "MUSIQ: Multi-scale Image Quality Transformer", ICCV, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2108.05997)]
    * **TReS**: "No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency", arXiv, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2108.06858)]
* Image Retrieval:
    * **ViT-Retrieval**: "Investigating the Vision Transformer Model for Image Retrieval Tasks", arXiv, 2021 (*Democritus University of Thrace*). [[Paper](https://arxiv.org/abs/2101.03771)]
    * **IRT**: "Training Vision Transformers for Image Retrieval", arXiv, 2021 (*Facebook + INRIA*). [[Paper](https://arxiv.org/abs/2102.05644)]
    * **RRT**: "Instance-level Image Retrieval using Reranking Transformers", arXiv, 2021 (*University of Virginia*). [[Paper](https://arxiv.org/abs/2103.12236)]
    * **TransHash**: "TransHash: Transformer-based Hamming Hashing for Efficient Image Retrieval", arXiv, 2021 (*Shanghai Jiao Tong University*). [[Paper](https://arxiv.org/abs/2105.01823)]
* NAS:
    * **HR-NAS**: "HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers", CVPR, 2021 (*HKU*). [[Paper](https://arxiv.org/abs/2106.06560)][[PyTorch](https://github.com/dingmyu/HR-NAS)]
    * **CATE**: "CATE: Computation-aware Neural Architecture Encoding with Transformers", ICML, 2021 (*Michigan State University*). [[Paper](https://arxiv.org/abs/2102.07108)]
    * **AutoFormer**: "AutoFormer: Searching Transformers for Visual Recognition", ICCV, 2021 (*Microsoft*). [[Paper](https://arxiv.org/abs/2107.00651)][[Code (in construction)](https://github.com/microsoft/AutoML)]
    * **BossNAS**: "BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search", arXiv, 2021 (*Monash University*). [[Paper](https://arxiv.org/abs/2103.12424)][[PyTorch](https://github.com/changlin31/BossNAS)]
    * **ViTAS**: "Vision Transformer Architecture Search", arXiv, 2021 (*The University of Sydney + SenseTime*). [[Paper](https://arxiv.org/abs/2106.13700)]
    * **GLiT**: "GLiT: Neural Architecture Search for Global and Local Image Transformer", arXiv, 2021 (*The University of Sydney + SenseTime*). [[Paper](https://arxiv.org/abs/2107.02960)]
    * **PSViT**: "PSViT: Better Vision Transformer via Token Pooling and Attention Sharing", arXiv, 2021 (*The University of Sydney + SenseTime*). [[Paper](https://arxiv.org/abs/2108.03428)]
    * **ViT-ResNAS**: "Searching for Efficient Multi-Stage Vision Transformers", arXiv, 2021 (*MIT*). [[Paper](https://arxiv.org/abs/2109.00642)][[PyTorch](https://github.com/yilunliao/vit-search)]
* Deepfake-related:
    * **?**: "Video Transformer for Deepfake Detection with Incremental Learning", ACMMM, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2108.05307)]
    * **CViT**: "Deepfake Video Detection Using Convolutional Vision Transformer", arXiv, 2021 (*Jimma University*). [[Paper](https://arxiv.org/abs/2102.11126)]
    * **ViT-Distill**: "Deepfake Detection Scheme Based on Vision Transformer and Distillation", arXiv, 2021 (*Sookmyung Women’s University*). [[Paper](https://arxiv.org/abs/2104.01353)]
    * **M2TR**: "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection", arXiv, 2021 (*Fudan University*). [[Paper](https://arxiv.org/abs/2104.09770)]
    * **Cross-ViT**: "Combining EfficientNet and Vision Transformers for Video Deepfake Detection", arXiv, 2021 (*University of Pisa*). [[Paper](https://arxiv.org/abs/2107.02612)]
* 3D Reconstruction:
    * **PlaneTR**: "PlaneTR: Structure-Guided Transformers for 3D Plane Recovery", ICCV, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2107.13108)][[PyTorch](https://github.com/IceTTTb/PlaneTR3D)]
    * **VolT**: "Multi-view 3D Reconstruction with Transformer", arXiv, 2021 (*University of British Columbia*). [[Paper](https://arxiv.org/abs/2103.12957)]
    * **LegoFormer**: "LegoFormer: Transformers for Block-by-Block Multi-view 3D Reconstruction", arXiv, 2021 (*TUM + Google*). [[Paper](https://arxiv.org/abs/2106.12102)]
    * **TransformerFusion**: "TransformerFusion: Monocular RGB Scene Reconstruction using Transformers", arXiv, 2021 (*TUM*). [[Paper](https://arxiv.org/abs/2107.02191)][[Website](https://aljazbozic.github.io/transformerfusion/)]
* Trajectory Prediction:
    * **mmTransformer**: "Multimodal Motion Prediction with Stacked Transformers", CVPR, 2021 (*CUHK + SenseTime*). [[Paper](https://arxiv.org/abs/2103.11624)][[Code (in construction)](https://github.com/decisionforce/mmTransformer)][[Website](https://decisionforce.github.io/mmTransformer/)]
    * **AgentFormer**: "AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting", ICCV, 2021 (*CMU*). [[Paper](https://arxiv.org/abs/2103.14023)][[Code (in construction)](https://github.com/Khrylx/AgentFormer)][[Website](https://www.ye-yuan.com/agentformer/)]
* 3D Motion Synthesis:
    * **ACTOR**: "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE", ICCV, 2021 (*Univ Gustave Eiffel*). [[Paper](https://arxiv.org/abs/2104.05670)][[PyTorch](https://github.com/Mathux/ACTOR)][[Website](https://imagine.enpc.fr/~petrovim/actor/)]
* Fashion:
    * **Kaleido-BERT**: "Kaleido-BERT: Vision-Language Pre-training on Fashion Domain", CVPR, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2103.16110)][[Tensorflow](https://github.com/mczhuge/Kaleido-BERT)]
    * **CIT**: "Cloth Interactive Transformer for Virtual Try-On", arXiv, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2104.05519)][[Code (in construction)](https://github.com/Amazingren/CIT)]
* Crowd Counting:
    * **TransCrowd**: "TransCrowd: Weakly-Supervised Crowd Counting with Transformer", arXiv, 2021 (*Huazhong University of Science and Technology*). [[Paper](https://arxiv.org/abs/2104.09116)][[PyTorch (in construction)](https://github.com/dk-liang/TransCrowd)]
    * **TAM-RTM**: "Boosting Crowd Counting with Transformers", arXiv, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2105.10926)]
    * **CC-AV**: "Audio-Visual Transformer Based Crowd Counting", arXiv, 2021 (*University of Kansas*). [[Paper](https://arxiv.org/abs/2109.01926)]
* Pruning:
    * **VTP**: "Visual Transformer Pruning", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2104.08500)]
* Remote Sensing:
    * **DCFAM**: "Transformer Meets DCFAM: A Novel Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images", arXiv, 2021 (*Wuhan University*). [[Paper](https://arxiv.org/abs/2104.12137)]
    * **WiCNet**: "Looking Outside the Window: Wider-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images", arXiv, 2021 (*University of Trento*). [[Paper](https://arxiv.org/abs/2106.15754)]
    * **?**: "Vision Transformers For Weeds and Crops Classification Of High Resolution UAV Images", arXiv, 2021 (*University of Orleans, France*). [[Paper](https://arxiv.org/abs/2109.02716)]
* Place Recognition:
    * **SVT-Net**: "SVT-Net: A Super Light-Weight Network for Large Scale Place Recognition using Sparse Voxel Transformers", arXiv, 2021 (*Renmin University of China*). [[Paper](https://arxiv.org/abs/2105.00149)]
* Traffic:
    * **ViTAL**: "Novelty Detection and Analysis of Traffic Scenario Infrastructures in the Latent Space of a Vision Transformer-Based Triplet Autoencoder", IV, 2021 (*Technische Hochschule Ingolstadt*). [[Paper](https://arxiv.org/abs/2105.01924)]
    * **?**: "Predicting Vehicles Trajectories in Urban Scenarios with Transformer Networks and Augmented Information", IVS, 2021 (*Universidad de Alcala*). [[Paper](https://arxiv.org/abs/2106.00559)]
* Character Recognition:
    * **BTTR**: "Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer", arXiv, 2021 (*Peking*). [[Paper](https://arxiv.org/abs/2105.02412)]
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
    * **LocalTrans**: "LocalTrans: A Multiscale Local Transformer Network for Cross-Resolution Homography Estimation", arXiv, 2021 (*Tsinghua*). [[Paper](https://arxiv.org/abs/2106.04067)]
* Time Series:
    * **MissFormer**: "MissFormer: (In-)attention-based handling of missing observations for trajectory filtering and prediction", arXiv, 2021 (*Fraunhofer IOSB, Germany*). [[Paper](https://arxiv.org/abs/2106.16009)]
* Geo-Localization:
    * **EgoTR**: "Cross-view Geo-localization with Evolving Transformer", arXiv, 2021 (*Shenzhen University*). [[Paper](https://arxiv.org/abs/2107.00842)]
* Out-Of-Distribution:
    * **OODformer**: "OODformer: Out-Of-Distribution Detection Transformer", arXiv, 2021 (*LMU Munich*). [[Paper](https://arxiv.org/abs/2107.08976)]
* Layoyt Generation:
    * **VTN**: "Variational Transformer Networks for Layout Generation", CVPR, 2021 (*Google*). [[Paper](https://arxiv.org/abs/2104.02416)]
    * **LayoutTransformer**: "LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity", CVPR, 2021 (*NTU*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html)][[PyTorch](https://github.com/davidhalladay/LayoutTransformer)]
* Scene Graph:
    * **BGT-Net**: "BGT-Net: Bidirectional GRU Transformer Network for Scene Graph Generation", CVPRW, 2021 (*ETHZ*). [[Paper](https://arxiv.org/abs/2109.05346)]
    * **STTran**: "Spatial-Temporal Transformer for Dynamic Scene Graph Generation", ICCV, 2021 (*Leibniz University Hannover, Germany*). [[Paper](https://arxiv.org/abs/2107.12309)]
* Zero-Shot:
    * **ViT-ZSL**: "Multi-Head Self-Attention via Vision Transformer for Zero-Shot Learning", IMVIP, 2021 (*University of Exeter, UK*). [[Paper](https://arxiv.org/abs/2108.00045)]
* Domain Adaptation/Generalization:
    * **TransDA**: "Transformer-Based Source-Free Domain Adaptation", arXiv, 2021 (*Haerbin Institute of Technology*). [[Paper](https://arxiv.org/abs/2105.14138)][[PyTorch](https://github.com/ygjwd12345/TransDA)]
    * **TVT**: "TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation", arXiv, 2021 (*UT Arlington + Kuaishou*). [[Paper](https://arxiv.org/abs/2108.05988)]
    * **ResTran**: "Discovering Spatial Relationships by Transformers for Domain Generalization", arXiv, 2021 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2108.10046)]
    * **CDTrans**: "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation", arXiv, 2021 (*Alibaba*). [[Paper](https://arxiv.org/abs/2109.06165)]
* Digital Holography:
    * **?**: "Convolutional Neural Network (CNN) vs Visual Transformer (ViT) for Digital Holography", ICCCR, 2022 (*UBFC, France*). [[Paper](https://arxiv.org/abs/2108.09147)]
* Curriculum Learning:
    * **SSTN**: "Spatial Transformer Networks for Curriculum Learning", arXiv, 2021 (*TU Kaiserslautern, Germany*). [[Paper](https://arxiv.org/abs/2108.09696)]
* 3D Human Texture Estimation:
    * **Texformer**: "3D Human Texture Estimation from a Single Image with Transformers", ICCV, 2021 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2109.02563)][[Website](https://www.mmlab-ntu.com/project/texformer/)]
* Camera Calibration:
    * **CTRL-C**: "CTRL-C: Camera calibration TRansformer with Line-Classification", ICCV, 2021 (*Kakao + Kookmin University*). [[Paper](https://arxiv.org/abs/2109.02259)][[PyTorch](https://github.com/jwlee-vcl/CTRL-C)]
* Animation-related:
    * **AnT**: "The Animation Transformer: Visual Correspondence via Segment Matching", ICCV, 2021 (*Cadmium*). [[Paper](https://arxiv.org/abs/2109.02614)]

---

* [Survey notes (general)](survey/Survey-Visual-Transformer.md)
* References:
    * Survey:
        * "A Survey of Transformers", arXiv, 2021 (*Fudan*). [[Paper](https://arxiv.org/abs/2106.04554)]
        * "A Survey on Visual Transformer", arXiv, 2021 (*Huawei*). [[Paper](https://arxiv.org/abs/2012.12556)]
        * "Transformers in Vision: A Survey", arXiv, 2021. [[Paper](https://arxiv.org/abs/2101.01169)]
        * "Efficient Transformers: A Survey", arXiv, 2020 (*Google*). [[Paper](https://arxiv.org/abs/2009.06732)]
    * Online Resources:
        * [Awesome Visual-Transformer (GitHub)](https://github.com/dk-liang/Awesome-Visual-Transformer)
        * [Awesome Transformer for Vision Resources List (GitHub)](https://github.com/lijiaman/awesome-transformer-for-vision)
