# DualVector

OpenAccess[[pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_DualVector_Unsupervised_Vector_Font_Synthesis_With_Dual-Part_Representation_CVPR_2023_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Liu_DualVector_Unsupervised_Vector_CVPR_2023_supplemental.zip)]
Arxiv[[pdf](https://arxiv.org/abs/2305.10462)]

Code release for DualVector: Unsupervised Vector Font Synthesis with Dual-Part Representation (CVPR 2023)


## Requirements

In general, other versions of the packages listed below may also work, but are not tested. 

Important python related packages
- python 3.8
- pytorch 1.9.0
- torchvision 0.10.0
- [diffvg](https://github.com/BachiLi/diffvg) (Note: please first replace the original `diffvg/pydiffvg/save_svg.py` with [DeepVecFont's save_svg.py](https://github.com/yizhiwang96/deepvecfont/blob/master/data_utils/save_svg.py), and the original `parse_svg.py` with [our parse_svg.py](./eval/diffvg_parse_svg.py) and then install.)
- [svgpathtools](https://github.com/mathandy/svgpathtools) (We use version 1.4.4 and make several modifications. Please replace the source code with [our modified version](./eval/svgpathtools_modified). Basically we add a `forceZ` option to ensure a closed path.)

We recommend you to install these dependencies with `pip`.  You can refer to `requirements.txt` for all the packages' version but some of them may not be used in code. 

Javascript related packages

- npm 8.5.5
- node v17.9.0
- [Paper.js](https://github.com/paperjs/paper.js)

If NPM is already installed, just run

```bash
cd js
npm install
```

## Dataset

Please download it from [here](https://drive.google.com/file/d/1quWgaU7-QiLQ8TNGuGnTJPFsGf4um0pK/view?usp=share_link) and extract it to `data/dvf_png`.

The dataset is the same as that used in [DeepVecFont](https://github.com/yizhiwang96/deepvecfont), a subset from [SVG-VAE](https://github.com/magenta/magenta/tree/main/magenta/models/svg_vae).  But because some of the fonts' lowercase letters are not fully contained by the images, we re-render them to images with a resolution of 1536*1024. The training data required for [Multi-Implicits](https://github.com/preddy5/multi_implicit_fonts) is also included.

## Run

### Train

**Font Reconstruction**

```bash
python train.py --config configs/reconstruct.yaml --name recon --gpu 0
```

This will save the training records under `save/recon`, including the source code, tensorboard log, image visualization at the validation epoch and the checkpoint.

**Font Generation (Font Style Transfer)**

Replace the following path in [generation_latent_guide.yaml](./configs/generation_latent_guide.yaml) with the reconstruction checkpoint:

```yaml
   submodule_config:
    - 
      ckpt: ./save/recon/ckpt/epoch-last.pth           # replace this line with your checkpoint
      ckpt_key: [encoder, img_decoder, decoder, encoder]
      module_name: [latent_encoder, img_decoder, decoder, img_encoder]
      freeze: [true, true, true, true]
```

Then train a model with the latent guidance:

```bash
python train.py --config configs/generation_latent_guide.yaml --name gen_latent --gpu 0
```

Finally resume from this checkpoint and train the whole model. Please select config if said `Model/Optimizer configs are different`.

```
python train.py --config configs/generation.yaml --name gen --gpu 0 --resume save/gen_latent/ckpt/epoch-last.pth
```

### Test

**Font reconstruction**

First synthesize the reconstructed image and initial SVG:

```bash
python eval/eval_reconstruction.py --resume save/recon/ckpt/epoch-last.pth --outdir eval/save/test_recon # initial SVG
```

Then run the contour refinement:

```bash
python eval/post_refinement.py --outdir eval/save/test_recon/refined --input eval/save/test_recon/rec_init/ --fmin 0 --fmax 200 # refinement
```

**Font generation**

Similar with font reconstruction, run the following two steps:

```bash
python eval/eval_generation_multiref.py --outdir eval/save/test_gen/ --resume save/gen/ckpt/epoch-last.pth # initial SVG
python eval/post_refinement.py --outdir eval/save/test_gen/refined --input eval/save/test_gen/rec_init/ --fmin 0 --fmax 200 # refinement
```

**Sampling new fonts**

Specify the number of fonts you want:

```bash
python eval/sample_dvf.py --outdir eval/save/test_sample/svg --n-sample 10 # initial SVG
```

Then refine it:

```bash
python eval/post_refinement.py --outdir eval/save/test_sample/refined --input eval/save/test_sample/svg/ --fmin 0 --fmax 20 # refinement
```

**Evaluate the metrics**

Take the font reconstruction as an example. To render the glyph with a resolution of 256*256 and evaluate the L1-error, SSIM and s-IOU, run:

```bash
cd eval
python run_metrics.py --name ours --pred_lowercase --pred save/test_recon/refined --ff {0:02d}_p4.svg --fontmin 0 --fontmax 200 --glyph 52 --res 256
```

The result will be saved at `eval/eval/save/`.

# Citation
If you find DualVector helpful, please consider citing:
```
@InProceedings{dualvector,
    author    = {Liu, Ying-Tian and Zhang, Zhifei and Guo, Yuan-Chen and Fisher, Matthew and Wang, Zhaowen and Zhang, Song-Hai},
    title     = {DualVector: Unsupervised Vector Font Synthesis With Dual-Part Representation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14193-14202}
}
```