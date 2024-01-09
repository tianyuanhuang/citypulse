# CityPulse

[[Paper]](https://arxiv.org/abs/2401.01107) [[Citing]](https://github.com/tianyuanhuang/citypulse?tab=readme-ov-file#citing) [Project Page] (under construction)

Welcome to the official repository of CityPulse! 

## Download Data

## Environment Setup
```sh
conda env create -f citypulse.yaml
```

## Finetune


```sh
python finetune.py --mode train
```

To test the finetuned model on the test set, please run the following command:
```sh
python finetune.py --mode test --checkpoint /YOUR/CKPT/PATH.ckpt
```

## Inference with Our Finetuned Dinov2-Siamese Model
```sh
python SiamDINOv2_inference.py --data_idx /YOUR/DATA/INDEX/CSV --checkpoint /Your/CKPT/PATH.ckpt --with_label True
```

## Citing

If you found this project useful, please consider citing:

```bibtex
@article{huang2024citypulse,
  title={CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series},
  author={Huang, Tianyuan and Wu, Zejia and Wu, Jiajun and Hwang, Jackelyn and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2401.01107},
  year={2024}
}
```
