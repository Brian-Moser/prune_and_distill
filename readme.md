# Distill the Best, Ignore the Rest: Improving Dataset Distillation with Loss-Value-Based Pruning

## Requirements
Run `install.sh` as prolog to the following experiments. You need to insert your W&D key in the scripts `distill_...`.

## Get losses for pruned datasets
Setup the dataset directory in the following line of `get_losses_imagenet.py`:
```bash
data_dir = '/ds/imagenet/train'
```
and then run the script to get the lost values.

## IPC=1 Experiments
### DC
Run for LD3M
```bash
python ./src/distill_dc_LD3M.py --dataset=imagenet-a --space=wp --layer=16 --ipc=1 --data_path=/ds/imagenet --percent=60 --order=asc
```
and for GLaD
```bash
python ./src/distill_dc.py --dataset=imagenet-a --space=wp --layer=16 --ipc=1 --data_path=/ds/imagenet --percent=60 --order=asc
```

### DM
Run for LD3M
```bash
python ./src/distill_dm_LD3M.py --dataset=imagenet-a --space=wp --layer=20 --ipc=1 --data_path=/ds/imagenet  --percent=20 --order=asc
```
and for GLaD
```bash
python ./src/distill_dm.py --dataset=imagenet-a --space=wp --layer=20 --ipc=1 --data_path=/ds/imagenet  --percent=20 --order=asc
```

### MTT
Setup expert trajectories by running
```bash
python ./src/buffer_mtt.py --dataset=imagenet-a --subset a --model ConvNet --depth 5 --res 128 --norm_train=instancenorm --train_epochs=15 --num_experts=5000 --buffer_path=/netscratch/bmoser/pruning_and_distillation/storage/60_asc/ --data_path=/ds/imagenet --percent=60 --order=asc
```
Next, run for LD3M
```bash
python ./src/distill_mtt_LD3M.py --dataset=imagenet-a --space=wp --layer=5 --ipc=1 --batch_real=256 --batch_train=256 --buffer_path=/pruning_and_distillation/storage/60_asc --data_path=/ds/imagenet --percent=60 --order=asc
```
and for GLaD
```bash
python ./src/distill_mtt.py --dataset=imagenet-a --space=wp --layer=5 --ipc=1 --batch_real=256 --batch_train=256 --buffer_path=/pruning_and_distillation/storage/60_asc --data_path=/ds/imagenet --percent=60 --order=asc
```

## IPC=10 Experiments
### DC
Run for LD3M
```bash
python ./src/distill_dc_LD3M.py --dataset=imagenet-a --space=wp --layer=16 --ipc=10 --data_path=/ds/imagenet --percent=60 --order=asc
```
and for GLaD
```bash
python ./src/distill_dc.py --dataset=imagenet-a --space=wp --layer=16 --ipc=10 --data_path=/ds/imagenet --percent=60 --order=asc
```

### DM
Run for LD3M
```bash
python ./src/distill_dm_LD3M.py --dataset=imagenet-a --space=wp --layer=20 --ipc=10 --data_path=/ds/imagenet  --percent=20 --order=asc
```
and for GLaD
```bash
python ./src/distill_dm.py --dataset=imagenet-a --space=wp --layer=20 --ipc=10 --data_path=/ds/imagenet  --percent=20 --order=asc
```

## 256x256 Experiments
Run
```bash
python ./src/distill_dc_LD3M.py --dataset=imagenet-a --res=256 --depth=6 --space=wp --layer=16 --ipc=1 --data_path=/ds/imagenet --percent=60 --order=asc
```
add 
```bash
--ffhq=True
```
for FFHQ experiments and 
```bash
--rand_g=True
```
for random initialization.

## References
If you find this repo helpful, please acknowledge us in your work:


```
@article{moser2024distill,
  title={Distill the Best, Ignore the Rest: Improving Dataset Distillation with Loss-Value-Based Pruning},
  author={Moser, Brian B and Raue, Federico and Nauen, Tobias C and Frolov, Stanislav and Dengel, Andreas},
  journal={arXiv preprint arXiv:2411.12115},
  year={2024}
}
```


The work is based on LD3M and GLaD, please acknowledge this and their work accordingly:

```
@inproceedings{cazenavette2023generalizing,
  title={Generalizing dataset distillation via deep generative prior},
  author={Cazenavette, George and Wang, Tongzhou and Torralba, Antonio and Efros, Alexei A and Zhu, Jun-Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3739--3748},
  year={2023}
}

@article{moser2024latent,
  title={Latent dataset distillation with diffusion models},
  author={Moser, Brian B and Raue, Federico and Palacio, Sebastian and Frolov, Stanislav and Dengel, Andreas},
  journal={arXiv preprint arXiv:2403.03881},
  year={2024}
}
```
