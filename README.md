# Projeto Final - Modelos Preditivos Conexionistas 2022.1

### Rodrigo Prado

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|**Qtde de imagens por classe**|**Qtde de classes**|
|--|--|--|--|--|
|Deteção de Objetos|YOLOv5|PyTorch| ~100 | 6 |

### Informações Adicionais
Classes:
- Bike:  121 fotos
- Car: 94 fotos
- Motorcycle: 103
- Men: 124
- Women: 160
- Children: 96

## Performance

O modelo treinado possui performance de **74%**. Fotos originais coloridas.

Model summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs
|**Class**|**Images**|**Instances**|**P**|**R**|**mAP50**|**mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]**|
|:--:|:--:|:--:|:--|:--:|:--:|:--:|
|all|        136|        264|       0.66|      0.722|      **0.736**|       0.36|
|bike|        136     |   22  |    0.697  |    0.864 |     0.859|      0.459|
|car|      |  136     |    87     |  0.71  |    0.667  |    0.698   |   0.489|
|child  |      136  |       27 |     0.673 |     0.667|      0.742|      0.278|
|man |       136 |        49|      0.646 |     0.531|      0.622|      0.233|
|motorcycle |       136  |       45|      0.887 |     0.871|      0.894|      0.482|
| woman |       136  |       34 |     0.348 |     0.735|      0.602|      0.216|

O modelo treinado com escala de cinza possui performance de **??%**.

Model summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs
|**Class**|**Images**|**Instances**|**P**|**R**|**mAP50**|**mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]**|
|:--:|:--:|:--:|:--|:--:|:--:|:--:|
|all|        136|        264|       0.66|      0.722|      **0.736**|       0.36|
|bike|        136     |   22  |    0.697  |    0.864 |     0.859|      0.459|
|car|      |  136     |    87     |  0.71  |    0.667  |    0.698   |   0.489|
|child  |      136  |       27 |     0.673 |     0.667|      0.742|      0.278|
|man |       136 |        49|      0.646 |     0.531|      0.622|      0.233|
|motorcycle |       136  |       45|      0.887 |     0.871|      0.894|      0.482|
| woman |       136  |       34 |     0.348 |     0.735|      0.602|      0.216|

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
    wandb: Currently logged in as: rgp. Use `wandb login --relogin` to force relogin
train: weights=yolov5s.pt, cfg=, data=/content/yolov5/CarViewDetection-3/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=2000, batch_size=64, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.2-224-g82a5585 Python-3.7.15 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.13.4
wandb: Run data is saved locally in /content/yolov5/wandb/run-20221102_203240-2knv862j
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run divine-monkey-3
wandb: ⭐️ View project at https://wandb.ai/rgp/YOLOv5
wandb: 🚀 View run at https://wandb.ai/rgp/YOLOv5/runs/2knv862j
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 25.9MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 171MB/s]

Overriding model.yaml nc=80 with nc=6

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     29667  models.yolo.Detect                      [6, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7035811 parameters, 7035811 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/yolov5/CarViewDetection-3/train/labels' images and labels...544 found, 0 missing, 1 empty, 0 corrupt: 100% 544/544 [00:00<00:00, 1999.81it/s]
train: New cache created: /content/yolov5/CarViewDetection-3/train/labels.cache
train: Caching images (0.7GB ram): 100% 544/544 [00:02<00:00, 184.68it/s]
val: Scanning '/content/yolov5/CarViewDetection-3/valid/labels' images and labels...136 found, 0 missing, 1 empty, 0 corrupt: 100% 136/136 [00:00<00:00, 809.81it/s]
val: New cache created: /content/yolov5/CarViewDetection-3/valid/labels.cache
val: Caching images (0.2GB ram): 100% 136/136 [00:02<00:00, 63.56it/s]

AutoAnchor: 3.94 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 2000 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     0/1999        14G     0.1183    0.03271    0.06053        101        640: 100% 9/9 [00:10<00:00,  1.20s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:04<00:00,  2.12s/it]
                   all        136        264   0.000916      0.116   0.000769   0.000256

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     1/1999      12.5G    0.09541    0.03373    0.05279        113        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.33s/it]
                   all        136        264    0.00407      0.499     0.0105    0.00287

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     2/1999      12.5G    0.07538    0.03504    0.04362        107        640: 100% 9/9 [00:06<00:00,  1.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.47s/it]
                   all        136        264      0.569      0.178     0.0887     0.0231

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     3/1999      12.5G    0.06436    0.03177     0.0352         92        640: 100% 9/9 [00:06<00:00,  1.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.32s/it]
                   all        136        264       0.12      0.441      0.266      0.091

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     4/1999      12.5G    0.05995    0.02734    0.03173        107        640: 100% 9/9 [00:06<00:00,  1.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.39s/it]
                   all        136        264      0.211      0.507      0.282      0.105

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     5/1999      12.5G    0.05971    0.02563    0.02647         98        640: 100% 9/9 [00:06<00:00,  1.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.15s/it]
                   all        136        264      0.241      0.539      0.342      0.124

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     6/1999      12.5G    0.05782    0.02502    0.02571        103        640: 100% 9/9 [00:06<00:00,  1.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.233      0.541      0.315      0.125

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     7/1999      12.5G     0.0571    0.02262    0.02286         95        640: 100% 9/9 [00:06<00:00,  1.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.16s/it]
                   all        136        264      0.236      0.558      0.331      0.112

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     8/1999      12.5G    0.05541    0.02286    0.02047        115        640: 100% 9/9 [00:06<00:00,  1.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.276      0.634      0.379      0.158

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     9/1999      12.5G    0.05448    0.02213     0.0202        119        640: 100% 9/9 [00:06<00:00,  1.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.08s/it]
                   all        136        264      0.277      0.643      0.387      0.158

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    10/1999      12.5G     0.0543    0.02109    0.01952         97        640: 100% 9/9 [00:06<00:00,  1.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.285      0.613      0.341      0.111

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    11/1999      12.5G    0.05204    0.02132    0.01838        103        640: 100% 9/9 [00:06<00:00,  1.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.15s/it]
                   all        136        264      0.212      0.535       0.27      0.103

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    12/1999      12.5G    0.05122    0.02176    0.01934        106        640: 100% 9/9 [00:06<00:00,  1.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.15s/it]
                   all        136        264       0.29      0.563      0.395      0.153

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    13/1999      12.5G    0.04707    0.02076    0.01803        104        640: 100% 9/9 [00:06<00:00,  1.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.302      0.557      0.392      0.157

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    14/1999      12.5G    0.04565    0.01998    0.01741         98        640: 100% 9/9 [00:08<00:00,  1.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.291      0.636      0.364      0.152

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    15/1999      12.5G    0.04422    0.02125    0.01748        110        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.288      0.424       0.34      0.155

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    16/1999      12.5G    0.04361    0.02051    0.01719        106        640: 100% 9/9 [00:06<00:00,  1.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.251      0.508      0.303      0.122

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    17/1999      12.5G    0.04295    0.01996    0.01781        110        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.239       0.49      0.352      0.142

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    18/1999      12.5G    0.04156     0.0209    0.01652        131        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.259       0.45      0.344      0.154

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    19/1999      12.5G    0.04093    0.02049    0.01552        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.325      0.481      0.414      0.197

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    20/1999      12.5G    0.04079    0.02056    0.01627         80        640: 100% 9/9 [00:06<00:00,  1.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.383      0.542      0.512      0.242

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    21/1999      12.5G    0.03974     0.0199    0.01587        106        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264       0.33      0.636      0.531      0.244

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    22/1999      12.5G    0.03958    0.01949    0.01658         86        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264       0.41      0.586      0.508       0.22

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    23/1999      12.5G    0.03992     0.0194    0.01589        106        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264       0.42      0.627      0.523       0.24

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    24/1999      12.5G    0.03914    0.01993    0.01558         97        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.532      0.584      0.549      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    25/1999      12.5G    0.03892    0.01887    0.01454        106        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.466      0.523      0.484      0.203

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    26/1999      12.5G     0.0405    0.02014    0.01488        117        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.484      0.494      0.489      0.207

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    27/1999      12.5G    0.03826    0.01797    0.01468         87        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.588      0.534      0.539      0.239

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    28/1999      12.5G    0.03805    0.01896    0.01324        130        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.546      0.534      0.515      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    29/1999      12.5G    0.03743     0.0191    0.01388        114        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.484      0.519      0.511      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    30/1999      12.5G    0.03758    0.01973    0.01219        107        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.558        0.5        0.5       0.24

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    31/1999      12.5G    0.03772    0.01912    0.01158         92        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264       0.64      0.473      0.535      0.247

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    32/1999      12.5G    0.03658    0.01942    0.01018        108        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264        0.5      0.482      0.495       0.22

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    33/1999      12.5G    0.03742    0.01907    0.01109        129        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.09s/it]
                   all        136        264      0.465      0.545      0.488      0.229

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    34/1999      12.5G    0.03573    0.01809    0.01008        101        640: 100% 9/9 [00:07<00:00,  1.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.18s/it]
                   all        136        264      0.532      0.519      0.539      0.221

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    35/1999      12.5G    0.03545    0.01895    0.01018        118        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.587       0.53      0.555      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    36/1999      12.5G    0.03582    0.01799   0.009851         86        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.644      0.602      0.617      0.265

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    37/1999      12.5G    0.03544    0.01872   0.009714         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.612      0.582      0.603       0.27

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    38/1999      12.5G    0.03691     0.0191    0.01074        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.569      0.477      0.503      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    39/1999      12.5G    0.03546    0.01881   0.009071        114        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.514      0.584      0.518      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    40/1999      12.5G    0.03492     0.0179   0.009537         89        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.592      0.624      0.607      0.284

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    41/1999      12.5G    0.03418    0.01802   0.009182         89        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.531       0.43      0.454      0.206

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    42/1999      12.5G     0.0344    0.01739   0.009347         98        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.586       0.61      0.578      0.272

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    43/1999      12.5G    0.03403      0.017   0.009213         93        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.476      0.506      0.482      0.219

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    44/1999      12.5G    0.03433    0.01801   0.008873        116        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.538      0.525      0.514      0.231

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    45/1999      12.5G    0.03438    0.01792   0.007677        103        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.631      0.505      0.547      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    46/1999      12.5G    0.03457    0.01768   0.007835        122        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.592      0.543      0.563      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    47/1999      12.5G    0.03341    0.01711   0.007517        107        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.583      0.633      0.631      0.292

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    48/1999      12.5G    0.03363    0.01714   0.008527         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.595      0.632      0.625      0.281

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    49/1999      12.5G    0.03481    0.01784   0.008247        111        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.545      0.567      0.556      0.247

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    50/1999      12.5G    0.03432    0.01786   0.008753        114        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.605       0.53      0.568      0.284

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    51/1999      12.5G    0.03204    0.01803   0.006719        133        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.622      0.502      0.588      0.255

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    52/1999      12.5G    0.03269    0.01794   0.007484        102        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.594      0.523      0.563      0.263

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    53/1999      12.5G      0.033     0.0166   0.007257         90        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.67s/it]
                   all        136        264      0.637      0.619      0.643      0.315

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    54/1999      12.5G    0.03281    0.01746   0.007854        109        640: 100% 9/9 [00:06<00:00,  1.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.628      0.592      0.642      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    55/1999      12.5G    0.03259      0.017   0.008137         79        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.624      0.537      0.546      0.238

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    56/1999      12.5G    0.03257    0.01732   0.008031         91        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.612       0.61      0.628      0.288

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    57/1999      12.5G    0.03169    0.01676   0.007535         89        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.513       0.54      0.558      0.245

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    58/1999      12.5G    0.03293    0.01754   0.008743         97        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.656      0.501      0.551      0.264

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    59/1999      12.5G    0.03245    0.01646   0.007643        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.611      0.444      0.478      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    60/1999      12.5G    0.03183    0.01709   0.006395        101        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.607       0.58      0.584      0.265

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    61/1999      12.5G    0.03292    0.01704   0.007469        110        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.589      0.538      0.564      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    62/1999      12.5G    0.03181    0.01723   0.007369        113        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264       0.52      0.544      0.557      0.258

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    63/1999      12.5G     0.0314    0.01689   0.005934        104        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.694       0.59      0.616      0.277

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    64/1999      12.5G    0.03116    0.01689   0.006935        108        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.683      0.575      0.624      0.273

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    65/1999      12.5G    0.03105    0.01648   0.006074        102        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.539      0.563      0.541      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    66/1999      12.5G    0.03128     0.0168   0.006414        116        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.597      0.538      0.581      0.254

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    67/1999      12.5G    0.03166    0.01614   0.005549         90        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.655      0.539      0.593      0.272

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    68/1999      12.5G    0.03145    0.01714   0.005658        111        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.14it/s]
                   all        136        264      0.665      0.597      0.649      0.308

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    69/1999      12.5G    0.03168    0.01664   0.006885        112        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.686       0.57      0.625      0.301

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    70/1999      12.5G    0.03137    0.01596   0.006464        103        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.613      0.508      0.541      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    71/1999      12.5G    0.03026    0.01668   0.006154        108        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.686      0.547      0.587      0.265

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    72/1999      12.5G    0.02979    0.01598   0.006213         93        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.676      0.551      0.608      0.292

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    73/1999      12.5G    0.03041    0.01661   0.005701        106        640: 100% 9/9 [00:07<00:00,  1.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.37s/it]
                   all        136        264       0.67       0.55      0.597       0.27

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    74/1999      12.5G    0.03151    0.01571   0.005608        109        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.666      0.633       0.66      0.306

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    75/1999      12.5G    0.03128     0.0157    0.00673        103        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.685      0.633      0.656      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    76/1999      12.5G     0.0317    0.01616    0.00671        103        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.657      0.617      0.664      0.314

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    77/1999      12.5G    0.03112    0.01716   0.006845        105        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.631      0.537      0.603      0.289

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    78/1999      12.5G    0.02984    0.01598    0.00662        105        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264       0.63      0.464      0.509      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    79/1999      12.5G    0.02968    0.01637   0.007101        125        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.599      0.546      0.565      0.265

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    80/1999      12.5G    0.02995    0.01698   0.005957        119        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.624      0.547      0.569      0.273

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    81/1999      12.5G    0.03086    0.01595   0.005386        147        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.665      0.618      0.626       0.29

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    82/1999      12.5G    0.02949    0.01661   0.004989        109        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.737      0.546      0.636      0.306

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    83/1999      12.5G     0.0302    0.01597   0.004916         95        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.676      0.515      0.587      0.296

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    84/1999      12.5G    0.02918    0.01625   0.005128        128        640: 100% 9/9 [00:06<00:00,  1.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.639      0.583      0.602      0.283

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    85/1999      12.5G     0.0304    0.01529   0.005315        109        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.638      0.684      0.674      0.302

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    86/1999      12.5G    0.02944    0.01608   0.005408        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.711      0.604      0.678      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    87/1999      12.5G    0.02985     0.0149   0.005299         95        640: 100% 9/9 [00:06<00:00,  1.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.11s/it]
                   all        136        264      0.739      0.582      0.648      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    88/1999      12.5G    0.02893    0.01533   0.005568        115        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.11it/s]
                   all        136        264      0.691      0.586      0.607      0.285

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    89/1999      12.5G    0.02938    0.01565   0.005628        113        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.11it/s]
                   all        136        264      0.554      0.582      0.556      0.251

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    90/1999      12.5G    0.03032      0.016     0.0063        116        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.659      0.556      0.613      0.282

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    91/1999      12.5G    0.02945    0.01548   0.005457         76        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.699      0.557      0.612      0.289

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    92/1999      12.5G     0.0299    0.01505    0.00492         94        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.651      0.584      0.625      0.311

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    93/1999      12.5G     0.0292    0.01582   0.005233         93        640: 100% 9/9 [00:08<00:00,  1.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.16s/it]
                   all        136        264      0.685       0.62      0.637      0.295

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    94/1999      12.5G    0.02917    0.01577   0.005918         95        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.662      0.555      0.626      0.291

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    95/1999      12.5G     0.0293    0.01553   0.005212         92        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.689      0.529       0.57      0.284

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    96/1999      12.5G    0.02735    0.01566   0.004259         99        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.651      0.617      0.645      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    97/1999      12.5G    0.02758    0.01402   0.004655         83        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.703      0.519      0.606      0.288

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    98/1999      12.5G    0.02912    0.01681    0.00526        124        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.661      0.611      0.622        0.3

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    99/1999      12.5G     0.0288     0.0147   0.005463        109        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.754       0.51      0.619      0.302

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   100/1999      12.5G    0.02875    0.01521   0.005411        115        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.09s/it]
                   all        136        264      0.684      0.633      0.658      0.306

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   101/1999      12.5G    0.02945    0.01487   0.005087         93        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.728      0.612      0.673      0.294

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   102/1999      12.5G    0.02896    0.01568   0.005019        105        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.11s/it]
                   all        136        264      0.721      0.611      0.661      0.309

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   103/1999      12.5G    0.02836    0.01542   0.005102         79        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.701      0.649       0.67      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   104/1999      12.5G    0.02851     0.0146   0.004879         90        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264       0.62      0.658      0.634      0.311

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   105/1999      12.5G    0.02828     0.0156   0.004499        112        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.673      0.643      0.647      0.306

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   106/1999      12.5G    0.02772    0.01603   0.005045        132        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.627      0.643      0.628       0.29

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   107/1999      12.5G    0.02762    0.01534   0.003753        112        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.679      0.587      0.622      0.292

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   108/1999      12.5G     0.0275    0.01517   0.004567        101        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.725      0.599      0.663      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   109/1999      12.5G    0.02709    0.01496   0.004678         96        640: 100% 9/9 [00:06<00:00,  1.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.695      0.682      0.672      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   110/1999      12.5G    0.02898    0.01513   0.004594         87        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.621      0.532       0.56      0.269

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   111/1999      12.5G    0.02751    0.01541   0.005147        102        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.651      0.576      0.591      0.276

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   112/1999      12.5G     0.0284    0.01531     0.0043        109        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.47s/it]
                   all        136        264      0.648       0.48      0.559       0.25

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   113/1999      12.5G    0.02779    0.01524   0.004309        107        640: 100% 9/9 [00:06<00:00,  1.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.12s/it]
                   all        136        264      0.627      0.597      0.611      0.278

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   114/1999      12.5G    0.02767    0.01542   0.004626        118        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.18s/it]
                   all        136        264      0.728      0.615      0.688      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   115/1999      12.5G    0.02811    0.01456   0.003988        111        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.678      0.568      0.655      0.324

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   116/1999      12.5G    0.02757    0.01525   0.003952        120        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.758      0.625        0.7      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   117/1999      12.5G     0.0276    0.01556   0.004478         98        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.686      0.649      0.679      0.326

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   118/1999      12.5G    0.02656    0.01477   0.004008        142        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.677      0.665      0.685      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   119/1999      12.5G    0.02671    0.01462   0.004044        110        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.662      0.623      0.653      0.313

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   120/1999      12.5G    0.02582    0.01457   0.004239        106        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.649      0.655      0.674      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   121/1999      12.5G    0.02603    0.01443   0.004934         85        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.657      0.596      0.632      0.314

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   122/1999      12.5G    0.02622    0.01393   0.004142         91        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.621      0.616      0.583      0.279

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   123/1999      12.5G    0.02762    0.01455   0.004425         87        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.656      0.583      0.621      0.307

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   124/1999      12.5G    0.02653    0.01417   0.003356         97        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.642      0.624      0.644       0.31

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   125/1999      12.5G     0.0265    0.01389   0.003788         92        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.691      0.547      0.614      0.309

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   126/1999      12.5G    0.02682    0.01473    0.00396        123        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.647      0.582      0.654      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   127/1999      12.5G    0.02614    0.01427   0.004283         92        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.676       0.54      0.589      0.303

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   128/1999      12.5G    0.02651    0.01426   0.003858         95        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.559      0.624      0.574      0.294

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   129/1999      12.5G    0.02756     0.0148   0.003824        127        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.637      0.593      0.599      0.297

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   130/1999      12.5G      0.026    0.01434   0.005033         89        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.645      0.614      0.666      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   131/1999      12.5G    0.02615    0.01429   0.004037         88        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.694      0.621      0.669      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   132/1999      12.5G    0.02665    0.01456    0.00395        108        640: 100% 9/9 [00:07<00:00,  1.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.56s/it]
                   all        136        264      0.718      0.662      0.685      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   133/1999      12.5G    0.02583    0.01366   0.003941        114        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.694       0.63      0.649      0.304

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   134/1999      12.5G    0.02608    0.01404   0.004683        108        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.631      0.591      0.608      0.288

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   135/1999      12.5G     0.0266    0.01466   0.004104        103        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.14s/it]
                   all        136        264      0.722      0.517      0.581      0.296

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   136/1999      12.5G    0.02627    0.01504    0.00379        106        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.673      0.632      0.661      0.318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   137/1999      12.5G    0.02578    0.01368   0.004331         83        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.769      0.562      0.653       0.33

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   138/1999      12.5G    0.02784    0.01408   0.004868         93        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.667      0.657      0.673      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   139/1999      12.5G    0.02676    0.01415   0.003982         97        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.665      0.626      0.693      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   140/1999      12.5G    0.02606    0.01427   0.004641        107        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.18it/s]
                   all        136        264      0.707      0.642      0.686      0.321

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   141/1999      12.5G     0.0268     0.0145   0.004254        106        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.711      0.556      0.621      0.287

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   142/1999      12.5G    0.02634    0.01358   0.005149         90        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.691      0.611      0.652      0.305

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   143/1999      12.5G    0.02597    0.01452   0.004514        108        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.561      0.585       0.56      0.254

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   144/1999      12.5G    0.02487    0.01408   0.003762        107        640: 100% 9/9 [00:06<00:00,  1.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.686      0.513      0.564      0.262

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   145/1999      12.5G    0.02604    0.01381   0.004431        101        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.636      0.558      0.598      0.296

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   146/1999      12.5G    0.02575    0.01434   0.003513         94        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264       0.71      0.618      0.641      0.311

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   147/1999      12.5G    0.02662    0.01485    0.00429        114        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.652      0.597      0.645      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   148/1999      12.5G    0.02491    0.01379   0.003827        104        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.675      0.569       0.62      0.293

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   149/1999      12.5G    0.02482    0.01367   0.004334        116        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.703      0.594      0.688      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   150/1999      12.5G    0.02586     0.0135   0.003552        121        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.13it/s]
                   all        136        264      0.658      0.627      0.675      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   151/1999      12.5G    0.02642    0.01346   0.003438        114        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.19s/it]
                   all        136        264      0.657       0.69      0.684      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   152/1999      12.5G    0.02516    0.01376   0.003396        124        640: 100% 9/9 [00:07<00:00,  1.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.09s/it]
                   all        136        264      0.739      0.576      0.683       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   153/1999      12.5G    0.02562    0.01387   0.003622        106        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.09s/it]
                   all        136        264      0.658      0.686       0.68       0.34

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   154/1999      12.5G    0.02524    0.01354   0.003408         97        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264        0.7      0.603      0.664      0.318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   155/1999      12.5G    0.02545    0.01364   0.003987         97        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.642      0.585      0.618      0.302

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   156/1999      12.5G    0.02526    0.01359   0.004122         97        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.18it/s]
                   all        136        264       0.68      0.637      0.667      0.323

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   157/1999      12.5G    0.02472    0.01295   0.003872         96        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.744       0.64      0.656      0.329

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   158/1999      12.5G    0.02462    0.01319   0.003371         99        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.691      0.653      0.694      0.354

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   159/1999      12.5G    0.02471    0.01375   0.003472         94        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264       0.69      0.644      0.669      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   160/1999      12.5G    0.02512     0.0136    0.00317        106        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.667      0.665      0.671      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   161/1999      12.5G    0.02542    0.01425   0.002695         90        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.08s/it]
                   all        136        264      0.693      0.585      0.621      0.304

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   162/1999      12.5G    0.02546    0.01376   0.003075        118        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.671       0.51       0.58      0.296

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   163/1999      12.5G    0.02533    0.01395   0.003446        129        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.703      0.582      0.641      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   164/1999      12.5G    0.02426    0.01418   0.003004        110        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.796      0.564       0.68      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   165/1999      12.5G    0.02436    0.01287    0.00344         90        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.15it/s]
                   all        136        264      0.733      0.525      0.604      0.299

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   166/1999      12.5G    0.02479    0.01369   0.003344        114        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.627      0.535      0.581      0.284

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   167/1999      12.5G    0.02416     0.0139   0.003661        125        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.728      0.548      0.593      0.294

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   168/1999      12.5G    0.02467    0.01386   0.003255        100        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.644      0.615      0.612      0.296

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   169/1999      12.5G    0.02542    0.01369   0.004049        104        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.699      0.599      0.621      0.315

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   170/1999      12.5G    0.02453    0.01449   0.003243        116        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.673      0.645      0.665      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   171/1999      12.5G    0.02462    0.01374   0.002883        108        640: 100% 9/9 [00:07<00:00,  1.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.58s/it]
                   all        136        264      0.679      0.557       0.62      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   172/1999      12.5G    0.02415     0.0131   0.003236         87        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.752      0.581      0.687      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   173/1999      12.5G    0.02425    0.01405   0.003554         84        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.695        0.6      0.669      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   174/1999      12.5G    0.02543     0.0138    0.00491        131        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.737      0.571      0.657      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   175/1999      12.5G    0.02406    0.01368   0.003465         98        640: 100% 9/9 [00:07<00:00,  1.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.17it/s]
                   all        136        264      0.772      0.559      0.658      0.332

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   176/1999      12.5G    0.02499      0.013   0.004004         80        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.682      0.625      0.652      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   177/1999      12.5G    0.02481    0.01369   0.004193         87        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.738      0.624      0.688      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   178/1999      12.5G     0.0247    0.01307   0.003203         99        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.649      0.668      0.673      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   179/1999      12.5G    0.02544    0.01342   0.003201        106        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.752      0.651      0.689      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   180/1999      12.5G     0.0239    0.01339   0.003727        102        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.664      0.623      0.654      0.319

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   181/1999      12.5G    0.02395    0.01347   0.002586        123        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.11it/s]
                   all        136        264      0.668      0.572      0.633      0.321

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   182/1999      12.5G    0.02385      0.013   0.003052         87        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.717      0.566      0.639      0.323

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   183/1999      12.5G    0.02376     0.0128   0.003132         84        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264       0.71      0.602      0.631      0.317

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   184/1999      12.5G    0.02393    0.01339   0.003288         98        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.704      0.643      0.652      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   185/1999      12.5G    0.02355    0.01387   0.003533        122        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.733      0.608      0.661      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   186/1999      12.5G     0.0235    0.01377    0.00301        105        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.698      0.642      0.669      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   187/1999      12.5G    0.02344    0.01249   0.003762        108        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.691      0.617      0.652      0.318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   188/1999      12.5G    0.02391    0.01307   0.003457        110        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264       0.72      0.628      0.661      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   189/1999      12.5G    0.02339    0.01244   0.002957         79        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.726       0.61       0.66      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   190/1999      12.5G     0.0233    0.01324   0.002271        108        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.666       0.63      0.648      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   191/1999      12.5G    0.02366    0.01342   0.003008        108        640: 100% 9/9 [00:08<00:00,  1.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.17s/it]
                   all        136        264      0.736      0.596       0.63      0.317

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   192/1999      12.5G     0.0226    0.01256   0.002383         96        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.655      0.686      0.644      0.312

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   193/1999      12.5G    0.02289    0.01266   0.002375        124        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.693      0.608      0.632      0.311

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   194/1999      12.5G    0.02361    0.01285   0.002663         92        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.674      0.586      0.599      0.302

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   195/1999      12.5G    0.02278    0.01265   0.003013        116        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264       0.65      0.562       0.58      0.304

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   196/1999      12.5G    0.02302    0.01317   0.002987        101        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.713      0.552      0.603      0.314

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   197/1999      12.5G    0.02372    0.01252   0.003088        109        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.695       0.57      0.615      0.313

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   198/1999      12.5G    0.02358    0.01304   0.002891        106        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.668      0.598       0.61      0.314

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   199/1999      12.5G    0.02373    0.01276   0.003186         95        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:04<00:00,  2.02s/it]
                   all        136        264      0.668       0.65      0.666      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   200/1999      12.5G    0.02426    0.01288   0.002962         98        640: 100% 9/9 [00:06<00:00,  1.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264       0.69      0.642      0.653      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   201/1999      12.5G    0.02249    0.01276   0.003167        101        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.687      0.573      0.625      0.318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   202/1999      12.5G    0.02408    0.01273   0.003079         93        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.648      0.593      0.603      0.298

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   203/1999      12.5G    0.02376    0.01252   0.002733         99        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.672      0.576      0.627      0.301

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   204/1999      12.5G    0.02305    0.01256   0.002629         83        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.718      0.564      0.646      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   205/1999      12.5G    0.02305    0.01219   0.002206         99        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.644       0.61      0.623      0.318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   206/1999      12.5G    0.02235    0.01243   0.002379        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.657      0.601      0.603      0.311

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   207/1999      12.5G    0.02374    0.01293    0.00264         92        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.702      0.647      0.655      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   208/1999      12.5G    0.02318    0.01324   0.002921        114        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.705      0.573      0.652      0.329

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   209/1999      12.5G    0.02329    0.01292   0.002455        126        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.53s/it]
                   all        136        264      0.692       0.58      0.622      0.315

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   210/1999      12.5G    0.02248    0.01326   0.003267        118        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.693      0.623      0.668      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   211/1999      12.5G     0.0217    0.01303   0.002515         79        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.699      0.599       0.64       0.34

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   212/1999      12.5G    0.02224    0.01269    0.00302         91        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.666      0.623      0.663      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   213/1999      12.5G     0.0232    0.01269   0.002212        124        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.674      0.625      0.673      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   214/1999      12.5G    0.02349    0.01348   0.002498        100        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.663       0.62      0.658      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   215/1999      12.5G    0.02343    0.01282   0.003196        102        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.668      0.616      0.652      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   216/1999      12.5G    0.02313    0.01236   0.003512         76        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.681      0.624      0.628      0.307

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   217/1999      12.5G    0.02182     0.0121   0.002464        117        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.782      0.591      0.674      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   218/1999      12.5G    0.02256    0.01236   0.002274        113        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.762      0.626      0.676      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   219/1999      12.5G    0.02144    0.01198   0.002316         80        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.755      0.612      0.664      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   220/1999      12.5G    0.02229    0.01284   0.002953        126        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.706      0.619       0.67      0.332

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   221/1999      12.5G    0.02293    0.01233   0.002233        106        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264       0.76      0.581      0.672      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   222/1999      12.5G    0.02278    0.01258   0.002567        114        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.669      0.634      0.648      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   223/1999      12.5G    0.02184    0.01194   0.001919         94        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.717      0.604       0.68      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   224/1999      12.5G    0.02197    0.01171   0.002319         99        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.715      0.662      0.703      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   225/1999      12.5G    0.02304    0.01247   0.002385         88        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.14it/s]
                   all        136        264      0.658      0.611      0.639      0.315

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   226/1999      12.5G    0.02283    0.01155   0.002785         94        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.692       0.59      0.636      0.313

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   227/1999      12.5G     0.0232    0.01211    0.00223         94        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.663      0.612      0.616      0.319

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   228/1999      12.5G    0.02213    0.01246   0.002325        124        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.735      0.639      0.711      0.357

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   229/1999      12.5G    0.02122     0.0117   0.002293         98        640: 100% 9/9 [00:07<00:00,  1.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.10s/it]
                   all        136        264      0.712      0.626      0.704      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   230/1999      12.5G    0.02248    0.01167    0.00254        100        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.757      0.656      0.711      0.344

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   231/1999      12.5G    0.02248     0.0129   0.002542        108        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264       0.75      0.642      0.712      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   232/1999      12.5G    0.02227    0.01249   0.002235         99        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.701      0.673      0.709      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   233/1999      12.5G    0.02167    0.01228   0.002314        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.12s/it]
                   all        136        264      0.691      0.629      0.664       0.33

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   234/1999      12.5G     0.0215    0.01203   0.002136        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.693      0.651      0.683      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   235/1999      12.5G    0.02203    0.01225   0.002346        103        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.712      0.629      0.682      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   236/1999      12.5G    0.02178    0.01227   0.002226        115        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264        0.7      0.622      0.667      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   237/1999      12.5G    0.02216    0.01169   0.002755         85        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.688      0.535      0.623      0.313

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   238/1999      12.5G    0.02158    0.01189   0.003186        112        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.695      0.615      0.678      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   239/1999      12.5G    0.02166    0.01195    0.00279        103        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264       0.72      0.578      0.638      0.314

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   240/1999      12.5G    0.02242    0.01256   0.002442        106        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.722      0.619      0.686      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   241/1999      12.5G    0.02134    0.01247   0.002304        121        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.698      0.594       0.62      0.305

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   242/1999      12.5G    0.02197    0.01269   0.002241        103        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.715      0.628      0.655      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   243/1999      12.5G    0.02127    0.01272   0.001946         91        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264        0.7      0.668      0.681      0.342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   244/1999      12.5G    0.02146    0.01199   0.002671        115        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.14it/s]
                   all        136        264      0.727      0.645      0.671      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   245/1999      12.5G    0.02232    0.01194   0.002704        127        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.794      0.567      0.655      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   246/1999      12.5G    0.02255    0.01182   0.002751        112        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.683      0.648      0.676      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   247/1999      12.5G    0.02253    0.01247    0.00266         94        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.724      0.671      0.699      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   248/1999      12.5G     0.0218    0.01212   0.002104         99        640: 100% 9/9 [00:07<00:00,  1.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.08s/it]
                   all        136        264      0.678      0.666      0.685      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   249/1999      12.5G     0.0213    0.01216    0.00278         93        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.744       0.59      0.671      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   250/1999      12.5G    0.02168     0.0122   0.002832        110        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.734      0.624      0.682      0.348

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   251/1999      12.5G    0.02156    0.01183   0.002245         99        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.723      0.615      0.686      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   252/1999      12.5G    0.02169    0.01273   0.002728         88        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.676      0.622      0.683      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   253/1999      12.5G     0.0205    0.01152   0.001943        114        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.612      0.715      0.708      0.351

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   254/1999      12.5G    0.02214    0.01251     0.0023        130        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.727      0.601       0.67      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   255/1999      12.5G     0.0208    0.01164   0.002136        108        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.694      0.644      0.661      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   256/1999      12.5G    0.02151    0.01108   0.001928         93        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.697       0.62      0.691      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   257/1999      12.5G     0.0215    0.01207   0.002805         89        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264       0.71      0.618      0.673      0.342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   258/1999      12.5G    0.02192    0.01234   0.001888         96        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.736      0.626      0.691      0.364

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   259/1999      12.5G    0.02135     0.0114   0.002516         74        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.708      0.612      0.663      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   260/1999      12.5G     0.0218    0.01232   0.001877        109        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.746      0.601       0.67      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   261/1999      12.5G     0.0209    0.01209   0.002406        109        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264        0.7      0.623      0.663      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   262/1999      12.5G    0.02142    0.01235   0.002358        100        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.773      0.591      0.696      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   263/1999      12.5G    0.02256    0.01154   0.002718         92        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.767      0.543      0.661      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   264/1999      12.5G    0.02167    0.01237   0.002719        108        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.701      0.585      0.643      0.323

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   265/1999      12.5G    0.02221    0.01206   0.001876         97        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.699       0.59      0.637       0.32

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   266/1999      12.5G    0.02105    0.01168   0.003037         89        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.706      0.602      0.648      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   267/1999      12.5G    0.02086    0.01176   0.001962        117        640: 100% 9/9 [00:08<00:00,  1.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264        0.7      0.667      0.662      0.344

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   268/1999      12.5G    0.02241    0.01191   0.002439         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.693      0.666      0.676      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   269/1999      12.5G    0.02127    0.01163   0.002788         89        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.753      0.595      0.668      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   270/1999      12.5G    0.02103    0.01208    0.00232        110        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.677      0.606      0.644      0.329

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   271/1999      12.5G     0.0213    0.01226   0.002383        106        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.668      0.586      0.622      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   272/1999      12.5G    0.02114    0.01156    0.00237         85        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.696      0.594       0.63      0.324

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   273/1999      12.5G    0.02072    0.01168   0.001998         96        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264       0.68      0.611      0.646      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   274/1999      12.5G    0.02021     0.0114   0.001614        123        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.664      0.641      0.659      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   275/1999      12.5G    0.02145    0.01231   0.002294        122        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.732      0.607      0.645      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   276/1999      12.5G     0.0217    0.01181    0.00225        112        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.723      0.557      0.621      0.312

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   277/1999      12.5G    0.02126    0.01119   0.002634         81        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.727      0.599      0.644      0.324

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   278/1999      12.5G    0.02098    0.01141   0.002464        112        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.736        0.6      0.646      0.313

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   279/1999      12.5G    0.02145    0.01224   0.002627        120        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.06s/it]
                   all        136        264      0.721      0.658      0.682      0.323

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   280/1999      12.5G    0.02031    0.01177   0.001948        109        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.732       0.65      0.698      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   281/1999      12.5G    0.02039    0.01161   0.001688         83        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.693      0.674       0.72      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   282/1999      12.5G    0.02081    0.01224   0.002009        118        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.736      0.684      0.709      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   283/1999      12.5G    0.02087    0.01205   0.001861        109        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.697      0.671      0.683      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   284/1999      12.5G    0.01987    0.01203   0.002198        114        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.656       0.67      0.677      0.358

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   285/1999      12.5G    0.02033     0.0112   0.001723        129        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.728      0.622      0.677      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   286/1999      12.5G    0.02069    0.01129   0.001672         98        640: 100% 9/9 [00:07<00:00,  1.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.27s/it]
                   all        136        264      0.709      0.635      0.673      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   287/1999      12.5G    0.02056    0.01095   0.001917         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.704      0.618      0.657      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   288/1999      12.5G    0.02059    0.01146   0.001743        105        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.682       0.64      0.663      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   289/1999      12.5G    0.02059    0.01093   0.001519         82        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.645      0.641      0.651      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   290/1999      12.5G    0.02171    0.01198   0.002106        116        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.681      0.584      0.634      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   291/1999      12.5G     0.0216    0.01141   0.002074         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.763      0.563      0.669       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   292/1999      12.5G    0.02135    0.01167   0.002103         92        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.752      0.544      0.655       0.34

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   293/1999      12.5G    0.02104    0.01138   0.002154        100        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.672      0.632      0.666      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   294/1999      12.5G    0.02072    0.01165   0.001847        111        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.702      0.594      0.633      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   295/1999      12.5G    0.02006     0.0109   0.001586         99        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.729      0.599      0.639      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   296/1999      12.5G    0.02094    0.01056    0.00169         93        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.695      0.632       0.66      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   297/1999      12.5G    0.02026    0.01085   0.001376        108        640: 100% 9/9 [00:06<00:00,  1.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.656       0.67      0.665      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   298/1999      12.5G    0.02147     0.0115    0.00191        113        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.677      0.594      0.627       0.32

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   299/1999      12.5G    0.02179     0.0116   0.002261         97        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.664      0.592       0.62      0.306

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   300/1999      12.5G    0.01978    0.01192   0.002196        138        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.788      0.553      0.651      0.318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   301/1999      12.5G    0.02114    0.01181   0.002212        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.716      0.606      0.658      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   302/1999      12.5G    0.01962     0.0108   0.001621         94        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264       0.75       0.63      0.655       0.33

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   303/1999      12.5G    0.02045    0.01102   0.002262         80        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264       0.73      0.655      0.689      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   304/1999      12.5G    0.02017    0.01097   0.001798        101        640: 100% 9/9 [00:06<00:00,  1.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.777      0.634      0.679      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   305/1999      12.5G     0.0199    0.01124   0.002001         88        640: 100% 9/9 [00:07<00:00,  1.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.19s/it]
                   all        136        264      0.688      0.655      0.666      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   306/1999      12.5G    0.02055    0.01113   0.002259        110        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.685      0.653      0.672      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   307/1999      12.5G    0.02023    0.01137   0.001822        124        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.649      0.651      0.661      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   308/1999      12.5G    0.02011    0.01078   0.001891        100        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.663       0.65      0.644       0.33

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   309/1999      12.5G     0.0205    0.01098   0.001666        106        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.644      0.636      0.623      0.304

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   310/1999      12.5G    0.02055    0.01089   0.001705         82        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.668      0.671      0.658      0.329

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   311/1999      12.5G    0.01989    0.01126   0.001768        119        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.686      0.665      0.677      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   312/1999      12.5G    0.01966    0.01109   0.001972        126        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.665      0.625      0.641      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   313/1999      12.5G    0.02003    0.01085   0.002106        112        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.18it/s]
                   all        136        264      0.735      0.587      0.658       0.32

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   314/1999      12.5G     0.0199    0.01149   0.002507        122        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.731      0.596      0.656      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   315/1999      12.5G    0.02019    0.01084   0.002386        113        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.684      0.641      0.659      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   316/1999      12.5G    0.01983    0.01117    0.00217         92        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.11it/s]
                   all        136        264      0.701      0.633      0.673      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   317/1999      12.5G    0.02013      0.011   0.001969         84        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.754      0.559      0.652       0.34

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   318/1999      12.5G    0.02113    0.01173   0.002375         97        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.13it/s]
                   all        136        264      0.671      0.625      0.642      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   319/1999      12.5G    0.01966    0.01048   0.001814         89        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.648      0.654      0.653      0.332

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   320/1999      12.5G    0.02003    0.01053   0.001996         86        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.645      0.616      0.631      0.332

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   321/1999      12.5G    0.01985    0.01105   0.001636         87        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.667      0.626      0.636      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   322/1999      12.5G    0.01996    0.01222   0.001736         96        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.766      0.578      0.636      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   323/1999      12.5G    0.01902    0.01115   0.002135        120        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.37s/it]
                   all        136        264      0.709      0.573      0.611      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   324/1999      12.5G    0.01977    0.01052   0.001842        122        640: 100% 9/9 [00:07<00:00,  1.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.18s/it]
                   all        136        264      0.771      0.593      0.669      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   325/1999      12.5G    0.02016    0.01158   0.002066        101        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.672      0.658      0.675      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   326/1999      12.5G    0.01973     0.0119   0.002223        130        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.704      0.632      0.653      0.329

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   327/1999      12.5G    0.01998    0.01127   0.001859        105        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.668      0.657      0.661      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   328/1999      12.5G    0.01971    0.01109   0.001772         93        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.705      0.618      0.648      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   329/1999      12.5G    0.02087    0.01129   0.002351        135        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.678      0.594      0.626      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   330/1999      12.5G    0.01987      0.011    0.00248        114        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.637      0.662      0.633      0.324

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   331/1999      12.5G    0.01985    0.01107   0.001914        107        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264       0.66       0.56      0.628      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   332/1999      12.5G    0.02072    0.01162   0.002709        113        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.728      0.542      0.613      0.323

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   333/1999      12.5G    0.02005    0.01023   0.002243         99        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264       0.65      0.626      0.614       0.32

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   334/1999      12.5G    0.01997    0.01079   0.002204        106        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.07s/it]
                   all        136        264      0.666      0.605      0.643       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   335/1999      12.5G    0.01987    0.01081   0.002578         69        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.738      0.603      0.676      0.352

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   336/1999      12.5G    0.01981      0.011   0.001984        108        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.718      0.596      0.665      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   337/1999      12.5G    0.01981    0.01165   0.002199        119        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.685      0.603      0.651      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   338/1999      12.5G    0.01979    0.01096   0.001666        110        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.707      0.616      0.633      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   339/1999      12.5G    0.02026     0.0116   0.002302         94        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.752      0.626      0.686      0.351

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   340/1999      12.5G    0.01866    0.01041   0.001886        103        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.16it/s]
                   all        136        264      0.703      0.595       0.65      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   341/1999      12.5G    0.01932     0.0109   0.001772        110        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.694       0.59      0.646      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   342/1999      12.5G    0.01911    0.01111   0.002241        113        640: 100% 9/9 [00:06<00:00,  1.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.56s/it]
                   all        136        264      0.763      0.555      0.651      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   343/1999      12.5G     0.0198     0.0116   0.002144        119        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.721      0.613      0.665      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   344/1999      12.5G    0.02038    0.01177   0.002251        139        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.723      0.598       0.66      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   345/1999      12.5G    0.02092     0.0111    0.00226         73        640: 100% 9/9 [00:07<00:00,  1.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.27it/s]
                   all        136        264      0.731      0.584      0.651      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   346/1999      12.5G    0.01995    0.01101   0.002061        115        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.639      0.641      0.631       0.32

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   347/1999      12.5G    0.02045    0.01124   0.002664         94        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264       0.73      0.605      0.657      0.326

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   348/1999      12.5G    0.01973    0.01104   0.003027        119        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.747      0.601      0.671      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   349/1999      12.5G    0.02045     0.0107   0.002261         85        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.698      0.638      0.643      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   350/1999      12.5G    0.02046    0.01115   0.002153        112        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.695       0.61      0.659      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   351/1999      12.5G    0.01963    0.01054   0.001592        108        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.708       0.63      0.682       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   352/1999      12.5G    0.01924    0.01056     0.0016         83        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.661      0.722      0.737       0.36

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   353/1999      12.5G    0.01889     0.0106   0.001809         94        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.738      0.597      0.704      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   354/1999      12.5G    0.02054    0.01055   0.002063        103        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.743      0.619      0.695      0.357

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   355/1999      12.5G    0.01905    0.01111   0.001956        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.655      0.657      0.686      0.352

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   356/1999      12.5G    0.02068    0.01136   0.002114         83        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.631      0.642      0.666      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   357/1999      12.5G    0.01907    0.01069   0.002291        101        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.672      0.628      0.661      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   358/1999      12.5G    0.01895    0.01121   0.001838        113        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.689      0.622       0.68      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   359/1999      12.5G    0.01876    0.01087   0.001697         83        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.711      0.589      0.664      0.352

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   360/1999      12.5G    0.01909    0.01066   0.001583         91        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.697      0.561      0.636      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   361/1999      12.5G    0.01945    0.01015   0.001684         97        640: 100% 9/9 [00:07<00:00,  1.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.11s/it]
                   all        136        264      0.736      0.581      0.655      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   362/1999      12.5G     0.0188    0.01104   0.001219        123        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.686      0.651      0.671      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   363/1999      12.5G    0.01908    0.01073   0.001615        109        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.765      0.613      0.682      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   364/1999      12.5G    0.02005    0.01085   0.001268         87        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.755      0.568      0.663      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   365/1999      12.5G    0.01877    0.01069    0.00151         81        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.00s/it]
                   all        136        264      0.663      0.624      0.652      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   366/1999      12.5G    0.01865    0.01027    0.00151        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.672      0.616      0.656      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   367/1999      12.5G    0.01929    0.01025   0.002055        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.674      0.587      0.642      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   368/1999      12.5G    0.01938    0.01021   0.001473         87        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.652      0.563      0.597      0.303

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   369/1999      12.5G    0.01855    0.01036   0.002031        108        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.725      0.566      0.634      0.324

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   370/1999      12.5G    0.01941    0.01098   0.001858        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.765      0.587      0.655      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   371/1999      12.5G    0.01841    0.01041   0.001477         90        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264       0.68      0.602      0.651      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   372/1999      12.5G    0.01905    0.01096   0.001622        108        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.701      0.599      0.649      0.319

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   373/1999      12.5G    0.01936    0.01031   0.001582         90        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.703       0.63      0.657      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   374/1999      12.5G    0.02004    0.01116   0.001722        122        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.759      0.609      0.662      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   375/1999      12.5G    0.01869    0.01111   0.002204        109        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.10s/it]
                   all        136        264      0.696      0.649      0.654       0.33

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   376/1999      12.5G    0.01872    0.01102   0.001945        112        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.02s/it]
                   all        136        264      0.709      0.622      0.656      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   377/1999      12.5G    0.01849    0.01066    0.00177        120        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.727      0.646       0.69      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   378/1999      12.5G    0.01864    0.01113   0.001782        116        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.744      0.637      0.688      0.351

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   379/1999      12.5G    0.01858    0.00992   0.001313         82        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.57s/it]
                   all        136        264      0.678      0.656      0.661      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   380/1999      12.5G    0.01869    0.01156   0.001847        125        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.10s/it]
                   all        136        264      0.721      0.611      0.661      0.344

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   381/1999      12.5G    0.01825    0.01051   0.001668         88        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.784      0.579      0.671      0.348

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   382/1999      12.5G    0.01873    0.01039   0.001792        107        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.726      0.658      0.679      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   383/1999      12.5G    0.01886    0.01091   0.001492        113        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.698       0.65      0.681      0.351

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   384/1999      12.5G    0.01907    0.01027   0.001804         96        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.724      0.617      0.675      0.342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   385/1999      12.5G    0.01798    0.01028   0.001308         93        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.751      0.611      0.686      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   386/1999      12.5G    0.01942    0.01049   0.001204         98        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.712      0.669      0.684      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   387/1999      12.5G    0.01851    0.01064   0.002122        105        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.732      0.589      0.674      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   388/1999      12.5G    0.01898    0.01096   0.001999        121        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.709      0.581       0.64      0.329

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   389/1999      12.5G    0.01883    0.01041    0.00196        120        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.04s/it]
                   all        136        264      0.665      0.635      0.674      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   390/1999      12.5G    0.01856    0.01082   0.002217        125        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.669      0.622      0.664      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   391/1999      12.5G    0.01885    0.01027   0.002187        107        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.667      0.606      0.653      0.326

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   392/1999      12.5G    0.01866    0.01026   0.001714         99        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264       0.65      0.619       0.66      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   393/1999      12.5G    0.01829    0.01058   0.001417        118        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264       0.68      0.616      0.655      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   394/1999      12.5G    0.01902    0.01034   0.001842         96        640: 100% 9/9 [00:06<00:00,  1.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.765       0.61      0.701      0.348

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   395/1999      12.5G    0.01889    0.01116   0.001469        102        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.733      0.627      0.692      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   396/1999      12.5G    0.01846    0.01052    0.00172        117        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.722      0.602      0.674      0.344

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   397/1999      12.5G    0.01842    0.01052   0.001267        118        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.705      0.667      0.671       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   398/1999      12.5G    0.01909    0.01128   0.001677        125        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.57s/it]
                   all        136        264      0.669      0.687      0.702      0.362

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   399/1999      12.5G    0.01972    0.01092   0.001333        122        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:04<00:00,  2.00s/it]
                   all        136        264      0.711       0.63      0.672      0.344

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   400/1999      12.5G    0.01873    0.01069   0.001434        106        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.687      0.694      0.694      0.342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   401/1999      12.5G    0.01894    0.01058   0.001287        104        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.19it/s]
                   all        136        264      0.725      0.638       0.68      0.332

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   402/1999      12.5G    0.01824    0.01017   0.001299         87        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264       0.72      0.664      0.715      0.335

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   403/1999      12.5G    0.01849     0.0106   0.001553        109        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.693      0.656      0.699       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   404/1999      12.5G    0.01863    0.01076    0.00161         85        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.692      0.651      0.703      0.348

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   405/1999      12.5G    0.01855    0.01009   0.001365         91        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.739      0.615      0.703      0.344

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   406/1999      12.5G    0.01856    0.01053   0.001716        109        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.735      0.597      0.686      0.351

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   407/1999      12.5G    0.01893    0.01132   0.001455        116        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.691      0.643       0.68       0.34

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   408/1999      12.5G    0.01807    0.01028   0.001629         98        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.689      0.651      0.662      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   409/1999      12.5G    0.01816     0.0106   0.001393        103        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.12it/s]
                   all        136        264      0.689      0.627      0.667       0.34

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   410/1999      12.5G     0.0191    0.01042   0.001762        100        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.751       0.61      0.668      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   411/1999      12.5G    0.01864     0.0104   0.002313        121        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.693      0.629      0.664      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   412/1999      12.5G    0.01876    0.01096   0.001588         94        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.723      0.589      0.636       0.33

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   413/1999      12.5G    0.01797    0.01016   0.001523         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.728      0.571      0.637      0.332

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   414/1999      12.5G    0.01893    0.01094   0.002235        105        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.715      0.607      0.675      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   415/1999      12.5G    0.01893     0.0106    0.00165        122        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.684      0.651      0.668      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   416/1999      12.5G    0.01966    0.01034   0.001833        100        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.695      0.625      0.671      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   417/1999      12.5G    0.01914    0.01067   0.001322        117        640: 100% 9/9 [00:07<00:00,  1.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.05s/it]
                   all        136        264      0.707      0.618      0.663      0.327

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   418/1999      12.5G    0.01779    0.01086   0.001692        108        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                   all        136        264      0.652      0.631      0.657      0.331

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   419/1999      12.5G    0.01809    0.01041   0.001545        101        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.01s/it]
                   all        136        264      0.728       0.59      0.657      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   420/1999      12.5G    0.01869     0.0103   0.002132         98        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all        136        264      0.704      0.606      0.659      0.325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   421/1999      12.5G    0.01865    0.01048   0.001896        108        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.678      0.679      0.678      0.328

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   422/1999      12.5G    0.01936    0.01031   0.001748        123        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.699      0.641      0.676      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   423/1999      12.5G    0.01858    0.01083   0.001963        116        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.14it/s]
                   all        136        264      0.716      0.618      0.667      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   424/1999      12.5G    0.01824    0.01058    0.00221         93        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.693      0.633      0.659      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   425/1999      12.5G    0.01857    0.01051   0.002004         78        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.708      0.629      0.665      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   426/1999      12.5G    0.01882    0.01055   0.001589        100        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.744        0.6       0.67      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   427/1999      12.5G    0.01942    0.01029   0.001469        111        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.682      0.649      0.671      0.342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   428/1999      12.5G    0.01815    0.01056   0.001628         93        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264       0.68      0.657      0.669      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   429/1999      12.5G    0.01838    0.01047   0.001417         99        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.723      0.633      0.685      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   430/1999      12.5G    0.01793    0.01013   0.001346        101        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.745      0.637      0.694      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   431/1999      12.5G    0.01886    0.01031   0.002122        101        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.11it/s]
                   all        136        264      0.783      0.619      0.681      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   432/1999      12.5G    0.01843    0.01032   0.001406         93        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.00it/s]
                   all        136        264      0.725      0.662      0.707      0.354

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   433/1999      12.5G    0.01805    0.01031   0.001563        116        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264       0.71      0.711      0.711      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   434/1999      12.5G    0.01889   0.009962   0.001516        103        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.695      0.691       0.71      0.355

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   435/1999      12.5G    0.01802    0.01009   0.001363        125        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:03<00:00,  1.59s/it]
                   all        136        264      0.702      0.649       0.68      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   436/1999      12.5G    0.01814   0.009868   0.001479         95        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.699      0.621       0.68      0.348

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   437/1999      12.5G    0.01828    0.01062   0.001665        118        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.689      0.672      0.691      0.351

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   438/1999      12.5G    0.01802    0.01056   0.001536        116        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.06it/s]
                   all        136        264      0.702       0.64      0.671      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   439/1999      12.5G    0.01781     0.0099  0.0009814         89        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.678      0.658      0.674      0.346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   440/1999      12.5G    0.01845    0.01031   0.001251        112        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.03it/s]
                   all        136        264      0.709      0.637       0.67      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   441/1999      12.5G    0.01816    0.01005   0.001715         92        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.775      0.609      0.692      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   442/1999      12.5G    0.01879    0.01044   0.001478         96        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.677      0.643       0.66      0.339

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   443/1999      12.5G    0.01822    0.01033   0.001658        104        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.20it/s]
                   all        136        264      0.754      0.604      0.674      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   444/1999      12.5G     0.0182    0.01008   0.002042         89        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.05it/s]
                   all        136        264      0.681      0.682      0.696      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   445/1999      12.5G    0.01765    0.01032   0.001715        102        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.01it/s]
                   all        136        264      0.683      0.697      0.707      0.356

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   446/1999      12.5G     0.0188    0.01053   0.001743         87        640: 100% 9/9 [00:06<00:00,  1.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.09it/s]
                   all        136        264      0.737      0.631      0.692      0.342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   447/1999      12.5G    0.01779    0.01025   0.001325         89        640: 100% 9/9 [00:06<00:00,  1.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264      0.729      0.652      0.703      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   448/1999      12.5G    0.01826    0.01042   0.001439        111        640: 100% 9/9 [00:06<00:00,  1.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.13it/s]
                   all        136        264      0.696       0.62      0.687      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   449/1999      12.5G    0.01786   0.009847   0.001705        110        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.08it/s]
                   all        136        264      0.715       0.67      0.706      0.353

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   450/1999      12.5G    0.01768    0.01031   0.001108        105        640: 100% 9/9 [00:06<00:00,  1.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.04it/s]
                   all        136        264      0.716      0.649      0.693      0.347

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   451/1999      12.5G    0.01852   0.009759   0.001569        100        640: 100% 9/9 [00:06<00:00,  1.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.17it/s]
                   all        136        264      0.716       0.64       0.69      0.354

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   452/1999      12.5G    0.01728    0.01007   0.001507        110        640: 100% 9/9 [00:06<00:00,  1.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.10it/s]
                   all        136        264      0.718      0.618      0.698      0.362
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 352, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

453 epochs completed in 1.158 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.5MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.5MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.02it/s]
                   all        136        264       0.66      0.722      0.736       0.36
                  bike        136         22      0.697      0.864      0.859      0.459
                   car        136         87       0.71      0.667      0.698      0.489
                 child        136         27      0.673      0.667      0.742      0.278
                   man        136         49      0.646      0.531      0.622      0.233
            motorcycle        136         45      0.887      0.871      0.894      0.482
                 woman        136         34      0.348      0.735      0.602      0.216
Results saved to runs/train/exp
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▁▅▆▇▄▆▆▆▇▇▆██▇▇▇▆▇▇██▇▇▇▇▇▇▆▇▇▇▇▇▇█▇▇▇█
wandb: metrics/mAP_0.5:0.95 ▁▁▅▅▆▄▆▆▆▇▇▇▇█▇▇▇▇█▇██▇███▇█▇█▇█▇█▇█████
wandb:    metrics/precision ▁▂▄▆▆▆▇▇▇▇▇▇▇▇▇█▇▇▇██▇▇▇█▇▇▇▇██▇▇▇▇▇▇▇▇▇
wandb:       metrics/recall ▁▄▆▄▆▁▄▃▄▇▅▄▆█▅▄▆▄▇▆▇▇▇▇▆▆▅▇▅▆▆▅▅▇▆▇▇▇▇▇
wandb:       train/box_loss █▆▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▁▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/cls_loss █▅▄▃▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/obj_loss █▅▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         val/box_loss █▅▄▂▁▇▅▄▄▄▅▃▄▃▄▄▃▃▃▂▃▄▄▃▂▃▄▄▃▄▄▄▃▃▃▅▄▄▅▃
wandb:         val/cls_loss ▆▃▁▁▂▃▂▆▃▅▄▅▄▅▁▄▄█▅▂▅█▃▄█▆▄▅▇▆▄▅▄▆▇▅▄▆▄▆
wandb:         val/obj_loss █▁▁▂▂▂▃▃▃▂▂▃▂▂▂▄▅▄▃▄▃▄▃▃▄▃▆▃▇▄▄▅▆▃▅▄▄▄▅▆
wandb:                x/lr0 █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                x/lr1 ▁█████████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆
wandb:                x/lr2 ▁█████████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆
wandb: 
wandb: Run summary:
wandb:           best/epoch 352
wandb:         best/mAP_0.5 0.7366
wandb:    best/mAP_0.5:0.95 0.36022
wandb:       best/precision 0.66095
wandb:          best/recall 0.72225
wandb:      metrics/mAP_0.5 0.73626
wandb: metrics/mAP_0.5:0.95 0.35954
wandb:    metrics/precision 0.66011
wandb:       metrics/recall 0.72227
wandb:       train/box_loss 0.01728
wandb:       train/cls_loss 0.00151
wandb:       train/obj_loss 0.01007
wandb:         val/box_loss 0.05528
wandb:         val/cls_loss 0.03459
wandb:         val/obj_loss 0.01286
wandb:                x/lr0 0.00777
wandb:                x/lr1 0.00777
wandb:                x/lr2 0.00777
wandb: 
wandb: Synced divine-monkey-3: https://wandb.ai/rgp/YOLOv5/runs/2knv862j
wandb: Synced 5 W&B file(s), 79 media file(s), 1 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20221102_203240-2knv862j/logs

  ```
  
</details>

### Evidências do treinamento

Métricas do treinamento:
<details>
  <summary>Click to expand!</summary>
  
  ```text
     Total de 452 épocas - Melhor época: 352
  ```
  ![Descrição](https://github.com/rgprado/RedesNeuraisRGP/blob/main/assets/Me%CC%81tricas%20do%20treinamento.png)
</details>

Matriz de confusão:
<details>
  <summary>Click to expand!</summary>
  
  ```text
    Matriz de confusão das objetos detectados
  ```
  ![Descrição](https://github.com/rgprado/RedesNeuraisRGP/blob/main/assets/confusion_matrix.png)
</details>

Validação do Modelo:
<details>
  <summary>Click to expand!</summary>
  
  ```text
    Validação do Batch-0
  ```
  ![Descrição](https://github.com/rgprado/RedesNeuraisRGP/blob/main/assets/val_batch0_pred.jpg)
</details>
<details>
  <summary>Click to expand!</summary>
  
  ```text
    Validação do Batch-1
  ```
  ![Descrição](https://github.com/rgprado/RedesNeuraisRGP/blob/main/assets/val_batch1_pred.jpg)
</details>




## Roboflow
Dataset fotos coloridas
[Roboflow - Stree View Detection](https://app.roboflow.com/cesarschool/carviewdetection/3)

Dataset fotos em escala de cinza
[Roboflow - Stree View Detection - GrayScale](https://app.roboflow.com/cesarschool/carviewdetection/4)

## HuggingFace

HuggingFace - 
[Cesar - Street View Detection](https://huggingface.co/spaces/rgp/Street-View-Detection)

HuggingFace - GrayScale
[Cesar - Street View Detection](https://huggingface.co/spaces/rgp/Street-View-Detection-GrayScale)
