# Shopee Code League: Product Detection

## Prepare data

1. `mkdir data`
2. Download the data ([Link 1](https://drive.google.com/drive/folders/1V_sHZN2MmhcfeVao3hoJpjRWLSnm1_Pe), [Link 2](https://drive.google.com/drive/folders/1PIJHZ6QXU5rjskT7dIimyEUdQJ84YvoR))
3. Extract the zip into the newly created `data`
4. `python split.py -csv data/shopee-product-detection-dataset/train.csv -out data/shopee-product-detection-dataset/`

## Train

Edit the `configs/train/shopee.yaml` file for configuration.

To start training, run:
```
    python train.py --config configs/train/shopee.yaml --gpus 0
```

Use Tensorboard on the `runs` directory (auto generated) to monitor the training processes. For example, run:
```
    tensorboard --logdir runs --port 3698
```
and view Tensorboard at `localhost:3698` (on your web browser).

## Evaluate

Run:

```
    python val.py -g 0 -d data/shopee-product-detection-dataset/train/train -c data/list/train_val.csv -w backup/baseline/best_metric_Accuracy.pth
```

## Test

Run:

```
    python test.py -d data/shopee-product-detection-dataset/test/test -w backup/baseline/best_metric_Accuracy.pth -g 0
```