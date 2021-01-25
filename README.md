# GNEM: A Generic One-to-Set Neural Entity Matching Framework

### Training
```
python train.py --seed 28 --log_freq 5 --lr 0.0001 --embed_lr 0.00002 --epochs 10 --batch_size 2 \
                --tableA_path data/abt_buy/tableA.csv --tableB_path data/abt_buy/tableB.csv \
                --train_path data/abt_buy/train.csv --test_path data/abt_buy/test.csv \
                --val_path data/abt_buy/valid.csv --gpu 0 1 \
                --gcn_layer 1 --test_score_type mean min max
```

### Testing
Pre-trained model can be found 
```
python test.py  --gcn_layer 1 --tableA_path data/abt_buy/tableA.csv --tableB_path data/abt_buy/tableB.csv \
                --train_path data/abt_buy/train.csv --test_path data/abt_buy/test.csv \
                --val_path data/abt_buy/valid.csv --gpu 0 --checkpoint_path <path_to_checkpoint>
```
