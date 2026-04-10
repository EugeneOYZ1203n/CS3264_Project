Base model
```python train.py --data_root ./asl-signs --save_dir checkpoints/google_asl_base --scheduler reduceonplateau```

Stoch Drop
```python train.py --data_root ./asl-signs --save_dir checkpoints/google_asl_stoch_drop --scheduler reduceonplateau --stoch_drop```