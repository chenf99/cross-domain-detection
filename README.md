```bash
python3 pseudo_label.py --root ./dataset/clipart --data_folder ./json/clipart/ --result ./dataset/clipart_dt_pl/ --gpu 7 --checkpoint result/clipart_dt.pth.tar 

python3 eval.py --data_folder ./json/clipart/ --checkpoint result/clipart.tar --gpu 7 

python3 train.py --data_folder ./json/clipart_dt_pl/ --checkpoint result/clipart_dt.pth.tar --gpu 5 --iteration 20000 --result result/clipart_dt_pl 
```

|          | clipart | watercolor | comic |
| :------: | :-----: | :--------: | :---: |
| baseline | 28.5 | 50.1 | 23.1 |
| ideal case | 53.3 | 59.1 | 44.7 |
| DT       | 35.8 | 46.3 | 29.3 |
| DT + PL  | 45.1 | 56.9 | 39.7 |
| P_DT       | 37.1 | 49.4 | 29.9 |
| P_DT + PL  | 45.7 | 57.6 | 39.9 |

P_DT implies joint training with PASCAL VOC dataset
