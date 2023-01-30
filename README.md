## 概要
ECoGデータを用いて事前学習済したモデルを脳の電極の識別のタスクに応用。


## 使い方  
このレポジトリのディレクトリを`$share`とする。  
例：`$share=/home/hoge/share` (ホームディレクトリのエイリアス`~`は使用不可)

0. インストール ([参考](https://github.com/s3prl/s3prl#installation))  
```
cd $share
pip install -e .
```

1. 事前学習
 1-1
 /$share/pretrain/tera/config_runner.yamlのrootの部分を書き換える
 (ここではgeorge_20120724_session1_CommonAve_ReginLabelを用いる。)
 1-2 
 cd /home/hoge/$share　でカレントディレクトリを移動
 1-3
 `$TERA_ECOG/s3prl/downstream/speaker_linear_utter_libri/config.yaml`内の`$TERA_ECOG`の部分を書き換える。
 (ここではgeorge_20120724_session1_CommonAve_ReginLabelを用いる。)
 1-4
 python run_pretrain.py -u tera -g pretrain/tera/config_model.yaml -n 'resultに保存するファイル名'
 で事前学習の実行

2. Fine-Tunig  
`cd s3prl`
`python3 run_downstream.py -m train -n ExpName -u tera_local -k $TERA_ECOG/s3prl/result/pretrain/YourModelName/states-500000.ckpt -d speaker_linear_utter_libri`  
といったようなコマンドで学習が実行される

3. テスト  
`python3 run_downstream.py -m evaluate -t "test" -e '$share/result/downstream/ExpName/states-250.ckpt'`  
といったようなコマンドでテストが実行される


## Troubleshooting

### CUDA error
```RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```
PyTorchとCUDAのバージョンが合っていないため、[公式](https://pytorch.org/get-started/locally/)に従って再インストールする。
```
pip uninstall torch torchvision torchaudio
pip install ...
```
