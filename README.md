# AI Painter

CycleGANを用いた画像スタイル変換アプリ

[こちら](https://share.streamlit.io/navitacion/ai-painter/app.py)
からお試し

## アプリについて

本アプリはCycleGANによる画像スタイルを変換することができます。

モネやゴッホなどといった画家のスタイルを学習することで、写真を絵画調にリアレンジします。

## 使用データ

データについては下記のものを使用しています。

- [モネ](https://www.kaggle.com/c/gan-getting-started)

- [ゴッホ](https://www.kaggle.com/ipythonx/van-gogh-paintings)


## 起動方法（ローカル）

下記コマンドを実行する

```
docker build -t ai-painter .
docker run -it --rm -v $(pwd):/workspace ai-painter -p 8501:8501 bash
streamlit run app.py
```

実行後、下記URLにアクセスする

```
http://localhost:8501/
```


## 学習要件

- エポック数: 500
- 学習率：0.0002
- 画像サイズ: (256, 256)

## 学習ログ

[こちら](https://www.comet.ml/navitacion/ai-painter-train/view/new)
を参照してください
