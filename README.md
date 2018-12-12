# VNect VRM Unity Sample

VNect で推定した姿勢を VRM ファイルに反映するサンプルです。

ウェブカムや動画ファイルの映像をリアルタイムで処理できます。

## 環境構築

- VRM は VRoid からエクスポートしたモデルを使用しています。

https://vroid.pixiv.net/

- UniVRM を使用しています。

下記のリポジトリから unitypackage をダウンロードし、 Unity にインポートしてください

https://github.com/dwango/UniVRM/releases

- TensorFlowSharp を使用しています。

下記のリポジトリから unitypackage をダウンロードし、 Unity にインポートしてください

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Basic-Guide.md

TensorFlowSharp に含まれる libtensorflow.dll は GPU 用に差し替えると高速化できます。

DLL は下記などで公開されています。 SIMD や CUDA 、 cuDNN がお使いの環境と合ったものを選択してください。

https://github.com/fo40225/tensorflow-windows-wheel

- TensorFlowSharp を使うため、 VNect のウェイトを Caffe 用から TensorFlow 用に変換します

1. 下記のスクリプトで .caffemodel を .pkl に変換します

https://github.com/timctho/VNect-tensorflow/blob/master/caffe_weights_to_pickle.py

2. 下記のスクリプトを修正し、 .pkl を .txt に変換します

https://github.com/timctho/VNect-tensorflow/blob/master/models/vnect_model.py

saver = tf.train.Saver() の後に以下の処理を追加してください。

```
model.load_weights(sess, model_file)
tf.train.write_graph(sess.graph.as_graph_def(), str('./'), 'input_graph.txt', as_text=True)
save_path = saver.save(sess, "./tf_vnect")
```

3. freeze_graph.py で Variable を Const に変更します

```
python path\to\freeze_graph.py --input_graph=path\to\input_graph.txt --input_checkpoint=path\to\tf_vnect --output_graph=path\to\frozen_graph.bytes --output_node_names=Placeholder,split_2
```

frozen_graph.bytes が出力されるので、 Unity の Resources フォルダに入れてください。

## 参考

- VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera

http://gvv.mpi-inf.mpg.de/projects/VNect/

- timctho/VNect-tensorflow

https://github.com/timctho/VNect-tensorflow

## ライセンス

ライブラリやモデル、ウェイト、学習データなどのライセンスは各配布元に従ってください。
