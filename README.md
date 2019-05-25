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

bin フォルダの tensorflow.dll を libtensorflow.dll にリネームし、既存ファイルと差し替えてください。

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
python path\to\freeze_graph.py --input_graph=path\to\input_graph.txt --input_checkpoint=path\to\tf_vnect --output_graph=path\to\vnect_frozen.bytes --output_node_names=Placeholder,split_2
```

vnect_frozen.bytes が出力されるので、 Unity の Resources フォルダに入れてください。

## 参考

- VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera

http://gvv.mpi-inf.mpg.de/projects/VNect/

- timctho/VNect-tensorflow

https://github.com/timctho/VNect-tensorflow

## ライセンス

ライブラリやモデル、ウェイト、学習データなどのライセンスは各配布元に従ってください。

## Overview

A sample that reflects the attitude estimated by VNect in the VRM file.
You can process video of webcam and movie files in real time.

## Environment
- VRM uses the model exported from VRoid.

https://vroid.pixiv.net/

- I am using UniVRM.
Download unitypackage from the following repository and import it to Unity

https://github.com/dwango/UniVRM/releases

- I am using TensorFlowSharp
Download unitypackage from the following repository and import it to Unity

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Basic-Guide.md

You can speed up libtensorflow.dll included in TensorFlowSharp by replacing it for GPU.
DLL is released as below. Please select the one that SIMD, CUDA, cuDNN matches your environment.

https://github.com/fo40225/tensorflow-windows-wheel

Rename tensorflow.dll in bin folder to libtensorflow.dll and replace with existing file.

In order to use TensorFlowSharp, we convert the weight of VNect for Caffe for TensorFlow

1. Convert .caffemodel to .pkl with the following script

https://github.com/timctho/VNect-tensorflow/blob/master/caffe_weights_to_pickle.py

2. Modify the script below and convert. Pkl to. Txt

https://github.com/timctho/VNect-tensorflow/blob/master/models/vnect_model.py

Please add the following processing after saver = tf.train.Saver ().
  ````
  model.load_weights(sess, model_file)
  tf.train.write_graph(sess.graph.as_graph_def(), str('./'), 'input_graph.txt', as_text=True)
  save_path = saver.save(sess, "./tf_vnect")
  ````
  3. In freeze_graph.py change Variable to Const
  ````
  python path\to\freeze_graph.py --input_graph=path\to\input_graph.txt --input_checkpoint=path\to\tf_vnect --output_graph=path\to\vnect_frozen.bytes --output_node_names=Placeholder,split_2
  ````
Since vnect_frozen.bytes is output, please put it in Unity's Resources folder.


# references

- VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera

http://gvv.mpi-inf.mpg.de/projects/VNect/

- timctho/VNect-tensorflow

https://github.com/timctho/VNect-tensorflow


# license
Licenses such as libraries, models, weights, learning data, etc. should follow the distributor.
