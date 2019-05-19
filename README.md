# VNect VRM Unity Sample

A sample that reflects the attitude estimated by VNect in the VRM file.
You can process video of webcam and movie files in real time.

## Environment
-VRM uses the model exported from VRoid.
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


- In order to use TensorFlowSharp, we convert the weight of VNect for Caffe for TensorFlow
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


#license
Licenses such as libraries, models, weights, learning data, etc. should follow the distributor.
