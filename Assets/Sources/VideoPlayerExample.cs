using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
using UnityEngine.UI;
using TensorFlow;

public class VideoPlayerExample :MonoBehaviour {
    //ウェブカムを使う場合はtrue。動画ファイルを使う場合はfalse
    public bool UseWebcam = false;

    public float ModelPositionX = 0.0f;
    public float ModelPositionY = 1.8f;

    //前フレームのジョイント位置に対して今フレームのジョイント位置候補が、
    //画面サイズの何割離れていたら誤検出とみなすかの距離
    public float JointDistanceLimit = 0.3f;

    //NNの出力をジョイントとみなす閾値
    public float JointThreshold = 0.3f;

    //前フレームに対してのスムージング
    public float Joint2DLerp = 0.5f;
    public float Joint3DLerp = 0.5f;

    //入力映像の色調整
    public float AdjInputR = 0.5f;
    public float AdjInputG = 0.5f;
    public float AdjInputB = 0.5f;

    //サイズを変更した入力画像を複数枚用意する。検出後に平均して推定誤差を減らすため
    public bool UseMultiScale = true;

    //閾値以上をラベリングして重心をジョイント候補にする。使わなくてもそこそこ精度は出る
    public bool UseLabeling = false;

    //次フレームの入力映像の注目領域として使い、フレーム間の誤差を減らす
    //不安定なので一旦保留
    public bool UseBoundingBox = false;

    //VRMに姿勢を反映する場合はtrue
    public bool EnableHead = false;
    public bool EnableNeck = false;
    public bool EnableArms = true;
    public bool EnableElbows = true;    
    public bool EnableWrists = false;
    public bool EnableLegs = true;
    public bool EnableKnees = false;
    public bool EnableFoots = false;
    public bool EnableToeBases = false;

    private Dictionary<string, bool> enableJoints = new Dictionary<string, bool>();

    //デバッグ用の描画フラグ
    public bool DrawInputTensorBuff = false;
    public bool DebugDrawHeatmap = false;
    public bool DebugDrawHeatmapBuff = false;
    public bool DrawHeatmapLabel = false;
    public bool DrawResults2D = true;
    public bool DrawResults3D = true;

    private DebugRenderer debugRenderer;

    private WebCamTexture webcamTexture;
    private RenderTexture videoTexture;
    private Texture2D texture;
    private float videoWidth, videoHeight;
    private Rect boundingBox;

    private bool isPosing;

    private VNectManager vnect = new VNectManager();
    private VRMManager vrm = new VRMManager();

    //ジョイントの各種情報
    private Dictionary<string, JointInfo> jointInfos = new Dictionary<string, JointInfo>();

    void Start() {
        Application.runInBackground = true;

        RectTransform rectTransform = GetComponent<RectTransform>();
        Renderer renderer = GetComponent<Renderer>();

        if (UseWebcam) {
            //ウェブカメラの設定初期化
            WebCamDevice[] devices = WebCamTexture.devices;
            webcamTexture = new WebCamTexture(devices[0].name);
            renderer.material.mainTexture = webcamTexture;
            webcamTexture.Play();

            texture = new Texture2D(webcamTexture.width, webcamTexture.height);

        } else {
            //動画ファイルの設定初期化
            VideoPlayer videoPlayer = GetComponent<VideoPlayer>();
            int width = (int)rectTransform.rect.width;
            int height = (int)rectTransform.rect.height;
            videoTexture = new RenderTexture(width, height, 24);
            videoPlayer.targetTexture = videoTexture;
            renderer.material.mainTexture = videoTexture;
            videoPlayer.Play();

            texture = new Texture2D(videoTexture.width, videoTexture.height);
        }

        //バウンディングボックスの初期値は入力映像の長辺の正方形
        videoWidth = texture.width;
        videoHeight = texture.height;
        float padWidth = (videoWidth > videoHeight) ? 0 : (videoHeight - videoWidth) / 2;
        float padHeight = (videoWidth > videoHeight) ? (videoWidth - videoHeight) / 2 : 0;
        //第三、四引数は幅、高さなので（右、上の位置ではないので）パディング分は二倍する
        boundingBox = new Rect(-padWidth, -padHeight, videoWidth + padWidth * 2, videoHeight + padHeight * 2);

        //ジョイント情報の初期化
        JointInfo.Init(jointInfos);

        //VNectのモデルを読み込み
        vnect.Init(jointInfos, UseMultiScale);

        //推定結果の描画用プレーン
        debugRenderer = GameObject.Find("DebugRenderer").GetComponent<DebugRenderer>();

        //VRoidのモデルを読み込み
        vrm.Init(jointInfos, ModelPositionX, ModelPositionY);
    }

    void Update() {
        if (UseWebcam) {
            //ウェブカメラの映像をテクスチャに反映
            var color32 = webcamTexture.GetPixels32();
            texture.SetPixels32(color32);
            texture.Apply();

        } else {
            //動画ファイルの映像をテクスチャに反映
            Graphics.SetRenderTarget(videoTexture);
            texture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
            texture.Apply();
            Graphics.SetRenderTarget(null);
        }

        //前フレームの処理が完了していたら今フレームの処理を呼び出す
        if (isPosing) { return; }
        isPosing = true;
        StartCoroutine("PoseUpdate", texture);
    }

    bool initialized = false;
    IEnumerator PoseUpdate(Texture2D texture) {
        //二フレーム目からバウンディングボックスを計算する
        //※最後に計算するとinputTensorとボックスのサイズに差異が出るためOnRenderObjectで範囲外アクセスが発生する
        if (UseBoundingBox && initialized) { vnect.UpdateBoundingBox(ref boundingBox, videoWidth, videoHeight); }
        initialized = true;

        Color adjColor = new Color(AdjInputR, AdjInputG, AdjInputB);
        Texture2D resizedTexture = ResizeTexture(texture);
        vnect.Update(resizedTexture, JointDistanceLimit, JointThreshold, Joint2DLerp, Joint3DLerp, adjColor, UseLabeling);
        Destroy(resizedTexture);

        isPosing = false;

        SetEnableFlags();
        vrm.Update(vnect.joint2D, vnect.joint3D, jointInfos, vnect.extractedJoints, enableJoints, vnect.NN_INPUT_WIDTH_MAX, vnect.NN_INPUT_HEIGHT_MAX);

        yield return null;
    }

    private void SetEnableFlags(){
        enableJoints.Clear();
        enableJoints["Head"] = EnableHead;
        enableJoints["Neck"] = EnableNeck;
        enableJoints["RightArm"] = EnableArms;
        enableJoints["LeftArm"] = EnableArms;
        enableJoints["RightElbow"] = EnableElbows;    
        enableJoints["LeftElbow"] = EnableElbows;    
        enableJoints["RightWrist"] = EnableWrists;
        enableJoints["LeftWrist"] = EnableWrists;
        enableJoints["RightLeg"] = EnableLegs;
        enableJoints["LeftLeg"] = EnableLegs;
        enableJoints["RightKnee"] = EnableKnees;
        enableJoints["LeftKnee"] = EnableKnees;
        enableJoints["RightFoot"] = EnableFoots;
        enableJoints["LeftFoot"] = EnableFoots;
        enableJoints["RightTooBase"] = EnableToeBases;
        enableJoints["LeftTooBase"] = EnableToeBases;
    }

    //デバッグ用
    public void OnRenderObject() {
        int scaleNum = 0;
        //NNに入力するデータの確認用
        if(DrawInputTensorBuff){
            debugRenderer.DebugDrawInputTensorBuff(vnect.nnInputBuff, vnect.NN_INPUT_HEIGHT_MAX, vnect.NN_INPUT_WIDTH_MAX);
        }
        //NNから出力されたデータの確認用
        if(DebugDrawHeatmap){
            //debugRenderer.DebugDrawHeatmap(vnect.nnOutputPtr, vnect.SHAPE_SCALES, scaleNum, vnect.heatmapHeight, vnect.heatmapWidth, vnect.NN_JOINT_COUNT);
            debugRenderer.DebugDrawHeatmap2(vnect.nnOutputPtr, vnect.nnShapeScales, scaleNum, vnect.heatmapHeight, vnect.heatmapWidth, vnect.NN_POOL_SIZE, jointInfos);
        }
        //処理用バッファの確認用
        else if(DebugDrawHeatmapBuff){
            //debugRenderer.DebugDrawHeatmapBuff(vnect.heatmapBuff, vnect.heatmapHeight, vnect.heatmapWidth, vnect.NN_JOINT_COUNT);
            debugRenderer.DebugDrawHeatmapBuff2(vnect.heatmapBuff, vnect.heatmapHeight, vnect.heatmapWidth, vnect.NN_POOL_SIZE, jointInfos);
        }
        //ラベリング用バッファの確認用
        else if(DrawHeatmapLabel){
            debugRenderer.DebugDrawHeatmapLabel(vnect.heatmapLabel, vnect.heatmapLabelCount, vnect.heatmapHeight, vnect.heatmapWidth, vnect.NN_JOINT_COUNT);
            //debugRenderer.DebugDrawHeatmapLabel2(vnect.heatmapLabel, vnect.heatmapLabelCount, vnect.heatmapHeight, vnect.heatmapWidth, vnect.NN_POOL_SIZE, jointInfos);
        }
        //2Dジョイントの確認用
        if(DrawResults2D){
            debugRenderer.DrawResults2D(vnect.joint2D, jointInfos);
        }
        //3Dジョイントの確認用
        if(DrawResults3D){
            debugRenderer.DrawResults3D(vnect.joint3D, jointInfos);
        }
    }

    //ウェブカメラや動画ファイルの入力映像サイズからNNの入力サイズにリサイズする
    private Texture2D ResizeTexture(Texture2D src) {
        float bbLeft = boundingBox.xMin;
        float bbRight = boundingBox.xMax;
        float bbTop = boundingBox.yMin;
        float bbBottom = boundingBox.yMax;
        float bbWidth = boundingBox.width;
        float bbHeight = boundingBox.height;

        float videoLongSide = (videoWidth > videoHeight) ? videoWidth : videoHeight;
        float videoShortSide = (videoWidth > videoHeight) ? videoHeight : videoWidth;
        float aspectWidth = videoWidth / videoShortSide;
        float aspectHeight = videoHeight / videoShortSide;

        float left = bbLeft;
        float right = bbRight;
        float top = bbTop;
        float bottom = bbBottom;

        //短辺を1としてマッピングしたいのでvideoShortSideで割る
        //（0未満や1超になることはあり得る。勝手にパディングされる）
        left /= videoShortSide;
        right /= videoShortSide;
        top /= videoShortSide;
        bottom /= videoShortSide;

        src.filterMode = FilterMode.Trilinear;
        src.Apply(true);

        RenderTexture rt = new RenderTexture(vnect.NN_INPUT_WIDTH_MAX, vnect.NN_INPUT_HEIGHT_MAX, 32);
        Graphics.SetRenderTarget(rt);
        GL.LoadPixelMatrix(left, right, bottom, top);
        //RotateTexture();
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, aspectWidth, aspectHeight), src);

        Rect dstRect = new Rect(0, 0, vnect.NN_INPUT_WIDTH_MAX, vnect.NN_INPUT_HEIGHT_MAX);
        Texture2D dst = new Texture2D((int)dstRect.width, (int)dstRect.height, TextureFormat.ARGB32, true);
        dst.ReadPixels(dstRect, 0, 0, true);
        Graphics.SetRenderTarget(null);
        Destroy(rt);

        return dst;
    }
    //現状使わない
    private void RotateTexture(){
        Vector3 t = new Vector3(0, 1, 0);
        Quaternion r = Quaternion.Euler(0, 0, -90);
        Vector3 s = Vector3.one;
        Matrix4x4 m = Matrix4x4.identity;
        m.SetTRS(t, r, s);
        GL.MultMatrix(m);
    }
}