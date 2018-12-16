using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRM;

class VRMManager {
    private Animator animator;
    private float modelPositionX;
    private float modelPositionY;
    private float positionScale = 10;

    private Dictionary<string, Vector3> bindDirs = new Dictionary<string, Vector3>();
    private Dictionary<string, Vector3> parent2ownDirs = new Dictionary<string, Vector3>();
    private Dictionary<string, Vector3> own2childDirs = new Dictionary<string, Vector3>();

    private Dictionary<string, Transform> joints = new Dictionary<string, Transform>();

    private Dictionary<string, GameObject> gizmoSphere = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderX = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderY = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderZ = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderB = new Dictionary<string, GameObject>();

    private const bool ShowGizmo = false;

    public Dictionary<string, Vector3> Init(Dictionary<string, JointInfo> jointInfos, 
                                            float modelPositionX, float modelPositionY) {
        this.modelPositionX = modelPositionX;
        this.modelPositionY = modelPositionY;

        GameObject model = GameObject.Find("VRoid");
        animator = model.GetComponent<Animator>();

        //表情は笑顔に設定しておく
        var proxy = model.GetComponent<VRMBlendShapeProxy>();
        proxy.SetValue(BlendShapePreset.Fun, 1.0f);

        Dictionary<string, GameObject> objs = new Dictionary<string, GameObject>();
        GetJoints(model, ref objs);

        foreach(string key in jointInfos.Keys){
            string ownName = jointInfos[key].vroid;
            if(!objs.ContainsKey(ownName)){ continue; }
            Vector3 ownPos = objs[ownName].transform.position;

            string childName = jointInfos[key].child;
            if(!jointInfos.ContainsKey(childName)){ continue; }
            childName = jointInfos[childName].vroid;
            if(!objs.ContainsKey(childName)){ continue; }
            Vector3 childPos = objs[childName].transform.position;

            bindDirs[key] = (childPos - ownPos).normalized;

            HumanBodyBones humanBodyBones = jointInfos[key].human;
            joints[key] = animator.GetBoneTransform(humanBodyBones).transform;

            if(ShowGizmo){
                gizmoSphere[key] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                gizmoSphere[key].name = key + "Sphere";
                gizmoSphere[key].transform.SetParent(objs[ownName].transform);
                gizmoSphere[key].transform.localPosition = new Vector3(0, 0, 0);
                gizmoSphere[key].transform.localRotation = Quaternion.Euler(new Vector3(0, 0, 0));
                gizmoSphere[key].transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
                Material mat = gizmoSphere[key].GetComponent<Renderer>().material; 
                mat.color = jointInfos[key].color; 

                gizmoCylinderX[key] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                gizmoCylinderX[key].name = key + "CylinderX";
                gizmoCylinderX[key].transform.SetParent(gizmoSphere[key].transform);
                gizmoCylinderX[key].transform.localPosition = new Vector3(1, 0, 0);
                gizmoCylinderX[key].transform.localRotation = Quaternion.Euler(new Vector3(0, 0, 90));
                gizmoCylinderX[key].transform.localScale = new Vector3(0.1f, 1.0f, 0.1f);
                mat = gizmoCylinderX[key].GetComponent<Renderer>().material; 
                mat.color = Color.red; 

                gizmoCylinderY[key] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                gizmoCylinderY[key].name = key + "CylinderY";
                gizmoCylinderY[key].transform.SetParent(gizmoSphere[key].transform);
                gizmoCylinderY[key].transform.localPosition = new Vector3(0, 1, 0);
                gizmoCylinderY[key].transform.localRotation = Quaternion.Euler(new Vector3(0, 0, 0));
                gizmoCylinderY[key].transform.localScale = new Vector3(0.1f, 1.0f, 0.1f);
                mat = gizmoCylinderY[key].GetComponent<Renderer>().material; 
                mat.color = Color.green; 

                gizmoCylinderZ[key] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                gizmoCylinderZ[key].name = key + "CylinderZ";
                gizmoCylinderZ[key].transform.SetParent(gizmoSphere[key].transform);
                gizmoCylinderZ[key].transform.localPosition = new Vector3(0, 0, 1);
                gizmoCylinderZ[key].transform.localRotation = Quaternion.Euler(new Vector3(90, 0, 0));
                gizmoCylinderZ[key].transform.localScale = new Vector3(0.1f, 1.0f, 0.1f);
                mat = gizmoCylinderZ[key].GetComponent<Renderer>().material; 
                mat.color = Color.blue; 

                gizmoCylinderB[key] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                gizmoCylinderB[key].name = key + "CylinderB";
                gizmoCylinderB[key].transform.SetParent(gizmoSphere[key].transform);
                gizmoCylinderB[key].transform.localPosition = bindDirs[key] * 2;
                gizmoCylinderB[key].transform.localRotation = Quaternion.FromToRotation(Vector3.up, bindDirs[key]);
                gizmoCylinderB[key].transform.localScale = new Vector3(0.1f, 2.0f, 0.1f);
                mat = gizmoCylinderB[key].GetComponent<Renderer>().material; 
                mat.color = Color.yellow; 
            }
        }

        return bindDirs;
    }

    //モデルに紐付くGameObjectを集める
    private void GetJoints(GameObject obj, ref Dictionary<string, GameObject> dst) {
        Transform children = obj.GetComponentInChildren<Transform>();
        if (children.childCount == 0) { return; }

        foreach (Transform ob in children) {
            dst[ob.name] = ob.gameObject;
            GetJoints(ob.gameObject, ref dst);
        }
    }

    private Quaternion CalcRotateY(Dictionary<string, Vector3> joint3D, Dictionary<string, JointInfo> jointInfos, string rightName, string leftName){
        Vector3 right = joint3D[jointInfos[rightName].name];
        Vector3 left = joint3D[jointInfos[leftName].name];
        //右部位から見た左部位の方向とワールド座標上の左方向の差がY軸の回転角度
        Vector3 dir = new Vector3(left.x - right.x, 0, left.z - right.z);
        Quaternion rotate = Quaternion.FromToRotation(Vector3.left, dir);

        return rotate;
    }

    //TODO：ジョイント名のハードコードはやめる
    public void Update(Dictionary<string, Vector2> joint2D, Dictionary<string, Vector3> joint3D, 
                        Dictionary<string, JointInfo> jointInfos, Dictionary<string, bool> extractedJoints, 
                        Dictionary<string, bool> enableJoints, float nnInputWidth, float nnInputHeight) {

        if (animator == null) { return; }

        //※ローカル座標での回転処理は未実装
        bool UseWorldRotate = true;

        Quaternion hipsRotateY = CalcRotateY(joint3D, jointInfos, "RightLeg", "LeftLeg");
        Quaternion chestRotateY = CalcRotateY(joint3D, jointInfos, "RightArm", "LeftArm");

        own2childDirs.Clear();
        if(!UseWorldRotate){ parent2ownDirs.Clear(); }

        //自身と親、子の部位の向きを出しておく
        foreach(string key in jointInfos.Keys){
            if(!bindDirs.ContainsKey(key)){ continue; }

            Vector3 ownPos = joint3D[key];
            string childName = jointInfos[key].child;
            Vector3 childPos = joint3D.ContainsKey(childName) ? joint3D[childName] : ownPos;
            own2childDirs[key] = (childPos - ownPos).normalized;

            if(!UseWorldRotate){
                string parentName = jointInfos[key].parent;
                Vector3 parentPos = joint3D.ContainsKey(parentName) ? joint3D[parentName] : ownPos;
                parent2ownDirs[key] = (ownPos - parentPos).normalized;
            }
        }
        //自身と親、子の部位の向きから回転角度を出す
        foreach(string key in jointInfos.Keys){
            if (!bindDirs.ContainsKey(key)) { continue; }
            if (enableJoints.ContainsKey(key) && !enableJoints[key]){

                //脚が検出できない場合は下方向に向けておく
                if (key == "RightLeg" || key == "LeftLeg" || key == "RightKnee" || key == "LeftKnee") {
                    Quaternion rot = Quaternion.FromToRotation(bindDirs[key], Vector3.down);
                    joints[key].rotation = rot;
                }

                continue;
            }
            if (extractedJoints[key] == false) { continue;  }

            if(UseWorldRotate){
                Quaternion rot = Quaternion.FromToRotation(bindDirs[key], own2childDirs[key]);

                //両脚の角度から腰の、両腕の角度から胸のY軸回転角度を出して掛ける
                if(key == "Hips" || key == "Spine" || key == "LeftLeg" || key == "RightLeg" || 
                    key == "LeftKnee" || key == "RightKnee" ){ rot *= hipsRotateY; }

                if(key == "Neck" || key == "Head"){ rot *= chestRotateY; }
                joints[key].rotation = rot;

                //モデルの位置調整
                if(key == "Hips") {
                    float x = joint2D[key].x / nnInputWidth * -positionScale;
                    float y = joint2D[key].y / nnInputHeight * -positionScale;
                    joints[key].localPosition = new Vector3(0 + modelPositionX, y + modelPositionY, 0);
                }


            }else{
                Quaternion rot = Quaternion.FromToRotation(parent2ownDirs[key], own2childDirs[key]);
                if(key == "Hips"){ rot *= hipsRotateY; }
                joints[key].localRotation = rot;
            }
        }
    }
}
