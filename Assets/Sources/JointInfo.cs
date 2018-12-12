using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JointInfo {
    public string name;
    public int index;
    public string child;
    public string parent;
    public Color color;
    public HumanBodyBones human;
    public string vroid;
    public bool enable;

    public JointInfo(string name, int index, string child, string parent, Color color, HumanBodyBones human, string vroid, bool enable) {
        this.name = name;
        this.index = index;
        this.child = child;
        this.parent = parent;
        this.color = color;
        this.human = human;
        this.vroid = vroid;
        this.enable = enable;
    }

    static public void Init(Dictionary<string, JointInfo> jointInfos){
        //NNから出力されるジョイントの順番に合わせる
        int index = 0;
        Add(jointInfos, "Head", index++, "", "Neck", Color.red, HumanBodyBones.Head, "J_Bip_C_Head");    //0
        Add(jointInfos, "Neck", index++, "Head", "Spine", Color.red, HumanBodyBones.Neck, "J_Bip_C_Neck");    //1
        Add(jointInfos, "RightArm", index++, "RightElbow", "Neck", Color.green, HumanBodyBones.RightUpperArm, "J_Bip_R_UpperArm");    //2
        Add(jointInfos, "RightElbow", index++, "RightWrist", "RightArm", Color.green, HumanBodyBones.RightLowerArm, "J_Bip_R_LowerArm");    //3
        Add(jointInfos, "RightWrist", index++, "RightHand", "RightElbow", Color.green, HumanBodyBones.RightHand, "J_Bip_R_Hand");    //4
        Add(jointInfos, "LeftArm", index++, "LeftElbow", "Neck", Color.blue, HumanBodyBones.LeftUpperArm, "J_Bip_L_UpperArm");    //5
        Add(jointInfos, "LeftElbow", index++, "LeftWrist", "LeftArm", Color.blue, HumanBodyBones.LeftLowerArm, "J_Bip_L_LowerArm");    //6
        Add(jointInfos, "LeftWrist", index++, "LeftHand", "LeftElbow", Color.blue, HumanBodyBones.LeftHand, "J_Bip_L_Hand");    //7
        Add(jointInfos, "RightLeg", index++, "RightKnee", "Hips", Color.yellow, HumanBodyBones.RightUpperLeg, "J_Bip_R_UpperLeg");    //8
        Add(jointInfos, "RightKnee", index++, "RightFoot", "RightLeg", Color.yellow, HumanBodyBones.RightLowerLeg, "J_Bip_R_LowerLeg");    //9
        Add(jointInfos, "RightFoot", index++, "RightToeBase", "RightKnee", Color.yellow, HumanBodyBones.RightFoot, "J_Bip_R_Foot");    //10
        Add(jointInfos, "LeftLeg", index++, "LeftKnee", "Hips", Color.cyan, HumanBodyBones.LeftUpperLeg, "J_Bip_L_UpperLeg");    //11
        Add(jointInfos, "LeftKnee", index++, "LeftFoot", "LeftLeg", Color.cyan, HumanBodyBones.LeftLowerLeg, "J_Bip_L_LowerLeg");    //12
        Add(jointInfos, "LeftFoot", index++, "LeftToeBase", "LeftKnee", Color.cyan, HumanBodyBones.LeftFoot, "J_Bip_L_Foot");    //13
        Add(jointInfos, "Hips", index++, "Spine", "", Color.magenta, HumanBodyBones.Hips, "J_Bip_C_Hips");    //14
        Add(jointInfos, "Spine", index++, "Neck", "Hips", Color.magenta, HumanBodyBones.Spine, "J_Bip_C_Spine");    //15
        Add(jointInfos, "Eyes", index++, "", "Neck", Color.red, HumanBodyBones.Head, "");    //16
        Add(jointInfos, "RightHand", index++, "", "RightWrist", Color.gray, HumanBodyBones.RightHand, "J_Bip_R_Middle1");    //17
        Add(jointInfos, "LeftHand", index++, "", "LeftWrist", Color.gray, HumanBodyBones.LeftHand, "J_Bip_L_Middle1");    //18
        Add(jointInfos, "RightToeBase", index++, "", "RightFoot", Color.gray, HumanBodyBones.RightToes, "J_Bip_R_ToeBase");    //19 
        Add(jointInfos, "LeftToeBase", index++, "", "LeftFoot", Color.gray, HumanBodyBones.LeftToes, "J_Bip_L_ToeBase");    //20 
    }
    static private void Add(Dictionary<string, JointInfo> jointInfos, string name, int index, string child, string parent, 
                            Color color, HumanBodyBones human, string vroid, bool enable = true) {

        jointInfos[name] = new JointInfo(name, index, child, parent, color, human, vroid, enable);
    }

}


