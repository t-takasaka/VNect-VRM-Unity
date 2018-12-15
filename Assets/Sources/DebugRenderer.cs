using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DebugRenderer :MonoBehaviour {
    private float displayScale = 0.02f * 256.0f / 368.0f;

    private Vector3 joint3DPosition = new Vector3(-4, -0.5f, 2);
    private float joit3DScale = 0.4f;

    private Material lineMaterial;

    private Dictionary<string, GameObject> gizmoSphere = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinder = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderX = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderY = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> gizmoCylinderZ = new Dictionary<string, GameObject>();

    private bool ShowGizmo = false;

    private void CreateLineMaterial() {
        if (lineMaterial) { return; }

        Shader shader = Shader.Find("Hidden/Internal-Colored");
        lineMaterial = new Material(shader);
        lineMaterial.hideFlags = HideFlags.HideAndDontSave;
        lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        lineMaterial.SetInt("_ZWrite", 0);
    }

    public void DrawBegin() {
        CreateLineMaterial();
        lineMaterial.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(transform.localToWorldMatrix);
    }
    public void DrawEnd() {
        GL.PopMatrix();
    }
    public void DebugDrawInputTensorBuff(float[] nnInputBuff, int height, int width) {
        //CreateShapes()で0.0～1.0の範囲の色情報から0.4引いているのでやや暗くなる
        //正確な色を確認したい場合は該当箇所を0.0に変更する

        DrawBegin();
        Color color = Color.black;
        for (int y = 0; y < height; ++y) {
            GL.Begin(GL.LINE_STRIP);
            for (int x = 0; x < width; ++x) {
                int pos = (y * width + x) * 3;
                color.r = nnInputBuff[pos + 0];
                color.g = nnInputBuff[pos + 1];
                color.b = nnInputBuff[pos + 2];
                GL.Color(color);
                GL.Vertex3(x * displayScale, y * displayScale, 0f);
            }
            GL.End();
        }
        DrawEnd();
    }
    public unsafe void DebugDrawHeatmap(IntPtr nnOutputPtr, float[] scales, int scaleNum, int height, int width, int jointCount) {
        int slide = 70; //nnInputWidthが46なのでこれくらい

        float* src = (float*)nnOutputPtr;
        int padHeight = (int)((height - (height * scales[scaleNum])) / 2);
        int padWidth = (int)((width - (width * scales[scaleNum])) / 2);

        DrawBegin();
        Color color = Color.black;
        int srcChannel = scaleNum * height * width;

        for (int j = 0; j < jointCount; ++j) {
            for (int y = 0; y < height; ++y) {
                GL.Begin(GL.LINE_STRIP);
                int srcHeight = ((int)(y * scales[scaleNum]) + padHeight) * width;
                for (int x = 0; x < width; ++x) {
                    int srcWidth = ((int)(x * scales[scaleNum]) + padWidth);
                    int srcPos = (srcChannel + srcHeight + srcWidth) * jointCount;

                    float v = Mathf.Min(1.0f, Mathf.Max(0.0f, *(src + srcPos + j)));
                    color.r = color.g = color.b = v;
                    GL.Color(color);
                    GL.Vertex3((x + (j % 5) * slide) * displayScale, (y + (j / 5) * slide) * displayScale, 0f);
                }
                GL.End();
            }
        }

        for (int y = 0; y < height; ++y) {
            GL.Begin(GL.LINE_STRIP);
            int srcHeight = ((int)(y * scales[scaleNum]) + padHeight) * width;
            for (int x = 0; x < width; ++x) {
                int srcWidth = ((int)(x * scales[scaleNum]) + padWidth);
                int srcPos = (srcChannel + srcHeight + srcWidth) * jointCount;
                float v = 0;
                for (int j = 0; j < jointCount; ++j) {
                    v += *(src + srcPos + j);
                }
                color.r = color.g = color.b = Mathf.Min(1.0f, Mathf.Max(0.0f, v));
                GL.Color(color);
                GL.Vertex3((x + 4 * slide) * displayScale, (y + 4 * slide) * displayScale, 0f);
            }
            GL.End();
        }
        DrawEnd();
    }
    public unsafe void DebugDrawHeatmap2(IntPtr nnOutputPtr, float[] scales, int scaleNum, int height, int width, int poolsSize, Dictionary<string, JointInfo> jointInfos) {
        int jointCount = jointInfos.Count;

        float* src = (float*)nnOutputPtr;
        int padHeight = (int)((height - (height * scales[scaleNum])) / 2);
        int padWidth = (int)((width - (width * scales[scaleNum])) / 2);

        DrawBegin();
        Color color = Color.red;
        int srcChannel = scaleNum * height * width;
        for (int y = 0; y < height * poolsSize; ++y) {
            GL.Begin(GL.LINE_STRIP);
            int srcHeight = ((int)((y / poolsSize) * scales[scaleNum]) + padHeight) * width;
            for (int x = 0; x < width * poolsSize; ++x) {
                int srcWidth = ((int)((x / poolsSize) * scales[scaleNum]) + padWidth);
                int srcPos = (srcChannel + srcHeight + srcWidth) * jointCount;
                float v = 0;
                foreach (string key in jointInfos.Keys) {
                    if (jointInfos[key].enable == false) { continue; }
                    v += *(src + srcPos + jointInfos[key].index);
                }
                v *= 10;
                v = Mathf.Pow(v, 2);
                v /= 10;
                color.a = Mathf.Min(1.0f, Mathf.Max(0.0f, v));
                GL.Color(color);
                GL.Vertex3(x * displayScale, y * displayScale, 0f);
            }
            GL.End();
        }

        DrawEnd();
    }
    public void DebugDrawHeatmapBuff(float[,,,] heatmapBuff, int height, int width, int jointCount) {
        int heatmapType = 0;
        int slide = 70; //nnInputWidthが46なのでこれくらい

        DrawBegin();
        Color color = Color.black;
        for (int j = 0; j < jointCount; ++j) {
            for (int y = 0; y < height; ++y) {
                GL.Begin(GL.LINE_STRIP);
                for (int x = 0; x < width; ++x) {
                    float v = Mathf.Min(1.0f, Mathf.Max(0.0f, heatmapBuff[y, x, j, heatmapType]));
                    color.r = color.g = color.b = v;
                    GL.Color(color);
                    GL.Vertex3((x + (j % 5) * slide) * displayScale, (y + (j / 5) * slide) * displayScale, 0f);
                }
                GL.End();
            }
        }

        for (int y = 0; y < height; ++y) {
            GL.Begin(GL.LINE_STRIP);
            for (int x = 0; x < width; ++x) {
                float v = 0;
                for (int j = 0; j < jointCount; ++j) {
                    v += heatmapBuff[y, x, j, heatmapType];
                }
                color.r = color.g = color.b = Mathf.Min(1.0f, Mathf.Max(0.0f, v));
                GL.Color(color);
                GL.Vertex3((x + 4 * slide) * displayScale, (y + 4 * slide) * displayScale, 0f);
            }
            GL.End();
        }
        DrawEnd();
    }
    public void DebugDrawHeatmapBuff2(float[,,,] heatmapBuff, int height, int width, int poolsSize, Dictionary<string, JointInfo> jointInfos) {
        int jointCount = jointInfos.Count;
        int heatmapType = 3;

        DrawBegin();
        Color color = Color.red;
        for (int y = 0; y < height * poolsSize; ++y) {
            GL.Begin(GL.LINE_STRIP);
            for (int x = 0; x < width * poolsSize; ++x) {
                float v = 0;
                foreach (string key in jointInfos.Keys) {
                    if (jointInfos[key].enable == false) { continue; }
                    int j = jointInfos[key].index;
                    v += heatmapBuff[(y / poolsSize), (x / poolsSize), j, heatmapType];
                }
                color.a = Mathf.Min(1.0f, Mathf.Max(0.0f, v));
                GL.Color(color);
                GL.Vertex3((x) * displayScale, (y) * displayScale, 0f);
            }
            GL.End();
        }

        DrawEnd();
    }

    public void DebugDrawHeatmapLabel(int[,,] heatmapLabel, int[] heatmapLabelCount, int height, int width, int jointCount) {
        int slide = 70; //nnInputWidthが46なのでこれくらい

        DrawBegin();
        Color color = Color.black;
        for (int j = 0; j < jointCount; ++j) {
            for (int y = 0; y < height; ++y) {
                GL.Begin(GL.LINE_STRIP);
                for (int x = 0; x < width; ++x) {
                    int count = heatmapLabelCount[j];
                    if (count == 0) { continue; }
                    float v = heatmapLabel[j, y, x] / (float)count;
                    color.r = color.g = color.b = v;
                    GL.Color(color);
                    GL.Vertex3((x + (j % 5) * slide) * displayScale, (y + (j / 5) * slide) * displayScale, 0f);
                }
                GL.End();
            }
        }
        DrawEnd();
    }
    public void DebugDrawHeatmapLabel2(int[,,] heatmapLabel, int[] heatmapLabelCount, int height, int width, int poolSize, Dictionary<string, JointInfo> jointInfos) {
        int jointCount = jointInfos.Count;

        DrawBegin();
        Color color = Color.red;
        for (int y = 0; y < height * poolSize; ++y) {
            GL.Begin(GL.LINE_STRIP);
            for (int x = 0; x < width * poolSize; ++x) {
                float v = 0;
                foreach (string key in jointInfos.Keys) {
                    if (jointInfos[key].enable == false) { continue; }
                    int count = heatmapLabelCount[jointInfos[key].index];
                    if (count == 0) { continue; }

                    int j = jointInfos[key].index;
                    v += heatmapLabel[j, (y / poolSize), (x / poolSize)] / (float)count;
                }
                color.a = Mathf.Min(1.0f, Mathf.Max(0.0f, v));
                GL.Color(color);
                GL.Vertex3((x) * displayScale, (y) * displayScale, 0f);
            }
            GL.End();
        }
        DrawEnd();
    }

    public void DrawResults2D(Dictionary<string, Vector2> joint2D, Dictionary<string, JointInfo> jointInfos) {
        DrawSkeleton(joint2D, jointInfos);
        DrawKeypoint(joint2D, jointInfos);
    }
    private void DrawKeypoint(Dictionary<string, Vector2> joint2d, Dictionary<string, JointInfo> jointInfos) {
        float radius = 0.08f;

        DrawBegin();
        foreach (string key in jointInfos.Keys) {
            if (jointInfos[key].enable == false) { continue; }

            var joint = joint2d[key];

            GL.Begin(GL.LINES);
            GL.Color(jointInfos[key].color);
            for (float theta = 0.0f; theta < (2 * Mathf.PI); theta += 0.01f) {
                float x = Mathf.Cos(theta) * radius + joint.x * displayScale * 8;
                float y = Mathf.Sin(theta) * radius + joint.y * displayScale * 8;
                GL.Vertex3(x, y, 0f);
            }
            GL.End();
        }
        DrawEnd();
    }
    private void DrawSkeleton(Dictionary<string, Vector2> joint2d, Dictionary<string, JointInfo> jointInfos) {
        DrawBegin();
        GL.Begin(GL.QUADS);
        foreach (string key in jointInfos.Keys) {
            string child = jointInfos[key].child;
            if (!joint2d.ContainsKey(child)) { continue; }

            Vector2 v0 = joint2d[key];
            Vector2 v1 = joint2d[child];
            GL.Color(jointInfos[key].color);
            DrawLine2D(new Vector3(v0.x * displayScale * 8, v0.y * displayScale * 8),
                        new Vector3(v1.x * displayScale * 8, v1.y * displayScale * 8), 0.02f);
        }
        GL.End();
        DrawEnd();
    }
    private void DrawLine2D(Vector3 v0, Vector3 v1, float lineWidth) {
        Vector3 n = ((new Vector3(v1.y, v0.x, 0.0f)) - (new Vector3(v0.y, v1.x, 0.0f))).normalized * lineWidth;
        GL.Vertex3(v0.x - n.x, v0.y - n.y, 0.0f);
        GL.Vertex3(v0.x + n.x, v0.y + n.y, 0.0f);
        GL.Vertex3(v1.x + n.x, v1.y + n.y, 0.0f);
        GL.Vertex3(v1.x - n.x, v1.y - n.y, 0.0f);
    }

    private void CreategizmoJoints(Dictionary<string, JointInfo> jointInfos) {
        foreach (string key in jointInfos.Keys) {
            gizmoSphere[key] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            gizmoSphere[key].name = key + "Sphere";
            gizmoSphere[key].transform.localPosition = new Vector3(0, 0, 0);
            gizmoSphere[key].transform.localScale = new Vector3(0.2f, 0.2f, 0.2f);
            Material mat = gizmoSphere[key].GetComponent<Renderer>().material;
            mat.color = jointInfos[key].color;

            gizmoCylinder[key] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            gizmoCylinder[key].name = key + "Cylinder";
            gizmoCylinder[key].transform.SetParent(gizmoSphere[key].transform);
            gizmoCylinder[key].transform.localPosition = new Vector3(0, 2, 0);
            gizmoCylinder[key].transform.localScale = new Vector3(0.5f, 2.0f, 0.5f);
            mat = gizmoCylinder[key].GetComponent<Renderer>().material;
            mat.color = jointInfos[key].color;

            if (ShowGizmo) {
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
            }
        }

        foreach (string key in jointInfos.Keys) {
            if (jointInfos[key].parent == "") { continue; }
            gizmoSphere[key].transform.SetParent(gizmoSphere[jointInfos[key].parent].transform);
        }
    }
    public void DrawResults3D(Dictionary<string, Vector3> joint3D, Dictionary<string, JointInfo> jointInfos) {
        if (gizmoSphere.Count == 0) { CreategizmoJoints(jointInfos); }

        //関節部分のスフィアを移動
        foreach (string key in jointInfos.Keys) {
            if (!jointInfos[key].enable) { continue; }
            string childName = jointInfos[key].child;
            if (!joint3D.ContainsKey(childName)) { continue; }

            Vector3 ownPos = joint3D[key];
            Vector3 childPos = joint3D[childName];
            Vector3 dir = childPos - ownPos;
            Quaternion rot = Quaternion.FromToRotation(Vector3.up, dir);
            gizmoSphere[key].transform.position = ownPos * joit3DScale + joint3DPosition;
            gizmoSphere[key].transform.rotation = rot;
        }

        //スフィアを全部移動し終わってから骨部分のシリンダーの長さを変更
        foreach (string key in jointInfos.Keys) {
            if (!jointInfos[key].enable) { continue; }

            float distance = 1;

            string childName = jointInfos[key].child;
            Vector3 ownPos = gizmoSphere[key].transform.position;
            if (gizmoSphere.ContainsKey(childName)) {
                Vector3 childPos = gizmoSphere[childName].transform.position;
                distance = (childPos - ownPos).magnitude * 2;
            }

            gizmoCylinder[key].transform.localPosition = new Vector3(0, distance, 0);
            gizmoCylinder[key].transform.localScale = new Vector3(0.5f, distance, 0.5f);
        }
    }
}
