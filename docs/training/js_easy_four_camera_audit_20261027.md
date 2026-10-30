# JS-easy四条用户指定训练窗口的相机核查

日期：2026-10-27。仅CPU诊断，没有修改训练、源pose或VAE。

详细中文报告和抽帧、背景跟踪、相机曲线：

`/datacc05/lizihao/worldmodel/fastvideowm_train_gallery_20261027/camera_audit_four_20261027/相机标注核查报告_zh.md`

机器可读汇总：同目录`results.json`。
复现入口：`tools/audit_gallery_camera_four_20261027.py`；单元测试：
`tests/local_tests/test_gallery_camera_audit_20261027.py`。

## 主要证据

四条来自当前train split，原始VGGT到实际训练C2W重建误差小于1.2e-7，K误差小于6.4e-7。
静态背景支持旋转主方向：D文件实际右转37.3度；AJ后半窗口左转24.5度；
I短视频上转12.5度；AI前半窗口实际左转11.9度，不符合计划的纯A。
文件名和生成prompt是请求，不是最终生成视频的运动真值。

历史build_fastvideowm_approved_uncached_dataset.py把非零平移除以整视频最大中心位移，
四条乘数约50.4、210.9、1172.9、159.1。I视频的背景旋转预测误差中位约0.83px，
但训练平移条件多轴折返、路径长约8.03训练单位，高度可疑。
单目尺度任意，不能把乘数说成米制放大，不能单凭这些数断言真实平移严格为0。

后续需批量审计低可信平移及归一化，不能按文件名强制改pose。
原视频窗口未变时VAE可复用，只改派生相机条件与索引；该修复尚未执行，当前训练输入未改变。
原记录gpu_qc_pass=true不等于人工几何审核，四条quality_approved均为false。
画廊小角度键名阈值偏敏感是另一个展示问题，不改变矩阵本身。

11项CPU坐标／投影／画廊测试与pre-commit通过。背景ROI是人工限定，
2D单应结果不能替代完整3D相机标定或确定真实相机公转中心。
