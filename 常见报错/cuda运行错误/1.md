## 1. cuda 报错（RuntimeError: CUDA error: unknown error）

**错误代码：**

```cmd
  File "G:\wav2lip384\preprocess.py", line 85, in mp_handler
    process_video_file(vfile, args, gpu_id)
  File "G:\wav2lip384\preprocess.py", line 59, in process_video_file
    preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
  File "G:\wav2lip384\face_detection\api.py", line 66, in get_detections_for_batch
    detected_faces = self.face_detector.detect_from_batch(images.copy())
    bboxlists = batch_detect(self.face_detector, images, device=self.device)
  File "G:\wav2lip384\face_detection\detection\sfd\detect.py", line 65, in batch_detect
    imgs = torch.from_numpy(imgs).float().to(device)
RuntimeError: CUDA error: unknown error
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```

**为什么报错：**

​	1. torch 与torchmetrics ，及其torchversion 之间的版本不兼容

**解决方案：**

	1. 卸载 torch 与torchmetrics ，及其torchversion
	1. 三个库版本需要保持一致

