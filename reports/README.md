# 单任务实验交付物

## 可直接提交

- [实验报告 PDF](experiment_report.pdf)
- [实验报告 Word](experiment_report.docx)
- [实验报告 Markdown](experiment_report.md)
- [视频汇报口播稿 Word](video_report_script.docx)
- [视频汇报口播稿 Markdown](video_report_script.md)
- [20 秒无声汇总视频](single_task_video_report.mp4)
- [四模型同步对比视频](comparison_episode_00.mp4)

## 实验依据

- [统一评测结果](evaluation/results.json)
- [结果图](figures/model_comparison.png)
- [全部 State BC 视频](evaluation/videos/state/)
- [全部 RNN BC 视频](evaluation/videos/rnn/)
- [全部 Transformer BC 视频](evaluation/videos/transformer/)
- [全部 ACT 视频](evaluation/videos/act/)

## 重新评测

```bash
python my_bc/evaluate_single_task.py \
  --episodes 10 \
  --max-steps 220 \
  --record-resolution 512 \
  --save-videos \
  --output-dir reports/evaluation
```

Vision BC 因当前工作区缺少 `vision_bc_policy.pt` 而被自动跳过。重新训练并生成该 checkpoint 后，再运行上述命令即可补齐结果和视频。
