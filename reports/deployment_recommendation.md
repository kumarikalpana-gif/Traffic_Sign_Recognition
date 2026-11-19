
# Traffic Sign Recognition — Deployment Recommendation

## 1. Recommended Model

**Model:** Baseline CNN

**Model file:** `models/cnn_baseline.keras`

The Baseline CNN is recommended for deployment because it achieved
the highest average accuracy across the four corrupted test conditions.

## 2. Performance Summary

| Metric | Result |
|---|---:|
| Clean test accuracy | 99.18% |
| Average corrupted accuracy | 92.80% |
| Overall average accuracy | 94.08% |

## 3. Why This Model Was Selected

The Baseline CNN achieved:
- Strong clean-test accuracy of 99.18%.
- The highest average corrupted-test accuracy among the three models.
- Better overall average accuracy than the EfficientNetB0 and
  heavy-augmentation CNN models in this evaluation.

## 4. Robustness Considerations

The Baseline CNN performed well under Gaussian blur and brightness
changes. However, its accuracy decreased under occlusion and combined
corruption.

This indicates that the model can be sensitive when important parts
of a traffic sign are blocked or when multiple image distortions occur.

## 5. Deployment Recommendation

Use `cnn_baseline.keras` as the primary model for a prototype
traffic-sign recognition application.

Before deployment in a real vehicle or safety-critical environment,
additional testing is necessary, including:
- Testing on traffic signs captured in real-world driving conditions.
- Evaluating different lighting, weather, distance, and camera angles.
- Testing partially hidden, damaged, and unfamiliar signs.
- Measuring prediction latency and memory requirements on the target device.
- Adding a safe response for low-confidence or uncertain predictions.

The model should be treated as a prototype, not as a validated
safety-critical driving system.

## 6. Final Conclusion

The Baseline CNN provides the best balance of clean-test accuracy
and average corrupted-test accuracy among the three evaluated models.
It is therefore the recommended model for the next deployment
prototype stage.
