"""
使用 DirectML 进行 GPU 加速的情绪识别器.
支持 EmotiEffLib 预训练模型和自定义训练模型.
"""

import onnx
import onnxruntime as ort
import numpy as np
from onnx import TensorProto, helper, numpy_helper

from emotiefflib.facial_analysis import EmotiEffLibRecognizerOnnx, EmotiEffLibRecognizerBase
from emotiefflib.utils import get_model_path_onnx

_GPU_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]


class EmotiEffLibRecognizerOnnxGPU(EmotiEffLibRecognizerOnnx):
    """
    使用 DirectML GPU 推理的情绪识别器.
    支持两种模式:
    1. 标准模式: 加载 EmotiEffLib 预训练 ONNX 模型 (自动剥离 GEMM 分类头)
    2. 自定义模式: 加载自定义训练的 ONNX backbone + .npz 分类器权重
    """

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf",
                 custom_onnx: str = None, custom_weights: str = None,
                 custom_labels: dict = None) -> None:
        # 跳过父类 __init__ (其硬编码 CPU provider 并下载默认模型)
        EmotiEffLibRecognizerBase.__init__(self, model_name)

        if custom_onnx is not None:
            self._init_custom(custom_onnx, custom_weights, custom_labels)
        else:
            self._init_standard(model_name)

    def _init_standard(self, model_name):
        path = get_model_path_onnx(model_name)
        model = onnx.load(path)
        graph = model.graph
        gemm_node = graph.node[-1]
        if len(gemm_node.input) < 3:
            raise RuntimeError("Unexpected gemm node!")
        new_output_name = gemm_node.input[0]
        weight_tensor = next((t for t in graph.initializer if t.name == gemm_node.input[1]), None)
        bias_tensor = next((t for t in graph.initializer if t.name == gemm_node.input[2]), None)
        self.classifier_weights = numpy_helper.to_array(weight_tensor) if weight_tensor else None
        self.classifier_bias = numpy_helper.to_array(bias_tensor) if bias_tensor else None

        graph.node.remove(gemm_node)
        graph.output.remove(graph.output[0])
        new_output = helper.make_tensor_value_info(
            new_output_name, TensorProto.FLOAT, [None, self.classifier_weights.shape[1]]
        )
        graph.output.append(new_output)

        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        self.ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=_GPU_PROVIDERS,
            sess_options=options,
        )

    def _init_custom(self, custom_onnx, custom_weights, custom_labels):
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        self.ort_session = ort.InferenceSession(custom_onnx, providers=_GPU_PROVIDERS,
                                                 sess_options=options)

        data = np.load(custom_weights)
        self.classifier_weights = data["weights"]
        self.classifier_bias = data["bias"]

        input_shape = self.ort_session.get_inputs()[0].shape
        if input_shape and input_shape[-1] in (112, 224, 260):
            self.img_size = input_shape[-1]
            if self.img_size == 112:
                self.mean = [0.5, 0.5, 0.5]
                self.std = [0.5, 0.5, 0.5]
            else:
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]

        if custom_labels:
            self.idx_to_emotion_class = custom_labels
        elif self.classifier_weights.shape[0] == 7:
            self.idx_to_emotion_class = {
                0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happiness",
                4: "Neutral", 5: "Sadness", 6: "Surprise",
            }

    def __repr__(self) -> str:
        return f"EmotiEffLibRecognizerOnnxGPU(providers={self.ort_session.get_providers()})"
