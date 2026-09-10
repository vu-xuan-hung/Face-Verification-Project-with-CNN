"""Reusable CPU-only MiniFASNet session with explicit fail-closed decisions."""
import hashlib

import numpy as np

from .model_registry import DEFAULT_CONFIG, ModelContractError, load_pad_contract
from .pad_preprocessing import prepare_pad_input


class MiniFASNetONNX:
    def __init__(self, contract):
        try:
            import onnxruntime as ort

            # Hash the exact bytes handed to ORT, avoiding a check/load file race.
            artifact = contract.model_path.read_bytes()
            if hashlib.sha256(artifact).hexdigest() != contract.checksum:
                raise ValueError("checksum mismatch")
            self._session = ort.InferenceSession(artifact, providers=["CPUExecutionProvider"])
            inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("invalid model signature")
            if inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, 80, 80]:
                raise ValueError("invalid model input")
            if outputs[0].type != "tensor(float)" or outputs[0].shape != [1, 3]:
                raise ValueError("invalid model output")
            if self._session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("unexpected provider")
            self._input_name, self._output_name = inputs[0].name, outputs[0].name
            self.contract = contract
        except Exception as exc:
            raise ModelContractError("PAD runtime unavailable") from exc

    @classmethod
    def from_config(cls, config_path=DEFAULT_CONFIG):
        return cls(load_pad_contract(config_path))

    @property
    def ready(self):
        return True

    def predict(self, *, image, bbox):
        from .anti_spoof import PadResult, PadStatus

        fields = {
            "model_version": self.contract.model_version,
            "threshold": self.contract.threshold,
        }
        try:
            batch = prepare_pad_input(image, bbox, self.contract.crop_scale)
        except Exception:
            return PadResult(status=PadStatus.ERROR, reason="INVALID_INPUT", **fields)
        try:
            outputs = self._session.run([self._output_name], {self._input_name: batch})
            logits = np.asarray(outputs[0])
            if len(outputs) != 1 or logits.shape != (1, 3) or logits.dtype != np.float32:
                raise ValueError("invalid logits")
            if not np.isfinite(logits).all() or self.contract.real_class_index != 1:
                raise ValueError("invalid scores")
            logits = logits.astype(np.float64)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            predicted = int(probabilities.argmax(axis=1)[0])
            score = float(probabilities[0, self.contract.real_class_index])
            if predicted != self.contract.real_class_index:
                status, reason = PadStatus.FAKE, "SPOOF"
            elif score < self.contract.threshold:
                status, reason = PadStatus.UNCERTAIN, "PAD_UNCERTAIN"
            else:
                status, reason = PadStatus.REAL, "PAD_REAL"
            return PadResult(status=status, score=score, class_index=predicted, reason=reason, **fields)
        except Exception:
            return PadResult(status=PadStatus.ERROR, reason="MODEL_ERROR", **fields)
