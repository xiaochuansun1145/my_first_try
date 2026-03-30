from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, RTDetrForObjectDetection
from transformers.modeling_outputs import BaseModelOutput

from src.semantic_rtdetr.contracts import EncoderFeatureBundle


@dataclass(frozen=True)
class RoundTripComparison:
    max_abs_logit_diff: float
    max_abs_box_diff: float

    def to_dict(self) -> dict[str, float]:
        return {
            "max_abs_logit_diff": self.max_abs_logit_diff,
            "max_abs_box_diff": self.max_abs_box_diff,
        }


class RTDetrBaseline:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        local_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.device = self._resolve_device(device)
        self.model_source = self._resolve_model_source(model_name, local_path)
        load_kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_source, **load_kwargs)
        self.model = RTDetrForObjectDetection.from_pretrained(self.model_source, **load_kwargs).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_model_source(model_name: str, local_path: str | Path | None) -> str:
        if local_path is None:
            return model_name

        candidate = Path(local_path)
        if not candidate.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / candidate

        if candidate.is_dir():
            return str(candidate)
        return model_name

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def prepare_inputs(self, image: Image.Image) -> dict[str, torch.Tensor]:
        processed = self.image_processor(images=image, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in processed.items()}

    def prepare_frame_tensor_batch(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [batch, channels, height, width]")

        images = [to_pil_image(frame.detach().cpu().clamp(0.0, 1.0)) for frame in frames]
        processed = self.image_processor(images=images, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in processed.items()}

    @torch.no_grad()
    def predict(self, inputs: dict[str, torch.Tensor]):
        return self.model(**inputs, return_dict=True)

    @torch.no_grad()
    def extract_encoder_feature_bundle(self, inputs: dict[str, torch.Tensor]) -> EncoderFeatureBundle:
        core_model = self.model.model
        pixel_values = inputs["pixel_values"]
        pixel_mask = inputs.get("pixel_mask")

        if pixel_mask is None:
            batch_size, _, height, width = pixel_values.shape
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        backbone_features = core_model.backbone(pixel_values, pixel_mask)
        projected_features = [
            core_model.encoder_input_proj[level](source)
            for level, (source, _mask) in enumerate(backbone_features)
        ]
        encoder_outputs = core_model.encoder(
            projected_features,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        feature_maps = list(encoder_outputs.last_hidden_state)
        decoder_sources = self._build_decoder_sources(feature_maps)
        spatial_shapes, level_start_index = self._build_decoder_indices(decoder_sources)

        return EncoderFeatureBundle(
            feature_maps=feature_maps,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            strides=list(core_model.encoder.out_strides),
        )

    @torch.inference_mode()
    def predict_from_encoder_feature_bundle(
        self,
        inputs: dict[str, torch.Tensor],
        feature_bundle: EncoderFeatureBundle,
    ):
        return self.forward_from_encoder_feature_bundle(inputs, feature_bundle)

    def forward_from_encoder_feature_bundle(
        self,
        inputs: dict[str, torch.Tensor],
        feature_bundle: EncoderFeatureBundle,
    ):
        encoder_outputs = BaseModelOutput(last_hidden_state=feature_bundle.feature_maps)
        return self.model(
            pixel_values=inputs["pixel_values"],
            pixel_mask=inputs.get("pixel_mask"),
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

    @torch.inference_mode()
    def compare_direct_and_roundtrip(self, inputs: dict[str, torch.Tensor]) -> RoundTripComparison:
        direct_outputs = self.predict(inputs)
        feature_bundle = self.extract_encoder_feature_bundle(inputs)
        roundtrip_outputs = self.predict_from_encoder_feature_bundle(inputs, feature_bundle)

        return self.compare_outputs(direct_outputs, roundtrip_outputs)

    @staticmethod
    def compare_outputs(direct_outputs, roundtrip_outputs) -> RoundTripComparison:
        return RoundTripComparison(
            max_abs_logit_diff=float((direct_outputs.logits - roundtrip_outputs.logits).abs().max().item()),
            max_abs_box_diff=float((direct_outputs.pred_boxes - roundtrip_outputs.pred_boxes).abs().max().item()),
        )

    def post_process(
        self,
        outputs,
        image_size: tuple[int, int],
        threshold: float,
    ) -> list[dict[str, Any]]:
        target_sizes = torch.tensor([image_size[::-1]], device=outputs.logits.device)
        detections = self.image_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

        serialized: list[dict[str, Any]] = []
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            label_id = int(label.item())
            serialized.append(
                {
                    "score": float(score.item()),
                    "label_id": label_id,
                    "label_name": self.model.config.id2label.get(label_id, str(label_id)),
                    "box_xyxy": [float(value) for value in box.tolist()],
                }
            )
        return serialized

    def save_visualization(
        self,
        image: Image.Image,
        detections: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        for detection in detections:
            x0, y0, x1, y1 = detection["box_xyxy"]
            label = detection["label_name"]
            score = detection["score"]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, max(0.0, y0 - 14.0)), f"{label} {score:.2f}", fill="red")

        canvas.save(output_path)

    def _build_decoder_sources(self, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        core_model = self.model.model
        sources = [
            core_model.decoder_input_proj[level](feature_map)
            for level, feature_map in enumerate(feature_maps)
        ]

        while len(sources) < self.model.config.num_feature_levels:
            next_level = len(sources)
            sources.append(core_model.decoder_input_proj[next_level](sources[-1]))

        return sources

    @staticmethod
    def _build_decoder_indices(sources: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = sources[0].device
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)

        for level, source in enumerate(sources):
            spatial_shapes[level, 0] = source.shape[-2]
            spatial_shapes[level, 1] = source.shape[-1]

        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        return spatial_shapes, level_start_index