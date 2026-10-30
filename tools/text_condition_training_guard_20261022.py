"""Opt-in training-input audit; no extra loss or transformer forward."""
import json
from pathlib import Path

from fastvideo.distributed import get_world_rank
from fastvideo.train.callbacks.callback import Callback
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class TextConditionTrainingGuard(Callback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.microbatches = 0
        self.text_tokens = 0

    def on_train_start(self, method, iteration=0):
        student = method.student
        if student.vision_only_conditioning or student.neutralize_text_conditioning:
            raise ValueError('This run requires actual, non-neutralized scene text')
        original = student.prepare_batch

        def checked_prepare_batch(raw_batch, **kwargs):
            result = original(raw_batch, **kwargs)
            layout = result.minimax_h3_layout
            tags = layout.token_tags[layout.text_indices]
            ntext, nvision = int((tags == 1).sum()), int((tags == 0).sum())
            expected = int(raw_batch['text_attention_mask'][0].sum())
            source_tags = raw_batch['text_token_tags'][0]
            if ntext <= 0 or nvision <= 0 or len(tags) != expected or len(source_tags) != expected:
                raise ValueError('Text missing, silently truncated, or tag/embedding alignment broken')
            self.microbatches += 1
            self.text_tokens = ntext
            if get_world_rank() == 0 and (self.microbatches <= 8 or self.microbatches % 800 == 0):
                record = dict(microbatch=self.microbatches, text_tokens=ntext, vision_tokens=nvision,
                    full_condition_rows=len(tags), actual_text_entered_dit=True)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with (self.output_dir / 'training_text_input_audit.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                logger.info('TRAINING_TEXT_CONDITION_VERIFIED %s', record)
            return result

        student.prepare_batch = checked_prepare_batch

    def on_training_step_end(self, method, loss_dict, iteration=0):
        loss_dict['conditioning/text_tokens'] = float(self.text_tokens)
