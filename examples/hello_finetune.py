from mistral_fp16_torch import MistralConfig, Mistral
from mistral_fp16_torch.sentencepiece import SentencePieceTokenizer

from torch_prelude import gpu, smp, f16, f32, i64
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from fire import Fire

from glob import glob
from itertools import repeat
import os
from typing import Optional

class HelloDataset(Dataset):
    text = ("Thucydides, an Athenian, wrote the history of the war between"
           " the Peloponnesians and the Athenians, beginning at the moment"
           " that it broke out, and believing that it would be a great war"
           " and more worthy of relation than any that had preceded it."
           " This belief was not without its grounds. The preparations of"
           " both the combatants were in every department in the last state"
           " of perfection; and he could see the rest of the Hellenic race"
           " taking sides in the quarrel; those who delayed doing so at once"
           " having it in contemplation. Indeed this was the greatest movement"
           " yet known in history, not only of the Hellenes, but of a large"
           " part of the barbarian world-- I had almost said of mankind."
           " For though the events of remote antiquity, and even those that"
           " more immediately preceded the war, could not from lapse of time"
           " be clearly ascertained, yet the evidences which an inquiry carried"
           " as far back as was practicable leads me to trust, all point to"
           " the conclusion that there was nothing on a great scale, either"
           " in war or in other matters.")

    def __init__(self, max_seq_len, tokenizer):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        text_tok = self.tokenizer(self.text)
        assert len(text_tok) <= self.max_seq_len
        # NB: assuming below that 0 is the pad token.
        for _ in range(len(text_tok), self.max_seq_len):
            text_tok.append(0)
        return torch.asarray(text_tok, dtype=i64, device=smp)

def main(MODEL_PATH: str, model_format: Optional[str] = "pickle"):
    batch_size = 1
    max_seq_len = 256
    nstep = 4

    cfg = MistralConfig.mistral_7b()
    cfg.linear_scale = 16.0
    print("INFO:     model config = {}".format(cfg))

    #loss_scale = None
    loss_scale = 1024.0
    if loss_scale is not None:
        print("INFO:     loss scale = {}".format(loss_scale))

    tokenizer = SentencePieceTokenizer(os.path.join(MODEL_PATH, "tokenizer.model"))
    assert len(tokenizer) == cfg.tok_dim

    data = DataLoader(HelloDataset(max_seq_len, tokenizer), sampler = repeat(0), batch_size = batch_size)

    params = dict()
    if model_format == "safetensors":
        import safetensors.torch
        for tensors_file in sorted(glob("model-*-of-*.safetensors", root_dir = MODEL_PATH)):
            tensors_path = os.path.join(MODEL_PATH, tensors_file)
            params.update(safetensors.torch.load_file(tensors_path))
    elif model_format == "pickle":
        state = dict()
        for pickle_file in sorted(glob("pytorch_model-*-of-*.bin", root_dir = MODEL_PATH)):
            pickle_path = os.path.join(MODEL_PATH, pickle_file)
            state.update(torch.load(pickle_path))
        for k, v in state.items():
            if k.find(".weight") != -1:
                params[k] = v
        del state
    else:
        raise ValueError("unsupported value for --model-format")
    save_params = params

    params = dict()
    with torch.device(smp):
        for save_k in list(save_params.keys()):
            k = save_k
            if k.find("model.") == 0:
                k = k[6:]
            params[k] = save_params[save_k].to(dtype=f32)
            if cfg.linear_scale is not None and (
                    k.find("_proj") != -1 or k.find("_head") != -1
            ):
                params[k].data.mul_(cfg.linear_scale)
            del save_params[save_k]
    del save_params

    gpu_params = dict()
    with torch.device(gpu):
        for k, v in params.items():
            gpu_params[k] = v.to(dtype=f16, device=gpu)
            v.detach_()

    param_nan_ct = 0
    for k in params:
        param_nan_ct += torch.count_nonzero(gpu_params[k].data.isnan()).to(device=smp)
    if param_nan_ct > 0:
        print("DEBUG:    param nan ct = {}".format(param_nan_ct))
    assert param_nan_ct == 0

    with torch.device(smp):
        adamw = optim.AdamW(
                params.values(),
                lr = 2.0e-5,
                betas = (0.9, 0.95),
                eps = 1.0e-6,
                weight_decay = 0.1,
        )

    with torch.device(gpu):
        loss_fn = nn.CrossEntropyLoss(reduction = "sum")
        gpu_model = Mistral(cfg, batch_size, max_seq_len)
        gpu_model.load_state_dict(gpu_params)
        gpu_params = dict(gpu_model.named_parameters())
        for _, v in gpu_params.items():
            assert v.requires_grad

    for step, text_tok in enumerate(data):
        print("INFO:     step = {}/{}".format(step, nstep))

        in_tok = text_tok.to(device=gpu)
        out_logit = gpu_model(in_tok)
        out_lm_tok = torch.argmax(out_logit, 2, keepdim=False).to(dtype=i64)
        text_lm_tok = out_lm_tok.to(device=smp)

        in_tok_lens = []
        loss_denom = 0
        for batch_idx in range(batch_size):
            # NB: assuming below that 0 is the pad token.
            in_tok_len = torch.count_nonzero(text_tok[batch_idx,:])
            assert in_tok_len > 0
            in_tok_lens.append(in_tok_len)
            loss_denom += in_tok_len - 1

        for batch_idx in range(1):
            in_tok_len = in_tok_lens[batch_idx]
            for i in range(in_tok_len - 1):
                ts = [text_tok[batch_idx,i], text_lm_tok[batch_idx,i], text_tok[batch_idx,i+1]]
                ps = []
                for t in ts:
                    p = tokenizer[int(t)]
                    if ord(p[0]) == 9601:
                        p = " {}".format(p[1:])
                    ps.append(p)
                print("INFO:         pos={} {} {} {} \"{}\" \"{}\" \"{}\"".format(i, *ts, *ps))

        loss = 0.0
        for batch_idx in range(batch_size):
            in_tok_len = in_tok_lens[batch_idx]
            lm_logit = out_logit[batch_idx,:in_tok_len-1,:].to(dtype=f32)
            lm_target = in_tok[batch_idx,1:in_tok_len]
            loss += loss_fn(lm_logit, lm_target)
        loss /= loss_denom
        org_loss = torch.empty((), dtype=f32, device=gpu)
        org_loss.data.copy_(loss)
        if loss_scale is not None:
            loss *= loss_scale

        if step >= nstep:
            pass
        else:
            gpu_model.zero_grad()
            loss.backward()

        print("INFO:     loss = {}".format(org_loss.to(device=smp).item()))
        if loss_scale is not None:
            print("INFO:     loss = {} (scaled x {})".format(loss.to(device=smp).item(), loss_scale))

        if step >= nstep:
            break

        grad_nan_ct = 0
        for k in params:
            grad_nan_ct += torch.count_nonzero(gpu_params[k].grad.data.isnan()).to(device=smp)
        if grad_nan_ct > 0:
            print("DEBUG:    grad nan ct = {}".format(grad_nan_ct))
        assert grad_nan_ct == 0

        for k in params:
            if params[k].grad is not None:
                params[k].grad.data.copy_(gpu_params[k].grad.data)
            else:
                params[k].grad = gpu_params[k].grad.data.to(dtype=f32, device=smp)
            if loss_scale is not None:
                params[k].grad.data.mul_(1.0 / loss_scale)

        with torch.device(smp):
            adamw.step()

        param_nan_ct = 0
        for k, v in params.items():
            param_nan_ct += torch.count_nonzero(v.data.isnan())
        if param_nan_ct > 0:
            print("DEBUG:    param nan ct = {}".format(param_nan_ct))
        assert param_nan_ct == 0

        for k in params:
            gpu_params[k].data.copy_(params[k].data)

if __name__ == "__main__":
    Fire(main)
