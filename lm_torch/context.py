from lm_torch.prelude import gpu, smp, i64

import torch

class Context:
    def __init__(self, tag, tokenizer, model, max_seq_len, device = gpu):
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = max_seq_len
        self.device = device
        self.tag = tag
        self.restart = 0
        self.tok = []
        self.buf = []

    def rollout(self, text_buf: List[str], stop_ctx: str):
        assert len(text_buf) > 0

        if self.restart > 0:
            if len(self.buf) > 0 and self.buf[0] == text_buf[0]:
                pass
            else:
                self.restart = 0
        start = self.restart
        text_tok = []
        if self.restart > 0:
            self.tok = self.tok[:self.restart]
        else:
            text_tok.extend(self.tokenizer(text_buf[0], prepend=True))
            self.restart = len(text_tok)
            self.tok.clear()
            self.tok.extend(text_tok)
        self.buf.clear()
        for i, text in enumerate(text_buf):
            if i > 0:
                t = self.tokenizer(text, prepend=False)
                text_tok.extend(t)
                self.tok.extend(t)
            self.buf.append(text)
            print("{}".format(text), end="", flush=True)
        assert start + len(text_tok) <= self.max_seq_len
        text_tok = torch.asarray(text_tok, dtype=i64, device=smp).reshape((1, -1))

        stop_pat = "</{}>\n".format(stop_ctx)
        stop_buf = None

        torch.cuda.empty_cache()
        while True:
            with torch.no_grad():
                with torch.device(self.device):
                    #torch.cuda.empty_cache()
                    in_tok = text_tok.to(device=self.device)
                    in_len = in_tok.shape[1]
                    out_logit = self.model(in_tok, start, cache_tag = self.tag)
                    out_lm_tok = torch.argmax(out_logit[:,in_len-1:in_len,:], 2, keepdim=False).to(dtype=i64)
                    text_lm_tok = out_lm_tok.to(device=smp)
                    text_tok = text_lm_tok
                    t = int(text_tok[0,0])
                    self.tok.append(t)
                    if t <= 0 or t == 2:
                        break
                    s = self.tokenizer[t]
                    print("{}".format(s), end="", flush=True)
                    self.buf.append(s)
                    if stop_buf is None:
                        if s.find("</") == 0:
                            stop_buf = s
                    else:
                        stop_buf += s
                        if stop_buf.find(stop_pat) == 0:
                            break
                        elif len(stop_buf) >= len(stop_pat):
                            stop_buf = None
                    start += in_len
                    if start >= self.max_seq_len:
                        break
