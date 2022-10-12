from typing import Optional

import torch.nn as nn
from torch import Tensor


class TransformerEncoder(nn.TransformerEncoder):
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("encoder", pos is None)
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoder(nn.TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        output_attention: Optional[bool] = False,
    ) -> Tensor:
        output = tgt
        attentions = []
        for mod in self.layers:
            output, attention = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_pos=tgt_pos,
                memory_pos=memory_pos,
                output_attention=True,
            )
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        if output_attention:
            return output, attentions
        return output


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        output_attention: Optional[bool] = False,
    ) -> Tensor:
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_pos))
        x_, attn = self._mha_block(
            x,
            memory,
            memory_mask,
            memory_key_padding_mask,
            x_pos=tgt_pos,
            mem_pos=memory_pos,
            output_attention=True,
        )
        x = self.norm2(x + x_)
        x = self.norm3(x + self._ff_block(x))
        if output_attention:
            return x, attn
        return x

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("decoder", pos is None)
        return tensor if pos is None else tensor + pos

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos: Optional[Tensor],
    ) -> Tensor:
        q = k = self.with_pos_embed(x, pos)
        x = self.self_attn(
            q,
            k,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        x_pos: Optional[Tensor],
        mem_pos: Optional[Tensor],
        output_attention: Optional[bool] = False,
    ) -> Tensor:
        x, attn = self.multihead_attn(
            self.with_pos_embed(x, x_pos),
            self.with_pos_embed(mem, mem_pos),
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        if output_attention:
            return self.dropout2(x), attn
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
