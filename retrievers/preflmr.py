import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from retrievers.colbert.modeling.colbert import colbert_score
from retrievers.colbert import ColBERT, ColBERTConfig
from retrievers.layers.vit import CLIPVisionTower
from transformers import AutoConfig
from retrievers.colbert.infra import ColBERTConfig
from retrievers.colbert.infra.config.core_config import DefaultVal

def build_vision_tower(model_name, **kwargs):
    return CLIPVisionTower(model_name, **kwargs)
    
def build_text_tower(text_tower_cfg: ColBERTConfig, **kwargs):
    return ColBERT(name=text_tower_cfg.checkpoint, colbert_config=text_tower_cfg)

class FLMRMultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron with an activation function. This can be used as the mapping network in the FLMR model.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(FLMRMultiLayerPerceptron, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        
class PreFLMRSampler(nn.Module):
    def __init__(
        self, t_dim, v_dim, num_tokens=32, out_dim=128
    ):
        super().__init__()
        self.out_dim = out_dim
        self.context_vis_proj = FLMRMultiLayerPerceptron((
            v_dim,
            (out_dim * num_tokens) // 2,
            out_dim * num_tokens
        ))
        
        try:
            from transformers import BertConfig
            from transformers.models.bert.modeling_bert import BertEncoder
        except Exception as e:
            raise ImportError(f"Failed to import BertConfig and BertEncoder from transformers. {e}")
        
        transformer_mapping_config = BertConfig.from_pretrained("bert-base-uncased")
        transformer_mapping_config.num_hidden_layers = 1
        # add cross attention
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
   
        self.transformer_mapping_input_linear = nn.Linear(
            v_dim, transformer_mapping_config.hidden_size
        )
        
        # The transformer encoder
        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, out_dim
        )
        self.transformer_mapping_cross_attention_length = 32
        
    def forward(self, t, v, t_attn_mask, return_relevance=False):
        v_cls = v[:,0:1]
        vision_embeddings = self.context_vis_proj(v_cls)
        vision_embeddings = vision_embeddings.view(v.size(0), -1, self.out_dim)
        
        vision_second_last_layer_hidden_states = v[:,1:]
        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )
        
        # Cross attention only attends to the first 32 tokens
        encoder_mask = torch.ones_like(t_attn_mask).to(t_attn_mask.device, dtype=t_attn_mask.dtype)
        cross_attention_length = self.transformer_mapping_cross_attention_length
        if t.shape[1] > cross_attention_length:
            t = t[:, :cross_attention_length]
            encoder_mask = encoder_mask[:, :cross_attention_length]
        
        # Obtain cross attention mask
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
        # Pass through the transformer mapping
        transformer_mapping_outputs = self.transformer_mapping_network(
            transformer_mapping_input_features,
            encoder_hidden_states = t,
            encoder_attention_mask = encoder_extended_attention_mask
        )
        transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
        # Convert the dimension to FLMR dim
        transformer_mapping_output_features = self.transformer_mapping_output_linear(
            transformer_mapping_output_features
        )
        # Merge with the vision embeddings
        vision_embeddings = torch.cat([vision_embeddings, transformer_mapping_output_features], dim=1)
        
        return vision_embeddings
        
    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(torch.float32).min

        return encoder_extended_attention_mask
    
    
class PreFLMR(ColBERT):
    def __init__(self, config, colbert_config=None):
        super().__init__(name=config.colbert_checkpoint, colbert_config=colbert_config)
        
        vision_config = AutoConfig.from_pretrained(config.vision_model_name).vision_config
        vis_hidden_size = vision_config.hidden_size
        hidden_size = config.hidden_size
        self.config = config
        self.pretraining = self.config.pretraining
            
        self.vision_tower = build_vision_tower(config.vision_model_name)

        self.query_fusion = PreFLMRSampler(
            t_dim=hidden_size, v_dim=vis_hidden_size,
            num_tokens=config.num_tokens
        )
    
    def freeze_parameters(self, frozen=True):
        for _, param in self.vision_tower.named_parameters():
            param.requires_grad = False
        
        for _, param in self.model.named_parameters():
            param.requires_grad = not frozen
            
        self.linear.requires_grad = not frozen
    
    def forward(
        self,
        query_ids=None,
        query_attention_mask=None,
        query_images=None,
        query_image_mask=None,
        doc_ids=None,
        doc_attention_mask=None,
        doc_images=None,
        doc_image_mask=None,
        query_image_embs=None,
        doc_image_embs=None,
        return_loss=False,
        return_ib_acc=False,
    ):
        # query
        Q_t, Q_i = self.embed_VL(query_ids, query_attention_mask,
                                query_images, query_image_embs)
        
        # document
        D_t, D_i = self.embed_VL(doc_ids, doc_attention_mask, doc_images)
        if doc_image_embs is not None:
            D_i = doc_image_embs
        
        mask = torch.tensor(self.mask(doc_ids, skiplist=self.skiplist), device=self.device)
        D, D_mask = self.doc(D_t, D_i, mask, doc_image_mask, keep_dims='return_mask')
            
        Q = self.query(Q_t, Q_i, query_attention_mask, query_image_mask)

        nway = D.size(0) // Q.size(0)
        # Repeat each query encoding for every corresponding document.
        if Q.size(0)!=D.size(0):
            Q_duplicated = Q.repeat_interleave(nway, dim=0).contiguous()
        else:
            Q_duplicated = Q
            
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            
        scores = scores.view(-1, nway)
                
        loss = None
        accuracy = None
        if return_loss:
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            # scores /= 0.37
            loss = F.cross_entropy(scores, labels)
            
            if self.colbert_config.use_ib_negatives:
                loss += ib_loss
            
            if return_ib_acc:
                accuracy = self.inbatch_accuracy(Q, D, D_mask)
            else:
                accuracy = torch.tensor(1.0)
        
        return {'loss': loss, 'score': scores, "accuracy": accuracy}
    
    def neftune(self, embeddings, neftune_noise_alpha=5.0):
        dims = torch.tensor(embeddings.size(1)*embeddings.size(2))
        mag_norm = neftune_noise_alpha / torch.sqrt(dims)
        embeddings = embeddings + torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
        return embeddings
    
    def embed_VL(self, input_ids, attention_mask, images=None, image_embeds=None):
        assert input_ids is not None or images is not None
        T, I = None, None
        
        if images is not None:
            # final [CLS]; outputs in the penultimate layer
            with torch.no_grad():
                I = self.vision_tower(images.to(self.device))
        
        if image_embeds is not None:
            I = image_embeds

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            T = self.bert(input_ids, attention_mask=attention_mask)[0]

        return T, I
    
    def query(self, Q_t, Q_i, attention_mask, image_attn_mask=None):
        if Q_i is not None:
            Q_i = self.query_fusion(Q_t, Q_i, t_attn_mask=attention_mask)
            if self.pretraining:
                Q = Q_i
            else:
                Q_t = self.linear(Q_t)
                Q_t = Q_t * attention_mask.unsqueeze(2).float()
                Q = torch.cat([Q_i, Q_t], dim=1)
        else:
            Q = self.linear(Q_t)
            Q = Q * attention_mask.unsqueeze(2).float()
        
        return torch.nn.functional.normalize(Q, p=2, dim=2)
    
    def doc(self, D_t, D_i, mask, image_attn_mask=None, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']
        
        if D_i is not None:
            D = torch.cat([D_i, D_t], dim=1)
            if image_attn_mask is not None:
                image_attn_mask = image_attn_mask.expand(-1, D_i.size(1))
            else:
                image_attn_mask = torch.ones((D_i.size(0), D_i.size(1)))
            mask = torch.cat([image_attn_mask, mask], dim=1)
        else:
            D = D_t
        
        D = self.linear(D)
        D = D * mask.unsqueeze(2).float()
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D
    
    def inbatch_accuracy(self, Q, D_padded, D_mask):
        B, S, E = Q.shape
        B, K, E = D_padded.shape
        Q = Q.view(-1, E)
        D_padded = D_padded.view(-1, E)
        
        scores = D_padded @ Q.t() # (B K) x (B S)
        scores = scores.view(B,K,B,S).permute(0,2,1,3) # B x B x K x S
        
        # D_mask -> [B x K]
        D_mask = D_mask.unsqueeze(1).expand(-1,B,-1)
        D_padding = ~D_mask.view(scores.size(0), scores.size(1), scores.size(2)).bool()
        scores[D_padding] = -9999
        
        scores = scores.max(2).values.sum(-1)
        _, max_idx = torch.max(scores, 1)
        labels = torch.arange(len(scores), device=scores.device)
        accuracy = (max_idx==labels.detach()).sum() / scores.size(0)
        
        return accuracy
    
    def score(self, Q, D_padded, D_mask):                
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)