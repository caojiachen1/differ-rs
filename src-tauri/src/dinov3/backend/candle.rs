//! Candle native backend for DINOv3 inference.
//!
//! Implements the DINOv3 ViT forward pass natively using Hugging Face's Candle
//! framework. Loads Hugging Face `transformers` DINOv3 safetensors weights and
//! supports CUDA/CPU devices.
//!
//! The architecture matches the reference GGML implementation:
//!   - Token order: [CLS, registers, patches]
//!   - RoPE applied to Q/K in attention (no absolute position embedding)
//!   - LayerScale (gamma) on attention and MLP residual branches
//!   - LayerNorm eps = 1e-5, exact-erf GELU

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module};

use crate::dinov3::config::ModelConfig;
use crate::dinov3::model::{ModelInfo, TransformerLayer, VitWeights};
use crate::dinov3::{preprocess_image, InferenceBackend};

/// LayerNorm epsilon (HF DINOv3 `layer_norm_eps`).
const LAYER_NORM_EPS: f64 = 1e-5;

/// RoPE frequency base (HF DINOv3 `rope_theta`).
const ROPE_FREQ_BASE: f32 = 100.0;

// ============================================================
// Helper utilities
// ============================================================

/// Retrieve a tensor from a map by trying multiple name patterns.
fn get_tensor(tensors: &HashMap<String, Tensor>, names: &[&str]) -> Result<Tensor> {
    for name in names {
        if let Some(t) = tensors.get(*name) {
            return Ok(t.clone());
        }
    }
    anyhow::bail!("Could not find tensor with any of these names: {:?}", names)
}

/// Try to retrieve an optional tensor (returns None if not found).
fn try_get_tensor(tensors: &HashMap<String, Tensor>, names: &[&str]) -> Option<Tensor> {
    for name in names {
        if let Some(t) = tensors.get(*name) {
            return Some(t.clone());
        }
    }
    None
}

/// SiLU activation: x * sigmoid(x).
fn silu(x: &Tensor) -> candle_core::Result<Tensor> {
    let sigmoid = (x.neg()?.exp()? + 1.0f64)?.recip()?;
    x * sigmoid
}

/// Build DINOv3 RoPE cos/sin tables of shape `[1, 1, seq_len, head_dim]`.
///
/// Matches the reference GGML implementation exactly: CLS and register tokens
/// are identity rows (cos=1, sin=0); patch tokens use 2D axial RoPE over the
/// normalized patch-center coordinates. The rotate-half sign is folded into the
/// sin table (first half negated) so application is a plain roll-by-D/2.
fn build_rope_tables(
    config: &ModelConfig,
    head_dim: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let grid_h = config.input_height / config.patch_size;
    let grid_w = config.input_width / config.patch_size;
    let n_prefix = 1 + config.num_register_tokens; // CLS + registers
    let seq_len = 1 + grid_h * grid_w + config.num_register_tokens;
    let d = head_dim;

    let mut cos = vec![0f32; seq_len * d];
    let mut sin = vec![0f32; seq_len * d];

    // Prefix tokens (CLS + registers): identity rotation.
    for t in 0..n_prefix {
        for k in 0..d {
            cos[t * d + k] = 1.0;
            sin[t * d + k] = 0.0;
        }
    }

    let two_pi = std::f32::consts::TAU;
    for row in 0..grid_h {
        for col in 0..grid_w {
            let t = n_prefix + row * grid_w + col;
            let cy = 2.0 * (row as f32 + 0.5) / grid_h as f32 - 1.0;
            let cx = 2.0 * (col as f32 + 0.5) / grid_w as f32 - 1.0;
            let base = t * d;
            for j in 0..d / 4 {
                let inv_freq = ROPE_FREQ_BASE.powf(-4.0 * j as f32 / d as f32);
                let ay = two_pi * cy * inv_freq;
                let ax = two_pi * cx * inv_freq;
                // first half: sign folded in (-sin)
                cos[base + j] = ay.cos();
                sin[base + j] = -ay.sin();
                cos[base + d / 4 + j] = ax.cos();
                sin[base + d / 4 + j] = -ax.sin();
                // second half: +sin
                cos[base + d / 2 + j] = ay.cos();
                sin[base + d / 2 + j] = ay.sin();
                cos[base + 3 * d / 4 + j] = ax.cos();
                sin[base + 3 * d / 4 + j] = ax.sin();
            }
        }
    }

    let cos_t = Tensor::from_vec(cos, (1, 1, seq_len, d), dev)?;
    let sin_t = Tensor::from_vec(sin, (1, 1, seq_len, d), dev)?;
    Ok((cos_t, sin_t))
}

/// Apply DINOv3 RoPE: `x*cos + roll(x, D/2)*sin`.
///
/// `x`: `[B, num_heads, N, head_dim]`; `cos`/`sin`: `[1, 1, N, head_dim]`.
/// The rotate-half sign is pre-folded into `sin`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    // roll(x, D/2): rotated[i] = x[(i + D/2) mod D]
    let rotated = Tensor::cat(&[&x2, &x1], D::Minus1)?;
    x.broadcast_mul(cos)?
        .add(&rotated.broadcast_mul(sin)?)
}

// ============================================================
// Model Components
// ============================================================

/// Patch embedding via Conv2d (kernel=stride=patch_size).
struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.proj.forward(x)
    }
}

/// Multi-head self-attention with DINOv3 RoPE on Q and K.
struct Attention {
    qkv: Linear, // combined [3*H, H]
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        // Combined QKV projection: [B, N, C] -> [B, N, 3*C]
        let qkv = self.qkv.forward(x)?;
        // -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        let qkv = qkv
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;

        // DINOv3 RoPE on Q and K (identity for CLS/register rows).
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // Scaled dot-product attention.
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_weights = (q.matmul(&k_t)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v)?; // [B, num_heads, N, head_dim]

        // -> [B, N, C]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, n, c))?;
        self.o_proj.forward(&out)
    }
}

/// MLP block with optional SwiGLU gating (for larger models).
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    fc3: Option<Linear>, // SwiGLU gate projection
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match &self.fc3 {
            Some(gate) => {
                // SwiGLU: fc2(silu(fc1(x)) * gate(x))
                let x1 = self.fc1.forward(x)?;
                let g = gate.forward(x)?;
                let x1 = silu(&x1)?;
                self.fc2.forward(&(&x1 * &g)?)
            }
            None => {
                // Standard MLP with exact-erf GELU: fc2(gelu(fc1(x)))
                let x = self.fc1.forward(x)?;
                let x = x.gelu_erf()?;
                self.fc2.forward(&x)
            }
        }
    }
}

/// Transformer encoder block with LayerScale (DINOv3).
struct TransformerBlock {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    ls1: Option<Tensor>, // LayerScale gamma for attention [hidden_size]
    ls2: Option<Tensor>, // LayerScale gamma for MLP [hidden_size]
}

impl TransformerBlock {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        // Pre-norm attention with residual + optional LayerScale.
        let residual = x;
        let attn_out = self.attn.forward(&self.norm1.forward(x)?, cos, sin)?;
        let attn_out = match &self.ls1 {
            Some(gamma) => attn_out.broadcast_mul(gamma)?,
            None => attn_out,
        };
        let x = (residual + attn_out)?;

        // Pre-norm MLP with residual + optional LayerScale.
        let residual = &x;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&x)?)?;
        let mlp_out = match &self.ls2 {
            Some(gamma) => mlp_out.broadcast_mul(gamma)?,
            None => mlp_out,
        };
        residual + mlp_out
    }
}

// ============================================================
// Full DINOv3 Vision Transformer
// ============================================================

/// Native Candle implementation of the DINOv3 Vision Transformer.
struct CandleVit {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    register_tokens: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    blocks: Vec<TransformerBlock>,
    norm: LayerNorm,
    config: ModelConfig,
    device: Device,
}

impl CandleVit {
    /// Forward pass: pixel values -> flattened feature tokens.
    fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        // 1. Patch embedding: [B, 3, H, W] -> [B, hidden_size, pH, pW]
        let patches = self.patch_embed.forward(pixel_values)?;
        let (b, c, ph, pw) = patches.dims4()?;
        // Flatten spatial dims and transpose: [B, num_patches, hidden_size]
        let patches = patches.reshape((b, c, ph * pw))?.permute((0, 2, 1))?;

        // 2. Token assembly (HF DINOv3 order: [CLS, registers, patches]).
        let cls = self
            .cls_token
            .broadcast_as((b, 1, self.config.hidden_size))?;
        let regs = self.register_tokens.broadcast_as((
            b,
            self.config.num_register_tokens,
            self.config.hidden_size,
        ))?;
        let mut x = Tensor::cat(&[&cls, &regs, &patches], 1)?;

        // 3. Transformer blocks (RoPE applied inside attention; no abs pos embed).
        for block in &self.blocks {
            x = block.forward(&x, &self.rope_cos, &self.rope_sin)?;
        }

        // 4. Final layer norm.
        x = self.norm.forward(&x)?;

        Ok(x)
    }

    /// Extract the flattened `last_hidden_state` feature vector.
    ///
    /// Returns all tokens flattened (hidden-contiguous, token order
    /// [CLS, registers, patches]) to match the reference GGML/ONNX backends,
    /// which return the full `last_hidden_state` rather than a pooled vector.
    fn extract_features(&self, pixel_values: &Tensor) -> candle_core::Result<Vec<f32>> {
        let output = self.forward(pixel_values)?; // [1, seq_len, hidden_size]
        output.flatten_all()?.to_vec1::<f32>()
    }

    // --------------------------------------------------------
    // Construction from VitWeights (internal format)
    // --------------------------------------------------------

    fn from_vit_weights(weights: &VitWeights, config: &ModelConfig, dev: &Device) -> Result<Self> {
        let hs = config.hidden_size;

        // Patch embedding conv2d.
        let pe_weight = Tensor::from_vec(
            weights.patch_embed_weight.clone(),
            (hs, 3, config.patch_size, config.patch_size),
            dev,
        )?;
        let pe_bias = Tensor::from_vec(weights.patch_embed_bias.clone(), hs, dev)?;
        let patch_embed = PatchEmbed {
            proj: Conv2d::new(
                pe_weight,
                Some(pe_bias),
                Conv2dConfig {
                    stride: config.patch_size,
                    padding: 0,
                    ..Default::default()
                },
            ),
        };

        // CLS token: stored as [hidden_size], reshape to [1, 1, hidden_size].
        let cls_token = Tensor::from_vec(weights.cls_token.clone(), (1, 1, hs), dev)?;

        // Register tokens: [num_register, hidden_size] -> [1, num_register, hidden_size].
        let num_reg = config.num_register_tokens;
        let register_tokens =
            Tensor::from_vec(weights.register_tokens.clone(), (1, num_reg, hs), dev)?;

        // Transformer blocks.
        let blocks = weights
            .layers
            .iter()
            .map(|layer| Self::build_block(layer, config, dev))
            .collect::<Result<Vec<_>>>()?;

        // Final layer norm.
        let norm_weight = Tensor::from_vec(weights.norm_weight.clone(), hs, dev)?;
        let norm_bias = Tensor::from_vec(weights.norm_bias.clone(), hs, dev)?;
        let norm = LayerNorm::new(norm_weight, norm_bias, LAYER_NORM_EPS);

        let head_dim = hs / config.num_heads;
        let (rope_cos, rope_sin) = build_rope_tables(config, head_dim, dev)?;

        Ok(CandleVit {
            patch_embed,
            cls_token,
            register_tokens,
            rope_cos,
            rope_sin,
            blocks,
            norm,
            config: config.clone(),
            device: dev.clone(),
        })
    }

    /// Build a single transformer block from VitWeights layer.
    fn build_block(
        layer: &TransformerLayer,
        config: &ModelConfig,
        dev: &Device,
    ) -> Result<TransformerBlock> {
        let hs = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hs / num_heads;

        // LayerNorm 1.
        let n1_w = Tensor::from_vec(layer.norm1_weight.clone(), hs, dev)?;
        let n1_b = Tensor::from_vec(layer.norm1_bias.clone(), hs, dev)?;
        let norm1 = LayerNorm::new(n1_w, n1_b, LAYER_NORM_EPS);

        // QKV projection (combined).
        let qkv_w = Tensor::from_vec(layer.qkv_weight.clone(), (3 * hs, hs), dev)?;
        let qkv_b = Tensor::from_vec(layer.qkv_bias.clone(), 3 * hs, dev)?;
        let qkv = Linear::new(qkv_w, Some(qkv_b));

        // Attention output projection.
        let proj_w = Tensor::from_vec(layer.proj_weight.clone(), (hs, hs), dev)?;
        let proj_b = Tensor::from_vec(layer.proj_bias.clone(), hs, dev)?;
        let o_proj = Linear::new(proj_w, Some(proj_b));

        let attn = Attention {
            qkv,
            o_proj,
            num_heads,
            head_dim,
        };

        // LayerNorm 2.
        let n2_w = Tensor::from_vec(layer.norm2_weight.clone(), hs, dev)?;
        let n2_b = Tensor::from_vec(layer.norm2_bias.clone(), hs, dev)?;
        let norm2 = LayerNorm::new(n2_w, n2_b, LAYER_NORM_EPS);

        // MLP fc1.
        let fc1_w = Tensor::from_vec(
            layer.mlp_fc1_weight.clone(),
            (config.intermediate_size, hs),
            dev,
        )?;
        let fc1_b = Tensor::from_vec(layer.mlp_fc1_bias.clone(), config.intermediate_size, dev)?;
        let fc1 = Linear::new(fc1_w, Some(fc1_b));

        // MLP fc2.
        let fc2_w = Tensor::from_vec(
            layer.mlp_fc2_weight.clone(),
            (hs, config.intermediate_size),
            dev,
        )?;
        let fc2_b = Tensor::from_vec(layer.mlp_fc2_bias.clone(), hs, dev)?;
        let fc2 = Linear::new(fc2_w, Some(fc2_b));

        // MLP fc3 (SwiGLU gate, optional).
        let fc3 = match (&layer.mlp_fc3_weight, &layer.mlp_fc3_bias) {
            (Some(w), Some(b)) => {
                let fc3_w = Tensor::from_vec(w.clone(), (config.intermediate_size, hs), dev)?;
                let fc3_b = Tensor::from_vec(b.clone(), config.intermediate_size, dev)?;
                Some(Linear::new(fc3_w, Some(fc3_b)))
            }
            _ => None,
        };

        let mlp = Mlp { fc1, fc2, fc3 };

        // LayerScale gammas (optional).
        let ls1 = layer
            .ls1_gamma
            .as_ref()
            .map(|g| Tensor::from_vec(g.clone(), hs, dev))
            .transpose()?;
        let ls2 = layer
            .ls2_gamma
            .as_ref()
            .map(|g| Tensor::from_vec(g.clone(), hs, dev))
            .transpose()?;

        Ok(TransformerBlock {
            norm1,
            attn,
            norm2,
            mlp,
            ls1,
            ls2,
        })
    }

    // --------------------------------------------------------
    // Construction from safetensors (HF Transformers DINOv3 format)
    // --------------------------------------------------------

    fn from_safetensors(path: &Path, config: &ModelConfig, dev: &Device) -> Result<Self> {
        let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, dev)
            .map_err(|e| anyhow::anyhow!("Failed to load safetensors: {}", e))?;

        let hs = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hs / num_heads;

        // --- Patch embedding ---
        let pe_weight = get_tensor(
            &tensors,
            &[
                "embeddings.patch_embeddings.weight",
                "embeddings.patch_embeddings.projection.weight",
                "patch_embed.proj.weight",
            ],
        )?;
        let pe_bias = get_tensor(
            &tensors,
            &[
                "embeddings.patch_embeddings.bias",
                "embeddings.patch_embeddings.projection.bias",
                "patch_embed.proj.bias",
            ],
        )?;
        let patch_embed = PatchEmbed {
            proj: Conv2d::new(
                pe_weight,
                Some(pe_bias),
                Conv2dConfig {
                    stride: config.patch_size,
                    padding: 0,
                    ..Default::default()
                },
            ),
        };

        // --- CLS token --- ([1, 1, hidden_size])
        let cls_token = get_tensor(&tensors, &["embeddings.cls_token", "cls_token"])?;
        let cls_token = reshape_token(&cls_token, 1, hs)?;

        // --- Register tokens --- ([1, num_register, hidden_size])
        let num_reg = config.num_register_tokens;
        let register_tokens =
            match try_get_tensor(&tensors, &["embeddings.register_tokens", "register_tokens"]) {
                Some(t) => reshape_token(&t, num_reg, hs)?,
                None => Tensor::zeros((1, num_reg, hs), DType::F32, dev)?,
            };

        // --- Transformer blocks ---
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("layer.{}", i);

            let norm1 = load_layer_norm(&tensors, &[&format!("{p}.norm1")])?;
            let norm2 = load_layer_norm(&tensors, &[&format!("{p}.norm2")])?;

            // Attention: separate q/k/v projections concatenated into one QKV.
            // Note: HF DINOv3 omits the K projection bias.
            let qkv = Self::load_qkv(&tensors, &p, hs, dev)?;

            let o_proj = load_linear(
                &tensors,
                &format!("{p}.attention.o_proj"),
                true,
                hs,
                dev,
            )?;

            let attn = Attention {
                qkv,
                o_proj,
                num_heads,
                head_dim,
            };

            // MLP: up_proj (fc1) / down_proj (fc2).
            let fc1 = load_linear(
                &tensors,
                &format!("{p}.mlp.up_proj"),
                true,
                config.intermediate_size,
                dev,
            )?;
            let fc2 = load_linear(&tensors, &format!("{p}.mlp.down_proj"), true, hs, dev)?;
            let mlp = Mlp {
                fc1,
                fc2,
                fc3: None,
            };

            // LayerScale gammas.
            let ls1 = try_get_tensor(
                &tensors,
                &[
                    &format!("{p}.layer_scale1.lambda1"),
                    &format!("{p}.ls1.gamma"),
                ],
            );
            let ls2 = try_get_tensor(
                &tensors,
                &[
                    &format!("{p}.layer_scale2.lambda1"),
                    &format!("{p}.ls2.gamma"),
                ],
            );

            blocks.push(TransformerBlock {
                norm1,
                attn,
                norm2,
                mlp,
                ls1,
                ls2,
            });
        }

        // --- Final layer norm ---
        let norm = load_layer_norm(&tensors, &["norm", "layernorm", "dinov2.layernorm"])?;

        let (rope_cos, rope_sin) = build_rope_tables(config, head_dim, dev)?;

        Ok(CandleVit {
            patch_embed,
            cls_token,
            register_tokens,
            rope_cos,
            rope_sin,
            blocks,
            norm,
            config: config.clone(),
            device: dev.clone(),
        })
    }

    /// Load Q/K/V projections and concatenate into a combined QKV linear.
    ///
    /// Supports both separate q/k/v projections (HF DINOv3, where K has no bias)
    /// and a pre-combined qkv tensor (timm-style).
    fn load_qkv(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        hs: usize,
        dev: &Device,
    ) -> Result<Linear> {
        // Pre-combined qkv (timm-style).
        if let Some(w) = try_get_tensor(
            tensors,
            &[
                &format!("{prefix}.attention.qkv.weight"),
                &format!("{prefix}.attn.qkv.weight"),
            ],
        ) {
            let b = try_get_tensor(
                tensors,
                &[
                    &format!("{prefix}.attention.qkv.bias"),
                    &format!("{prefix}.attn.qkv.bias"),
                ],
            );
            let b = match b {
                Some(b) => b,
                None => Tensor::zeros(3 * hs, DType::F32, dev)?,
            };
            return Ok(Linear::new(w, Some(b)));
        }

        // Separate q/k/v projections (HF Transformers DINOv3).
        let q_w = get_tensor(tensors, &[&format!("{prefix}.attention.q_proj.weight")])?;
        let k_w = get_tensor(tensors, &[&format!("{prefix}.attention.k_proj.weight")])?;
        let v_w = get_tensor(tensors, &[&format!("{prefix}.attention.v_proj.weight")])?;

        let q_b = try_get_tensor(tensors, &[&format!("{prefix}.attention.q_proj.bias")])
            .unwrap_or(Tensor::zeros(hs, DType::F32, dev)?);
        // HF DINOv3 has no K bias.
        let k_b = try_get_tensor(tensors, &[&format!("{prefix}.attention.k_proj.bias")])
            .unwrap_or(Tensor::zeros(hs, DType::F32, dev)?);
        let v_b = try_get_tensor(tensors, &[&format!("{prefix}.attention.v_proj.bias")])
            .unwrap_or(Tensor::zeros(hs, DType::F32, dev)?);

        let weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?; // [3*hs, hs]
        let bias = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?; // [3*hs]
        Ok(Linear::new(weight, Some(bias)))
    }
}

/// Reshape a token tensor of any leading batch dims into `[1, expected_tokens, hidden_size]`.
fn reshape_token(t: &Tensor, expected_tokens: usize, hs: usize) -> Result<Tensor> {
    let flat = t.flatten_all()?;
    let n = flat.elem_count();
    anyhow::ensure!(
        n == expected_tokens * hs,
        "token tensor has {} elements, expected {}",
        n,
        expected_tokens * hs
    );
    Ok(flat.reshape((1, expected_tokens, hs))?)
}

/// Load a `Linear` from `{prefix}.weight` (+ optional `{prefix}.bias`).
fn load_linear(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    with_bias: bool,
    out_dim: usize,
    dev: &Device,
) -> Result<Linear> {
    let w = get_tensor(tensors, &[&format!("{prefix}.weight")])?;
    let b = if with_bias {
        match try_get_tensor(tensors, &[&format!("{prefix}.bias")]) {
            Some(b) => Some(b),
            None => Some(Tensor::zeros(out_dim, DType::F32, dev)?),
        }
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

/// Load a `LayerNorm` from `{prefix}.weight` and `{prefix}.bias`.
fn load_layer_norm(tensors: &HashMap<String, Tensor>, prefixes: &[&str]) -> Result<LayerNorm> {
    let w_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
    let b_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
    let w = get_tensor_from_map(tensors, &w_names)?;
    let b = get_tensor_from_map(tensors, &b_names)?;
    Ok(LayerNorm::new(w, b, LAYER_NORM_EPS))
}

/// Helper: get tensor from map trying multiple name patterns (String variant).
fn get_tensor_from_map(tensors: &HashMap<String, Tensor>, names: &[String]) -> Result<Tensor> {
    for name in names {
        if let Some(t) = tensors.get(name) {
            return Ok(t.clone());
        }
    }
    anyhow::bail!("Could not find tensor with any of these names: {:?}", names)
}

// ============================================================
// CandleBackend: InferenceBackend trait implementation
// ============================================================

/// Candle-based native Rust ViT inference backend.
pub struct CandleBackend {
    model: Option<Arc<CandleVit>>,
    config: ModelConfig,
    device: Device,
    info: Option<ModelInfo>,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self {
            model: None,
            config: ModelConfig::vit_small_16(),
            device: Device::Cpu,
            info: None,
        }
    }

    /// Create a backend with a specific config.
    pub fn with_config(config: ModelConfig) -> Self {
        Self {
            model: None,
            config,
            device: Device::Cpu,
            info: None,
        }
    }

    /// Load model from VitWeights (internal format).
    pub fn load_from_weights(&mut self, weights: &VitWeights) -> Result<()> {
        let device = Self::detect_device();
        let model = CandleVit::from_vit_weights(weights, &self.config, &device)?;
        self.info = Some(ModelInfo {
            name: format!("DINOv3 Candle ({})", self.config.hidden_size),
            backend: "candle".to_string(),
            config: self.config.clone(),
            quantization: None,
        });
        self.device = device;
        self.model = Some(Arc::new(model));
        Ok(())
    }

    /// Detect best available device (CUDA > CPU).
    fn detect_device() -> Device {
        match Device::cuda_if_available(0) {
            Ok(dev) if dev.is_cuda() => {
                log::info!("Candle: using CUDA device");
                dev
            }
            _ => {
                log::info!("Candle: using CPU device");
                Device::Cpu
            }
        }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CandleBackend {
    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        let device = Self::detect_device();
        let ext = model_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let model = match ext {
            "safetensors" => {
                log::info!("Loading safetensors model from {:?}", model_path);
                CandleVit::from_safetensors(model_path, &self.config, &device)?
            }
            _ => {
                // Try loading as safetensors anyway (some files may not have extension).
                log::info!("Attempting to load model from {:?}", model_path);
                CandleVit::from_safetensors(model_path, &self.config, &device).with_context(
                    || {
                        format!(
                            "Failed to load model from {:?}. \
                             Supported formats: safetensors. \
                             For VitWeights, use load_from_weights() instead.",
                            model_path
                        )
                    },
                )?
            }
        };

        self.info = Some(ModelInfo {
            name: format!(
                "DINOv3 Candle {} ({})",
                model_path.file_stem().unwrap_or_default().to_string_lossy(),
                self.config.hidden_size
            ),
            backend: "candle".to_string(),
            config: self.config.clone(),
            quantization: None,
        });
        self.device = device;
        self.model = Some(Arc::new(model));
        Ok(())
    }

    fn extract_features(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded. Call load_model() first."))?;

        // 1. Preprocess image using existing pipeline.
        let pixels = preprocess_image(image_data, &self.config)?;

        // 2. Convert to candle Tensor [1, 3, H, W].
        let h = self.config.input_height;
        let w = self.config.input_width;
        let tensor = Tensor::from_vec(pixels, (1, 3, h, w), &self.device)?;

        // 3. Run forward pass.
        let features = model
            .extract_features(&tensor)
            .map_err(|e| anyhow::anyhow!("Candle inference failed: {}", e))?;

        // 4. L2 normalize.
        let mut features = features;
        crate::dinov3::l2_normalize(&mut features);

        Ok(features)
    }

    fn model_info(&self) -> ModelInfo {
        self.info.clone().unwrap_or(ModelInfo {
            name: "Candle DINOv3 (not loaded)".to_string(),
            backend: "candle".to_string(),
            config: self.config.clone(),
            quantization: None,
        })
    }
}
