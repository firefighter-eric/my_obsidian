# Black Forest Labs - 2026 - FLUX.2 Overview

- Source HTML: `raw/html/Black Forest Labs - 2026 - FLUX.2 Overview.html`
- Source URL: https://docs.bfl.ai/flux_2/flux2_overview
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

Skip to main content

🚀 FLUX.2 [klein] — Sub-second generation. Open weights, Apache 2.0, API from $0.014/image. Learn more →

Black Forest Labs home page

Search...

- Help Center

- API Status

- API Pricing

- Get API Key

Get API Key

Search...

Navigation

FLUX.2

Overview

##### Get Started

Overview

Quick Start

Image Generation

API Pricing

##### Account Management

Organizations & Projects

Team Management

Credits & Billing

##### FLUX.2

Overview

FLUX.2 Image Editing

FLUX.2 Text to Image

FLUX.2 [klein] Training

FLUX.2 [klein] Style Training

##### FLUX.1 Kontext

Introduction

Image Generation

Image Editing

##### FLUX1.1 [pro] Models

FLUX1.1 [pro]

FLUX1.1 [pro] Ultra / Raw

##### FLUX.1 Tools

FLUX.1 Fill [pro]

##### API Integration

Integration Guide

MCP Integration

Agent Skills

Errors

On this page

- What Can You Do?

- Which Model to Choose?

- Compare FLUX.2 Models

- At a Glance

- Use Cases

- FLUX.2 [klein] Models

- API Models

- Open Weights (Community)

- Preview Endpoints

- Technical Specifications

- Getting Started

FLUX.2

# Overview

Copy page

FLUX.2 model family overview — from sub-second generation to highest quality, with multi-reference editing, color control, and up to 4MP output.

Copy page

FLUX.2 spans the full spectrum of image generation—from sub-second inference with [klein] to highest quality with [max]. Generate photorealistic images with precise control over colors, poses, and composition, or edit existing images by referencing up to 10 sources simultaneously.
Choose [klein] for real-time, high-volume generation, [pro] for production at scale, [flex] for fine-grained control, or [max] for maximum quality and grounding search.

Want to try first? Test FLUX.2 [max], [pro], and [flex] in our playground. [klein] is available via our API and on Hugging Face.

​

Multi-Reference

Photorealism & Detail

Grounding Search

Typography & Text

Exact Color Control

Structured Prompting

Combine elements from multiple images while maintaining identity across complex scenes. Create ad variants with consistent faces, product mockups in any context, or fashion editorials where models stay consistent.

Generate photorealistic images with enhanced detail, texture, and lighting. FLUX.2 produces images that merge seamlessly with real photography—ideal for e-commerce and product marketing.

Generate images grounded in real-time information with FLUX.2 [max]. It searches the web when needed, so you can create visuals of yesterday’s football game, the weather in real-time of any cities, or re-create historical events.

Reliable text rendering for infographics, UI mockups, and marketing materials.

Specify brand colors via hex codes with precision matching. No approximation—get the exact colors you need.Example: Gradient colors with hex codesPrompt: A vase on a table in living room, the color of the vase is a gradient of color, starting with color #02eb3c and finishing with color #edfa3c. The flowers inside the vase have the color #ff0088

Use structured prompts for precise control over generation. Perfect for production workflows and automation.

Example: Structured Prompting

```
{
 "subject": "Mona Lisa painting by Leonardo da Vinci",
 "background": "museum gallery wall, ornate gold frame",
 "lighting": "soft gallery lighting, warm spotlights",
 "style": "digital art, high contrast",
 "camera_angle": "eye level view",
 "composition": "centered, portrait orientation"
}
```

​

[klein][max][pro][flex][dev]

Best forReal-time, high-volumeHighest quality, final assetsProduction at scaleQuality with controlLocal development

Multi-referenceUp to 4Up to 8 (API), 10 (playground)Up to 8 (API), 10 (playground)Up to 8 (API), 10 (playground)Recommended max 6

ControlsStandardStandardStandardAdjustable steps & guidanceFull customization

Grounding searchNoYesNoNoNo

Pricingfrom $0.014 / imagefrom $0.07 / MPfrom $0.03 / MP$0.06 / MPFree (non-commercial)

FLUX.2 [klein] delivers sub-second inference with open weights. 4B runs on consumer GPUs (~13GB VRAM). Apache 2.0 for 4B, FLUX NCL for 9B. See model details below.

FLUX.2 [max] includes grounding search: when prompted, it performs web searches to access real-time information to visualize trending products, current events, or the latest styles without manually sourcing reference material.

​

## [klein]

Sub-second inference. Our fastest models with open weights. Runs on consumer GPUs (~13GB VRAM). From $0.014/image via API, or run locally with Apache 2.0 (4B) / FLUX NCL (9B).

## [max]

Maximum performance. Highest editing consistency across tasks. Vast world knowledge. Strongest prompt following and faithful style representation.

## [pro]

Top performance at affordable price. The high quality, production-grade image editing and generation model.

## [flex]

Specialized for typography. Best for text rendering and preserving small details.

​

Use CaseFLUX.2 [klein]FLUX.2 [max]FLUX.2 [pro]FLUX.2 [flex]

Product MarketingBulk catalog generation, A/B testing variantsHighest quality hero shots for marketplacesCreate ads at scale for social campaignsText overlay while preserving details

Movie MakingRapid storyboarding, concept explorationTop quality cinematic pre-visualizationRapid ideation and static movie bannersIntros, credits, static advertising

Creative PlatformsCost-efficient generation for all tiersPremium model for highest-tier subsHigh quality backbone at scaleSpecialized text placement

E-commerceHigh-volume product variations, thumbnailsPremium product photographyProduction-grade catalog imagesPrice tags, labels, descriptions

Editorial & FashionRapid mood boards, style explorationFinal hero imagesCampaign imagery at scaleText-heavy layouts

​

Open weights available: [klein] 4B is fully open under Apache 2.0. [klein] 9B is available under the FLUX Non-Commercial License. Download from Hugging Face.

​

[klein] 4B[klein] 9B

Best forHigh volume, local deploymentBalanced quality and speed

Architecture4B flow model9B flow model + 8B Qwen3 text embedder

Inference steps4 (step-distilled)4 (step-distilled)

VRAM~13GB~24GB

SpeedSub-secondSub-second

API Pricing0.014+0.014 + 0.014+0.001/MP0.015+0.015 + 0.015+0.002/MP

LicenseApache 2.0FLUX Non-Commercial License

​

[klein] Base 4B[klein] Base 9B

Best forFine-tuning, research, custom pipelinesMaximum quality, research

Output diversityHighHighest

Step-distilledNo (full capacity)No (full capacity)

LicenseApache 2.0FLUX Non-Commercial License

AvailabilityHugging FaceHugging Face

Base models are available as open weights for local development and research. They are not offered on the public API.

FLUX.2 [klein] does not include prompt upsampling. Write detailed, descriptive prompts for best results. See our prompting guide for techniques.

​

EndpointDescription

flux-2-pro-previewOur latest FLUX.2 [pro] model.

flux-2-proA fixed snapshot of FLUX.2 [pro]. This endpoint will not change, making it suitable for workflows that require reproducibility.

flux-2-klein-9b-previewOur latest FLUX.2 [klein] 9B model with KV caching for improved performance.

flux-2-klein-9bA fixed snapshot of FLUX.2 [klein] 9B. Choose this when you need reproducibility.

Which endpoint should I use? For most use cases, the preview endpoints (flux-2-pro-preview, flux-2-klein-9b-preview) give you the best results. Choose the non-preview endpoints when you need a pinned model — for example, if your workflow depends on consistent outputs across runs or you have compliance requirements around model stability.

The flux-2-pro and flux-2-klein-9b endpoints are unchanged. If you are already using them, no action is required.

​

## Resolution

- Output: Up to 4MP

- Input: 64x64 minimum

- Any aspect ratio

## Multi-Reference

- Up to 10 input images ([klein]: 4)

- Character consistency

- Style transfer

## Advanced Controls

- Pose guidance

- Hex color matching

- Structured prompting

- Grounding search ([max] only)

​

## Try in Playground

Test FLUX.2 [max], [pro], and [flex] in your browser. No setup required.

## Download [klein] Weights

Get [klein] weights from Hugging Face for local inference.

## Text-to-Image API

Generate images from text prompts.

## Image Editing API

Edit images with multi-reference support.

## [klein] Prompting Guide

Master narrative prompting for best [klein] results.

## Local Development

Download [dev] weights for local inference.

Was this page helpful?

YesNo

Credits & BillingFLUX.2 Image Editing

⌘I

Black Forest Labs home page

xgithublinkedin

Legal

Company

xgithublinkedin

Powered byThis documentation is built and hosted on Mintlify, a developer documentation platform

Assistant

Responses are generated using AI and may contain mistakes.
