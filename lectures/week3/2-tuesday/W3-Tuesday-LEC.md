# Week 3 Tuesday -- AWS Bedrock Foundations and Governance

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- Bedrock Foundations: Console & API Overview, CLI/Credentials/Boto3 Setup, Available Foundation Models, Model Selection Criteria, Managed vs Self-hosted
- Bedrock Governance: IAM & Resource Policies, Policy Configuration, Guardrails & PII Redaction, Cost Management, CloudWatch Monitoring, API Quotas & Throttling

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Bedrock Foundations: Console Overview, Boto3 Setup, Foundation Models, Model Invocation | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | Building with Bedrock: FraudShield Investigation Summaries, Model Comparison, Streaming | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Governance and Guardrails: IAM, Content Filtering, PII Redaction, Cost, Monitoring | 55 |
| Buffer | Course Wrap-up, Review Preparation | 10 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

On Wednesday you built attention and transformer architectures from scratch. You know what BERT, GPT, and T5 are. Bedrock gives you managed access to production-ready foundation models -- Claude, Titan, Llama, Mistral -- without managing any infrastructure. Today is about using these models effectively and governing them for enterprise deployment. We apply Bedrock to FraudShield: generating fraud investigation summaries from raw transaction data.

Associates continue as ML engineers at FraudShield. Over two weeks you trained custom models, deployed endpoints, tuned hyperparameters, monitored drift, and built pipelines. Today flips the question: instead of training your own model, how do you leverage someone else's massive pre-trained model through a managed API? And once you do, how do you govern that access for an enterprise?

1. **"What is Bedrock and why would I use it instead of deploying my own LLM?"** (Managed foundation model access, no infrastructure, pay per token)
2. **"Which model should I choose for my task?"** (Model selection criteria: latency, cost, capability, context window)
3. **"How do I invoke a foundation model from Python?"** (Boto3 bedrock-runtime, Messages API)
4. **"How do I ensure responsible, cost-effective, and secure usage?"** (Guardrails, IAM, cost management, monitoring)

Stage 1 establishes what Bedrock is and how to interact with it programmatically. Stage 2 applies Bedrock to FraudShield by generating investigation summaries and comparing models. Stage 3 covers enterprise governance: IAM policies, guardrails for content filtering and PII redaction, cost management, and monitoring.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] AWS account with Bedrock model access enabled (Claude, Titan at minimum)
- [ ] Boto3 and AWS CLI credentials configured
- [ ] Companion lecture notebook (`W3-Tuesday-notebook.ipynb`) open and tested
- [ ] Foundation model access requests approved in the Bedrock console (model access can take minutes to hours)
- [ ] IAM role with `bedrock:*` and `bedrock-runtime:*` permissions verified
- [ ] Budget verified and active
- [ ] This instructor guide open in a second tab

### Student Prerequisites

- [ ] Completed readings: Console & API Overview CT, CLI/Credentials/Boto3 Setup CT, Available Foundation Models CT, Model Selection Criteria CT, Managed vs Self-hosted CT, IAM & Resource Policies CT, Policy Configuration CT, Guardrails & PII Redaction CT, Cost Management CT, CloudWatch Monitoring CT, API Quotas & Throttling CT
- [ ] Wednesday's notebook completed (understand transformers, BERT/GPT/T5)
- [ ] AWS credentials configured, boto3 installed
- [ ] Familiarity with SageMaker endpoint deployment from Weeks 1-2

---

# STAGE 1 -- Bedrock Foundations (55 min)

> **Goal:** Understand what AWS Bedrock is, explore available foundation models programmatically, apply model selection criteria, and invoke a foundation model for text generation.

**Exit Criteria Addressed:**
- Explain what AWS Bedrock is and how it differs from self-hosted model deployment (Required)
- Navigate the Bedrock console conceptually and describe its key sections (Required)
- Configure boto3 clients for Bedrock and Bedrock Runtime (Required)
- List and filter available foundation models using the Bedrock API (Required)
- Apply model selection criteria to choose the right model for a given task (Required)
- Invoke a foundation model using the Messages API and interpret the response (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "For two weeks you have been training, tuning, and deploying your own models. You control everything: the algorithm, the hyperparameters, the instance type, the endpoint. That is powerful, but it is also expensive and time-consuming -- especially for large language models. Training GPT-scale models costs millions of dollars. Hosting them requires GPU instances running 24/7."

> "Bedrock flips that model. Instead of training and hosting, you get API access to foundation models built by Anthropic, Amazon, Meta, Mistral, and others. No infrastructure to manage. No model weights to store. You send a prompt, you get a response, you pay per token. The models you studied on Wednesday -- transformer-based architectures like GPT and BERT -- these are the same architectures powering the foundation models in Bedrock, just scaled to billions of parameters and trained on massive datasets."

---

## STEP 1 -- What is AWS Bedrock (8 minutes)

**Pacing: conceptual, notebook markdown.**

> "Bedrock is a fully managed service that provides API access to foundation models from multiple providers. You do not train these models. You do not host them. You call an API."

Key characteristics:

| Aspect | Description |
|--------|-------------|
| **What it is** | Managed API access to foundation models |
| **What you do NOT manage** | Infrastructure, model weights, scaling, GPU instances |
| **What you DO control** | Which model, prompt design, guardrails, IAM policies |
| **Pricing** | Pay per input/output token (on-demand) or provisioned throughput |
| **Providers** | Anthropic (Claude), Amazon (Titan), Meta (Llama), Mistral, Cohere, AI21, Stability AI |

> "Think of it as 'SageMaker endpoints, but someone else trained and hosts the model.' You focus on prompt engineering, guardrails, and governance -- not infrastructure."

**Discussion Prompt:** "When would you still train your own model on SageMaker instead of using Bedrock?" (When you need domain-specific performance that prompt engineering cannot achieve, when you need full control over the model architecture, when your data cannot leave your VPC, or when inference cost at scale favors a smaller custom model.)

---

## STEP 2 -- Console Overview (5 minutes)

**Pacing: conceptual, notebook markdown. Describe the console sections -- do not live-demo the console.**

> "The Bedrock console has several key sections. We will not navigate the console live, but you should understand its layout."

Walk through the console sections conceptually:

| Console Section | Purpose |
|----------------|---------|
| **Model access** | Request and manage access to foundation models from each provider |
| **Playgrounds** | Interactive testing area -- send prompts to models and see responses |
| **Guardrails** | Create and manage content filtering and PII redaction policies |
| **Model catalog** | Browse all available models with descriptions, pricing, and capabilities |
| **Custom models** | Fine-tune or continue pre-training foundation models on your data |
| **Provisioned throughput** | Reserve dedicated capacity for consistent performance |

> "The console is useful for exploration and testing. For production workloads, you use the API through boto3 -- which is what we focus on today."

---

## STEP 3 -- Boto3 Setup for Bedrock (7 minutes)

**Pacing: live code in notebook.**

> "Bedrock uses two boto3 clients. This is a common source of confusion."

| Client | Service Name | Purpose |
|--------|-------------|---------|
| **bedrock** | `bedrock` | Management operations: list models, create guardrails, manage access |
| **bedrock-runtime** | `bedrock-runtime` | Inference operations: invoke models, stream responses |

> "The split is similar to SageMaker's `sagemaker` client for management and `sagemaker-runtime` for inference. Management and inference are separate API surfaces."

Run the cell that creates both clients. Verify the connection by calling `list_foundation_models`.

---

## STEP 4 -- List and Explore Foundation Models (10 minutes)

**Pacing: live code in notebook.**

> "Bedrock gives you access to models from multiple providers. Let us see what is available."

Run the cell that calls `bedrock.list_foundation_models()` and displays a table with:
- Model name
- Provider
- Input modalities (TEXT, IMAGE)
- Output modalities (TEXT, IMAGE, EMBEDDING)
- Model ID (the identifier you use when invoking)

> "The model ID is what you pass to `invoke_model`. For Claude, it looks like `anthropic.claude-3-sonnet-20240229-v1:0`. For Titan, it looks like `amazon.titan-text-express-v1`. You need the exact ID -- no shortcuts."

Walk through the providers and highlight key models:

| Provider | Key Models | Strengths |
|----------|-----------|-----------|
| **Anthropic** | Claude 3 (Haiku, Sonnet, Opus) | Reasoning, instruction following, safety |
| **Amazon** | Titan Text, Titan Embeddings | Cost-effective, AWS-native |
| **Meta** | Llama 3 | Open-weight, strong general performance |
| **Mistral** | Mistral, Mixtral | Efficient, fast inference |

---

## STEP 5 -- Model Selection Criteria (8 minutes)

**Pacing: conceptual with notebook markdown table.**

> "You have dozens of models available. How do you choose? Four dimensions matter."

| Criterion | What to Evaluate | FraudShield Example |
|-----------|-----------------|---------------------|
| **Latency** | Time to first token, total generation time | Real-time fraud alerts need low latency -- Haiku or Mistral |
| **Cost** | Price per 1K input/output tokens | High-volume batch summaries -- Titan or Haiku for cost efficiency |
| **Capability** | Reasoning depth, instruction following, output quality | Complex investigation summaries -- Sonnet or Opus |
| **Context Window** | Maximum input + output tokens | Long transaction histories -- models with 100K+ context |

> "There is no universally best model. The best model is the cheapest one that meets your quality and latency requirements. Start with the smallest model that works, then scale up only if quality is insufficient."

Present the selection heuristic:

```
Need the highest quality reasoning?
  YES --> Claude Sonnet or Opus
  NO  --> Is latency critical (< 1 second)?
            YES --> Claude Haiku or Mistral
            NO  --> Is cost the primary concern?
                      YES --> Titan Text or Haiku
                      NO  --> Claude Sonnet (balanced)
```

**Discussion Prompt:** "FraudShield wants to generate one-paragraph investigation summaries from transaction data. The summaries are reviewed by human analysts, not shown to customers. Which model would you start with, and why?" (Start with Haiku or Titan for cost. Quality requirements are moderate since humans review the output. Only upgrade to Sonnet if the summaries are insufficient.)

---

## STEP 6 -- Invoke a Foundation Model (12 minutes)

**Pacing: live code in notebook. Walk through every component of the request and response.**

> "This is the core operation: sending a prompt to a foundation model and getting a response. We use the Messages API format, which is the standard for Claude models on Bedrock."

Walk through the invocation step by step:

1. **Construct the request body** -- model ID, messages array, max tokens, temperature
2. **Call `invoke_model`** -- pass the body as JSON, specify the model ID and content type
3. **Parse the response** -- extract the generated text from the response body

Key parameters to explain:

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `modelId` | Which foundation model to invoke | `anthropic.claude-3-haiku-20240307-v1:0` |
| `messages` | Conversation history (role + content pairs) | `[{"role": "user", "content": "..."}]` |
| `max_tokens` | Maximum tokens to generate | 256-4096 depending on task |
| `temperature` | Randomness (0 = deterministic, 1 = creative) | 0.1-0.3 for factual tasks |
| `anthropic_version` | API version for Anthropic models | `bedrock-2023-05-31` |

> "The Messages API uses a conversation format: an array of messages with roles ('user' and 'assistant'). This is the same pattern whether you use Bedrock, the Anthropic API directly, or any chat-based LLM API."

Run the invocation cell. Show the response and explain the structure:

- `content[0].text` -- the generated text
- `usage.input_tokens` -- how many tokens the prompt consumed
- `usage.output_tokens` -- how many tokens the model generated
- Token counts directly determine cost

---

## STEP 7 -- Request/Response Format (2 minutes)

**Pacing: conceptual, notebook markdown.**

> "Every model provider has a slightly different request format. Bedrock abstracts some of this, but you need to know the structure for the models you use."

Summarize the Claude Messages API format:

```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 512,
  "temperature": 0.2,
  "messages": [
    {"role": "user", "content": "Your prompt here"}
  ]
}
```

Response structure:

```json
{
  "content": [{"type": "text", "text": "Model response here"}],
  "usage": {"input_tokens": 42, "output_tokens": 128},
  "stop_reason": "end_turn"
}
```

> "The `stop_reason` tells you why the model stopped: `end_turn` means it finished naturally. `max_tokens` means it hit the limit. If you see `max_tokens`, your output was truncated -- increase the limit."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- Building with Bedrock (55 min)

> **Goal:** Apply Bedrock to FraudShield by generating fraud investigation summaries from transaction data, compare models on the same task, understand managed vs self-hosted trade-offs, and implement streaming responses.

**Exit Criteria Addressed:**
- Design a structured prompt that converts raw transaction data into a fraud investigation summary (Required)
- Compare multiple foundation models on the same task and evaluate output quality (Required)
- Articulate the trade-offs between Bedrock (managed) and SageMaker (self-hosted) for LLM workloads (Required)
- Implement streaming responses using `invoke_model_with_response_stream` (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "In Stage 1 you invoked a model with a simple prompt. That is the 'hello world' of Bedrock. In production, the prompt is everything. A well-structured prompt turns raw data into actionable output. A vague prompt produces vague results. We are going to take FraudShield transaction data -- amounts, timestamps, merchant info, risk scores -- and generate investigation summaries that a fraud analyst can act on."

---

## STEP 8 -- FraudShield Use Case: Investigation Summaries (5 minutes)

**Pacing: conceptual, notebook markdown.**

> "FraudShield's fraud analysts currently write investigation summaries manually. Each summary requires reviewing raw transaction data, identifying suspicious patterns, and writing a narrative that explains why the transaction was flagged. This takes 15-20 minutes per case. With Bedrock, we can generate a draft summary in seconds."

The pipeline:

```
Raw Transaction Data (amount, hour, merchant, risk score, flag reason)
    |
    v
Structured Prompt (instructions + data + output format)
    |
    v
Foundation Model (Bedrock API)
    |
    v
Draft Investigation Summary (human analyst reviews and approves)
```

> "The model does not make the final fraud decision. It generates a draft that a human analyst reviews. This is the 'human in the loop' pattern -- the model accelerates the analyst's work without replacing their judgment."

---

## STEP 9 -- Structured Prompt for Fraud Summaries (12 minutes)

**Pacing: live code in notebook. Walk through prompt construction carefully.**

> "The quality of the output depends entirely on the quality of the prompt. We build a structured prompt with four components."

| Component | Purpose | Example |
|-----------|---------|---------|
| **System context** | Tell the model who it is and what it does | "You are a fraud investigation analyst..." |
| **Data section** | Provide the raw transaction data | Amount, hour, merchant, risk score |
| **Instructions** | Specify what output you want | "Generate a summary with: risk assessment, key indicators, recommended action" |
| **Format constraints** | Control the output structure | "Use bullet points. Keep to 3-4 sentences per section." |

Walk through the code that constructs the prompt and invokes the model. Show the generated summary.

> "Notice how specific the instructions are. We do not say 'summarize this transaction.' We say 'generate a risk assessment, list key indicators, recommend an action.' Specificity in the prompt produces specificity in the output."

**Discussion Prompt:** "What could go wrong if we used this summary directly without human review?" (The model might hallucinate details not in the data, misinterpret risk scores, or miss context that a human analyst would catch. AI-generated summaries should always be reviewed before action.)

---

## STEP 10 -- Compare Models on the Same Task (12 minutes)

**Pacing: live code in notebook.**

> "One of Bedrock's advantages is model choice. You can test multiple models on the same task and compare quality, speed, and cost. Let us run the same fraud summary prompt through two different models."

Run the same prompt through two models (e.g., Claude Haiku and Amazon Titan). Display both outputs side by side. Compare:

| Dimension | Model A | Model B |
|-----------|---------|---------|
| **Output quality** | How detailed and accurate is the summary? | |
| **Latency** | Time from request to complete response | |
| **Token usage** | Input + output tokens consumed | |
| **Estimated cost** | Based on per-token pricing | |

> "In production, you would run this comparison on a sample of 50-100 transactions and have domain experts rate the summaries. The goal is to find the cheapest model that produces acceptable quality. Do not default to the most expensive model."

**Discussion Prompt:** "If Model A produces slightly better summaries but costs 10x more per invocation, how would you decide?" (Depends on volume. At 100 summaries per day, the cost difference is negligible -- use Model A. At 100,000 per day, the 10x cost difference is significant -- use Model B and accept slightly lower quality, or invest in prompt engineering to close the quality gap.)

---

## STEP 11 -- Managed vs Self-hosted Trade-offs (8 minutes)

**Pacing: conceptual with notebook comparison table.**

> "You have now used both SageMaker endpoints (Weeks 1-2) and Bedrock (today). When should you use each?"

| Dimension | Bedrock (Managed) | SageMaker (Self-hosted) |
|-----------|-------------------|------------------------|
| **Infrastructure** | None -- fully managed | You provision and manage instances |
| **Model choice** | Pre-built foundation models only | Any model (custom, open-source, fine-tuned) |
| **Customization** | Prompt engineering, fine-tuning (limited) | Full control over architecture and training |
| **Latency** | Variable (shared infrastructure) | Predictable (dedicated instances) |
| **Cost model** | Per token (on-demand) | Per instance-hour (always running) |
| **Scale to zero** | Yes (pay only when used) | No (endpoint runs continuously) |
| **Data privacy** | Data processed on AWS infrastructure | Data stays on your instances |
| **Best for** | Text generation, summarization, Q&A, classification via prompting | Custom models, specialized architectures, high-volume inference |

> "Bedrock is the right choice when a foundation model can solve your problem through prompting and the volume does not justify dedicated infrastructure. SageMaker is the right choice when you need a custom model, predictable latency, or cost-efficient high-volume inference."

**Discussion Prompt:** "For FraudShield's XGBoost fraud classifier that scores 500 TPS, would you use Bedrock or SageMaker?" (SageMaker. XGBoost is a custom-trained model, not a foundation model. Bedrock does not host custom XGBoost models. Even if it did, 500 TPS of per-token billing would be far more expensive than a dedicated endpoint.)

---

## STEP 12 -- Streaming Responses (10 minutes)

**Pacing: live code in notebook.**

> "When generating long responses, waiting for the entire output before displaying anything creates a poor user experience. Streaming sends tokens as they are generated, so the user sees output appearing incrementally."

Key differences:

| Aspect | `invoke_model` | `invoke_model_with_response_stream` |
|--------|---------------|-------------------------------------|
| **Response delivery** | All at once after generation completes | Token by token as generated |
| **User experience** | Blank wait, then full response | Progressive display |
| **Latency to first token** | High (waits for full generation) | Low (streams immediately) |
| **Use case** | Backend processing, batch jobs | Interactive applications, chat interfaces |

Run the streaming cell. Show tokens appearing one at a time.

> "Streaming does not change the total generation time. It changes the perceived latency. The user sees the first tokens within milliseconds instead of waiting seconds for the full response."

---

## STEP 13 -- When to Use Bedrock vs SageMaker for LLMs (5 minutes)

**Pacing: discussion, notebook markdown.**

> "This is the decision framework for LLM workloads specifically."

| Scenario | Recommendation |
|----------|---------------|
| Prototyping a text generation feature | Bedrock (fast to start, no infrastructure) |
| High-volume production LLM inference (100K+ requests/day) | SageMaker or Bedrock provisioned throughput (cost comparison needed) |
| Need a model fine-tuned on proprietary data | SageMaker (full control) or Bedrock custom model (limited) |
| Multi-modal application (text + image + code) | Bedrock (access to multi-modal foundation models) |
| Strict latency SLA (< 100ms p99) | SageMaker (dedicated instances, predictable performance) |
| Regulatory requirement to keep data on dedicated infrastructure | SageMaker (your instances, your VPC) |

> "The two services are complementary, not competitive. Many production architectures use both: Bedrock for text generation and summarization, SageMaker for custom classification and scoring."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Governance and Guardrails (55 min)

> **Goal:** Secure Bedrock access with IAM policies, create guardrails for content filtering and PII redaction, understand cost management options, set up CloudWatch monitoring, and handle API throttling.

**Exit Criteria Addressed:**
- Configure IAM policies to control Bedrock model access (Required)
- Create a Bedrock guardrail with content filtering and PII detection (Required)
- Test a guardrail to verify content filtering and PII redaction behavior (Required)
- Compare on-demand and provisioned throughput pricing models (Required)
- Identify key CloudWatch metrics for Bedrock monitoring (Required)
- Implement retry logic with exponential backoff for API throttling (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "You can now invoke foundation models and generate useful output. But in an enterprise, access without governance is a liability. Who can access which models? What happens if a prompt contains personally identifiable information? How do you prevent the model from generating harmful content? How do you control costs when every invocation has a price? Stage 3 answers these questions."

---

## STEP 14 -- IAM Policies for Bedrock (8 minutes)

**Pacing: conceptual with notebook markdown table.**

> "Bedrock access is controlled through standard IAM policies. The key is understanding which actions map to which operations."

| Permission | Action | Use Case |
|------------|--------|----------|
| `bedrock:ListFoundationModels` | List available models | Discovery, model catalog |
| `bedrock:GetFoundationModel` | Get details about a specific model | Model selection |
| `bedrock:InvokeModel` | Send a prompt and get a response | Core inference |
| `bedrock:InvokeModelWithResponseStream` | Stream responses | Interactive applications |
| `bedrock:CreateGuardrail` | Create content filtering rules | Governance setup |
| `bedrock:ApplyGuardrail` | Apply a guardrail to content | Content moderation |
| `bedrock:CreateProvisionedModelThroughput` | Reserve dedicated capacity | Production workloads |

> "The principle of least privilege applies. A fraud analyst who only needs to invoke Claude Haiku should not have permissions to create guardrails or provision throughput. A governance admin who manages guardrails should not need inference permissions."

Present two example policies:

**Analyst policy (invoke only):**
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku*"
}
```

**Admin policy (full management):**
```json
{
  "Effect": "Allow",
  "Action": "bedrock:*",
  "Resource": "*"
}
```

> "Notice the resource-level control. You can restrict access to specific models by ARN. This means you can allow Haiku but deny Opus -- controlling cost at the IAM level."

---

## STEP 15 -- Create a Guardrail (12 minutes)

**Pacing: live code in notebook.**

> "Guardrails are Bedrock's mechanism for content filtering and PII protection. A guardrail is a reusable policy that you apply to model invocations. It inspects both the input prompt and the output response."

Walk through the guardrail creation:

1. **Content filters** -- block harmful content categories (hate, violence, sexual content, insults, misconduct)
2. **Denied topics** -- block specific topics you define (e.g., investment advice, medical diagnosis)
3. **PII detection** -- identify and redact personally identifiable information (names, SSNs, credit card numbers, email addresses)

Key configuration:

| Component | What It Does | FraudShield Relevance |
|-----------|-------------|----------------------|
| **Content filter** | Blocks harmful content by category | Prevent model from generating offensive investigation summaries |
| **Denied topics** | Blocks prompts about specific subjects | Prevent using fraud investigation tool for non-fraud queries |
| **PII detection** | Identifies and redacts PII in prompts/responses | Redact customer names and card numbers from summaries |

Run the cell that creates a guardrail with content filtering (MEDIUM strength) and PII detection (ANONYMIZE action for names, email, phone, SSN, credit card numbers).

> "The guardrail does not modify the model. It wraps the model invocation with pre-processing (inspect the prompt) and post-processing (inspect the response). If a violation is detected, the guardrail either blocks the request or redacts the sensitive content."

---

## STEP 16 -- Test PII Redaction (8 minutes)

**Pacing: live code in notebook.**

> "Let us test the guardrail with a prompt that contains PII. We want to verify that the guardrail detects and redacts personal information."

Run the cell that invokes the model with a prompt containing a customer name, email, and credit card number. Show:
- The original prompt (with PII)
- The guardrail's detection of PII entities
- The redacted output (PII replaced with placeholders)

> "The guardrail intercepted the PII before it reached the model and in the response. The analyst gets the investigation summary without exposed personal data. This is critical for compliance with GDPR, CCPA, PCI DSS, and other privacy regulations."

---

## STEP 17 -- Test Content Filtering (5 minutes)

**Pacing: live code in notebook.**

> "Now let us test the content filter with a prompt that requests something outside the intended use case."

Run the cell that sends a prompt on a denied topic. Show the guardrail blocking the request and returning an intervention message instead of model output.

> "The guardrail prevented the model from processing the off-topic request entirely. This is your first line of defense against prompt injection and misuse."

---

## STEP 18 -- Cost Management (8 minutes)

**Pacing: conceptual with notebook comparison table.**

> "Every Bedrock invocation has a cost. Understanding the pricing model is essential for budgeting and architecture decisions."

| Pricing Model | How It Works | Best For |
|---------------|-------------|----------|
| **On-demand** | Pay per input/output token, no commitment | Variable workloads, prototyping, low-to-moderate volume |
| **Provisioned throughput** | Reserve dedicated capacity (model units), pay hourly | Predictable high-volume workloads, latency-sensitive applications |

On-demand pricing example (illustrative -- check current AWS pricing):

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|----------------------|
| Claude 3 Haiku | $0.00025 | $0.00125 |
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3 Opus | $0.015 | $0.075 |
| Titan Text Express | $0.0002 | $0.0006 |

> "At 10,000 fraud summaries per day, each averaging 500 input tokens and 300 output tokens, the daily cost with Haiku is roughly: 10,000 * (0.5 * $0.00025 + 0.3 * $0.00125) = $5.00. With Opus, that same workload costs approximately $300. Model selection is a cost decision, not just a quality decision."

**Discussion Prompt:** "When does provisioned throughput become cheaper than on-demand?" (When your sustained usage exceeds the break-even point. Calculate: if provisioned costs $X/hour and on-demand would cost more than $X/hour at your volume, provision. Also relevant when you need guaranteed latency -- on-demand can have variable latency under load.)

---

## STEP 19 -- CloudWatch Monitoring (5 minutes)

**Pacing: conceptual with notebook markdown.**

> "Bedrock publishes metrics to CloudWatch automatically. These metrics let you track usage, detect anomalies, and set up alarms."

Key metrics:

| Metric | What It Measures | Alert Threshold Example |
|--------|-----------------|------------------------|
| `Invocations` | Total number of model invocations | Spike above 2x baseline suggests unexpected usage |
| `InvocationLatency` | Time from request to complete response | p99 > 10s may indicate throttling |
| `InvocationClientErrors` | 4xx errors (bad requests, access denied) | Any sustained errors need investigation |
| `InvocationServerErrors` | 5xx errors (service issues) | Any errors warrant an AWS support case |
| `InputTokenCount` | Total input tokens consumed | Budget monitoring |
| `OutputTokenCount` | Total output tokens generated | Budget monitoring |

> "Set up a CloudWatch alarm on `InputTokenCount + OutputTokenCount` with a daily threshold based on your budget. This prevents runaway costs from bugs, prompt injection, or unexpected traffic spikes."

---

## STEP 20 -- API Quotas and Throttling (5 minutes)

**Pacing: conceptual with notebook markdown.**

> "Bedrock has per-account, per-model, per-region rate limits. If you exceed them, you get throttled -- HTTP 429 (ThrottlingException). The solution is exponential backoff with jitter."

Key quota dimensions:

| Quota | Description | Typical Default |
|-------|-------------|----------------|
| **Requests per minute (RPM)** | How many API calls per minute | Varies by model (e.g., 100 RPM for Opus) |
| **Tokens per minute (TPM)** | Total tokens processed per minute | Varies by model |
| **Max input tokens** | Maximum tokens in a single request | Model-dependent (e.g., 200K for Claude 3) |

Retry strategy:

```
attempt = 0
while attempt < max_retries:
    try:
        response = invoke_model(...)
        break
    except ThrottlingException:
        wait = min(base_delay * 2^attempt + random_jitter, max_delay)
        sleep(wait)
        attempt += 1
```

> "Exponential backoff prevents thundering herd. If 100 requests all get throttled at the same time and all retry after exactly 1 second, they all get throttled again. Adding random jitter spreads the retries over time."

---

## STEP 21 -- Cleanup (2 minutes)

**Pacing: live code. Delete the guardrail.**

> "Delete the guardrail we created. In production, guardrails persist as long-lived governance resources. For training, we clean up."

Run the cleanup cell.

---

## Wrap-up (10 minutes)

### Course Summary (5 minutes)

> "This is the final teaching day of the course. Let us take a step back and look at what you have accomplished across three weeks."

> **Week 1:** You built the foundations. ML fundamentals: supervised learning, evaluation metrics (precision, recall, F1, ROC-AUC), neural networks, CNNs, and your first SageMaker training job. You learned to think about models as experiments with measurable outcomes.

> **Week 2:** You went deep on SageMaker and advanced ML. Feature Store for feature engineering, Experiments for tracking, transformers and attention from scratch, NLP pipelines, hyperparameter optimization, model deployment with five inference patterns, and model monitoring with drift detection. You built the complete MLOps lifecycle: train, deploy, monitor, detect drift, retrain.

> **Week 3 Tuesday (today):** You bridged from custom models to foundation models. Bedrock gives you managed access to production-ready LLMs. You generated fraud investigation summaries, compared models, and governed access with IAM, guardrails, cost management, and monitoring. You now have both sides of the ML deployment spectrum: build your own model (SageMaker) and use someone else's model (Bedrock).

### Review Preparation (3 minutes)

> "The remaining days are open review and project time. Here is how to prepare."

Key areas to review:
1. **ML Fundamentals:** Evaluation metrics, bias-variance trade-off, train/val/test splits
2. **Neural Networks and Deep Learning:** Backpropagation, CNNs, transformers, attention
3. **SageMaker Core:** Training jobs, model registry, endpoints, inference patterns
4. **SageMaker Advanced:** Feature Store, Experiments, HPO, monitoring, drift detection
5. **Bedrock:** Foundation models, model selection, guardrails, IAM, cost management
6. **MLOps:** The full lifecycle -- train, evaluate, register, deploy, monitor, retrain

> "Focus on understanding the 'why' behind each service and technique, not memorizing API signatures. If you understand when to use Bedrock vs SageMaker, when to use serverless vs real-time inference, and how monitoring connects to retraining, you are well prepared."

### Open Q&A (2 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Model access not enabled in Bedrock console | Model access requests must be submitted and approved before invocation works. Some models approve instantly, others take hours. Submit requests well before the lecture. |
| `AccessDeniedException` on `invoke_model` | Verify the IAM role has `bedrock:InvokeModel` permission and that model access is approved in the console. |
| `ValidationException` on model invocation | Check the request body format. Each model provider has a different schema. Claude requires `anthropic_version` and `messages` array. Titan uses a different format. |
| `ThrottlingException` during model comparison | Rate limits vary by model. Wait between invocations or use exponential backoff. Haiku has higher limits than Opus. |
| Guardrail creation fails | Verify the IAM role has `bedrock:CreateGuardrail` permission. Check that the guardrail name is unique and follows naming conventions. |
| PII redaction not working | Verify the guardrail has PII filters configured with the `ANONYMIZE` action. Check the guardrail version is in `READY` status. |
| Students confused about Bedrock vs SageMaker | Emphasize: Bedrock = use someone else's pre-trained model via API. SageMaker = train/host your own model. They are complementary. |
| Streaming response shows garbled output | Ensure the response chunks are decoded properly. Each chunk is a separate JSON event that must be parsed individually. |
| Cost concerns from students | On-demand Bedrock is pay-per-use. A few hundred invocations in class costs cents. The risk is leaving a provisioned throughput reservation active -- we do not create one today. |
| `ModelNotReadyException` | The model may be warming up. Retry after a few seconds. This is rare but can happen with less-popular models. |
