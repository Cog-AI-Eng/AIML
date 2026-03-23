# Interview Questions: Week 5 - Generative AI

## Beginner (Foundational)

### Q1: What is prompt engineering and why is it important?
**Keywords:** Input Design, AI Effectiveness, Clear Communication, Output Quality
<details>
<summary>Click to Reveal Answer</summary>

**Prompt engineering** is the practice of designing and refining inputs (prompts) to AI systems to achieve desired outputs.

**Why it matters:**
- Poorly crafted prompts lead to irrelevant, incorrect, or incomplete results
- Well-engineered prompts unlock the full potential of AI tools
- The same AI can produce dramatically different outputs based on prompt quality

**Core principles:**
1. Be specific
2. Provide context
3. Specify the format
4. Define the role
5. Set constraints
</details>

---

### Q2: What are tokens and context windows in LLMs?
**Keywords:** Basic Units, Text Processing, Maximum Length, Token Limits
<details>
<summary>Click to Reveal Answer</summary>

**Tokens** are the basic units LLMs use to process text. A token can be a word, part of a word, or a character.

- Roughly 1 token = 4 characters in English, or about 0.75 words
- Complex words may use multiple tokens ("programming" = ["program", "ming"])

**Context window** is the maximum number of tokens an LLM can process at once, including both input and output.

| Model | Context Window |
|-------|----------------|
| GPT-3.5 | 4K - 16K tokens |
| GPT-4 | 8K - 128K tokens |
| Claude 3 | Up to 200K tokens |

Your prompt + response must fit within the context window.
</details>

---

### Q3: What is the difference between zero-shot and few-shot prompting?
**Keywords:** No Examples, Providing Examples, Learning from Patterns
<details>
<summary>Click to Reveal Answer</summary>

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Zero-shot** | Ask directly without providing examples | Simple, well-defined tasks |
| **Few-shot** | Provide examples before asking | Complex formats, consistent style |

**Zero-shot example:**
```
Classify this sentiment: "I love this product!" -> Positive
```

**Few-shot example:**
```
Examples:
"Great service!" -> Positive
"Terrible experience" -> Negative
"It was okay" -> Neutral

Now classify: "I love this product!" ->
```

Few-shot helps when the AI needs to understand a specific format or pattern.
</details>

---

### Q4: What are the main cloud service models for AI tools?
**Keywords:** GPT, Claude, Copilot, Codeium, Open Source
<details>
<summary>Click to Reveal Answer</summary>

Major LLM providers and tools:

| Model/Tool | Provider | Key Characteristics |
|------------|----------|---------------------|
| **GPT-4** | OpenAI | Flagship model, broad capabilities |
| **Claude** | Anthropic | Long context, safety-focused |
| **Copilot** | GitHub/Microsoft | IDE integration, code-focused |
| **Codeium** | Codeium | Free tier, code completion |
| **Llama** | Meta | Open source, self-hostable |
| **Gemini** | Google | Multimodal, integrated with GCP |

Each has different strengths, context limits, and pricing models.
</details>

---

### Q5: What is an AI hallucination?
**Keywords:** Confident Incorrect Output, Fabricated Information, Plausible But Wrong
<details>
<summary>Click to Reveal Answer</summary>

**AI hallucinations** occur when models generate content that is plausible-sounding but factually incorrect, logically inconsistent, or entirely fabricated.

**Types of hallucinations:**
- **Factual**: Incorrect statements presented as facts
- **Code**: Non-existent modules, functions, or parameters
- **API**: Invalid API endpoints or method signatures
- **Library**: Recommendations for packages that do not exist

**Why they happen:**
- LLMs predict patterns, not facts
- When uncertain, the model still produces output
- Plausible patterns can be factually wrong
</details>

---

### Q6: What are the core principles of effective prompts?
**Keywords:** Specificity, Context, Format, Role, Constraints
<details>
<summary>Click to Reveal Answer</summary>

**Five core principles:**

1. **Be Specific**: Specify exactly what you want
   - Bad: "Write some code"
   - Good: "Write a Python function that returns the sum of even numbers"

2. **Provide Context**: Help AI understand your situation
   - Include language version, framework, use case

3. **Specify Format**: Tell AI how to structure output
   - "Return as a bulleted list" or "Include docstrings"

4. **Define Role**: Assign a persona
   - "As a senior Python developer..."

5. **Set Constraints**: Narrow the solution space
   - "Do not use external libraries" or "Must be O(n) complexity"
</details>

---

### Q7: What is chain-of-thought prompting?
**Keywords:** Step by Step, Reasoning, Show Work, Complex Problems
<details>
<summary>Click to Reveal Answer</summary>

**Chain-of-thought prompting** asks the AI to show its reasoning step by step before providing a final answer.

**Without chain-of-thought:**
```
What is 23 * 17?
```

**With chain-of-thought:**
```
What is 23 * 17? Think through this step by step.
```

**Benefits:**
- Improves accuracy on complex reasoning tasks
- Makes errors easier to identify
- Helps with math, logic, and multi-step problems
- Provides transparency into AI reasoning

Often triggered by phrases like "Let's think step by step" or "Walk me through your reasoning."
</details>

---

### Q8: What is the temperature parameter in LLMs?
**Keywords:** Randomness, Deterministic, Creative, Consistent
<details>
<summary>Click to Reveal Answer</summary>

**Temperature** controls how random or deterministic LLM outputs are:

| Temperature | Behavior |
|-------------|----------|
| **0** | Most probable tokens chosen; same input = same output |
| **0.1 - 0.3** | Low randomness; good for code and factual content |
| **0.7 - 0.9** | Higher creativity; more varied outputs |
| **1** | Full sampling; very creative but may be inconsistent |

**For code generation:** Use lower temperatures (0.1-0.3) for consistent, predictable results.

**For creative writing:** Use higher temperatures for variety.
</details>

---

### Q9: What are the benefits of AI pair programming?
**Keywords:** Code Completion, Explanation, Debugging, Productivity
<details>
<summary>Click to Reveal Answer</summary>

AI pair programming provides:

| Benefit | Description |
|---------|-------------|
| **Code completion** | AI suggests code as you type |
| **Explanation** | Understand unfamiliar code |
| **Debugging** | Identify and fix bugs |
| **Documentation** | Generate comments and docstrings |
| **Refactoring** | Suggest improvements |
| **Learning** | Discover new patterns and libraries |

**Key insight:** AI is an assistant, not a replacement. You remain responsible for code quality and correctness.
</details>

---

### Q10: What are the main security risks when using AI tools?
**Keywords:** Data Leakage, Prompt Injection, Sensitive Information, Validation
<details>
<summary>Click to Reveal Answer</summary>

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Data leakage** | Sending sensitive data to AI APIs | Never include credentials, PII, or secrets |
| **Prompt injection** | Malicious input manipulating AI behavior | Sanitize user inputs before AI processing |
| **Hallucinated code** | AI suggests insecure patterns | Always review security-critical code |
| **Dependency risks** | AI suggests vulnerable packages | Verify package reputation and security |
| **Overreliance** | Trusting AI without verification | Test and validate all AI outputs |

**Golden rule:** Treat AI output as untrusted input that requires validation.
</details>

---

## Intermediate (Application)

### Q11: How would you structure a prompt for effective code generation?
**Hint:** Think about the anatomy of a well-structured prompt.
<details>
<summary>Click to Reveal Answer</summary>

**Prompt structure:**

```
ROLE:
You are an experienced data engineer specializing in Python and SQL.

CONTEXT:
I am building an ETL pipeline that processes daily sales data from 
CSV files and loads them into a PostgreSQL database.

TASK:
Write a Python function that validates incoming CSV data before loading.

REQUIREMENTS:
- Check for required columns: date, product_id, quantity, price
- Validate date format (YYYY-MM-DD)
- Ensure quantity and price are positive numbers
- Return a tuple: (is_valid: bool, errors: list)

FORMAT:
Include:
- Type hints
- Comprehensive docstring
- Inline comments explaining validation logic
```

This structure provides context, specifies exactly what you need, and defines the expected output format.
</details>

---

### Q12: Describe strategies for detecting AI hallucinations in generated code.
**Hint:** Think about verification before trusting.
<details>
<summary>Click to Reveal Answer</summary>

**Detection strategies:**

1. **Try to run it**
   ```python
   try:
       from nonexistent.module import thing
   except ImportError:
       print("This module does not exist!")
   ```

2. **Check official documentation**
   - Verify function signatures exist
   - Confirm parameters are valid

3. **Search for existence**
   - Check PyPI for suggested packages
   - Search GitHub for library names

4. **Test claims**
   - Benchmark performance claims
   - Validate algorithm correctness

**Verification checklist:**
- Imports resolve without errors
- Functions/methods exist in the library
- Parameters are valid
- Code executes without runtime errors
</details>

---

### Q13: How do you effectively use AI for code review?
**Hint:** Think about prompting for specific review aspects.
<details>
<summary>Click to Reveal Answer</summary>

**Effective code review prompt:**

```
As a senior Python developer conducting a code review, analyze 
this code for:

1. Bugs and potential issues
2. Performance concerns
3. Security vulnerabilities
4. Best practices violations
5. Readability improvements

Provide specific line numbers and suggested fixes.

[code snippet]
```

**Best practices:**
- Ask for specific aspects (security, performance, style)
- Request actionable suggestions with line numbers
- Have AI explain its reasoning
- Verify AI suggestions independently
- Use as supplement, not replacement, for human review
</details>

---

### Q14: What are the key differences between major LLMs for development work?
**Hint:** Think about context windows, strengths, and use cases.
<details>
<summary>Click to Reveal Answer</summary>

| Model | Context | Strengths | Best For |
|-------|---------|-----------|----------|
| **GPT-4** | Up to 128K | Broad knowledge, reasoning | General coding, complex problems |
| **Claude 3** | Up to 200K | Long context, safety | Large codebases, documentation |
| **Copilot** | IDE-integrated | Real-time suggestions | Code completion, in-editor help |
| **Llama 2** | 4K | Open source, self-hostable | Privacy-sensitive, custom deployment |
| **Gemini** | Variable | Multimodal, GCP integration | Google Cloud workflows |

**Selection criteria:**
- Context window needs (large files?)
- Privacy requirements (can data leave your network?)
- Integration (IDE, API, chat?)
- Cost considerations
</details>

---

### Q15: How should you handle AI-generated code in a professional setting?
**Hint:** Think about verification, attribution, and responsibility.
<details>
<summary>Click to Reveal Answer</summary>

**Professional practices:**

1. **Always verify**
   - Test all AI-generated code
   - Check for edge cases
   - Validate security implications

2. **Take responsibility**
   - You own the code, not the AI
   - Review as if you wrote it yourself
   - Be prepared to explain every line

3. **Document appropriately**
   - Internal: Note AI assistance where relevant
   - Follow team/company policies on AI usage

4. **Handle responsibly**
   - Never commit untested AI code
   - Do not include sensitive data in prompts
   - Respect licensing of suggested code

5. **Iterate and improve**
   - Refine AI suggestions to match your style
   - Use AI as starting point, not final product
</details>

---

### Q16: What prompting techniques can reduce hallucinations?
**Hint:** Think about constraining AI behavior and requesting verification.
<details>
<summary>Click to Reveal Answer</summary>

**Hallucination reduction techniques:**

1. **Request uncertainty disclosure**
   ```
   When answering, indicate your confidence level.
   If uncertain, say "I believe" or "I'm not sure"
   ```

2. **Request sources**
   ```
   Provide your answer and cite specific documentation.
   Note: I will verify these, so only cite real sources.
   ```

3. **Constrain scope**
   ```
   Only use features from pandas 1.5.x
   Only reference the Python standard library
   Only suggest packages with over 1000 GitHub stars
   ```

4. **Ask for verification**
   ```
   After providing the code, list any libraries that 
   need to be installed and confirm they exist on PyPI.
   ```

5. **Use chain-of-thought**
   - Step-by-step reasoning exposes errors
</details>

---

### Q17: How do you integrate AI tools effectively into your IDE workflow?
**Hint:** Think about configuration, acceptance, and hybrid workflows.
<details>
<summary>Click to Reveal Answer</summary>

**Integration best practices:**

1. **Configure appropriately**
   - Adjust suggestion aggressiveness
   - Set keyboard shortcuts for accept/reject
   - Configure languages/projects

2. **Develop acceptance habits**
   - Read suggestions before accepting
   - Reject partial/incorrect suggestions
   - Tab to accept, Esc to dismiss

3. **Use hybrid approach**
   - Let AI handle boilerplate
   - Write critical logic yourself
   - Use AI for exploration, not final code

4. **Leverage context**
   - Keep relevant files open
   - Write descriptive function names
   - Add comments to guide suggestions

5. **Balance reliance**
   - Do not over-rely on suggestions
   - Maintain your own coding skills
   - Use AI to augment, not replace thinking
</details>

---

## Advanced (Deep Dive)

### Q18: Explain the LLM training process and how it affects the model's capabilities and limitations.
<details>
<summary>Click to Reveal Answer</summary>

**Training phases:**

**Phase 1: Pre-training**
- Train on massive text datasets (books, websites, code)
- Task: Predict the next token, billions of times
- Learns: Language structure, general knowledge, patterns

**Phase 2: Fine-tuning**
- Train on specific tasks (instruction following, chat)
- Uses curated prompt-response pairs
- Teaches: How to follow instructions, format outputs

**Phase 3: Alignment (RLHF)**
- Reinforcement Learning from Human Feedback
- Humans rank model outputs
- Model learns to prefer high-quality responses
- Improves: Helpfulness, harmlessness, honesty

**How this affects capabilities:**
- Pre-training creates broad knowledge (with cutoff date)
- Fine-tuning enables instruction following
- RLHF shapes behavior and safety

**Limitations that emerge:**
- Knowledge cutoff (training data has end date)
- Hallucinations (pattern matching, not fact retrieval)
- Inconsistency (probabilistic, not deterministic)
</details>

---

### Q19: Design a verification workflow for using AI to generate production code.
<details>
<summary>Click to Reveal Answer</summary>

**Production-Ready AI Code Workflow:**

```
1. Generate with constraints
   - Specify language version, dependencies
   - Request error handling
   - Ask for type hints and docstrings
            |
            v
2. Automated checks
   - Lint check (flake8, pylint)
   - Type check (mypy)
   - Import verification
            |
            v
3. Functional validation
   - Write/run unit tests
   - Check edge cases
   - Verify expected behavior
            |
            v
4. Security review
   - Check for injection vulnerabilities
   - Verify input validation
   - Review dependency security
            |
            v
5. Human review
   - Code review by team member
   - Verify logic correctness
   - Ensure maintainability
            |
            v
6. Integration testing
   - Deploy to staging
   - Run integration tests
   - Monitor for errors
```

**Key principle:** AI-generated code should pass the same quality gates as human-written code.
</details>

---

### Q20: Compare different prompting strategies for complex coding tasks and when each is most effective.
<details>
<summary>Click to Reveal Answer</summary>

**Prompting strategies comparison:**

| Strategy | Best For | Example |
|----------|----------|---------|
| **Zero-shot** | Simple, well-defined tasks | "Write a function to sort a list" |
| **Few-shot** | Consistent formatting, complex patterns | Provide 2-3 examples first |
| **Chain-of-thought** | Reasoning, algorithms | "Think step by step..." |
| **Role-based** | Domain expertise | "As a security expert..." |
| **Iterative** | Complex requirements | Build up through conversation |

**Strategy selection guide:**

```
Is the task simple and well-defined?
    Yes -> Zero-shot
    No  -> Does it require specific format?
              Yes -> Few-shot with examples
              No  -> Does it involve reasoning?
                        Yes -> Chain-of-thought
                        No  -> Does it need expertise?
                                  Yes -> Role-based
                                  No  -> Iterative refinement
```

**Advanced combinations:**
- Few-shot + Chain-of-thought for complex formatting with reasoning
- Role-based + Constraints for domain-specific code
- Iterative with verification for critical code
</details>

---

### Q21: How would you design an AI-assisted development workflow that balances productivity with code quality and security?
<details>
<summary>Click to Reveal Answer</summary>

**Balanced AI Development Workflow:**

**1. Planning Phase (Human-Led)**
- Define requirements and architecture
- Identify security-critical components
- AI role: Research, explore options

**2. Development Phase (AI-Assisted)**
```
For boilerplate/utilities:
   - Use AI code completion freely
   - Light review sufficient

For business logic:
   - Use AI for first draft
   - Thorough human review
   - Comprehensive testing

For security-critical code:
   - Minimal AI assistance
   - Expert human implementation
   - Security audit required
```

**3. Review Phase (Human-Led)**
- All AI code treated as untrusted
- Standard code review process
- Security scanning

**4. Guardrails**
- Never send production data to AI
- No credentials in prompts
- Verify all dependencies

**5. Continuous Improvement**
- Track AI suggestion acceptance rate
- Document false positives/hallucinations
- Refine prompting strategies

**Key balance:** Use AI to accelerate routine work while maintaining human oversight for quality and security.
</details>
