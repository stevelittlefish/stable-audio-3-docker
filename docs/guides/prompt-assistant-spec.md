# Prompt Assistant Reimplementation Specification

## 1. Purpose

This document specifies how to reimplement the Stable Audio 3 prompt assistant in another application using a hosted text-generation model.

The assistant accepts a short or incomplete audio description and replaces it with a concise prompt suitable for an instrumental audio-generation model. It also returns a recommended generation duration. It does **not** generate audio.

The implementation has four responsibilities:

1. Classify a non-empty request as music, an instrument/stem, a sound effect, or a one-shot.
2. Rewrite it using instructions specialized for that category.
3. Reject and retry malformed rewrites.
4. Extract the generated duration into structured data and remove it from the displayed prompt.

This specification distinguishes:

- **Parity requirements**, which reproduce the current behavior exactly.
- **Hosted-app recommendations**, which make the same behavior practical and safe behind a network API.

## 2. User-visible behavior

The UI contains:

- A prompt text field.
- A `Prompt Assistant` button.
- A duration control measured in whole seconds.
- A separate `Generate` button.

Pressing `Prompt Assistant` rewrites the prompt in place and may update the duration. It must never start audio generation. The user may edit the rewritten prompt before pressing `Generate`.

Only one prompt-assistant request should run at a time per UI/session. Disable the button or otherwise enforce a concurrency limit of one until the request finishes.

### 2.1 Empty input

If the prompt is empty after trimming whitespace:

1. Do not call the hosted model.
2. Randomly select one entry from a built-in pool of full-music example prompts.
3. Prefix it with `TrackType: Music, VocalType: Instrumental, `.
4. Extract its final duration as described in section 11.
5. Put the remaining text in the prompt field and the duration in the duration control.

This is an idea generator, not unconditional language-model generation.

### 2.2 Non-empty input

For non-empty input, run the classification, rewrite, validation, post-processing, prefixing, and duration-extraction pipeline described below. Replace the input only after the pipeline succeeds or exhausts its retries.

## 3. Categories and output prefixes

Use these internal categories and Stable Audio metadata prefixes:

| Classifier label | Internal category | Displayed prompt prefix |
|---|---|---|
| `music_genre` | `music` | `TrackType: Music, VocalType: Instrumental, ` |
| `instrument` | `instrument` | `TrackType: Instrument, ` |
| `sound` | `sfx` | `TrackType: SFX, ` |
| `one_shot` | `one_shot` | No prefix |

The lack of a one-shot prefix is required for strict parity.

The output is always intended to be instrumental. The validator rejects common words associated with voices, singing, rap, or lyrics.

## 4. Recommended hosted architecture

Keep model credentials and orchestration on the application server:

```text
Browser or native client
        |
        | POST /api/prompt-assistant
        v
Application backend
  - validates request
  - handles empty input
  - calls classifier
  - selects rewrite instructions
  - retries and validates
  - adds TrackType prefix
  - extracts duration
        |
        | HTTPS using server-side credentials
        v
Hosted chat/completions model
```

Do not call a paid hosted model directly from browser code because that exposes the API credential and makes request validation, rate limiting, and consistent retries difficult.

The classifier and rewriter may use the same hosted model. For closer parity, use the same model for both calls but different temperatures and token limits.

## 5. Public API contract

### 5.1 Request

`POST /api/prompt-assistant`

```json
{
  "prompt": "dark ambient forest",
  "max_audio_seconds": 380
}
```

Fields:

| Field | Type | Required | Rules |
|---|---|---:|---|
| `prompt` | string | yes | May be empty; impose an application limit such as 2,000 characters. |
| `max_audio_seconds` | integer | yes | Maximum supported by the currently selected audio model; must be at least 1. |

Do not trust a client-supplied maximum if the server already knows the selected audio model. Prefer deriving it from server-side model configuration.

### 5.2 Success response

```json
{
  "prompt": "TrackType: Music, VocalType: Instrumental, Dark ambient soundscape with low drones, distant bells, rustling forest textures, and sparse percussion creating eerie immersive tension. BPM: 50.",
  "seconds": 300,
  "category": "music",
  "source": "model",
  "attempts": 1
}
```

Fields:

| Field | Type | Meaning |
|---|---|---|
| `prompt` | string | Rewritten prompt with its final `Length` clause removed. |
| `seconds` | integer or `null` | Extracted duration, capped to `max_audio_seconds`; `null` means leave the UI duration unchanged. |
| `category` | enum or `null` | `music`, `instrument`, `sfx`, or `one_shot`. |
| `source` | enum | `model` or `random_example`. |
| `attempts` | integer | Number of rewrite calls. Use `0` for an empty-input random example. |

The current implementation internally retains the raw model output, but the browser does not need it. If returned for diagnostics, expose it only in authenticated development environments.

### 5.3 Errors

Use ordinary HTTP error semantics:

- `400` for an invalid request.
- `429` when the application or model provider rate-limits the user.
- `502` when the hosted provider returns an unusable response.
- `503` when the hosted provider is unavailable.
- `504` when the provider times out.

On failure, the client must retain the user's original prompt and duration. Do not partially update the form.

## 6. Model interface

The orchestration layer needs this provider-neutral operation:

```typescript
interface ChatMessage {
  role: "system" | "user";
  content: string;
}

interface GenerateOptions {
  maxNewTokens: number;
  temperature: number;
  sample: boolean;
}

interface HostedModel {
  generate(messages: ChatMessage[], options: GenerateOptions): Promise<string>;
}
```

For every model call, send exactly two logical messages:

```json
[
  {"role": "system", "content": "<selected instructions>"},
  {"role": "user", "content": "Input: <user prompt>\nOutput: "}
]
```

Do not interpolate the user input into the system message. If the provider supports a system or developer role, use it for the category instructions.

The reference local model disables hidden/reasoning output. With a hosted model, disable reasoning if the provider offers that option. Otherwise, ensure only the final response text is returned or strip a terminal `</think>` section as described in section 10.

## 7. Classification

### 7.1 Fast-path metadata detection

Convert the user's prompt to lowercase and apply these checks in order:

1. If it contains both `tracktype: music` and `vocaltype: instrumental`, use classifier label `music_genre` without a model call.
2. Else, if it contains `tracktype: instrument`, use `instrument` without a model call.
3. Else, if it contains `tracktype: sfx`, use `sound` without a model call.
4. Otherwise, call the hosted classifier.

These are substring checks, not a structured metadata parser.

### 7.2 Classifier generation settings

Use:

```json
{
  "maxNewTokens": 16,
  "temperature": 0.1,
  "sample": true
}
```

### 7.3 Classifier system instructions

Use the following functional system prompt. A production implementation may add examples, but must preserve the labels and output constraint.

```text
You are a classification assistant.

Classify the user's audio-generation input into exactly one category:

music_genre
instrument
sound
one_shot

Definitions:
- music_genre: A style or category of music, including full tracks and ambient music.
- instrument: A musical instrument, solo instrument, loop, or stem.
- sound: A non-instrumental sound effect, general sound, action, or ambience.
- one_shot: A single isolated audio sample, normally short and intended for music production.

Output only one label: music_genre, instrument, sound, or one_shot.
Do not explain the answer. If unsure, choose the closest category.
```

Representative classification cases:

| Input | Label |
|---|---|
| `jazz`, `techno`, `cinematic score` | `music_genre` |
| `solo cello`, `guitar loop`, `drum stem` | `instrument` |
| `rain in a forest`, `door creak`, `car passing` | `sound` |
| `kick`, `snare hit`, `synth stab`, `piano key hit` | `one_shot` |

### 7.4 Parsing classifier output

Trim and lowercase the response. Then select the rewrite category using substring checks in this exact order:

1. Contains `music_genre` -> `music`.
2. Else contains `one_shot` -> `one_shot`.
3. Else contains `instrument` -> `instrument`.
4. Else -> `sfx`.

Therefore invalid, empty, or unexplained classifier output falls back to SFX. This is a parity behavior, although a stricter implementation could instead retry classification.

## 8. Rewriting

### 8.1 Generation settings

Use these settings for every rewrite attempt:

```json
{
  "maxNewTokens": 128,
  "temperature": 1.11,
  "sample": true
}
```

The relatively high temperature intentionally produces variety. If the hosted provider uses a materially different temperature scale, tune it until outputs have similar diversity while reliably following the format.

### 8.2 Common output contract

All category prompts must tell the model to:

- Return only the rewritten audio prompt.
- Include no explanation, heading, Markdown, bullets, or quotation marks.
- Keep the result at no more than 45 whitespace-delimited words.
- Describe instrumental audio only, with no vocals or lyrics.
- End with `. Length: X seconds`, where `X` is an integer.

The exact validator accepts an optional final period after `seconds`, so both of these pass:

```text
... BPM: 120. Length: 150 seconds
... BPM: 120. Length: 150 seconds.
```

### 8.3 Music rewrite instructions

```text
You are an expert musician, musicologist, and prompt engineer. Transform the user's input into one detailed, vivid prompt for a full instrumental track.

Include:
1. Genre or style, with useful adjectives.
2. Main instruments.
3. Supporting layers, effects, harmonics, or field recordings where appropriate.
4. Rhythm and percussion.
5. Mood and energy integrated naturally into the description.
6. An appropriate integer BPM.
7. An integer duration. Normally use 120-180 seconds for energetic or dance music, 180-210 seconds for pop or rock, and 240-300 seconds for cinematic or ambient music.

Write one natural, fluid sentence. Avoid semicolons. Output only the final prompt, no formatting or explanation. Do not include vocals or lyrics. Use at most 45 words. End exactly with: . Length: X seconds

Form:
Genre/style with main instruments, supporting layers, and rhythm/percussion creating mood/energy. BPM: X. Length: Y seconds
```

Examples:

```text
Jazz ballad with smooth saxophone lead, piano chords, upright bass, brushed drums, and soft strings that swing gently for a warm and cozy evening. BPM: 85. Length: 180 seconds
EDM festival track with pulsing synth leads, plucked arpeggios, layered pads, side-chained bass, punchy kick and snare, and hi-hat rolls creating bright uplifting dance energy. BPM: 128. Length: 150 seconds
Dreamy ambient soundscape with layered pads, soft bell textures, gentle drones, and wind and water field recordings for ethereal spacious meditation. BPM: 40. Length: 300 seconds
```

### 8.4 Instrument/stem rewrite instructions

```text
You are a music metadata expert. Transform the user's instrument input into one descriptive prompt for a generative audio model.

Include:
1. The instrument or stem.
2. Playing style or technique.
3. Material, timbre, or texture.
4. Genre, context, mood, or emotional character.
5. Spatial or production qualities.
6. An appropriate integer BPM.
7. An integer duration: 6-20 seconds for loops and 20-180 seconds for stems or longer solo performances.

Output only the final prompt, no formatting or explanation. Do not include vocals or lyrics. Use at most 45 words. End exactly with: . Length: X seconds
```

Examples:

```text
Synth arpeggio loop with bright detuned oscillators and a tight electronic pulse. BPM: 120. Length: 8 seconds
Sombre acoustic guitar motif with cavernous reverb, delicate fingerpicking, and expressive melancholic tone. BPM: 70. Length: 45 seconds
String ensemble cinematic motif with rich wooden resonance and sweeping emotional movement. BPM: 80. Length: 120 seconds
```

### 8.5 SFX rewrite instructions

```text
You are a professional sound-design expert. Convert the user's input into a precise, vivid sound-effect description suitable for a generative audio model.

Describe the sound source, physical character and material, spatial qualities, temporal evolution, and motion when applicable. Be specific, concise, coherent, and realistic. Focus strictly on sound effects or ambience.

Use 1-3 seconds for very short impacts, clicks, and gunshots; 3-6 seconds for medium actions and transitions; and 6-15 seconds for ambience or environments.

Write one or two dense sentences. Output only the final prompt, no formatting or explanation. Do not include voices, vocals, or lyrics. Use at most 45 words. End exactly with: . Length: X seconds
```

Examples:

```text
Glass bottle shattering on concrete with a sharp initial impact, brittle fragments scattering outward, close dry perspective, and fast decay. Length: 2 seconds
Car speeding past at high velocity with a rising engine, realistic Doppler shift, and a broad stereo whoosh moving left to right. Length: 3 seconds
Quiet forest at dawn with birds chirping, soft wind moving through leaves, and a distant stream in a spacious outdoor ambience. Length: 15 seconds
```

The original example library contains some ambience durations longer than these written rules. For deterministic behavior in a new hosted app, the explicit duration rules should take priority.

### 8.6 One-shot rewrite instructions

```text
You are a music metadata expert. Given an instrument or sound, generate one descriptive prompt for a short isolated one-shot sample intended for music production.

Include:
1. Instrument or sound source.
2. Playing technique or hit type, such as pluck, slam, tap, or stab.
3. Material, timbre, or texture.
4. Spatial or production qualities such as dry, wet, room sound, or close miking.
5. An integer duration from 1 to 11 seconds.

Output only the final prompt, no formatting or explanation. Do not include voices, vocals, or lyrics. Use at most 45 words. End exactly with: . Length: X seconds
```

Examples:

```text
Dry hand clap with a crisp natural transient, close-miked in a dead room with no reflections and an extremely fast decay. Length: 1 second
Synth stab with a bright detuned attack, dense analog texture, and a short wet reverb tail. Length: 3 seconds
Cello expressive note with vibrato, warm resonant wooden body, close ribbon-mic detail, and a natural room decay. Length: 3 seconds
```

## 9. Validation and retry behavior

Validate the raw rewrite response before post-processing. A response is invalid if **any** of the following is true:

1. It contains `[`, `]`, `*`, or `#`.
2. It contains any forbidden vocal term as a whole word, case-insensitively.
3. It does not end in the required duration syntax.
4. It has more than 45 whitespace-delimited words.

Use these parity-compatible regular expressions:

```javascript
const ARTIFACT_RE = /[\[\]*#]/;
const VOCAL_RE = /\b(vocals?|singing|singer|female|male|voice|voices|chorus|rap|rapper|chant(ing)?|lyrics?)\b/i;
const LENGTH_RE = /\. Length: \d+ seconds\.?\s*$/;
```

The word count is equivalent to Python's `text.split()` length: trim is not required, and any run of whitespace separates words.

Try at most five rewrite calls. Every retry uses the same category system instructions, original user prompt, token limit, and temperature. For strict parity, do not send the rejected response or rejection reason back to the model.

If the fifth response remains invalid, continue with that fifth response. This permissive behavior is required for exact parity. A hardened hosted implementation may instead return `502`, but doing so changes user-visible behavior.

## 10. Post-processing

After the rewrite loop:

1. If the raw response contains `</think>`, discard everything through the first occurrence and keep the remainder after it.
2. Trim leading and trailing whitespace.
3. If the remaining text starts with `- `, remove those two characters.
4. Uppercase the first word character at the start of the string and after every period followed by whitespace.
5. Uppercase the first character once more if the string is non-empty.
6. Prepend the category prefix from section 3.

Do not otherwise normalize punctuation, whitespace, spelling, or metadata.

For exact parity, the post-processor does not strip quotation marks, code fences, explanations, or other invalid content after the retry limit is exhausted.

## 11. Duration extraction

Run duration extraction after post-processing and prefix insertion.

Use:

```javascript
const DURATION_RE = / Length: (\d+) seconds\.?\s*$/;
```

If it matches:

1. Parse capture group 1 as a base-10 integer.
2. Set `seconds = min(parsedSeconds, maxAudioSeconds)`.
3. Remove the matched substring from the displayed prompt, beginning with the space before `Length`.

If it does not match:

1. Leave the displayed prompt unchanged.
2. Return `seconds: null`, meaning that the UI must retain its current duration.

Do not enforce a lower bound during extraction for strict parity. Normal request validation should nevertheless require the audio model's maximum to be positive.

Example:

```text
Before: TrackType: SFX, Wooden door creaking in a reverberant hall. Length: 12 seconds
After:  TrackType: SFX, Wooden door creaking in a reverberant hall.
Seconds: min(12, maxAudioSeconds)
```

## 12. End-to-end pseudocode

```typescript
const PREFIX: Record<Category, string> = {
  music: "TrackType: Music, VocalType: Instrumental, ",
  instrument: "TrackType: Instrument, ",
  sfx: "TrackType: SFX, ",
  one_shot: "",
};

async function assist(
  originalPrompt: string,
  maxAudioSeconds: number,
  model: HostedModel,
): Promise<AssistResponse> {
  if (originalPrompt.trim() === "") {
    const example = randomChoice(MUSIC_EXAMPLES);
    return finalize(PREFIX.music + example, "music", "random_example", 0,
                    maxAudioSeconds);
  }

  const lower = originalPrompt.toLowerCase();
  let label: string;

  if (lower.includes("tracktype: music") &&
      lower.includes("vocaltype: instrumental")) {
    label = "music_genre";
  } else if (lower.includes("tracktype: instrument")) {
    label = "instrument";
  } else if (lower.includes("tracktype: sfx")) {
    label = "sound";
  } else {
    label = (await model.generate(
      messages(CLASSIFIER_SYSTEM_PROMPT, originalPrompt),
      {maxNewTokens: 16, temperature: 0.1, sample: true},
    )).trim().toLowerCase();
  }

  const category = categoryFromLabel(label);
  let raw = "";
  let attempts = 0;

  for (let attempt = 1; attempt <= 5; attempt++) {
    attempts = attempt;
    raw = await model.generate(
      messages(REWRITE_SYSTEM_PROMPT[category], originalPrompt),
      {maxNewTokens: 128, temperature: 1.11, sample: true},
    );
    if (!hasArtifacts(raw)) break;
  }

  const rewritten = postprocess(raw);
  return finalize(PREFIX[category] + rewritten, category, "model", attempts,
                  maxAudioSeconds);
}

function categoryFromLabel(label: string): Category {
  if (label.includes("music_genre")) return "music";
  if (label.includes("one_shot")) return "one_shot";
  if (label.includes("instrument")) return "instrument";
  return "sfx";
}

function hasArtifacts(text: string): boolean {
  return ARTIFACT_RE.test(text) ||
         VOCAL_RE.test(text) ||
         !LENGTH_RE.test(text) ||
         text.trim().split(/\s+/).length > 45;
}

function postprocess(raw: string): string {
  let value = raw.includes("</think>")
    ? raw.split("</think>", 2)[1].trim()
    : raw;
  if (value.startsWith("- ")) value = value.slice(2);
  value = value.trim().replace(/(^|\.\s)(\w)/g,
    (_match, prefix, char) => prefix + char.toUpperCase());
  if (value) value = value[0].toUpperCase() + value.slice(1);
  return value;
}

function finalize(
  value: string,
  category: Category,
  source: "model" | "random_example",
  attempts: number,
  maxAudioSeconds: number,
): AssistResponse {
  const match = value.match(/ Length: (\d+) seconds\.?\s*$/);
  if (!match || match.index === undefined) {
    return {prompt: value, seconds: null, category, source, attempts};
  }
  const seconds = Math.min(Number.parseInt(match[1], 10), maxAudioSeconds);
  return {
    prompt: value.slice(0, match.index),
    seconds,
    category,
    source,
    attempts,
  };
}
```

## 13. Random example pool

Store random examples as data rather than parsing them out of a system prompt. Every entry must already end with `. Length: X seconds` so the ordinary duration extraction path works.

Recommended minimum pool:

```json
[
  "Jazz ballad with smooth saxophone lead, piano chords, upright bass, brushed drums, and soft strings that swing gently for a warm cozy evening. BPM: 85. Length: 180 seconds",
  "EDM festival track with pulsing synth leads, plucked arpeggios, layered pads, side-chained bass, punchy kick and snare, and hi-hat rolls creating bright uplifting dance energy. BPM: 128. Length: 150 seconds",
  "Lo-fi hip-hop track with mellow electric piano, vinyl crackle, subtle pads, filtered drums, soft percussion, and plucked bass for a relaxed dreamy mood. BPM: 75. Length: 150 seconds",
  "Dreamy ambient soundscape with layered pads, soft bells, gentle drones, wind, and water field recordings for ethereal spacious meditation. BPM: 40. Length: 300 seconds",
  "Synthwave retro track with arpeggiated leads, analog pads, electric bass, punchy electronic drums, gated snares, and atmospheric effects for nostalgic vibrant energy. BPM: 110. Length: 180 seconds"
]
```

For exact output-distribution parity, copy all lines beginning with `- ` from the current `Music` system-prompt example section into this array. Random selection need not be cryptographically secure.

## 14. Client state machine

Use these client states:

```text
idle -> submitting -> success -> idle
                  \-> error -> idle
```

While `submitting`:

- Disable the assistant button to prevent duplicate requests.
- Keep the original field values visible.
- Optionally change the button text to `Improving prompt...`.

On success, update the prompt and duration atomically. Update the duration only when `seconds` is not `null`.

On error, retain both original values and show a non-destructive error message. The separate audio `Generate` action remains independent.

The local application's first-use button has a `Download Prompt Assistant (~4.2 GB)` state because it downloads Qwen locally. A hosted implementation does not need this state. Replace it with a server health/readiness check if cold starts are significant.

## 15. Security, privacy, and operations

- Keep provider API keys on the backend and in a secret manager or environment variable.
- Treat the prompt as untrusted input. Its content belongs only in the user message.
- Set request body and prompt-length limits.
- Apply per-user/IP rate limits because one button press can produce up to six hosted calls: one classification call and five rewrite attempts.
- Use a bounded provider timeout and cancellation when the client disconnects.
- Do not log prompts by default if they may contain private creative material.
- If logging is enabled, redact credentials and attach a request ID, category, attempt count, duration, latency, model identifier, and validation-failure reason.
- Pin a hosted model/version where the provider permits it. A silent provider model upgrade can change classifications, instruction following, duration choices, and output diversity.
- Record token use and retry rates. A high retry rate usually indicates weak format adherence or unsuitable temperature/model settings.
- Escape the returned prompt when rendering it. Treat it as text, never HTML.

## 16. Testing requirements

### 16.1 Unit tests

Test classification fast paths:

- Correct music metadata skips the classifier.
- `TrackType: Instrument` skips the classifier.
- `TrackType: SFX` skips the classifier.
- Music without `VocalType: Instrumental` does not take the music fast path.
- Matching is case-insensitive.

Test label parsing:

- Each valid label maps correctly.
- A response containing explanation plus a valid label still maps by substring.
- Unknown and empty labels map to SFX.
- If a pathological response contains multiple labels, the documented precedence wins.

Test validation:

- Each forbidden formatting character causes rejection.
- Every vocal term, including singular/plural variants, causes rejection as a whole word.
- `instrumental` does not match the vocal regex.
- Missing, malformed, or non-terminal length causes rejection.
- Both allowed terminal-period forms pass.
- Exactly 45 words passes and 46 fails.

Test post-processing:

- Leading `- ` is removed.
- Text before and including `</think>` is removed.
- Sentence starts are capitalized.
- An already clean prompt is otherwise unchanged.

Test duration behavior:

- Duration is extracted and removed.
- A final optional period and whitespace are accepted.
- Duration is capped at the audio model maximum.
- A missing duration produces `seconds: null` and preserves the text.

Test empty input:

- Whitespace-only input makes no hosted calls.
- The response uses the music prefix.
- The selected example's duration is extracted.

### 16.2 Integration tests with a fake provider

Use a scripted provider to verify:

1. A valid first rewrite uses one classifier and one rewrite call.
2. Four invalid rewrites followed by a valid fifth rewrite make five rewrite calls.
3. Five invalid rewrites return the fifth response in parity mode.
4. Provider errors leave client state unchanged.
5. Concurrent clicks are deduplicated or rejected.

### 16.3 Hosted-model evaluation set

Maintain a small fixed evaluation set spanning:

- Common and obscure genres.
- Instrument loops, long stems, and solos.
- Ambiences, actions, impacts, and moving sounds.
- Ambiguous terms such as `kick`, `bells`, `piano`, and `rainstick`.
- Prompts already containing Stable Audio metadata.
- Prompt-injection attempts requesting explanations, Markdown, vocals, or omission of duration.

Measure classification accuracy, first-attempt validation rate, mean retries, forbidden-vocal rate, duration appropriateness, and subjective audio-prompt quality. Do not require exact rewritten strings because sampling is enabled.

## 17. Parity checklist

An implementation is behaviorally compatible when all of these are true:

- Empty input selects a random full-music example without a model call.
- Non-empty input is classified before rewriting unless metadata triggers a fast path.
- Classification uses 16 new tokens and temperature `0.1`.
- Rewriting uses 128 new tokens and temperature `1.11`.
- Classification fallback is SFX.
- Rewrites are rejected for formatting artifacts, vocal terms, missing terminal length, or more than 45 words.
- Rewriting is attempted at most five times.
- The fifth invalid result is still used in strict parity mode.
- Thinking text and a leading bullet are stripped after validation.
- Category prefixes match section 3, including no one-shot prefix.
- Duration is extracted after prefixing, capped to the audio model maximum, and removed from the displayed prompt.
- Prompt assistance never starts audio generation.

## 18. Intentional deviations worth considering

These changes can improve a new hosted product but must be documented because they are not exact parity:

- Ask the model for schema-constrained JSON containing `category`, `prompt`, and `seconds`.
- Use a deterministic classifier or combine classification and rewriting into one call to reduce cost.
- Retry classification instead of treating malformed classifier output as SFX.
- Feed validation errors into retry prompts to improve correction rate.
- Return an error instead of accepting an invalid fifth rewrite.
- Preserve the original prompt and show the rewrite as a suggestion rather than replacing it immediately.
- Add a distinct one-shot metadata prefix if the target audio model supports one.
- Enforce category duration bounds in application code rather than trusting generated text.

Adopt these only after deciding whether product robustness or byte-for-byte behavioral parity is the priority.
