// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** A provider answer after its native JSON or SSE envelope has been removed. */
export interface ProviderAnswer {
  id: string;
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
  toolCalls: Json[];
}

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function appendText(answer: ProviderAnswer, value: unknown): void {
  if (typeof value === 'string') answer.text += value;
}

function toolCallOf(value: unknown, index: number): Json | null {
  if (!isRecord(value)) return null;
  const fn = isRecord(value.function) ? value.function : {};
  const name = typeof fn.name === 'string' ? fn.name : typeof value.name === 'string' ? value.name : '';
  const args = typeof fn.arguments === 'string' ? fn.arguments : isRecord(value.input) ? JSON.stringify(value.input) : isRecord(value.args) ? JSON.stringify(value.args) : '';
  const id = typeof value.id === 'string' && value.id !== '' ? value.id : `toolu_${index}`;
  return { id, type: 'function', function: { name, arguments: args } };
}

function mergeToolCall(answer: ProviderAnswer, value: unknown, fallbackIndex: number): void {
  const call = toolCallOf(value, fallbackIndex);
  if (call === null) return;
  const index = isRecord(value) && typeof value.index === 'number' ? value.index : fallbackIndex;
  const existing = answer.toolCalls[index];
  if (existing === undefined) {
    answer.toolCalls[index] = call;
    return;
  }
  const oldFn = isRecord(existing.function) ? existing.function : {};
  const newFn = isRecord(call.function) ? call.function : {};
  answer.toolCalls[index] = {
    ...existing,
    ...(typeof call.id === 'string' && call.id !== `toolu_${index}` ? { id: call.id } : {}),
    function: {
      ...oldFn,
      ...(typeof newFn.name === 'string' && newFn.name !== '' ? { name: newFn.name } : {}),
      arguments: `${typeof oldFn.arguments === 'string' ? oldFn.arguments : ''}${typeof newFn.arguments === 'string' ? newFn.arguments : ''}`,
    },
  };
}

function answerFor(id: string): ProviderAnswer {
  return { id, text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };
}

function finishOf(value: unknown, fallback: string): string {
  return typeof value === 'string' && value !== '' ? value : fallback;
}

/** Parse one non-streaming response from any of the three provider wires. */
export function parseProviderJson(wire: 'openai' | 'anthropic' | 'gemini', body: Json): ProviderAnswer {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const answer = answerFor(typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate');
    appendText(answer, message.content);
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    calls.forEach((call, index) => mergeToolCall(answer, call, index));
    const usage = isRecord(body.usage) ? body.usage : {};
    answer.inputTokens = finiteNumber(usage.prompt_tokens);
    answer.outputTokens = finiteNumber(usage.completion_tokens);
    answer.stop = finishOf(choice.finish_reason, 'stop');
    return answer;
  }
  if (wire === 'anthropic') {
    const answer = answerFor(typeof body.id === 'string' ? body.id : 'msg_aggregate');
    const content = Array.isArray(body.content) ? body.content : [];
    let callIndex = 0;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text') appendText(answer, block.text);
      else if (block.type === 'tool_use') mergeToolCall(answer, block, callIndex++);
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    answer.inputTokens = finiteNumber(usage.input_tokens);
    answer.outputTokens = finiteNumber(usage.output_tokens);
    answer.stop = finishOf(body.stop_reason, 'end_turn');
    return answer;
  }
  const answer = answerFor(typeof body.modelVersion === 'string' ? body.modelVersion : 'gemini-aggregate');
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let callIndex = 0;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    appendText(answer, part.text);
    if (isRecord(part.functionCall)) mergeToolCall(answer, { ...part.functionCall, type: 'function' }, callIndex++);
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  answer.inputTokens = finiteNumber(usage.promptTokenCount);
  answer.outputTokens = finiteNumber(usage.candidatesTokenCount);
  answer.stop = finishOf(candidate.finishReason, 'STOP');
  return answer;
}

/** Parse all SSE frames in a provider stream; provider streams are deliberately buffered before rewriting. */
export function parseProviderSse(wire: 'openai' | 'anthropic' | 'gemini', text: string): ProviderAnswer {
  const answer = answerFor(wire === 'anthropic' ? 'msg_stream' : wire === 'openai' ? 'chatcmpl-stream' : 'gemini-stream');
  const frames = text.replace(/\r\n/g, '\n').split('\n\n');
  let callIndex = 0;
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let value: unknown;
      try {
        value = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(value)) continue;
      if (wire === 'openai') {
        const choices = Array.isArray(value.choices) ? value.choices : [];
        const choice = isRecord(choices[0]) ? choices[0] : {};
        const delta = isRecord(choice.delta) ? choice.delta : {};
        appendText(answer, delta.content);
        const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
        for (const call of calls) mergeToolCall(answer, call, typeof call === 'object' && isRecord(call) && typeof call.index === 'number' ? call.index : callIndex++);
        if (typeof choice.finish_reason === 'string') answer.stop = choice.finish_reason;
        if (isRecord(value.usage)) {
          answer.inputTokens = finiteNumber(value.usage.prompt_tokens, answer.inputTokens);
          answer.outputTokens = finiteNumber(value.usage.completion_tokens, answer.outputTokens);
        }
      } else if (wire === 'anthropic') {
        if (value.type === 'message_start' && isRecord(value.message)) {
          answer.id = typeof value.message.id === 'string' ? value.message.id : answer.id;
          const usage = isRecord(value.message.usage) ? value.message.usage : {};
          answer.inputTokens = finiteNumber(usage.input_tokens, answer.inputTokens);
        } else if (value.type === 'content_block_delta' && isRecord(value.delta)) {
          if (value.delta.type === 'text_delta') appendText(answer, value.delta.text);
          if (value.delta.type === 'input_json_delta') mergeToolCall(answer, { index: value.index, function: { arguments: value.delta.partial_json } }, typeof value.index === 'number' ? value.index : callIndex++);
        } else if (value.type === 'content_block_start' && isRecord(value.content_block) && value.content_block.type === 'tool_use') {
          mergeToolCall(answer, { ...value.content_block, index: value.index }, typeof value.index === 'number' ? value.index : callIndex++);
        } else if (value.type === 'message_delta') {
          if (isRecord(value.delta)) answer.stop = finishOf(value.delta.stop_reason, answer.stop);
          if (isRecord(value.usage)) answer.outputTokens = finiteNumber(value.usage.output_tokens, answer.outputTokens);
        }
      } else {
        const candidates = Array.isArray(value.candidates) ? value.candidates : [];
        const candidate = isRecord(candidates[0]) ? candidates[0] : {};
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const part of parts) {
          if (!isRecord(part)) continue;
          appendText(answer, part.text);
          if (isRecord(part.functionCall)) mergeToolCall(answer, { ...part.functionCall, type: 'function' }, callIndex++);
        }
        if (typeof candidate.finishReason === 'string') answer.stop = candidate.finishReason;
        if (isRecord(value.usageMetadata)) {
          answer.inputTokens = finiteNumber(value.usageMetadata.promptTokenCount, answer.inputTokens);
          answer.outputTokens = finiteNumber(value.usageMetadata.candidatesTokenCount, answer.outputTokens);
        }
      }
    }
  }
  answer.toolCalls = answer.toolCalls.filter((call) => call !== undefined);
  return answer;
}

/** Translate an OpenAI Chat Completions request to Anthropic Messages. */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : raw.role === 'system' ? 'user' : 'user';
    const content = raw.content;
    if (role === 'user' && raw.role === 'system') continue;
    if (typeof content === 'string') messages.push({ role, content });
    else if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
          blocks.push(match === null ? { type: 'image', source: { type: 'url', url } } : { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } });
        }
      }
      if (blocks.length > 0) messages.push({ role, content: blocks });
    }
    if (Array.isArray(raw.tool_calls)) {
      const blocks = raw.tool_calls.flatMap((call) => {
        if (!isRecord(call) || !isRecord(call.function)) return [];
        return [{ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) }];
      });
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
    }
    if (raw.role === 'tool') messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') }] });
  }
  const system = source.filter((m) => isRecord(m) && m.role === 'system').flatMap((m) => (typeof m.content === 'string' ? [m.content] : [])).join('\n');
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1, messages };
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) out.tools = body.tools.flatMap((tool) => isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string' ? [{ name: tool.function.name, description: typeof tool.function.description === 'string' ? tool.function.description : '', input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} } }] : []);
  if (body.stream === true) out.stream = true;
  return out;
}

function geminiTextPart(value: unknown): Json[] {
  if (typeof value === 'string') return [{ text: value }];
  if (!Array.isArray(value)) return [];
  const out: Json[] = [];
  for (const part of value) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') out.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const match = /^data:([^;]+);base64,(.*)$/s.exec(part.image_url.url);
      if (match !== null) out.push({ inlineData: { mimeType: match[1], data: match[2] } });
      else out.push({ fileData: { fileUri: part.image_url.url } });
    }
  }
  return out;
}

function geminiToolsOf(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const declarations = tools.flatMap((tool) => {
    if (!isRecord(tool)) return [];
    const fn = isRecord(tool.function) ? tool.function : tool;
    if (typeof fn.name !== 'string') return [];
    return [{ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** Translate either client wire to the Gemini generateContent request. */
export function toGemini(body: Json, model: string, clientWire: 'openai' | 'anthropic', stream: boolean): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  if (clientWire === 'openai') {
    for (const raw of source) {
      if (!isRecord(raw) || raw.role === 'system') continue;
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const parts = geminiTextPart(raw.content);
      if (parts.length > 0) contents.push({ role, parts });
    }
  } else {
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const parts = geminiTextPart(raw.content);
      if (parts.length > 0) contents.push({ role, parts });
    }
  }
  const out: Json = { contents, generationConfig: {} };
  const system = clientWire === 'anthropic' ? textOf(body.system) : source.filter((m) => isRecord(m) && m.role === 'system').flatMap((m) => (typeof m.content === 'string' ? [m.content] : [])).join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config = out.generationConfig as Json;
  const max = clientWire === 'anthropic' ? body.max_tokens : typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  if (typeof max === 'number') config.maxOutputTokens = max;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientWire === 'anthropic' ? body.stop_sequences : body.stop;
  if (typeof stops === 'string') config.stopSequences = [stops];
  else if (Array.isArray(stops)) config.stopSequences = stops;
  const format = clientWire === 'openai' && isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  const tools = geminiToolsOf(body.tools);
  if (tools.length > 0) out.tools = tools;
  if (stream) out.stream = true;
  return out;
}

function finishForOpenAI(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_calls';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

function finishForAnthropic(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_use';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Render a normalized provider answer as a complete OpenAI JSON or SSE response. */
export function renderOpenAI(answer: ProviderAnswer, model: string, stream: boolean): string {
  const finish = finishForOpenAI(answer.stop);
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) message.tool_calls = answer.toolCalls;
  const usage = { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens };
  if (!stream) return JSON.stringify({ id: answer.id, object: 'chat.completion', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, message, finish_reason: finish }], usage });
  const frames: string[] = [];
  const chunk = (delta: Json, reason: string | null): Json => ({ id: answer.id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: reason }] });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  if (answer.text !== '') frames.push(sseEvent(null, chunk({ content: answer.text }, null)));
  answer.toolCalls.forEach((call, index) => frames.push(sseEvent(null, chunk({ tool_calls: [{ ...call, index }] }, null))));
  frames.push(sseEvent(null, chunk({}, finish)), sseEvent(null, { id: answer.id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage }), 'data: [DONE]\n\n');
  return frames.join('');
}

/** Render a normalized provider answer as a complete Anthropic JSON or SSE response. */
export function renderAnthropic(answer: ProviderAnswer, model: string, stream: boolean): string {
  const stop = finishForAnthropic(answer.stop);
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    const fn = isRecord(call.function) ? call.function : {};
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof fn.name === 'string' ? fn.name : '', input: argumentsOf(fn.arguments) });
  }
  if (!stream) return JSON.stringify({ id: answer.id, type: 'message', role: 'assistant', model, content, stop_reason: stop, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens } });
  const frames: string[] = [sseEvent('message_start', { type: 'message_start', message: { id: answer.id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } } })];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    if (answer.text !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    const fn = isRecord(call.function) ? call.function : {};
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: typeof call.id === 'string' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} } }));
    if (typeof fn.arguments === 'string' && fn.arguments !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }), sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
