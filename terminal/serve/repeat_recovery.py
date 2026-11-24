"""Bounded recovery of repetitive generation, without replaying published actions."""
from serve.frontend import Event
import time

NOTICE = '\n[Strata] 检测到重复输出，本轮已停止；会话仍可继续，请重试或发送下一条消息。'


def run(service, ids, thinking, tools, max_new, sampling, cancel, force=None,
        declared_only=False, deferred=False):
    used = reasoned = retries = 0
    visible = False
    has_images = any(getattr(service.embeddings, key, None) for key in ('path', 'hy4', 'inkling'))
    previous_timings = []
    last_sent = time.monotonic()
    while True:
        pending = []
        done = None
        source = service._run(ids, thinking, tools, max_new-used, sampling, cancel, force,
                              declared_only, deferred)
        try:
            for kind, value in source:
                if kind == 'done':
                    done = dict(value)
                    break
                if kind == 'event':
                    if value.kind in ('tool_start', 'tool_args'):
                        pending.append((kind,value))
                        if time.monotonic()-last_sent >= 10.0:
                            yield 'ping',None
                            last_sent = time.monotonic()
                        continue
                    if value.kind == 'tool_call':
                        for item in pending:
                            yield item
                        pending.clear()
                        visible = True
                    elif value.kind == 'content' and value.text:
                        visible = True
                yield kind,value
                last_sent = time.monotonic()
        finally:
            source.close()
        if done is None:
            return  # preserve the existing missing-terminal-event error, never invent success
        used += done['completion_tokens']
        reasoned += done.get('reasoning_tokens',0)
        if done.get('timings'):
            previous_timings.append(done['timings'])
        repeated = done.pop('repeated',False)
        effective_sampling = done.pop('_repeat_sampling',sampling)
        if (repeated and not visible and not retries and not has_images
                and not cancel.is_set() and done['finish'] != 'cancel' and used < max_new):
            retries = 1
            # Same input/history, no generated loop prefix; vary only this retry's sampler.
            sampling = {**(effective_sampling or {})}
            sampling['temperature'] = max(1.0,float(sampling.get('temperature') or 0))
            sampling['presence_penalty'] = max(1.5,float(sampling.get('presence_penalty') or 0))
            if sampling.get('seed') is not None:
                sampling['seed'] = (int(sampling['seed'])+1) % 2147483647
            print('[strata] repetitive unpublished draft discarded; retrying once from the original prompt',flush=True)
            with service.status_lock:
                service.totals['repeat_failure_retries'] = service.totals.get('repeat_failure_retries',0)+1
            yield 'ping',None
            continue
        done.update(completion_tokens=used,reasoning_tokens=reasoned)
        if retries:
            done['generation_recovery'] = {'reason':'repetition','retries':retries}
            if previous_timings:
                timing = dict(previous_timings[-1])
                for key in ('prompt_ms','predicted_ms','predicted_n'):
                    timing[key] = sum(t.get(key) or 0 for t in previous_timings)
                timing['prompt_n'] = sum(t.get('prompt_n') or 0 for t in previous_timings)
                timing['prompt_per_token_ms'] = (round(timing['prompt_ms']/timing['prompt_n'],3) if timing['prompt_n'] else None)
                timing['prompt_per_second'] = (round(1000*timing['prompt_n']/timing['prompt_ms'],1) if timing['prompt_ms'] else None)
                timing['predicted_per_token_ms'] = (round(timing['predicted_ms']/timing['predicted_n'],3) if timing['predicted_n'] else None)
                timing['predicted_per_second'] = (round(1000*timing['predicted_n']/timing['predicted_ms'],1)
                                                   if timing['predicted_ms'] else None)
                done['timings'] = timing
        empty_retry = retries and not visible and done['finish'] in ('stop','length')
        if (repeated or empty_retry) and not cancel.is_set() and done['finish'] != 'cancel':
            # A service notice is not a forged model tool call. Keep complete calls, discard unfinished ones.
            yield 'event',Event('content',NOTICE)
            done.update(finish='stop',generation_failure={'reason':'repetition' if repeated else 'retry_incomplete','retries':retries,'retryable':True})
            done.pop('unread_tool_markup',None)
            with service.status_lock:
                service.totals['repeat_failure_stops'] = service.totals.get('repeat_failure_stops',0)+1
        yield 'done',done
        return
