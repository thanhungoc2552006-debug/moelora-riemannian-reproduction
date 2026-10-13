"""Partial ScienceQA reproduction; imports the user's MoE-LoRA and RSGD code.

Paper hyperparameters, BF16 backbone, correctly aligned answer labels.
This is an explicitly documented protocol variant, not an exact score reproduction.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from moelora_repro.scienceqa.model import inject_moe_lora
from moelora_repro.scienceqa.optimizer import RiemannianSGD
from moelora_repro.scienceqa.train import make_collate_fn, parse_answer, set_seed

MODEL = 'meta-llama/Llama-3.2-3B'
DATASET = 'derek-thomas/ScienceQA'
MAX_LENGTH, PROMPT_BUDGET = 256, 240
ALPHA, DROPOUT = 16.0, 0.05
N_EXPERTS, TOP_K, RANK = 20, 10, 4
EXPERT_LR, GATE_LR, REG = 3e-5, 3e-8, 1e-6
SCHEDULER_HORIZON = 2222  # int(2000/9*10), as in published source
PROTOCOL = 'scienceqa-paper-hparams-corrected-labels-bf16-v1'


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def prompt(ex):
    context = ex.get('hint') or 'N/A'
    options = ' '.join(f'({chr(65+i)}) {choice}' for i, choice in enumerate(ex['choices']))
    instruction = f'Context: {context}\nQuestion: {ex["question"]}\nOptions: {options}\n'
    return ('Below is an instruction that describes a task. Write a response that '
            'appropriately completes the request.\n\n### Instruction:\n'
            + instruction + '\n\n### Response:\n')


def prompt_ids(ex, tokenizer):
    ids = tokenizer(prompt(ex), add_special_tokens=True, truncation=False)['input_ids']
    # Keep BOS, the end of the question/options and the response marker.
    # Apply the SAME rule at training and generation.
    if len(ids) > PROMPT_BUDGET:
        if ids[0] == tokenizer.bos_token_id:
            ids = [ids[0]] + ids[-(PROMPT_BUDGET-1):]
        else:
            ids = ids[-PROMPT_BUDGET:]
    return ids


def encode_training(ex, tokenizer):
    pids = prompt_ids(ex, tokenizer)
    text = f'Answer: The answer is {chr(65+int(ex["answer"]))}.'
    tids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    tids += [tokenizer.eos_token_id]
    if len(tids) > MAX_LENGTH - PROMPT_BUDGET:
        raise ValueError('Answer exceeds reserved budget; inspect tokenizer instead of truncating it.')
    ids = pids + tids
    labels = [-100]*len(pids) + tids
    assert len(ids) <= MAX_LENGTH and any(x != -100 for x in labels[1:])
    return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels)


def batch_plan(size, batch_size, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    return [order[i:i+batch_size] for i in range(0, size, batch_size)]


def to_device(batch):
    return {k: v.cuda() for k, v in batch.items()}


@torch.no_grad()
def val_loss(model, data, collate, batch_size):
    model.eval()
    loss_sum = token_count = 0
    for start in range(0, len(data), batch_size):
        batch = to_device(collate([data[i] for i in range(start, min(start+batch_size, len(data)))]))
        n = int((batch['labels'][:, 1:] != -100).sum().item())
        loss = model(**batch).loss.item()
        if not math.isfinite(loss):
            raise RuntimeError('Validation loss is not finite.')
        loss_sum += loss*n
        token_count += n
    return loss_sum / token_count


@torch.no_grad()
def accuracy(model, tokenizer, data, batch_size, prediction_path=None):
    model.eval()
    rows = []
    correct = invalid = 0
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        for start in range(0, len(data), batch_size):
            examples = [data[i] for i in range(start, min(start+batch_size, len(data)))]
            enc = tokenizer.pad([{'input_ids': prompt_ids(ex, tokenizer),
                                  'attention_mask': [1]*len(prompt_ids(ex, tokenizer))}
                                 for ex in examples], padding=True, return_tensors='pt')
            enc = to_device(enc)
            pred = model.generate(**enc, max_new_tokens=10, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
            texts = tokenizer.batch_decode(pred[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
            for idx, ex, text in zip(range(start, start+len(examples)), examples, texts):
                letter = parse_answer(text)
                gold = chr(65+int(ex['answer']))
                invalid += int(letter is None)
                correct += int(letter == gold)
                rows.append(dict(index=idx, gold=gold, prediction=letter, generated=text))
    finally:
        tokenizer.padding_side = old_side
    if prediction_path is not None:
        atomic_json(prediction_path, rows)
    return dict(accuracy=correct/len(data), correct=correct, total=len(data), invalid_predictions=invalid)


def save_checkpoint(out, model, gate_opt, expert_opt, cfg, step, epoch, next_batch):
    # Custom RiemannianSGD has no momentum or other hidden update state.
    # Save its LR/reg explicitly; gate optimizer has a normal state_dict.
    payload = dict(config=cfg, step=step, epoch=epoch, next_batch=next_batch,
                   trainable={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                   gate_optimizer=gate_opt.state_dict(),
                   expert_optimizer=dict(lr=expert_opt.lr, reg=expert_opt.reg),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temp = out / 'checkpoint_latest.pt.tmp'
    torch.save(payload, temp)
    latest, previous = out/'checkpoint_latest.pt', out/'checkpoint_previous.pt'
    if latest.exists():
        os.replace(latest, previous)
    os.replace(temp, latest)
    print(f'Checkpoint saved: step={step}, epoch={epoch}, next_batch={next_batch}', flush=True)


def load_checkpoint(out, model, gate_opt, expert_opt, cfg):
    paths = [out/'checkpoint_latest.pt', out/'checkpoint_previous.pt']
    payload = None
    for path in paths:
        if path.exists():
            try:
                payload = torch.load(path, map_location='cpu', weights_only=False)
                print(f'Resuming {path.name} at step {payload["step"]}', flush=True)
                break
            except Exception as error:
                print(f'Cannot read {path.name}: {type(error).__name__}; try previous checkpoint.', flush=True)
    if payload is None:
        raise FileNotFoundError('No readable checkpoint; use a new experiment directory to start again.')
    if payload['config'] != cfg:
        raise ValueError('Resume config/dependencies/source differ; restore the saved setup or use a new directory.')
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != payload['trainable'].keys():
        raise ValueError('Trainable parameter keys changed.')
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(payload['trainable'][n].to(p.device))
    gate_opt.load_state_dict(payload['gate_optimizer'])
    expert_opt.lr = payload['expert_optimizer']['lr']
    expert_opt.reg = payload['expert_optimizer']['reg']
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'])
    torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['step'], payload['epoch'], payload['next_batch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['riemannian', 'moe-riemannian'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--revisions', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=40)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every, args.save_every) < 1:
        parser.error('Step and batch settings must be positive.')
    assert torch.cuda.is_available(), 'Choose a GPU runtime.'
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    args.revisions.parent.mkdir(parents=True, exist_ok=True)
    if not args.revisions.exists():
        api = HfApi()
        atomic_json(args.revisions, dict(model=api.model_info(MODEL).sha,
                                        dataset=api.dataset_info(DATASET).sha))
    revisions = json.loads(args.revisions.read_text())
    import moelora_repro.scienceqa.model as source_model
    import moelora_repro.scienceqa.optimizer as source_opt
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cfg = dict(protocol=PROTOCOL, mode=args.mode, steps=args.steps, seed=args.seed,
               batch_size=args.batch_size, grad_accum=args.grad_accum,
               eval_every=args.eval_every, save_every=args.save_every,
               train_limit=args.train_limit, val_limit=args.val_limit, smoke=args.smoke,
               model=MODEL, dataset=DATASET, revisions=revisions,
               precision='BF16 backbone / FP32 adapters and router',
               n=N_EXPERTS, k=TOP_K, r=RANK, alpha=ALPHA, dropout=DROPOUT,
               expert_lr=EXPERT_LR, gate_lr=GATE_LR, reg=REG,
               scheduler_horizon=SCHEDULER_HORIZON, max_length=MAX_LENGTH, prompt_budget=PROMPT_BUDGET,
               source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               source_hashes=dict(driver=sha(__file__), model=sha(source_model.__file__), optimizer=sha(source_opt.__file__)),
               packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'datasets', 'numpy', 'accelerate']})
    config_path = args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError('Existing experiment has a different config; change EXPERIMENT_ID.')
        if (args.out/'summary.json').exists():
            print('Already completed:', (args.out/'summary.json').read_text(), flush=True)
            return
    else:
        atomic_json(config_path, cfg)
    atomic_json(args.out/'environment.json', dict(gpu=torch.cuda.get_device_name(),
                vram_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                cuda=torch.version.cuda, config=cfg))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revisions['model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    ds = load_dataset(DATASET, revision=revisions['dataset'])
    raw = {split: ds[split].filter(lambda x: x['image'] is None) for split in ['train', 'validation', 'test']}
    if args.train_limit:
        raw['train'] = raw['train'].shuffle(seed=args.seed).select(range(min(args.train_limit, len(raw['train']))))
    if args.val_limit:
        raw['validation'] = raw['validation'].select(range(min(args.val_limit, len(raw['validation']))))
    train = raw['train'].map(lambda ex: encode_training(ex, tokenizer), remove_columns=raw['train'].column_names,
                            load_from_cache_file=False)
    valid = raw['validation'].map(lambda ex: encode_training(ex, tokenizer), remove_columns=raw['validation'].column_names,
                                 load_from_cache_file=False)
    assert len(train) > 0 and len(valid) > 0
    atomic_json(args.out/'data_audit.json', dict(sizes={k: len(v) for k, v in raw.items()},
                zero_supervised_rows=sum(not any(y != -100 for y in ex['labels'][1:]) for ex in train),
                train_prompts_truncated=sum(len(tokenizer(prompt(ex))['input_ids']) > PROMPT_BUDGET for ex in raw['train']),
                test_prompts_truncated=sum(len(tokenizer(prompt(ex))['input_ids']) > PROMPT_BUDGET for ex in raw['test'])))
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revisions['model'], torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    inject_moe_lora(model, rank=RANK, num_experts=N_EXPERTS, top_k=TOP_K, alpha=ALPHA, mode=args.mode, dropout=DROPOUT)
    gates = [p for n, p in model.named_parameters() if '.gate.' in n and p.requires_grad]
    assert len(gates) > 0
    gate_opt = torch.optim.SGD(gates, lr=GATE_LR)
    expert_opt = RiemannianSGD(model, lr=EXPERT_LR, reg=REG)
    collate = make_collate_fn(tokenizer)
    step = epoch = next_batch = 0
    if (args.out/'checkpoint_latest.pt').exists() or (args.out/'checkpoint_previous.pt').exists():
        step, epoch, next_batch = load_checkpoint(args.out, model, gate_opt, expert_opt, cfg)
    else:
        save_checkpoint(args.out, model, gate_opt, expert_opt, cfg, step, epoch, next_batch)
    metrics_path = args.out/'metrics.jsonl'
    # Remove stale metrics written after an older checkpoint being recovered.
    if metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        metrics_path.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step'] <= step))
    if step == 0:
        initial = accuracy(model, tokenizer, raw['validation'].select(range(min(128, len(valid)))), args.batch_size)
        atomic_json(args.out/'baseline_validation.json', initial)
        print('Initial validation:', initial, flush=True)
    started = time.perf_counter()
    first_step = step
    while step < args.steps:
        plan = batch_plan(len(train), args.batch_size, args.seed, epoch)
        model.train()
        while next_batch < len(plan) and step < args.steps:
            count = min(args.grad_accum, len(plan)-next_batch)
            total_loss = 0.0
            gate_opt.zero_grad(set_to_none=True)
            expert_opt.zero_grad()
            for ids in plan[next_batch:next_batch+count]:
                batch = to_device(collate([train[i] for i in ids]))
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss; inspect data_audit and checkpoint.')
                total_loss += float(loss.detach())
                (loss/count).backward()
            torch.nn.utils.clip_grad_norm_(gates, 1.0)
            factor = max(0.0, 1.0-step/SCHEDULER_HORIZON)
            expert_opt.lr = EXPERT_LR*factor
            for group in gate_opt.param_groups:
                group['lr'] = GATE_LR*factor
            expert_opt.step()
            gate_opt.step()
            gate_opt.zero_grad(set_to_none=True)
            expert_opt.zero_grad()
            step += 1
            next_batch += count
            print(f'step={step}/{args.steps} epoch={epoch} loss={total_loss/count:.5f} '
                  f'expert_lr={expert_opt.lr:.3g} peak_GB={torch.cuda.max_memory_allocated()/2**30:.2f}', flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(args.out, model, gate_opt, expert_opt, cfg, step, epoch, next_batch)
            if step % args.eval_every == 0 or step == args.steps:
                row = dict(step=step, epoch=epoch, train_loss=total_loss/count,
                           validation_loss=val_loss(model, valid, collate, args.batch_size))
                if not args.smoke:
                    subset = raw['validation'].select(range(min(256, len(valid))))
                    row['validation_accuracy_first256'] = accuracy(model, tokenizer, subset, args.batch_size)
                with metrics_path.open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print('Validation:', row, flush=True)
                model.train()
        if next_batch == len(plan):
            epoch += 1
            next_batch = 0
    summary = dict(config=cfg, optimizer_steps=step,
                   elapsed_training_and_validation_seconds=time.perf_counter()-started,
                   steps_trained_this_session=step-first_step,
                   peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30)
    if not args.smoke:
        print('Final full text-only test evaluation (no test-based checkpoint selection)...', flush=True)
        summary.update(accuracy(model, tokenizer, raw['test'], args.batch_size, args.out/'test_predictions.json'))
    else:
        summary['note'] = 'Smoke test only; no Table 1 accuracy.'
    atomic_json(args.out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
