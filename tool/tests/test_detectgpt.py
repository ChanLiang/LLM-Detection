import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import torch
import tqdm
import random
import argparse

from functools import partial

from utils.detectgpt_utils import *
from utils.baseline_utils import *
from preprocessing import custom_datasets


from generators.LLMGenerator import LMGenerator
from detectors.GPTDetector import GPTDetector

import gradio as gr

def generate_samples(raw_data, args, base_tokenizer, base_model, mask_model, batch_size, sep=False):
    load_base_model(args, base_model, mask_model)

    if type(raw_data) == str:
        raw_data = [raw_data]

    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    # print (len(raw_data) // batch_size)
    for batch in range(len(raw_data) // batch_size):
        # print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(args, base_tokenizer, base_model, original_text, min_words=30 if args.dataset in ['pubmed'] else 55)

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_model()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_base_model()

    return data if not sep else [data["original"][0], data["sampled"][0]]


def generate_data(args, dataset, key):
    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)
    else:
        data = datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(args, data[:n_samples], batch_size=batch_size)


def load_base_model_and_tokenizer(args):
    name = args.base_model_name
    base_model_kwargs = {}
    if 'gpt-j' in name or 'neox' in name:
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in name:
        base_model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=args.cache_dir)

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=args.cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=args.cache_dir).to(args.DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=args.cache_dir)

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // args.batch_size), desc="Evaluating real"):
            batch_real = real[batch * args.batch_size:(batch + 1) * args.batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // args.batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * args.batch_size:(batch + 1) * args.batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': args.n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="")
    # parser.add_argument('--mask_filling_model_name', type=str, default="t5-3B")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-base")
    parser.add_argument('--batch_size', type=int, default=50) # bz of generation
    parser.add_argument('--chunk_size', type=int, default=20) # bz of mask_filling
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/misc/kfdata01/kf_grp/lchen/cache")
    parser.add_argument('--DEVICE', type=str, default="cuda")
    parser.add_argument('--algo', type=str, default="detectgpt")
    args = parser.parse_args()
    return args


def detectgpt():
    """Define and launch the gradio demo interface"""
    print ('run detectgpt...')
    args = get_args()

    # load generic generative model
    # base_model, base_tokenizer = load_base_model_and_tokenizer(args)

    generator = LMGenerator(args.base_model_name)

    # load mask filling t5 model
    # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model_name, cache_dir=args.cache_dir)
    # mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model_name, model_max_length=512, cache_dir=args.cache_dir)

    detector = GPTDetector(generator=generator, mask_filling_model_name=args.mask_filling_model_name)

    # generate_partial = partial(generate_samples, args=args, base_tokenizer=base_tokenizer, base_model=base_model, \
        # mask_model=mask_model, batch_size=1, sep=True)

    # detect_partial = partial(get_detect_results_for_demo, args=args, base_model=generator.model, mask_model=mask_model, \
    #     base_tokenizer=generator.tokenizer, mask_tokenizer=mask_tokenizer, span_length=2, n_perturbations=20, n_samples=1)

    detect_partial = partial(detector.get_detect_results_for_demo, args=args, span_length=2, n_perturbations=20, n_samples=1)


    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                """
                ## 💧 [Detecting GPT-2 Generations with DetectGPT](https://arxiv.org/abs/2301.10226) 🔍
                """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                """
                [![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jwkirchenbauer/lm-watermarking)
                """
                )

        gr.Markdown(f"Language model: {args.base_model_name}")

        # Construct state for parameters, define updates and toggles
        input_text = [
        "The fluency and factual knowledge of large language models (LLMs) heightens the need for corresponding systems to detect whether a piece of text is machine-written. For example, students may use LLMs to complete written assignments, leaving instructors unable to accurately assess student learning. In this paper, we first demonstrate that text sampled from an LLM tends to occupy negative curvature regions of the model's log probability function. Leveraging this observation, we then define a new curvature-based criterion for judging if a passage is generated from a given LLM. This approach, which we call DetectGPT, does not require training a separate classifier, collecting a dataset of real or generated passages, or explicitly watermarking generated text. It uses only log probabilities computed by the model of interest and random perturbations of the passage from another generic pre-trained language model (e.g, T5). We find DetectGPT is more discriminative than existing zero-shot methods for model sample detection, notably improving detection of fake news articles generated by 20B parameter GPT-NeoX from 0.81 AUROC for the strongest zero-shot baseline to 0.95 AUROC for DetectGPT. See this https URL for code, data, and other project information."
        ]

        default_prompt = input_text[0]

        with gr.Tab("Generate and Detect"):
            
            with gr.Row():
                prompt = gr.Textbox(label=f"Prompt", interactive=True, lines=10, max_lines=10, value=default_prompt)
            with gr.Row():
                generate_btn = gr.Button("Generate and Detect")
            # with gr.Row():
            #     detect_btn = gr.Button("Detect")
            with gr.Row():
                with gr.Column(scale=2):
                    human_text = gr.Textbox(label="Original Human Text", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    without_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=6,col_count=2)
                    
            with gr.Row():
                with gr.Column(scale=2):
                    machine_text = gr.Textbox(label="Continuation by Machine", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"],interactive=False,row_count=6,col_count=2)

        with gr.Tab("Detector Only"):
            with gr.Row():
                with gr.Column(scale=2):
                    detection_input = gr.Textbox(label="Text to detect", interactive=True,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=6,col_count=2)
            with gr.Row():
                    detect_btn_only = gr.Button("Detect")

        # Register main generation tab click, outputing generations as well as a the encoded+redecoded+potentially truncated prompt and flag
        generate_btn.click(fn=generator.generate_for_detectgpt_demo, inputs=[prompt], outputs=[human_text, machine_text])
        machine_text.change(fn=detect_partial, inputs=[human_text, machine_text], outputs=[without_watermark_detection_result, with_watermark_detection_result])
        # detect_btn_only.click(fn=detect_partial, inputs=[human_text, human_text], outputs=[detection_result])
        detect_btn_only.click(fn=detect_partial, inputs=[detection_input, detection_input], outputs=[detection_result])

    demo.queue(concurrency_count=3)

    demo.launch(share=True) # exposes app to the internet via randomly generated link


detectgpt()