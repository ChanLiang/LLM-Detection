import numpy as np
import transformers
import torch
import functools

from utils.detectgpt_utils import *
from utils.baseline_utils import *


class GPTDetector():
    def __init__(self, generator, mask_filling_model_name='t5-base', cache_dir='/misc/kfdata01/kf_grp/lchen/cache'):
        # load mask filling t5 model
        self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, cache_dir=cache_dir)
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=512, cache_dir=cache_dir)
        # LLM
        self.generator = generator

    def get_detect_results_for_demo(self, original_text, sampled_text, args, span_length=2, n_perturbations=20, n_samples=1):

        if not original_text or not sampled_text or len(original_text) < 1 or len(sampled_text) < 1:
            output = [["Error, string too short to compute metrics"]]
            output += [["",""] for _ in range(5)]
            return output

        torch.manual_seed(0)
        np.random.seed(0)

        load_mask_model(args, self.generator.model, self.mask_model)

        results = []

        # str
        if len(original_text.split(' ')) > 200:
            original_text = ' '.join(original_text.split(' ')[:200])
        if len(sampled_text.split(' ')) > 200:
            sampled_text = ' '.join(sampled_text.split(' ')[:200])

        original_text = [original_text]
        sampled_text = [sampled_text]

        perturb_fn = functools.partial(perturb_texts, args=args, mask_tokenizer=self.mask_tokenizer, \
            mask_model=self.mask_model, span_length=span_length, pct=args.pct_words_masked)

        p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
        p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
        for _ in range(args.n_perturbation_rounds - 1):
            try:
                p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
            except AssertionError:
                break

        assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
        assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "sampled": sampled_text[idx],
                "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
                "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
            })

        load_base_model(args, self.generator.model, self.mask_model)

        for res in results:
            p_sampled_ll = get_lls(args, self.generator.tokenizer, self.generator.model, res["perturbed_sampled"])
            p_original_ll = get_lls(args, self.generator.tokenizer, self.generator.model, res["perturbed_original"])
            res["original_ll"] = get_ll(args, self.generator.tokenizer, self.generator.model, res["original"])
            res["sampled_ll"] = get_ll(args, self.generator.tokenizer, self.generator.model, res["sampled"])
            res["all_perturbed_sampled_ll"] = p_sampled_ll
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

        output = run_perturbation_experiment(args, results, 'z', span_length=5, n_perturbations=10, n_samples=1)

        real_output = [
            ['original_ll', res["original_ll"]],
            ['perturbed_original_ll', res["perturbed_original_ll"]],
            ['perturbed_original_ll_std', res["perturbed_original_ll_std"]],
            ['discrepancy distance', output['predictions']['real'][0]],
            ['decision threshold', 0.25],
            ['prediction', 'human' if output['predictions']['real'][0] < 0.25 else 'machine']
        ]
        sample_output = [
            ['sampled_ll', res["sampled_ll"]],
            ['perturbed_sampled_ll', res["perturbed_sampled_ll"]],
            ['perturbed_sampled_ll_std', res["perturbed_sampled_ll_std"]],
            ['discrepancy distance', output['predictions']['samples'][0]],
            ['decision threshold', 0.25],
            ['prediction', 'human' if output['predictions']['samples'][0] < 0.25 else 'machine']
        ]
        # print (real_output, sample_output)
        if original_text != sampled_text:
            return real_output, sample_output
        else:
            return real_output


