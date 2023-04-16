import torch
from utils.base_class import WatermarkLogitsProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList
from functools import partial


class LMGenerator():
    def __init__(self, model_name_or_path, use_gpu=True, load_fp16=False):
        """Load the model and tokenizer"""
        self.is_seq2seq_model = any([(model_type in model_name_or_path) for model_type in ["t5","T0"]])
        self.is_decoder_only_model = any([(model_type in model_name_or_path) for model_type in ["gpt","opt","bloom"]])

        if self.is_seq2seq_model:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        elif self.is_decoder_only_model:
            if load_fp16:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16, device_map='auto')
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f"Unknown model type: {model_name_or_path}")

        if use_gpu:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if load_fp16: 
                pass
            else: 
                model = model.to(device)
        else:
            device = "cpu"
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def generate(self, prompt, prompt_max_length=None, max_new_tokens=200, use_sampling=False, sampling_temp=0.7, n_beams=1, generation_seed=123):
        gen_kwargs = dict(max_new_tokens=max_new_tokens)

        if use_sampling:
            gen_kwargs.update(dict(
                do_sample=True, 
                top_k=0,
                temperature=sampling_temp
            ))
        else:
            gen_kwargs.update(dict(
                num_beams=n_beams
            ))

        generate_without_watermark = partial(
            self.model.generate,
            **gen_kwargs
        )

        if prompt_max_length:
            pass
        elif hasattr(self.model.config,"max_position_embedding"):
            prompt_max_length = self.model.config.max_position_embeddings - max_new_tokens
        else:
            prompt_max_length = 2048 - max_new_tokens

        tokd_input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=prompt_max_length).to(self.device)
        truncation_warning = True if tokd_input["input_ids"].shape[-1] == prompt_max_length else False
        redecoded_input = self.tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

        torch.manual_seed(generation_seed)
        output_without_watermark = generate_without_watermark(**tokd_input)

        # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
        torch.manual_seed(generation_seed)

        if self.is_decoder_only_model:
            # need to isolate the newly generated tokens
            output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]

        decoded_output_without_watermark = self.tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
            
        return decoded_output_without_watermark 


    def generate_with_watermark(self, prompt, prompt_max_length=None, max_new_tokens=200, use_sampling=False, sampling_temp=0.7, n_beams=1, generation_seed=123, \
        gamma=0.25, delta=2.0, seeding_scheme='simple_1', select_green_tokens=True):

        watermark_processor = WatermarkLogitsProcessor(vocab=list(self.tokenizer.get_vocab().values()),
                                                    gamma=gamma,
                                                    delta=delta,
                                                    seeding_scheme=seeding_scheme,
                                                    select_green_tokens=select_green_tokens)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)

        if use_sampling:
            gen_kwargs.update(dict(
                do_sample=True, 
                top_k=0,
                temperature=sampling_temp
            ))
        else:
            gen_kwargs.update(dict(
                num_beams=n_beams
            ))

        generate_with_watermark = partial(
            self.model.generate,
            logits_processor=LogitsProcessorList([watermark_processor]), 
            **gen_kwargs
        )

        if prompt_max_length:
            pass
        elif hasattr(self.model.config,"max_position_embedding"):
            prompt_max_length = self.model.config.max_position_embeddings - max_new_tokens
        else:
            prompt_max_length = 2048 - max_new_tokens

        tokd_input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=prompt_max_length).to(self.device)
        truncation_warning = True if tokd_input["input_ids"].shape[-1] == prompt_max_length else False
        redecoded_input = self.tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

        # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
        torch.manual_seed(generation_seed)
        output_with_watermark = generate_with_watermark(**tokd_input)

        if self.is_decoder_only_model:
            # need to isolate the newly generated tokens
            output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

        decoded_output_with_watermark = self.tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
            
        return decoded_output_with_watermark


    def generate_for_watermark_demo(self, prompt):
        return [self.generate(prompt), self.generate_with_watermark(prompt)]

    def generate_for_detectgpt_demo(self, prompt):
        return [prompt, self.generate(prompt[:30])]
        