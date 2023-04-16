# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gradio as gr

from functools import partial

from utils.detectgpt_utils import *
from utils.baseline_utils import *

from generators.LLMGenerator import LMGenerator
from detectors.GPTDetector import GPTDetector
from detectors.WatermarkDetector import WatermarkDetector


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3B")
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
    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )

    return parser.parse_args()


def run_gradio(args, generator, watermark_detector, gpt_detector):
    """Define and launch the gradio demo interface"""

    gpt_detect_partial = partial(gpt_detector.get_detect_results_for_demo, args=args, span_length=2, n_perturbations=20, n_samples=1)

    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                """
                # A Tool for Large Language Models Detection
                """
                )
           

        with gr.Tab("Watermark algorithem"):
            with gr.Column(scale=9):
                gr.Markdown(
                """
                ## 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 
                """
                )

            # gr.Markdown(f"Language model: {args.base_model_name} ")

            # Construct state for parameters, define updates and toggles
            default_prompt = (
                "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
                "species of turtle native to the brackish coastal tidal marshes of the "
                "Northeastern and southern United States, and in Bermuda.[6] It belongs "
                "to the monotypic genus Malaclemys. It has one of the largest ranges of "
                "all turtles in North America, stretching as far south as the Florida Keys "
                "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
                "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
                "British English and American English. The name originally was used by "
                "early European settlers in North America to describe these brackish-water "
                "turtles that inhabited neither freshwater habitats nor the sea. It retains "
                "this primary meaning in American English.[8] In British English, however, "
                "other semi-aquatic turtle species, such as the red-eared slider, might "
                "also be called terrapins. The common name refers to the diamond pattern "
                "on top of its shell (carapace), but the overall pattern and coloration "
                "vary greatly. The shell is usually wider at the back than in the front, "
                "and from above it appears wedge-shaped. The shell coloring can vary "
                "from brown to grey, and its body color can be grey, brown, yellow, "
                "or white. All have a unique pattern of wiggly, black markings or spots "
                "on their body and head. The diamondback terrapin has large webbed "
                "feet.[9] The species is"
                )

            with gr.Tab("Generate and Detect"):
                
                with gr.Row():
                    prompt = gr.Textbox(label=f"Prompt", interactive=True,lines=10,max_lines=10, value=default_prompt)
                with gr.Row():
                    generate_btn = gr.Button("Generate")
                with gr.Row():
                    with gr.Column(scale=2):
                        output_without_watermark = gr.Textbox(label="Output Without Watermark", interactive=False,lines=14,max_lines=14)
                    with gr.Column(scale=1):
                        # without_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                        without_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
                with gr.Row():
                    with gr.Column(scale=2):
                        output_with_watermark = gr.Textbox(label="Output With Watermark", interactive=False,lines=14,max_lines=14)
                    with gr.Column(scale=1):
                        # with_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                        with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"],interactive=False,row_count=7,col_count=2)
            
            with gr.Tab("Detector Only"):
                with gr.Row():
                    with gr.Column(scale=2):
                        detection_input = gr.Textbox(label="Text to Analyze", interactive=True,lines=14,max_lines=14)
                    with gr.Column(scale=1):
                        # detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                        detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
                with gr.Row():
                        detect_btn = gr.Button("Detect")
            
            # Register main generation tab click, outputing generations as well as a the encoded+redecoded+potentially truncated prompt and flag
            # print (type(prompt))
            generate_btn.click(fn=generator.generate_for_watermark_demo, inputs=[prompt], outputs=[output_without_watermark, output_with_watermark])

            # Call detection when the outputs (of the generate function) are updated
            output_without_watermark.change(fn=watermark_detector.detect_for_web, inputs=[output_without_watermark], outputs=[without_watermark_detection_result])
            output_with_watermark.change(fn=watermark_detector.detect_for_web, inputs=[output_with_watermark], outputs=[with_watermark_detection_result])
            # Register main detection tab click
            detect_btn.click(fn=watermark_detector.detect_for_web, inputs=[detection_input], outputs=[detection_result])
        
        with gr.Tab("DetectGPT algorithem"):

            with gr.Row():
                with gr.Column(scale=9):
                    gr.Markdown(
                    """
                    ## 🔍 [Detecting GPT-2 Generations with DetectGPT](https://arxiv.org/abs/2301.10226) 
                    """
                    )

            # gr.Markdown(f"Language model: {args.base_model_name}")

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
            machine_text.change(fn=gpt_detect_partial, inputs=[human_text, machine_text], outputs=[without_watermark_detection_result, with_watermark_detection_result])
            # detect_btn_only.click(fn=detect_partial, inputs=[human_text, human_text], outputs=[detection_result])
            detect_btn_only.click(fn=gpt_detect_partial, inputs=[detection_input, detection_input], outputs=[detection_result])


    demo.queue(concurrency_count=3)

    if args.demo_public:
        demo.launch(share=True) # exposes app to the internet via randomly generated link
    else:
        demo.launch()


if __name__ == '__main__' :
    args = get_args()

    generator = LMGenerator(args.base_model_name)
    gpt_detector = GPTDetector(generator=generator, mask_filling_model_name=args.mask_filling_model_name)
    watermark_detector = WatermarkDetector(vocab=list(generator.tokenizer.get_vocab().values()),
                                    gamma=args.gamma,
                                    seeding_scheme=args.seeding_scheme,
                                    device=generator.device,
                                    tokenizer=generator.tokenizer,
                                    z_threshold=args.detection_z_threshold,
                                    normalizers=(args.normalizers.split(",") if args.normalizers else []),
                                    ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                    select_green_tokens=args.select_green_tokens)

    run_gradio(args, generator, watermark_detector, gpt_detector)


