import sys
sys.path.append("..")

from pprint import pprint

from tool.app import get_args

from detectors.WatermarkDetector import WatermarkDetector
from generators.LLMGenerator import LMGenerator

import gradio as gr


def test_local(args): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    # print(args)

    # if not args.skip_model_load:
    #     model, tokenizer, device = load_model(args)
    # else:
    #     model, tokenizer, device = None, None, None

    generator = LMGenerator(args.model_name_or_path)

    # Generate and detect, report to stdout
    input_text = (
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

    args.default_prompt = input_text

    term_width = 80
    print("#"*term_width)
    print("Prompt:")
    print(input_text)

    # _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
    #                                                                                     args, 
    #                                                                                     model=model, 
    #                                                                                     device=device, 
    #                                                                                     tokenizer=tokenizer)

    # _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generator.generate(input_text)

    decoded_output_without_watermark = generator.generate(input_text)
    decoded_output_with_watermark = generator.generate_with_watermark(input_text)


    # without_watermark_detection_result = detect(decoded_output_without_watermark, 
    #                                             args, 
    #                                             device=device, 
    #                                             tokenizer=tokenizer)
    # with_watermark_detection_result = detect(decoded_output_with_watermark, 
    #                                             args, 
    #                                             device=device, 
    #                                             tokenizer=tokenizer)

    detector = WatermarkDetector(vocab=list(generator.tokenizer.get_vocab().values()),
                                    gamma=args.gamma,
                                    seeding_scheme=args.seeding_scheme,
                                    device=generator.device,
                                    tokenizer=generator.tokenizer,
                                    z_threshold=args.detection_z_threshold,
                                    normalizers=args.normalizers,
                                    ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                    select_green_tokens=args.select_green_tokens)
                            
    without_watermark_detection_result = detector.detect_for_web(decoded_output_without_watermark)
    with_watermark_detection_result = detector.detect_for_web(decoded_output_with_watermark)

    print("#"*term_width)
    print("Output without watermark:")
    print(decoded_output_without_watermark)
    print("-"*term_width)
    print(f"Detection result @ {args.detection_z_threshold}:")
    pprint(without_watermark_detection_result)
    print("-"*term_width)

    print("#"*term_width)
    print("Output with watermark:")
    print(decoded_output_with_watermark)
    print("-"*term_width)
    print(f"Detection result @ {args.detection_z_threshold}:")
    pprint(with_watermark_detection_result)
    print("-"*term_width)


    # Launch the app to generate and detect interactively (implements the hf space demo)
    # if args.run_gradio:
        # run_gradio(args, model=model, tokenizer=tokenizer, device=device)

    test_gradio(args, generator, detector)

    return


def test_gradio(args, generator, detector):
    """Define and launch the gradio demo interface"""

    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                """
                ## 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍
                """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                """
                [![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jwkirchenbauer/lm-watermarking)
                """
                )
            # with gr.Column(scale=2):
            #     pass
            # ![visitor badge](https://visitor-badge.glitch.me/badge?page_id=tomg-group-umd_lm-watermarking) # buggy

        with gr.Accordion("Understanding the output metrics",open=False):
            gr.Markdown(
            """
            - `z-score threshold` : The cuttoff for the hypothesis test
            - `Tokens Counted (T)` : The number of tokens in the output that were counted by the detection algorithm. 
                The first token is ommitted in the simple, single token seeding scheme since there is no way to generate
                a greenlist for it as it has no prefix token(s). Under the "Ignore Bigram Repeats" detection algorithm, 
                described in the bottom panel, this can be much less than the total number of tokens generated if there is a lot of repetition.
            - `# Tokens in Greenlist` : The number of tokens that were observed to fall in their respective greenlist
            - `Fraction of T in Greenlist` : The `# Tokens in Greenlist` / `T`. This is expected to be approximately `gamma` for human/unwatermarked text.
            - `z-score` : The test statistic for the detection hypothesis test. If larger than the `z-score threshold` 
                we "reject the null hypothesis" that the text is human/unwatermarked, and conclude it is watermarked
            - `p value` : The likelihood of observing the computed `z-score` under the null hypothesis. This is the likelihood of 
                observing the `Fraction of T in Greenlist` given that the text was generated without knowledge of the watermark procedure/greenlists.
                If this is extremely _small_ we are confident that this many green tokens was not chosen by random chance.
            -  `prediction` : The outcome of the hypothesis test - whether the observed `z-score` was higher than the `z-score threshold`
            - `confidence` : If we reject the null hypothesis, and the `prediction` is "Watermarked", then we report 1-`p value` to represent 
                the confidence of the detection based on the unlikeliness of this `z-score` observation.
            """
            )

        with gr.Accordion("A note on model capability",open=True):
            gr.Markdown(
                """
                This demo uses open-source language models that fit on a single GPU. These models are less powerful than proprietary commercial tools like ChatGPT, Claude, or Bard. 

                Importantly, we use a language model that is designed to "complete" your prompt, and not a model this is fine-tuned to follow instructions. 
                For best results, prompt the model with a few sentences that form the beginning of a paragraph, and then allow it to "continue" your paragraph. 
                Some examples include the opening paragraph of a wikipedia article, or the first few sentences of a story. 
                Longer prompts that end mid-sentence will result in more fluent generations.
                """
                )
        
        gr.Markdown(f"Language model: {args.model_name_or_path} {'(float16 mode)' if args.load_fp16 else ''}")

        # Construct state for parameters, define updates and toggles
        default_prompt = args.__dict__.pop("default_prompt")

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
        output_without_watermark.change(fn=detector.detect_for_web, inputs=[output_without_watermark], outputs=[without_watermark_detection_result])
        output_with_watermark.change(fn=detector.detect_for_web, inputs=[output_with_watermark], outputs=[with_watermark_detection_result])
        # Register main detection tab click
        detect_btn.click(fn=detector.detect_for_web, inputs=[detection_input], outputs=[detection_result])
        
    demo.queue(concurrency_count=3)

    if args.demo_public:
        demo.launch(share=True) # exposes app to the internet via randomly generated link
    else:
        demo.launch()



if __name__ == "__main__":

    args = get_args()
    # print(args)

    test_local(args)
