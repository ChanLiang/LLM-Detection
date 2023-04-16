from generators.LLMGenerator import LMGenerator
from detectors.WatermarkDetector import WatermarkDetector

# Instantiate a generator
generator = LMGenerator('gpt2')

# Generate some machine text
prompt = (
    "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
    "species of turtle native to the brackish coastal tidal marshes of the "
    "Northeastern and southern United States, and in Bermuda.[6] It belongs "
    "to the monotypic genus Malaclemys. It has "
    )
output_wo_watermark = generator.generate(prompt)
output_w_watermark = generator.generate_with_watermark(prompt)

# Instantiate a detector
detector = WatermarkDetector(vocab=list(generator.tokenizer.get_vocab().values()), gamma=0.5, z_threshold=4.0, \
    device=generator.device, tokenizer=generator.tokenizer,)

# Check if the text is written by machine
print (detector.detect(output_wo_watermark))
print (detector.detect(output_w_watermark))