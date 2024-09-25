import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def run_captioning(images, concept_sentence=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image,
                           return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace(
            "The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        captions.append(caption_text)

    model.to("cpu")
    del model
    del processor
    return captions


def generate_captions_for_folder(folder_path, concept_sentence=""):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[
        1] in image_extensions]
    image_paths = [os.path.join(folder_path, f) for f in image_files]

    captions = run_captioning(image_paths, concept_sentence)

    for image_file, caption in zip(image_files, captions):
        base_name = os.path.splitext(image_file)[0]
        txt_file = os.path.join(folder_path, f"{base_name}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(caption)

    print(f"Generated captions for {len(captions)} images in {folder_path}")


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing images: ")
    concept_sentence = input(
        "Enter a concept sentence (optional, press Enter to skip): ")
    generate_captions_for_folder(folder_path, concept_sentence)
