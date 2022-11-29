import argparse
import os
from pathlib import Path
from typing import List

import PIL.Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                            stdoutToServer=True, stderrToServer=True, suspend=False)


def get_dataloader(data_path: Path, batch_size: int) -> DataLoader:
    tfm = transforms.Compose([  # TODO: rethink and see if Nomralization need or if it's done later
        transforms.ToTensor(),
        transforms.Resize((384, 384))
    ])
    dataset = ImageFolder(str(data_path), transform=tfm)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader


class TrOCRPredictor:
    def __init__(self, model_name: str = 'microsoft/trocr-base-handwritten'):
        # TODO: make Processor and Decoder exchangeable: https://huggingface.co/models?other=trocr
        # TODO: check why some weights are not initialized
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda()

    @staticmethod
    def preprocess_images(images: List[torch.Tensor], processor: TrOCRProcessor):
        processed_images = []
        for image in images:
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            processed_images.append(pixel_values)
        processed_images = torch.cat(processed_images, dim=0)
        return processed_images

    def predict(self, images: List[PIL.Image.Image]) -> List[str]:
        image_tensors = [F.to_tensor(img) for img in images]
        processed_images = self.preprocess_images(image_tensors, self.processor).cuda()
        generated_ids = self.model.generate(processed_images)
        predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return predicted_text


def main(args: argparse.Namespace):
    # dataloader = get_dataloader(args.dataset_path, batch_size=args.batch_size)
    dataset = ImageFolder(str(args.dataset_path))

    # processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    # model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').cuda()
    trocr_predictor = TrOCRPredictor()

    all_predictions = []
    # for batch_id, batch in enumerate(tqdm(dataloader)):
    num_samples = len(dataset)
    for i in range(0, num_samples, args.batch_size):
        images = [dataset[i][0] for i in range(i, min(i + args.batch_size, num_samples))]
        predicted_text = trocr_predictor.predict(images)
        all_predictions.extend(predicted_text)
    print(all_predictions)


if __name__ == '__main__':
    # TODO: clean file
    parser = argparse.ArgumentParser(description='Inference of TrOCR model on given dataset')
    parser.add_argument('dataset_path', type=Path, help='path to config with common train settings, such as LR')
    parser.add_argument('--image_size', type=int, nargs='+', default=[64, 256],
                        help='size to which dataset images should be scaled')
    parser.add_argument('--batch_size', type=int, default=8)
    parsed_args = parser.parse_args()
    main(parsed_args)
