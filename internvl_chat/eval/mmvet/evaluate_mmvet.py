import argparse
import json
import os
import random
import re
import time

import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

ds_collections = {
    'mmvet': {
        'root': 'data/mm-vet/images',
        'question': 'data/mm-vet/llava-mm-vet.jsonl',
        'metric': None,
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    }
}


def collate_fn(batches, has_history=False):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    batch = (pixel_values, questions, question_ids, annotations)
    if has_history:
        histories = [_['history'][0] for _ in batches]
        batch += (histories,)
    else:
        batch += (None,)

    return batch


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)

        image = os.path.join(self.root, image)
        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        question = question + ' ' + self.prompt
        return question_id, question, pixel_values, annotation, None


class MultipleInputsVQADataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, sep="__sep__"):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.sep = sep

    def __len__(self):
        return len(self.data)
    
    def _process_image(self, image):
        image = os.path.join(self.root, image.split("/")[-1])
        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = image
        return images

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)
        if annotation is not None:
            annotation_or_annotations = annotation.split(self.sep)
            if len(annotation_or_annotations)==1:
                a_history, annotation = annotation_or_annotations[0], None
            else:
                a_history, annotation = annotation_or_annotations
        else:
            a_history, annotation = None, None
        question_id = question_id.split(self.sep)[1]
        i_history, image = image.split(self.sep)
        q_history, question = question.split(self.sep)
        images = self._process_image(image)
        images_history = self._process_image(i_history)
        if self.dynamic_image_size:
            images=images_history+images
        else:
            images = [i_history, image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        question = '<image>\n' + question + ' ' + self.prompt
        q_history = '<image>\n' + q_history + ' ' + self.prompt
        return question_id, question, pixel_values, annotation, [(q_history, a_history)]


def evaluate_chat_model():
    random.seed(args.seed)
    prompt = ''

    for ds_name in args.datasets:
        if args.root_prefix!='':
            question_data = ds_collections[ds_name]['question']
            q_dir, q_file = '/'.join(question_data.split('/')[:-1]), question_data.split('/')[-1]
            question_data = f"{q_dir}/{args.root_prefix}{q_file}"
            dataset = MultipleInputsVQADataset(
                root=ds_collections[ds_name]['root'],
                data=question_data,
                prompt=prompt,
                input_size=image_size,
                dynamic_image_size=args.dynamic,
                use_thumbnail=use_thumbnail,
                max_num=args.max_num
            )
        else:
            dataset = VQADataset(
                root=ds_collections[ds_name]['root'],
                data=ds_collections[ds_name]['question'],
                prompt=prompt,
                input_size=image_size,
                dynamic_image_size=args.dynamic,
                use_thumbnail=use_thumbnail,
                max_num=args.max_num
            )

        outputs = {}
        for _, (question_id, question, pixel_values, _, histories) in tqdm(enumerate(dataset)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                output_attentions=args.output_attentions,
                output_hidden_states=args.output_hidden_states,
                verbose=False,
                history=histories,
            )
            if isinstance(response, dict):
                pred = response['response']
                if args.output_hidden_states:
                    state = response['hidden_states']
                if args.output_attentions:
                    att = response['attentions']
            else:
                pred = response
            # print("type(pred)", type(pred))
            # # print("pred", pred)
            # # print("pred['response']", pred['response'])
            # print("type(pred['hidden_states'])", type(pred['hidden_states']), len(pred['hidden_states']))
            # print("type(pred['hidden_states'][0])", type(pred['hidden_states'][0]), len(pred['hidden_states'][0]))
            # print("type(pred['hidden_states'][0][0])", type(pred['hidden_states'][0][0]), pred['hidden_states'][0][0].shape)
            # exit()

            outputs[f'v1_{question_id}'] = pred

        print(f'Evaluating {ds_name} ...')
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{ds_name}_{time_prefix}.json'
        results_file = os.path.join(args.out_dir, results_file)
        json.dump(outputs, open(results_file, 'w'))
        print('Results saved to {}'.format(results_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='pope')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--output-attentions', action='store_true')
    parser.add_argument('--output-hidden-states', action='store_true')
    parser.add_argument('--root-prefix', type=str, default="")
    args = parser.parse_args()
    args.root_prefix = re.sub('test', '', args.root_prefix)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    for ds_name in args.datasets:
        root_path = ds_collections[ds_name]['root']
        root_dir, root_name = '/'.join(root_path.split('/')[:-1]), root_path.split('/')[-1]
        # root_path = f'{os.getenv("HOME")}/{root_dir}/{args.root_prefix}{root_name}'
        root_path = f'{os.getenv("HOME")}/{root_dir}/{root_name}'
        ds_collections[ds_name]['root'] = root_path
        for ky2, vl2 in ds_collections[ds_name].items():
            if isinstance(vl2, str) and '/' in vl2 and ky2 != 'root':
                ds_collections[ds_name][ky2] = f'{os.getenv("HOME")}/{vl2}'
    # for ky in ds_collections.keys():
    #     for ky2, vl2 in ds_collections[ky].items():
    #         if isinstance(vl2, str) and '/' in vl2:
    #             ds_collections[ky][ky2] = f'{os.getenv("HOME")}/{vl2}'
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
