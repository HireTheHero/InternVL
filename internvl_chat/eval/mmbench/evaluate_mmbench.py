import argparse
import base64
import itertools
import json
import os
import pickle
import random
import re
import time
from functools import partial
from io import BytesIO

import pandas as pd
import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ds_collections = {
    'mmbench_dev_20230712': {
        'root': 'data/mmbench/mmbench_dev_20230712.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_dev_cn_20231003': {
        'root': 'data/mmbench/mmbench_dev_cn_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'cn'
    },
    'mmbench_dev_en_20231003': {
        'root': 'data/mmbench/mmbench_dev_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_test_cn_20231003': {
        'root': 'data/mmbench/mmbench_test_cn_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'cn'
    },
    'mmbench_test_en_20231003': {
        'root': 'data/mmbench/mmbench_test_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'en'
    },
    'ccbench_dev_cn': {
        'root': 'data/mmbench/CCBench_legacy.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'cn'
    }
}


def collate_fn(batches, has_history=False):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indexes = [_['index'] for _ in batches]
    options = [_['option'] for _ in batches]
    batch = (pixel_values, questions, answers, indexes, options)
    if has_history:
        histories = [_['history'][0] for _ in batches]
        batch += (histories,)
    else:
        batch += (None,)
    return batch


class MMBenchDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, language, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.df = pd.read_csv(root, sep='\t')
        self.prompt = prompt
        self.language = language
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        # catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']

        image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            question = hint + '\n' + question
        for key, item in options.items():
            question += f'\n{key}. {item}'
        if self.language == 'cn':
            question = question + '\n' + self.prompt['cn']
        else:
            question = question + '\n' + self.prompt['en']

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'index': index,
            'option': options,
            'history': None,
        }

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


class MultipleInputsMMBenchDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, language, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, sep="__sep__"):
        self.df = pd.read_csv(root, sep='\t')
        self.prompt = prompt
        self.language = language
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.sep = sep

    def __len__(self):
        return len(self.df)
    
    def _process_image(self, image):
        image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = image
        return images

    def __getitem__(self, idx):
        _, index = self.df.iloc[idx]['index'].split(self.sep)
        i_history, image = self.df.iloc[idx]['image'].split(self.sep)
        q_history, question = self.df.iloc[idx]['question'].split(self.sep)
        if 'answer' in self.df.iloc[0].keys():
            answer_or_answers = self.df.iloc[idx]['answer'].split(self.sep)
            if len(answer_or_answers)==1:
                a_history, answer = answer_or_answers[0], None
            else:
                a_history, answer = answer_or_answers
        else:
            a_history, answer = None, None
        # answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        # catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']

        images = self._process_image(image)
        images_history = self._process_image(i_history)
        if self.dynamic_image_size:
            images=images_history+images
        else:
            images = [i_history, image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            question = hint + '\n' + question
        for key, item in options.items():
            question += f'\n{key}. {item}'
        if self.language == 'cn':
            q_history = '<image>\n' + q_history + '\n' + self.prompt['cn']
            question = '<image>\n' + question + '\n' + self.prompt['cn']
        else:
            q_history = '<image>\n' + q_history + '\n' + self.prompt['en']
            question = '<image>\n' + question + '\n' + self.prompt['en']

        return {
            'question': question,
            'history': [(q_history, a_history)],
            'pixel_values': pixel_values,
            'answer': answer,
            'index': index,
            'option': options
        }

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        if args.root_prefix!='':
            dataset = MultipleInputsMMBenchDataset(
                root=ds_collections[ds_name]['root'],
                prompt=prompt,
                language=ds_collections[ds_name]['language'],
                input_size=image_size,
                dynamic_image_size=args.dynamic,
                use_thumbnail=use_thumbnail,
                max_num=args.max_num,
                sep=args.sep
            )
            has_history = True
        else:
            dataset = MMBenchDataset(
                root=ds_collections[ds_name]['root'],
                prompt=prompt,
                language=ds_collections[ds_name]['language'],
                input_size=image_size,
                dynamic_image_size=args.dynamic,
                use_thumbnail=use_thumbnail,
                max_num=args.max_num
            )
            has_history = False
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, has_history=has_history),
        )

        outputs = []
        for _, (pixel_values, questions, answers, indexes, options, histories) in tqdm(enumerate(dataloader)):
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
                question=questions[0],
                generation_config=generation_config,
                verbose=False,
                output_attentions=args.output_attentions,
                output_hidden_states=args.output_hidden_states,
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
            # print("pred.keys()", pred.keys())
            # # print("pred", pred)
            # print("pred['response']", pred['response'])
            # print("type(pred['hidden_states'])", type(pred['hidden_states']), len(pred['hidden_states']))
            # print("type(pred['hidden_states'][0])", type(pred['hidden_states'][0]), len(pred['hidden_states'][0]))
            # print("type(pred['hidden_states'][0][0])", type(pred['hidden_states'][0][0]), pred['hidden_states'][0][0].shape)
            # exit()
            preds = [post_process(pred, options[0])]

            for question, pred, answer, index in zip(questions, preds, answers, indexes):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'index': index
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:

            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.xlsx'
            output_path = os.path.join(args.out_dir, results_file)
            df = pd.read_table(ds_collections[ds_name]['root'])
            for col in df.columns:
                if df[col].dtype == 'object':  # 'object' usually means string in pandas
                    df[col] = df[col].apply(lambda x: x.split(args.sep)[-1] if isinstance(x, str) and args.sep in x else x)
            cur_df = df.copy()
            if 'mmbench' in ds_name:
                cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
                cur_df.insert(6, 'prediction', None)
            else:
                cur_df = cur_df.drop(columns=['category', 'image'])
                cur_df.insert(8, 'prediction', None)
            for item in merged_outputs:
                cur_df.loc[df['index'] == item['index'], 'prediction'] = item['answer']

            cur_df.to_excel(output_path, index=False, engine='openpyxl')
            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='mmbench_dev_20230712')
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
    parser.add_argument('--sep', type=str, default="__sep__")
    args = parser.parse_args()
    args.root_prefix = re.sub('test', '', args.root_prefix)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    for ds_name in args.datasets:
        root_path = ds_collections[ds_name]['root']
        root_dir, root_name = '/'.join(root_path.split('/')[:-1]), root_path.split('/')[-1]
        root_path = f"{root_dir}/{args.root_prefix}{root_name}"
        ds_collections[ds_name]['root'] = root_path
        for ky2, vl2 in ds_collections[ds_name].items():
            if isinstance(vl2, str) and '/' in vl2:
                ds_collections[ds_name][ky2] = f'{os.getenv("HOME")}/{vl2}'
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

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
    print(f'[test] max_num: {args.max_num}')

    prompt = {
        'en': "Answer with the option's letter from the given choices directly.",
        'cn': '请直接回答选项字母。'
    }
    evaluate_chat_model()
