import random
import copy
from pathlib import Path
from scripts.utils_run import FLORES101_CONVERT
from datasets import load_dataset
from tqdm import tqdm
from timeit import default_timer as timer
import os

class MTTask:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 testset: str,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.language_pair = f"{src_lang}-{tgt_lang}"
        self.testset = testset
        base_out_dir = Path(__file__).parent / "out"
        print(base_out_dir)
        assert base_out_dir.exists()
        self.out_dir = base_out_dir / self.testset
        self.out_dir.mkdir(exist_ok=True)

        self.out_dir = self.out_dir / self.language_pair
        self.out_dir.mkdir(exist_ok=True)
        self.load_converter = FLORES101_CONVERT

    def __str__(self):
        return f"{self.testset}-{self.src_lang}-{self.tgt_lang}"

    def evaluate(self,
                 translation_method: callable,
                 type='direct',
                 source_contrastive=1,
                 source_weight=None,
                 language_contrastive=None,
                 language_weight=None,
                 prompt_contrastive=None,
                 prefix=None,
                 small_dev=False) -> Path:
        assert type in {'direct', 'contrastive'}
        start_time = timer()

        if not os.path.isfile(str(self.out_dir)+"/"+"ref.text"):
            target_sentences = load_dataset('gsarti/flores_101', self.load_converter[self.tgt_lang])['devtest'][
                'sentence']
            with open(str(self.out_dir) + "/" + "ref.txt", 'w') as f:
                f.write("\n".join(target_sentences))
        if not os.path.isfile(str(self.out_dir)+"/"+"src.text"):
            target_sentences = load_dataset('gsarti/flores_101', self.load_converter[self.src_lang])['devtest'][
                'sentence']
            with open(str(self.out_dir) + "/" + "src.txt", 'w') as f:
                f.write("\n".join(target_sentences))

        ## load FLORES dataset
        source_sentences = load_dataset('gsarti/flores_101',self.load_converter[self.src_lang])['devtest']['sentence']
        if small_dev:
            assert isinstance(source_sentences, list)
            source_sentences = source_sentences[:5]

        if type == 'direct':
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            source_sentences=source_sentences,
            )
        elif type == 'contrastive':
            multi_source_sentences = [source_sentences]
            src_weights = [1]
            tgt_langs=[self.tgt_lang]
            src_langs=[self.src_lang]
            prompt_templates=[None]

            # randomly shuffled input to suppress hallucinations
            if source_contrastive:
                assert source_weight is not None, "Must set source_weight"
                for i in range(source_contrastive):
                    shuffled_sentences = copy.copy(source_sentences)
                    random.shuffle(shuffled_sentences)
                    multi_source_sentences.append(shuffled_sentences)
                    src_weights.append(source_weight/source_contrastive)
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)
                    prompt_templates.append(None)

            # input with wrong target language indicator to suppress off-target translation
            if language_contrastive:
                assert language_weight is not None, "Must set a language weight"
                for offtarget in language_contrastive:
                    # ignore contrastive variants that are identical to true translation direction
                    if offtarget == self.tgt_lang:
                        continue
                    # don't create contrastive variant for src language if language is already listed (avoid duplicates)
                    if offtarget == 'src' and self.src_lang in language_contrastive:
                        continue
                    multi_source_sentences.append(source_sentences)
                    src_weights.append(language_weight)
                    if offtarget == 'src':
                        tgt_langs.append(self.src_lang)
                    else:
                        tgt_langs.append(offtarget)
                    src_langs.append(self.src_lang)
                    prompt_templates.append(None)
            if prompt_contrastive:
                for template_weight, off_template in prompt_contrastive:
                    multi_source_sentences.append(source_sentences)
                    src_weights.append(template_weight)
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)
                    prompt_templates.append(off_template)

            translations = []
            for pair in tqdm(list(zip(*multi_source_sentences))):
                translation = translation_method(
                    src_langs=src_langs,
                    tgt_langs=tgt_langs,
                    src_weights=src_weights,
                    multi_source_sentences=pair,
                    prompt_templates=prompt_templates,
                    )
                translations.append(translation)
        else:
            raise NotImplementedError
        file_name = ""
        if prefix:
            file_name = f"{prefix}-"
        if small_dev:
            file_name = file_name+'dev-'
        if type == 'direct':
            file_name = file_name+'direct'
        elif type == 'contrastive':
            if prompt_contrastive:
                file_name = file_name + '-'.join([
                    'contrastive',
                    'prompt',
                ])
            else:
                file_name = file_name+'-'.join([
                    'contrastive',
                    f'src_cnt:{source_contrastive}',
                    f'src_w:{source_weight}',
                    f'lang_cnt:{language_contrastive}',
                    f'lang_w:{language_weight}',
                ])
        else:
            raise NotImplementedError
        end_time = timer()
        elapsed_seconds = end_time - start_time
        file_name = str(self.out_dir) + "/" + file_name + ".txt"
        if os.path.isfile(file_name):
            i = 2
            new_file_name = file_name + f"({i})"
            while os.path.isfile(new_file_name):
                i = i+1
                new_file_name = file_name + f"({i})"
            file_name = new_file_name
        side_file = file_name + ".run.txt"
        with open(side_file, 'w') as f:
            f.write(f"execution time (s):{elapsed_seconds}")
        with open(file_name, 'w') as f:
            f.write("\n".join(translations))
        return Path(f.name)
