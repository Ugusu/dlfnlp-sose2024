import ast
import copy
import fnmatch
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from enum import Enum
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union, Any
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import importlib_metadata
import numpy as np
import requests
import torch
import torch.nn as nn
from filelock import FileLock
from huggingface_hub.hf_api import HfFolder
from torch import Tensor
from tqdm.auto import tqdm
from torch.nn import functional as F

import re
from num2words import num2words
from transformers import AutoTokenizer
from words2num import w2n as word2num

import spacy

from rouge import Rouge
from collections import Counter

__version__ = "4.0.0"
_torch_version = importlib_metadata.version("torch")

hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "transformers")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}
HUGGINGFACE_CO_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"


def is_torch_available():
    return True


def is_tf_available():
    return False


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def http_get(
    url: str,
    temp_file: BinaryIO,
    proxies=None,
    resume_size=0,
    headers: Optional[Dict[str, str]] = None,
):
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=False,
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def hf_bucket_url(
    model_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    mirror=None,
) -> str:
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
        legacy_format = "/" not in model_id
        if legacy_format:
            return f"{endpoint}/{model_id}-{filename}"
        else:
            return f"{endpoint}/{model_id}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = "Bearer {}".format(use_auth_token)
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a huggingface token was not found."
            )
        headers["authorization"] = "Bearer {}".format(token)

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(
                url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout
            )
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        os.replace(temp_file.name, cache_path)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if (
            os.path.isdir(output_path_extracted)
            and os.listdir(output_path_extracted)
            and not force_extract
        ):
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(
                    "Archive format of {} could not be identified".format(output_path)
                )

        return output_path_extracted

    return output_path


def get_parameter_dtype(parameter: Union[nn.Module]):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def get_extended_attention_mask(attention_mask: Tensor, dtype) -> Tensor:
    # attention_mask [batch_size, seq_length]
    assert attention_mask.dim() == 2
    # [batch_size, 1, 1, seq_length] for multi-head attention
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class SwiGLU1(nn.Module):
    def __init__(self):
        super(SwiGLU1, self).__init__()

    def forward(self, x):
        # Split input tensor into two halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        if x.shape[0] == 1:
            output = F.silu(x)
        else:
            output = x1 * F.silu(x2)
        if output.shape != x.shape:
            output = output.view(x.shape[1::])
        return output


class SwiGLU2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU2, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x))) * torch.sigmoid(self.gate(x))

class GELU(nn.Module):
    def forward(self, x):
        print(x.shape)
        output = 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.tensor(torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        print(output.shape)
        return output


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.silu = nn.SiLU()  # Swish activation function

    def forward(self, x, gate):
        return self.silu(x) * gate


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config):
        super(SwiGLUFeedForward, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.linear2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.gate_linear = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.shape[-1]
        gate = nn.Linear(input_shape, self.config.decoder_ffn_dim)(x)
        x = nn.Linear(input_shape, self.config.decoder_ffn_dim)(x)
        x = self.swiglu(x, gate)
        x = self.dropout(x)
        x = nn.Linear(self.config.decoder_ffn_dim, input_shape)(x)
        x = self.dropout(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return self.weight * x

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return (x * cos) + (rotate_half(x) * sin)

def nums2word_word2nums(sentence: str, input_type: str = "digits", num_tag: bool = True) -> str:
    """
    Replace numbers with letters.
    :param sentence: Input sentence
    :param input_type: Type of input sentence (digits or words) - words only works if there are <num> </num> tokens
    :param num_tag: Tag the numbers with <num> </num> tokens
    """

    # TODO not perfect can be improved
    """def create_number_dictionary():
        number_dict = {}

        # Single digits and teens
        for i in range(0, 20):
            number_dict[num2words(i)] = str(i)

        # Tens
        for i in range(20, 101, 10):
            number_dict[num2words(i)] = str(i)

        # Hundreds
        for i in range(100, 1001, 100):
            number_dict[num2words(i)] = str(i)

        # Large numbers
        large_numbers = [1000, 1000000, 1000000000, 1000000000000]
        for num in large_numbers:
            number_dict[num2words(num)] = str(num)

        # Special cases
        number_dict['a'] = '1'
        number_dict['an'] = '1'
        number_dict['zero'] = '0'
        number_dict['oh'] = '0'
        number_dict['point'] = '.'
        number_dict['negative'] = '-'
        number_dict['million'] = 'million'

        return number_dict"""

    text = sentence
    def digits_to_words(match):

        number = match.group(0)
        #print(' I am the match: ', number)
        # if there is a string in the number, it probably has a scale
        scale = ''
        splits = number.split()
        if len(splits) > 1:
            number = splits[0]
            scale = ' ' + splits[1]

        if '.' in number:
            integer_part, decimal_part = number.split('.')
            integer_words = num2words(int(integer_part))
            decimal_words = ' point ' + num2words(int(decimal_part))
            word_num = integer_words + decimal_words
            if num_tag:
                # add a special token so that we can distinguish between the synthetic numbers and the rest of the text
                return '<num> ' + word_num + scale + ' </num>'
            else:
                return word_num + scale
        else:
            # add a special token so that we can distinguish between the synthetic numbers and the rest of the text
            word_num = num2words(int(number))
            if num_tag:
                return '<num> ' + word_num + scale + ' </num>'
            else:
                return word_num + scale

    def words_to_digits(text):
        # find the tagged numbers
        tagged_numbers = re.findall(r'<num> (.*?) <', text)
        for num in tagged_numbers:
            # convert the number to digits
            digit = word2num(num, 'en')
            #print('I am the digit: ', digit)
            # replace the tagged number with the digit
            text = text.replace(f'<num> {num} </num>', str(digit))

        return text

    if input_type == "digits":
        text = re.sub(r'\b-?\d+(?:\.\d+)?\b', digits_to_words, sentence)
        print("digit_to_word output: ", text)

    elif input_type == "words":
        text = words_to_digits(sentence)
        print("word_to_digit output: ", text)

    return text


#nums2word_word2nums("I have 25 apples and 30.05 oranges. The temperature is -2.3 degrees Celsius.", input_type="digits")
#nums2word_word2nums("I have <num> twenty-five </num> apples and <num> thirty point zero five </num> oranges. The temperature is -<num> two point three </num> degrees Celsius.", input_type="words")


def tag_pos(sentence: str):
    """
    Using spacy to tag the POS of the sentence
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    w_tokens = []
    w_pos = []

    for token in doc:
        w_tokens.append(token.text)
        w_pos.append(token.pos_)

    return w_tokens, w_pos



def get_important_tokens(sentence1_tokenized: str, sentence2_tokenized: str, all_sentence1_tokens: list) -> List[Any]:
    """
    Find the two most important tokens from sentence2 using ROUGE metrics.

    This function calculates the importance of tokens based on their ROUGE scores and frequency.
    It considers tokens with length greater than 3 and returns the two most important ones from sentence2.

    Args:
        sentence1_tokenized (str): The first tokenized sentence as a string representation of a list.
        sentence2_tokenized (str): The second tokenized sentence as a string representation of a list.

    Returns:
        List[Any]: A list containing up to two most important tokens from sentence2.

    Raises:
        ValueError: If there are fewer than two valid tokens (length > 3) in sentence2.
    """
    # Convert and clean the tokenized sentences
    sentence1_tokenized = ast.literal_eval(sentence1_tokenized)
    sentence2_tokenized = ast.literal_eval(sentence2_tokenized)

    # Combine both sentences for reference, but only consider sentence2 for valid tokens
    all_tokens = all_sentence1_tokens + sentence2_tokenized
    valid_tokens = [token for token in sentence2_tokenized if len(token) > 3]

    if len(valid_tokens) < 3:
        raise ValueError("Insufficient number of valid tokens (length > 4) in sentence2.")

    # Count occurrences of each valid token in sentence2
    token_counts = Counter(valid_tokens)

    # Calculate ROUGE scores for each token
    rouge = Rouge()
    reference = " ".join(all_tokens)
    token_scores = {}

    for token in set(valid_tokens):
        hypothesis = token
        scores = rouge.get_scores(hypothesis, reference)
        # Use ROUGE-l F1-score as the primary metric
        token_scores[token] = scores[0]['rouge-l']['f']

    # Combine ROUGE scores with inverse frequency for final importance score
    token_importance = {
        token: (score / token_counts[token])
        for token, score in token_scores.items()
    }

    # Sort tokens by importance score (descending) and return top 2
    important_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
    important_tokens = [token for token, _ in important_tokens[:2]]

    excluded_tokens = set(sentence1_tokenized + important_tokens)
    #print("Excluded tokens: ", excluded_tokens)
    handpicked_token = [token for token in valid_tokens if token not in excluded_tokens]
    if len(handpicked_token) > 1:
        # take two random tokens
        handpicked_token = random.sample(handpicked_token, 2)
        important_tokens.extend(handpicked_token)
        #print("Handpicked token: ", handpicked_token)
    elif len(handpicked_token) > 0:
        handpicked_token = random.choice(handpicked_token)
        important_tokens.append(handpicked_token)
    else:
        pass

    return important_tokens



'''def mask_important_tokens(tokens: list, important_tokens: list) -> str:
    """
    Mask the important tokens in the sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    masked_tokens = [tokenizer.mask_token if token in important_tokens else token for token in tokens]
    return tokenizer.convert_tokens_to_string(masked_tokens)'''

class PoolingStrategy(Enum):
    CLS = "cls"
    AVERAGE = "average"
    MAX = "max"
    ATTENTION = "attention"

class OptimizerType(Enum):
    ADAMW = "adamw"
    SOPHIA = "sophia"