from FlagEmbedding import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker
import cohere
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import json
import datetime
import random
from dotenv import load_dotenv
import os

load_dotenv()


def round_up_list(lst):
    return [round(x, 4) for x in lst]


def avg_list(lst):
    return sum(lst) / len(lst)


def diff_list(lst1, lst2):
    return round_up_list([x - y for x, y in zip(lst1, lst2)])


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class Request:
    docs: list[str]
    query: str

    def __init__(self, docs: list[str], query: str):
        self.docs = docs
        self.query = query

    def to_dict(self):
        return {"docs": self.docs, "query": self.query}


class Response:
    scores: list[list[float]]
    rerank_times: list[float]
    init_time: float

    def __init__(self):
        self.scores = []
        self.rerank_times = []
        self.init_time = 0.0

    def to_dict(self):
        return {
            "scores": self.scores,
            "rerank_times": self.rerank_times,
            "init_time": self.init_time,
        }


def gte_base_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    gte_model = "Alibaba-NLP/gte-multilingual-reranker-base"
    gte_tokenizer = AutoTokenizer.from_pretrained(gte_model)
    gte_base = AutoModelForSequenceClassification.from_pretrained(
        gte_model, trust_remote_code=True, torch_dtype=torch.float16
    )
    gte_base.eval()
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        gte_inputs = gte_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        gte_scores = gte_base(**gte_inputs, return_dict=True).logits.view(-1).float()
        gte_scores = round_up_list(sigmoid(gte_scores).tolist())
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(gte_scores)
        result.rerank_times.append(rerank_time)

    return result


def jina_v2_base_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    jina_v2_base = AutoModelForSequenceClassification.from_pretrained(
        "jinaai/jina-reranker-v2-base-multilingual",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    jina_v2_base.to("cuda")
    jina_v2_base.eval()
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        jina_v2_base_scores = round_up_list(
            jina_v2_base.compute_score(pairs, max_length=1024)
        )
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(jina_v2_base_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_base_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_base = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_base_scores = round_up_list(bge_base.compute_score(pairs, normalize=True))
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_base_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_large_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_large = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_large_scores = round_up_list(bge_large.compute_score(pairs, normalize=True))
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_large_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_v2_m3_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_v2_m3 = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_v2_m3_scores = round_up_list(bge_v2_m3.compute_score(pairs, normalize=True))
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_v2_m3_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_v2_gemma_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_v2_gemma = FlagLLMReranker("BAAI/bge-reranker-v2-gemma", use_fp16=True)
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_v2_gemma_scores = round_up_list(
            bge_v2_gemma.compute_score(pairs, normalize=True)
        )
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_v2_gemma_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_v2_minicpm_layerwise_20_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_v2_minicpm_layerwise = LayerWiseFlagLLMReranker(
        "BAAI/bge-reranker-v2-minicpm-layerwise", use_fp16=True
    )
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_v2_minicpm_layerwise_scores = round_up_list(
            bge_v2_minicpm_layerwise.compute_score(
                pairs,
                normalize=True,
                cutoff_layers=[20],
            )
        )
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_v2_minicpm_layerwise_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_v2_minicpm_layerwise_28_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_v2_minicpm_layerwise = LayerWiseFlagLLMReranker(
        "BAAI/bge-reranker-v2-minicpm-layerwise", use_fp16=True
    )
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_v2_minicpm_layerwise_scores = round_up_list(
            bge_v2_minicpm_layerwise.compute_score(
                pairs, normalize=True, cutoff_layers=[28]
            )
        )
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_v2_minicpm_layerwise_scores)
        result.rerank_times.append(rerank_time)

    return result


def bge_v2_minicpm_layerwise_40_rerank(requests: list[Request]) -> Response:
    init_start_time = time.time()
    bge_v2_minicpm_layerwise = LayerWiseFlagLLMReranker(
        "BAAI/bge-reranker-v2-minicpm-layerwise", use_fp16=True
    )
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    result: Response = Response()
    result.init_time = init_time

    for request in requests:
        docs = request.docs
        query = request.query
        pairs = []
        for doc in docs:
            pairs.append([query, doc])
        rerank_start_time = time.time()
        bge_v2_minicpm_layerwise_scores = round_up_list(
            bge_v2_minicpm_layerwise.compute_score(
                pairs, normalize=True, cutoff_layers=[40]
            )
        )
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.scores.append(bge_v2_minicpm_layerwise_scores)
        result.rerank_times.append(rerank_time)

    return result


def cohere_v35_rerank(requests: list[Request]) -> Response:
    co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
    result: Response = Response()

    for request in requests:
        query = request.query
        docs = request.docs
        rerank_start_time = time.time()
        cohere_result = co.rerank(
            model="rerank-v3.5", query=query, documents=docs, top_n=5
        )
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time
        result.rerank_times.append(rerank_time)
        cohere_v35_scores = []
        for r in sorted(cohere_result.results, key=lambda x: x.index):
            cohere_v35_scores.append(r.relevance_score)
        cohere_v35_scores = round_up_list(cohere_v35_scores)
        result.scores.append(cohere_v35_scores)

    return result


def main():
    with open("requests.json", "r") as f:
        requests_json = json.load(f)

    all_requests = [
        Request(request["docs"], request["query"]) for request in requests_json
    ]

    all_rerankers = {
        "gte_base": gte_base_rerank,
        "jina_v2_base": jina_v2_base_rerank,
        "bge_base": bge_base_rerank,
        "bge_large": bge_large_rerank,
        "bge_v2_m3": bge_v2_m3_rerank,
        "bge_v2_gemma": bge_v2_gemma_rerank,
        "bge_v2_minicpm_layerwise_20": bge_v2_minicpm_layerwise_20_rerank,
        "bge_v2_minicpm_layerwise_28": bge_v2_minicpm_layerwise_28_rerank,
        "bge_v2_minicpm_layerwise_40": bge_v2_minicpm_layerwise_40_rerank,  # This likely broken
        "cohere_v35": cohere_v35_rerank,
    }
    reranker_items = list(all_rerankers.items())
    random.shuffle(reranker_items)
    all_rerankers = dict(reranker_items)

    all_responses: dict[str, Response] = {}
    for reranker_name, reranker in all_rerankers.items():
        all_responses[reranker_name] = reranker(all_requests)

    all_responses_json = {k: v.to_dict() for k, v in all_responses.items()}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"responses_{timestamp}.json", "w") as f:
        json.dump(all_responses_json, f, indent=2)


if __name__ == "__main__":
    main()
